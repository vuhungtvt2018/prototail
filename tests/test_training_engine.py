import importlib.util
import unittest
from importlib import import_module
from pathlib import Path
import sys


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Install the PyTorch project dependencies")
class TrainingEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        cls.torch = import_module("torch")
        cls.contracts = import_module("prototail_gcd.contracts")
        cls.engine_module = import_module("prototail_gcd.training.engine")

    def test_batch_step_updates_baseline_parameters(self):
        torch = self.torch
        contracts = self.contracts

        class TinyBaseline:
            def forward(self, model, batch):
                logits = model(batch.views[0])
                return contracts.ModelOutput(
                    branches={
                        "student": contracts.BranchOutput(
                            global_features=logits,
                            logits=logits,
                        )
                    }
                )

            def compute_losses(self, output, batch, context, config):
                logits = output.branches["student"].logits
                target = torch.as_tensor(
                    batch.known_targets,
                    dtype=logits.dtype,
                )
                total = (logits.reshape(-1) - target).square().mean()
                return {"total": total}

        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        engine = self.engine_module.TrainingEngine(
            model=model,
            baseline=TinyBaseline(),
            optimizers={"main": optimizer},
            baseline_config={},
        )
        batch = contracts.TrainingBatch(
            views=(torch.ones(2, 2),),
            known_targets=torch.tensor([1.0, 0.0]),
            labeled_mask=torch.tensor([True, True]),
            sample_ids=torch.tensor([1, 2]),
        )
        before = model.weight.detach().clone()

        losses = engine.train_batch(batch, epoch=0)

        self.assertIn("total", losses)
        self.assertFalse(torch.equal(before, model.weight.detach()))

    def test_overlapping_optimizers_raise_value_error(self):
        torch = self.torch

        class DummyBaseline:
            pass

        model = torch.nn.Linear(2, 2)
        opt1 = torch.optim.SGD([model.weight], lr=0.1)
        opt2 = torch.optim.SGD([model.weight, model.bias], lr=0.01)

        with self.assertRaisesRegex(ValueError, "shares parameters"):
            self.engine_module.TrainingEngine(
                model=model,
                baseline=DummyBaseline(),
                optimizers={"main": opt1, "part": opt2},
                baseline_config={},
            )

    def test_momentum_teacher_supplies_logits_and_updates(self):
        torch = self.torch
        contracts = self.contracts
        from prototail_gcd.training.momentum_teacher import GlobalEMATeacher

        captured_context = {}

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(4, 4)
                self.projector = torch.nn.Linear(4, 2)
                self.finetune_from_block = 0

            def forward(self, x):
                feats = self.backbone(x)
                logits = self.projector(feats)
                return feats, None, logits, None

        class DummyBaseline:
            def forward(self, model, batch):
                imgs = torch.cat(tuple(batch.views), dim=0)
                _feats, _proj, logits, _patches = model(imgs)
                return contracts.ModelOutput(
                    branches={"student": contracts.BranchOutput(global_features=_feats, logits=logits)}
                )

            def compute_losses(self, output, batch, context, config):
                captured_context.update(context)
                logits = output.branches["student"].logits
                return {"total": logits.sum()}

        model = DummyModel()
        teacher = GlobalEMATeacher(model.backbone, model.projector, m0=0.9)
        initial_teacher_weight = teacher.backbone.weight.clone()

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        engine = self.engine_module.TrainingEngine(
            model=model,
            baseline=DummyBaseline(),
            optimizers={"main": opt},
            baseline_config={},
            teacher=teacher,
            context={"total_epochs": 10},
        )

        batch = contracts.TrainingBatch(
            views=(torch.ones(2, 4), torch.ones(2, 4)),
            known_targets=torch.tensor([0, 1]),
            labeled_mask=torch.tensor([True, True]),
            sample_ids=torch.tensor([1, 2]),
        )

        engine.train_batch(batch, epoch=0)

        self.assertIn("teacher_logits", captured_context)
        self.assertEqual(captured_context["teacher_logits"].shape, (4, 2))
        self.assertFalse(torch.equal(initial_teacher_weight, teacher.backbone.weight))
        for p in teacher.backbone.parameters():
            self.assertFalse(p.requires_grad)

    def test_promotion_store_applies_pseudo_labels_in_batch(self):
        torch = self.torch
        contracts = self.contracts
        from prototail_gcd.pseudo import PromotionStore

        store = PromotionStore(num_classes=3, num_known_classes=2)
        # Sample 2 (unlabeled in batch) is promoted to class 1
        store.add_accepted([(2, 1)])

        captured_batch = {}

        class DummyBaseline:
            def forward(self, model, batch):
                return contracts.ModelOutput(branches={})

            def compute_losses(self, output, batch, context, config):
                captured_batch["batch"] = batch
                return {"total": torch.tensor(0.0, requires_grad=True)}

        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        engine = self.engine_module.TrainingEngine(
            model=model,
            baseline=DummyBaseline(),
            optimizers={"main": opt},
            baseline_config={},
            context={"promotion_store": store},
        )

        batch = contracts.TrainingBatch(
            views=(torch.ones(2, 2),),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([1, 2]),
        )

        engine.train_batch(batch, epoch=0)

        final_batch = captured_batch["batch"]
        # Sample 2 is now supervised with target 1
        self.assertTrue(final_batch.labeled_mask[1].item())
        self.assertEqual(final_batch.known_targets[1].item(), 1)
        # Original labeled mask is preserved
        self.assertFalse(final_batch.original_labeled_mask[1].item())


if __name__ == "__main__":
    unittest.main()
