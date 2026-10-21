import importlib.util
import unittest
from importlib import import_module
from pathlib import Path
import sys


HAS_SIMGCD_DEPS = all(
    importlib.util.find_spec(package_name) is not None
    for package_name in ("numpy", "torch")
)


@unittest.skipUnless(
    HAS_SIMGCD_DEPS,
    "Install the SimGCD optional dependencies",
)
class SimGCDAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        cls.torch = import_module("torch")
        cls.contracts = import_module("prototail_gcd.contracts")
        cls.adapter_module = import_module(
            "prototail_gcd.baselines.simgcd.adapter"
        )

    def test_forward_preserves_two_views_as_named_student_branch(self):
        torch = self.torch

        class TinyNetwork:
            def __call__(self, images):
                batch_size = images.shape[0]
                features = torch.ones(batch_size, 8)
                projection = torch.ones(batch_size, 4)
                logits = torch.ones(batch_size, 3)
                patches = torch.ones(batch_size, 2, 8)
                return features, projection, logits, patches

        batch = self.contracts.TrainingBatch(
            views=(torch.ones(2, 3), torch.zeros(2, 3)),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([11, 12]),
        )

        adapter = self.adapter_module.SimGCDAdapter()
        output = adapter.forward(TinyNetwork(), batch)

        self.assertEqual(output.branches["student"].logits.shape, (4, 3))
        self.assertEqual(output.auxiliary["num_views"], 2)
        self.assertIsNone(output.branches["student"].patch_tokens)

    def test_forward_exposes_patch_tokens_when_prototail_is_attached(self):
        torch = self.torch

        class TinyNetwork:
            prototail = object()

            def __call__(self, images):
                batch_size = images.shape[0]
                return (
                    torch.ones(batch_size, 8),
                    torch.ones(batch_size, 4),
                    torch.ones(batch_size, 3),
                    torch.ones(batch_size, 2, 8),
                )

        batch = self.contracts.TrainingBatch(
            views=(torch.ones(2, 3), torch.zeros(2, 3)),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([11, 12]),
        )

        output = self.adapter_module.SimGCDAdapter().forward(
            TinyNetwork(),
            batch,
        )

        self.assertEqual(
            output.branches["student"].patch_tokens.shape,
            (4, 2, 8),
        )

    def test_native_loss_weights_match_simgcd_objective(self):
        torch = self.torch
        batch_size = 4
        class_count = 5
        projection = torch.randn(batch_size * 2, 12, requires_grad=True)
        logits = torch.randn(batch_size * 2, class_count, requires_grad=True)
        batch = self.contracts.TrainingBatch(
            views=(torch.empty(batch_size, 1), torch.empty(batch_size, 1)),
            known_targets=torch.tensor([0, 1, -1, -1]),
            labeled_mask=torch.tensor([True, True, False, False]),
            sample_ids=torch.arange(batch_size),
        )
        config = {
            "student_temp": 0.1,
            "memax_weight": 2.0,
            "contrastive_temp": 1.0,
            "supervised_contrastive_temp": 0.07,
            "sup_weight": 0.35,
        }
        context = {
            "batch_size": batch_size,
            "num_views": 2,
            "epoch": 0,
            "cluster_criterion": lambda student, teacher, epoch: (
                student - teacher
            ).square().mean(),
            "teacher_logits": torch.randn_like(logits),
        }

        losses = self.adapter_module.compute_simgcd_losses(
            projection=projection,
            logits=logits,
            batch=batch,
            context=context,
            config=config,
        )
        expected = (1 - config["sup_weight"]) * losses["cluster_objective"]
        expected = expected + config["sup_weight"] * losses["classification"]
        expected = expected + (
            (1 - config["sup_weight"]) * losses["contrastive"]
        )
        expected = expected + (
            config["sup_weight"] * losses["supervised_contrastive"]
        )

        torch.testing.assert_close(losses["total"], expected)
        losses["total"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(projection.grad)


if __name__ == "__main__":
    unittest.main()
