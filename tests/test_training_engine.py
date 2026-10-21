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


if __name__ == "__main__":
    unittest.main()
