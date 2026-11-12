import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prototail_gcd.methods.prototail import PartPrototypeBank, ProtoTailPlugin


class FixTailV3Tests(unittest.TestCase):
    def test_target_capacities_log_min_max(self):
        bank = PartPrototypeBank(num_classes=5, num_slots=3, feature_dim=16)
        freqs = torch.tensor([10.0, 20.0, 50.0, 100.0, 200.0])
        targets = bank.target_capacities(freqs)
        self.assertEqual(targets.shape, (5,))
        # Min frequency class must have target capacity 1.0
        self.assertAlmostEqual(targets[0].item(), 1.0, places=4)
        # Max frequency class must have target capacity 3.0 (num_slots)
        self.assertAlmostEqual(targets[-1].item(), 3.0, places=4)
        # Monotonically increasing
        for i in range(4):
            self.assertLess(targets[i].item(), targets[i + 1].item())

    def test_warm_start_from_estimated_support(self):
        bank = PartPrototypeBank(num_classes=5, num_slots=3, feature_dim=16)
        freqs = torch.tensor([10.0, 20.0, 50.0, 100.0, 200.0])
        initial_gates = bank.gates().clone()
        bank.warm_start_from_estimated_support(freqs)
        new_gates = bank.gates()
        # Head class sum of gates should be close to 3.0
        self.assertGreater(new_gates[-1].sum().item(), 2.5)
        # Tail class sum of gates should be close to 1.0
        self.assertLess(new_gates[0].sum().item(), 1.5)
        self.assertFalse(torch.allclose(initial_gates, new_gates))

    def test_shared_parameters_preserved(self):
        plugin = ProtoTailPlugin()

        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.Linear(16, 16) for _ in range(12)
                ])

            def named_parameters(self, prefix="", recurse=True):
                for i, b in enumerate(self.blocks):
                    for n, p in b.named_parameters():
                        yield f"blocks.{i}.{n}", p

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.contrastive_backbone = DummyBackbone()
                self.finetune_from_block = 11

        model = DummyModel()
        plugin.attach(
            model,
            {
                "num_slots": 3,
                "feature_dim": 16,
                "patch_dim": 16,
                "shared_backbone_learning_rate": 0.01,
            },
            num_classes=10,
            feature_dim=16,
        )
        groups = plugin.optimizer_groups(model)
        self.assertIn("part", groups)
        # Backbone fine-tuning lives SOLELY in the baseline optimizer now.
        # Adding the same Parameter objects to the method optimizer gave
        # them two momentum buffers and two updates per batch (double-step
        # bug). The part group must contain ONLY part-module parameters.
        part_params = {
            id(p) for group in groups["part"] for p in group["params"]
        }
        backbone_params = {
            id(p) for p in model.contrastive_backbone.parameters()
        }
        self.assertFalse(part_params & backbone_params)


if __name__ == "__main__":
    unittest.main()
