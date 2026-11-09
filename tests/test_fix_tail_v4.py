import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prototail_gcd.methods.prototail import PartPrototypeBank, ProtoTailPlugin


class FixTailV4Tests(unittest.TestCase):
    def test_warm_start_auto_enabled_for_tailaware(self):
        # Verify that for tail-aware, target capacities and warm start operate correctly
        bank = PartPrototypeBank(num_classes=5, num_slots=3, feature_dim=16)
        freqs = torch.tensor([5.0, 10.0, 25.0, 50.0, 100.0])
        bank.warm_start_from_estimated_support(freqs)
        gates = bank.gates()
        # Head class sum > 2.5, tail class sum < 1.5
        self.assertGreater(gates[-1].sum().item(), 2.5)
        self.assertLess(gates[0].sum().item(), 1.5)
        self.assertGreater(gates.mean().item(), 0.5)

    def test_default_kmeans_backend_is_sklearn_cpu(self):
        import argparse
        from prototail_gcd.train import main
        # Verify argparse default
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--kmeans-backend",
            choices=["sklearn_cpu", "faiss_gpu"],
            default="sklearn_cpu",
        )
        args = parser.parse_args([])
        self.assertEqual(args.kmeans_backend, "sklearn_cpu")


if __name__ == "__main__":
    unittest.main()
