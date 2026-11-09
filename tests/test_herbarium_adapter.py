"""Unit tests for Herbarium 19 Long-Tail Adapter and splits."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prototail_gcd.contracts import UNKNOWN_TARGET
from prototail_gcd.data.herbarium19 import HerbariumLongTailAdapter, resolve_herbarium_image_dir


class HerbariumLongTailAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_root = Path(r"D:\data\Herbarium 19")
        cls.splits_root = Path(__file__).resolve().parents[1] / "data_uq_idxs"
        cls.has_data = cls.data_root.is_dir()

    def test_split_files_exist_for_all_imb_ratios(self):
        for imb in [10, 20, 50, 100]:
            folder = self.splits_root / f"herbarium19_k341_imb{imb}"
            self.assertTrue(folder.is_dir(), f"Missing split folder: {folder}")
            self.assertTrue((folder / "l_k_uq_idxs.pt").is_file())
            self.assertTrue((folder / "unl_k_uq_idxs.pt").is_file())
            self.assertTrue((folder / "unl_unk_uq_idxs.pt").is_file())
            self.assertTrue((folder / "meta.json").is_file())
            self.assertTrue((folder / "split_meta.json").is_file())

            l_k = torch.load(folder / "l_k_uq_idxs.pt").numpy()
            unl_k = torch.load(folder / "unl_k_uq_idxs.pt").numpy()
            unl_unk = torch.load(folder / "unl_unk_uq_idxs.pt").numpy()

            # Zero overlap checks
            self.assertEqual(len(set(l_k) & set(unl_k)), 0)
            self.assertEqual(len(set(l_k) & set(unl_unk)), 0)
            self.assertEqual(len(set(unl_k) & set(unl_unk)), 0)

            # Positive non-empty splits
            self.assertGreater(len(l_k), 1000)
            self.assertGreater(len(unl_k), 1000)
            self.assertGreater(len(unl_unk), 2000)

    def test_adapter_build_and_batches(self):
        if not self.has_data:
            self.skipTest("Local D:\\data\\Herbarium 19 not found on this machine")

        adapter = HerbariumLongTailAdapter()
        config = {
            "data": {
                "root": str(self.data_root),
                "splits_root": str(self.splits_root),
                "image_size": 224,
                "crop_pct": 0.875,
                "eval_batch_size": 128,
                "protocol": {
                    "known_class_count": 341,
                    "imbalance_ratio": 20,
                },
            },
            "optimization": {
                "batch_size": 32,
                "num_workers": 0,
                "num_views": 2,
            },
        }

        bundle = adapter.build(config)
        self.assertEqual(bundle.num_classes, 683)
        self.assertEqual(len(bundle.known_class_ids), 341)

        batch = next(iter(bundle.train_loader))
        self.assertEqual(len(batch.views), 2)
        self.assertEqual(batch.views[0].shape, (32, 3, 224, 224))
        self.assertEqual(batch.known_targets.shape, (32,))

        # Verify labeled mask aligns with known targets
        is_labeled = batch.labeled_mask
        self.assertTrue(torch.all(batch.known_targets[~is_labeled] == UNKNOWN_TARGET))
        self.assertTrue(torch.all(batch.known_targets[is_labeled] >= 0))
        self.assertTrue(torch.all(batch.known_targets[is_labeled] < 341))


if __name__ == "__main__":
    unittest.main()
