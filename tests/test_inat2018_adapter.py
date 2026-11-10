"""Unit tests for iNaturalist 2018 Long-Tail Adapter and Split Generator."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prototail_gcd.contracts import UNKNOWN_TARGET
from prototail_gcd.data import get_dataset_adapter
from prototail_gcd.data.inat2018 import INat2018LongTailAdapter, INat2018Dataset
from make_inat2018_splits import generate_inat2018_splits, get_lt_dist


class INat2018AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_adapter_registry(self):
        adapter = get_dataset_adapter("inat2018")
        self.assertIsInstance(adapter, INat2018LongTailAdapter)
        self.assertEqual(adapter.name, "inat2018")

        adapter_alias = get_dataset_adapter("inat")
        self.assertIsInstance(adapter_alias, INat2018LongTailAdapter)

    def test_split_generation_and_adapter_build(self):
        # Create a synthetic mini iNat dataset with 10 classes (5 known, 5 novel)
        n_classes = 10
        k = 5
        images_info = []
        annotations_info = []

        img_id = 1
        for c in range(n_classes):
            n_samples = 10 + c * 5  # Class 0: 10 images, Class 9: 55 images
            cls_dir = self.temp_path / "train_val2018" / f"class_{c}"
            cls_dir.mkdir(parents=True, exist_ok=True)
            for s in range(n_samples):
                fn = f"train_val2018/class_{c}/img_{s}.jpg"
                img_path = self.temp_path / fn
                # Create small dummy image
                img = Image.new("RGB", (32, 32), color=(c * 20, s * 4, 100))
                img.save(img_path)

                images_info.append({
                    "id": img_id,
                    "file_name": fn,
                    "height": 32,
                    "width": 32,
                })
                annotations_info.append({
                    "id": img_id,
                    "image_id": img_id,
                    "category_id": c,
                })
                img_id += 1

        json_meta = {
            "images": images_info,
            "annotations": annotations_info,
            "categories": [{"id": c, "name": f"species_{c}"} for c in range(n_classes)],
        }
        train_json_path = self.temp_path / "train2018.json"
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(json_meta, f)

        # Generate splits using generate_inat2018_splits
        out_splits_dir = self.temp_path / "data_uq_idxs"
        summaries = generate_inat2018_splits(
            json_path=train_json_path,
            output_roots=[out_splits_dir],
            n_total=n_classes,
            k=k,
            img_max=20,
            seed=416,
            imbs=[10, 20],
            include_natural=True,
        )

        self.assertEqual(len(summaries), 3)  # natural, imb10, imb20

        # Verify natural split files
        nat_folder = out_splits_dir / f"inat2018_k{k}_natural"
        self.assertTrue((nat_folder / "l_k_uq_idxs.pt").is_file())
        self.assertTrue((nat_folder / "unl_k_uq_idxs.pt").is_file())
        self.assertTrue((nat_folder / "unl_unk_uq_idxs.pt").is_file())
        self.assertTrue((nat_folder / "meta.json").is_file())

        l_k = torch.load(nat_folder / "l_k_uq_idxs.pt").numpy()
        unl_k = torch.load(nat_folder / "unl_k_uq_idxs.pt").numpy()
        unl_unk = torch.load(nat_folder / "unl_unk_uq_idxs.pt").numpy()

        # Zero overlap
        self.assertEqual(len(set(l_k) & set(unl_k)), 0)
        self.assertEqual(len(set(l_k) & set(unl_unk)), 0)
        self.assertEqual(len(set(unl_k) & set(unl_unk)), 0)
        # In natural split, all samples are included
        self.assertEqual(len(l_k) + len(unl_k) + len(unl_unk), len(images_info))

        # Test building Adapter with synthetic dataset
        adapter = INat2018LongTailAdapter()
        config = {
            "data": {
                "root": str(self.temp_path),
                "splits_root": str(out_splits_dir),
                "image_size": 32,
                "crop_pct": 0.875,
                "eval_batch_size": 16,
                "protocol": {
                    "known_class_count": k,
                    "imbalance_ratio": "natural",
                },
            },
            "optimization": {
                "batch_size": 16,
                "num_workers": 0,
                "num_views": 2,
            },
        }

        bundle = adapter.build(config)
        self.assertEqual(bundle.num_classes, n_classes)
        self.assertEqual(len(bundle.known_class_ids), k)

        # Test batch from train_loader
        batch = next(iter(bundle.train_loader))
        self.assertEqual(len(batch.views), 2)
        self.assertEqual(batch.views[0].shape[0], 16)
        self.assertEqual(batch.views[0].shape[1], 3)
        self.assertEqual(batch.views[0].shape[2], 32)
        self.assertEqual(batch.known_targets.shape[0], 16)
        self.assertEqual(batch.sample_ids.shape[0], 16)
        self.assertEqual(batch.labeled_mask.shape[0], 16)

    def test_on_the_fly_partition_zero_pt_files(self):
        # Create a mini dataset without ANY .pt files
        n_classes = 6
        k = 3
        images_info = []
        annotations_info = []

        img_id = 1
        for c in range(n_classes):
            n_samples = 15
            cls_dir = self.temp_path / "zero_pt" / "train_val2018" / f"class_{c}"
            cls_dir.mkdir(parents=True, exist_ok=True)
            for s in range(n_samples):
                fn = f"train_val2018/class_{c}/img_{s}.jpg"
                img_path = self.temp_path / "zero_pt" / fn
                Image.new("RGB", (32, 32), color=(c * 30, s * 10, 50)).save(img_path)
                images_info.append({"id": img_id, "file_name": fn, "height": 32, "width": 32})
                annotations_info.append({"id": img_id, "image_id": img_id, "category_id": c})
                img_id += 1

        json_meta = {
            "images": images_info,
            "annotations": annotations_info,
            "categories": [{"id": c, "name": f"species_{c}"} for c in range(n_classes)],
        }
        train_json_path = self.temp_path / "zero_pt" / "train2018.json"
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump(json_meta, f)

        # Build Adapter with ZERO .pt files!
        adapter = INat2018LongTailAdapter()
        config = {
            "data": {
                "root": str(self.temp_path / "zero_pt"),
                "image_size": 32,
                "crop_pct": 0.875,
                "eval_batch_size": 16,
                "protocol": {
                    "known_class_count": k,
                },
            },
            "optimization": {
                "batch_size": 16,
                "num_workers": 0,
                "num_views": 2,
            },
        }

        bundle = adapter.build(config)
        self.assertEqual(bundle.num_classes, n_classes)
        self.assertEqual(len(bundle.known_class_ids), k)

        # Ensure batch works
        batch = next(iter(bundle.train_loader))
        self.assertEqual(len(batch.views), 2)
        self.assertEqual(batch.known_targets.shape[0], 16)

    def test_real_splits_in_data_uq_idxs(self):
        splits_dir = Path(__file__).resolve().parents[1] / "data_uq_idxs"
        expected = [
            ("inat2018_k4071_natural", 437513),
            ("inat2018_k4071_imb10", 159103),
            ("inat2018_k4071_imb20", 129212),
            ("inat2018_k4071_imb50", 101930),
            ("inat2018_k4071_imb100", 87820),
        ]
        for folder_name, expected_total in expected:
            folder = splits_dir / folder_name
            self.assertTrue(folder.is_dir(), f"Missing real split folder: {folder}")
            l_k = torch.load(folder / "l_k_uq_idxs.pt").numpy()
            unl_k = torch.load(folder / "unl_k_uq_idxs.pt").numpy()
            unl_unk = torch.load(folder / "unl_unk_uq_idxs.pt").numpy()

            self.assertEqual(len(set(l_k) & set(unl_k)), 0)
            self.assertEqual(len(set(l_k) & set(unl_unk)), 0)
            self.assertEqual(len(set(unl_k) & set(unl_unk)), 0)
            self.assertEqual(len(l_k) + len(unl_k) + len(unl_unk), expected_total)

            with open(folder / "meta.json") as f:
                meta = json.load(f)
            self.assertEqual(meta["num_classes"], 8142)
            self.assertEqual(meta["num_known_classes"], 4071)
            self.assertEqual(len(meta["buckets"]["K"]), 3)
            self.assertEqual(len(meta["buckets"]["U"]), 3)


if __name__ == "__main__":
    unittest.main()
