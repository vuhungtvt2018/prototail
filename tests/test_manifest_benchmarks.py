"""Synthetic images only: no benchmark downloads or training are necessary."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

source_root = Path(__file__).resolve().parents[1] / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

import numpy as np
from PIL import Image

from prototail_gcd.data.manifest_gcd import ManifestGCDAdapter


class ManifestBenchmarksTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest_dir = self.root / "manifests"
        self.manifest_dir.mkdir()
        self.train = []
        self.eval = []
        for class_id in range(4):
            for i in range(3):
                path = f"train/{class_id}/image_{i}.jpg"
                dest = self.root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (24, 24), (class_id * 45, i * 50, 0)).save(dest)
                self.train.append({"path": path, "category_id": class_id})
            path = f"val/{class_id}/image.jpg"
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), (class_id * 45, 10, 0)).save(dest)
            self.eval.append({"path": path, "category_id": class_id})
        shas = {}
        for name, rows in (("train", self.train), ("eval", self.eval)):
            file = self.manifest_dir / f"{name}.jsonl"
            file.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            shas[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        self.meta = {
            "dataset_id": "mock", "known_ids": [1, 0], "novel_ids": [3, 2],
            "labeled_train_indices": [0, 3],
            "train_sha256": shas["train"], "eval_sha256": shas["eval"],
        }
        self._write_meta()

    def _write_meta(self):
        (self.manifest_dir / "split.json").write_text(json.dumps(self.meta), encoding="utf-8")

    def _build(self):
        return ManifestGCDAdapter("mock", 4, 2).build({
            "data": {
                "root": str(self.root), "manifest_dir": str(self.manifest_dir),
                "image_size": 8, "eval_batch_size": 4,
            },
            "optimization": {"num_views": 2, "batch_size": 4, "num_workers": 0},
        })

    def test_masked_targets_distribution_and_label_mapping(self):
        bundle = self._build()
        self.assertEqual(bundle.num_classes, 4)
        self.assertEqual(bundle.known_class_ids, (0, 1))
        self.assertEqual(bundle.class_counts, (3, 3, 3, 3))
        rows = bundle.train_loader.dataset
        self.assertEqual(len(rows), 12)
        image_views, target, sample_id, is_labeled = rows[0]
        self.assertTrue(is_labeled)
        self.assertEqual(target, 1)  # Original class 0 mapped after known original class 1
        self.assertEqual(sample_id, 0)
        self.assertEqual(len(image_views), 2)
        self.assertEqual(tuple(image_views[0].shape), (3, 8, 8))
        self.assertEqual(rows[2][1], -1)
        dist = next(iter(bundle.evaluation_loaders["distribution"]))
        self.assertEqual(dist.known_targets.tolist()[:4], [1, 0, -1, -1])
        self.assertEqual(dist.labeled_mask.tolist()[:4], [True, True, False, False])
        evaluation = next(iter(bundle.evaluation_loaders["test"]))
        self.assertEqual(evaluation[1].tolist(), [1, 0, 3, 2])
        unlabeled_eval = next(iter(bundle.evaluation_loaders["unlabeled_train"]))
        self.assertTrue(all(int(y) >= 0 for y in unlabeled_eval[1]))

    def test_unknown_class_labels_cannot_enter_labeled_training(self):
        self.meta["labeled_train_indices"] = [0, 6]
        self._write_meta()
        with self.assertRaisesRegex(ValueError, "Novel ground truth"):
            self._build()

    def test_manifest_is_immutable_after_split_freeze(self):
        path = self.manifest_dir / "train.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self._build()

    def test_sample_ids_unique_across_labeled_unlabeled(self):
        bundle = self._build()
        dataset = bundle.train_loader.dataset
        ids = [dataset[i][2] for i in range(len(dataset))]
        self.assertEqual(set(ids), set(range(12)))


if __name__ == "__main__":
    unittest.main()
