"""Reusable, label-private GCD loader for precomputed benchmark manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from prototail_gcd.contracts import DistributionBatch, TrainingBatch, UNKNOWN_TARGET
from prototail_gcd.data.protocol import DatasetBundle


@dataclass(frozen=True)
class ImageRecord:
    path: str
    category_id: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[ImageRecord]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[ImageRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            relpath = str(item["path"])
            if not relpath or Path(relpath).is_absolute() or ".." in Path(relpath).parts:
                raise ValueError(f"Unsafe relative path in {path}:{line_no}")
            if relpath in seen:
                raise ValueError(f"Duplicate image path in {path}:{line_no}: {relpath}")
            seen.add(relpath)
            rows.append(ImageRecord(relpath, int(item["category_id"])))
    if not rows:
        raise ValueError(f"Empty image manifest: {path}")
    return rows


class _ImageView(Dataset):
    def __init__(
        self,
        root: Path,
        records: Sequence[ImageRecord],
        indices: Sequence[int],
        class_to_id: Mapping[int, int],
        transform: Any,
        *,
        labeled: bool,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.records = records
        self.indices = tuple(int(i) for i in indices)
        self.class_to_id = class_to_id
        self.transform = transform
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, int, int, bool]:
        idx = self.indices[position]
        item = self.records[idx]
        path = (self.root / item.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Image path escapes dataset root: {item.path}")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        image = self.transform(image)
        target = self.class_to_id[item.category_id] if self.labeled else UNKNOWN_TARGET
        return image, target, idx, self.labeled


class _TwoViewTransform:
    def __init__(self, transform: Any, num_views: int) -> None:
        self.transform = transform
        self.num_views = num_views

    def __call__(self, image: Any) -> tuple[torch.Tensor, ...]:
        return tuple(self.transform(image) for _ in range(self.num_views))


class _MergedDataset(Dataset):
    def __init__(self, labeled: Dataset, unlabeled: Dataset) -> None:
        self.labeled, self.unlabeled = labeled, unlabeled

    def __len__(self) -> int:
        return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, index: int) -> Any:
        if index < len(self.labeled):
            return self.labeled[index]
        return self.unlabeled[index - len(self.labeled)]


def _collate_training(rows: Sequence[tuple[Any, int, int, bool]]) -> TrainingBatch:
    num_views = len(rows[0][0])
    return TrainingBatch(
        views=tuple(torch.stack([row[0][v] for row in rows]) for v in range(num_views)),
        known_targets=torch.tensor([row[1] for row in rows], dtype=torch.long),
        sample_ids=torch.tensor([row[2] for row in rows], dtype=torch.long),
        labeled_mask=torch.tensor([row[3] for row in rows], dtype=torch.bool),
    )


def _collate_distribution(rows: Sequence[tuple[Any, int, int, bool]]) -> DistributionBatch:
    return DistributionBatch(
        images=torch.stack([row[0] for row in rows]),
        known_targets=torch.tensor([row[1] for row in rows], dtype=torch.long),
        sample_ids=torch.tensor([row[2] for row in rows], dtype=torch.long),
        labeled_mask=torch.tensor([row[3] for row in rows], dtype=torch.bool),
    )


def _transforms(image_size: int, crop_pct: float) -> tuple[Any, Any]:
    if not (0 < crop_pct <= 1):
        raise ValueError("crop_pct must be in (0,1]")
    interp = transforms.InterpolationMode.BICUBIC
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = transforms.Resize(int(image_size / crop_pct), interpolation=interp)
    train = transforms.Compose([
        resize, transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(), transforms.ToTensor(), normalize,
    ])
    evaluation = transforms.Compose([
        resize, transforms.CenterCrop(image_size), transforms.ToTensor(), normalize,
    ])
    return train, evaluation


class ManifestGCDAdapter:
    """Build a DatasetBundle; unlabeled GT is never exposed through training/distribution."""

    def __init__(self, dataset_id: str, n_classes: int, n_known: int) -> None:
        self.name = dataset_id
        self.n_classes = n_classes
        self.n_known = n_known

    def build(self, config: Mapping[str, Any]) -> DatasetBundle:
        data = config["data"]
        optimization = config["optimization"]
        root = Path(data["root"])
        manifest_dir = Path(data["manifest_dir"])
        train_path = manifest_dir / "train.jsonl"
        eval_path = manifest_dir / "eval.jsonl"
        metadata_path = manifest_dir / "split.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["dataset_id"] != self.name:
            raise ValueError(f"Manifest dataset mismatch: {metadata['dataset_id']} != {self.name}")
        if metadata["train_sha256"] != _sha256(train_path):
            raise ValueError("Training manifest changed after creating split")
        if metadata["eval_sha256"] != _sha256(eval_path):
            raise ValueError("Evaluation manifest changed after creating split")
        known = [int(x) for x in metadata["known_ids"]]
        novel = [int(x) for x in metadata["novel_ids"]]
        ordered_classes = known + novel
        if len(known) != self.n_known or len(ordered_classes) != self.n_classes:
            raise ValueError("Number of known/total classes does not match the benchmark")
        if len(set(ordered_classes)) != len(ordered_classes):
            raise ValueError("Repeated or overlapping known/novel class IDs")
        train_rows = _read_manifest(train_path)
        eval_rows = _read_manifest(eval_path)
        if set(x.category_id for x in train_rows) != set(ordered_classes):
            raise ValueError("Train manifest does not cover exactly the split classes")
        if set(x.category_id for x in eval_rows) != set(ordered_classes):
            raise ValueError("Eval manifest does not cover exactly the split classes")
        labeled_indices = [int(i) for i in metadata["labeled_train_indices"]]
        labeled_set = set(labeled_indices)
        if len(labeled_set) != len(labeled_indices) or not labeled_set:
            raise ValueError("Duplicate or empty labeled split")
        if min(labeled_set) < 0 or max(labeled_set) >= len(train_rows):
            raise ValueError("Out-of-range labeled index")
        if any(train_rows[i].category_id not in set(known) for i in labeled_set):
            raise ValueError("Novel ground truth was included in labeled training split")
        if not set(known).issubset({train_rows[i].category_id for i in labeled_set}):
            raise ValueError("Each known class must have labeled examples")
        unlabeled_indices = [i for i in range(len(train_rows)) if i not in labeled_set]
        if not unlabeled_indices:
            raise ValueError("Unlabeled split is empty")
        class_to_id = {class_id: index for index, class_id in enumerate(ordered_classes)}
        true_train_counts = np.bincount(
            [class_to_id[x.category_id] for x in train_rows], minlength=self.n_classes,
        )
        train_transform, eval_transform = _transforms(
            int(data.get("image_size", 224)), float(data.get("crop_pct", 0.875)),
        )
        num_views = int(optimization.get("num_views", 2))
        if num_views < 2:
            raise ValueError("Expected >=2 training views")
        multiview = _TwoViewTransform(train_transform, num_views)
        labeled_view = _ImageView(root, train_rows, labeled_indices, class_to_id, multiview, labeled=True)
        unlabeled_view = _ImageView(root, train_rows, unlabeled_indices, class_to_id, multiview, labeled=False)
        merged = _MergedDataset(labeled_view, unlabeled_view)
        # Same labeled/unlabeled mass balancing as the current CUB/Cars adapter.
        weights = torch.ones(len(merged), dtype=torch.double)
        weights[len(labeled_indices):] = len(labeled_indices) / len(unlabeled_indices)
        sampler = WeightedRandomSampler(weights, num_samples=len(merged), replacement=True)
        workers = int(optimization.get("num_workers", 2))
        batch_size = int(optimization["batch_size"])
        train_loader = DataLoader(
            merged, batch_size=batch_size, sampler=sampler, num_workers=workers,
            drop_last=True, pin_memory=True, collate_fn=_collate_training,
        )
        eval_workers = workers
        eval_batch_size = int(data.get("eval_batch_size", 256))
        eval_test = _ImageView(root, eval_rows, range(len(eval_rows)), class_to_id, eval_transform, labeled=True)
        eval_unlabeled = _ImageView(root, train_rows, unlabeled_indices, class_to_id, eval_transform, labeled=True)
        # Critically, no novel/hidden labels in this loader: only known supervised IDs.
        dist_labeled = _ImageView(root, train_rows, labeled_indices, class_to_id, eval_transform, labeled=True)
        dist_unlabeled = _ImageView(root, train_rows, unlabeled_indices, class_to_id, eval_transform, labeled=False)
        dist_dataset = _MergedDataset(dist_labeled, dist_unlabeled)
        evaluation_loaders = {
            "test": DataLoader(eval_test, batch_size=eval_batch_size, shuffle=False, num_workers=eval_workers),
            "unlabeled_train": DataLoader(eval_unlabeled, batch_size=eval_batch_size, shuffle=False, num_workers=eval_workers),
            "distribution": DataLoader(
                dist_dataset, batch_size=eval_batch_size, shuffle=False,
                num_workers=eval_workers, collate_fn=_collate_distribution,
            ),
        }
        return DatasetBundle(
            train_loader=train_loader,
            evaluation_loaders=evaluation_loaders,
            num_classes=self.n_classes,
            known_class_ids=tuple(range(self.n_known)),
            # Evaluation-only statistics. train.py MUST NOT place these in context.
            class_counts=tuple(int(n) for n in true_train_counts),
        )
