"""iNaturalist 2018 GCD data adapter.

Natively operates on the full natural long-tail distribution (437,513 images across 8,142 classes)
WITHOUT requiring any external precomputed .pt split files.
Splits are computed deterministically on the fly (0.1s) from train2018.json according to the standard GCD protocol:
- 4,071 Known classes: 50% labeled (l_k), 50% unlabeled (unl_k)
- 4,071 Novel classes: 100% unlabeled (unl_unk)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from prototail_gcd.contracts import (
    DistributionBatch,
    TrainingBatch,
    UNKNOWN_TARGET,
)
from prototail_gcd.data.protocol import DatasetBundle

N_CLASS = 8142
N_KNOWN = 4071


def resolve_inat2018_paths(root: Path) -> tuple[Path, Path]:
    """Auto-resolve the iNaturalist 2018 JSON annotation and image root directory."""
    root = Path(root).expanduser().resolve()
    candidates = [
        root,
        root / "iNaturalist 2018",
        root / "inat2018",
        Path("/bacon-storage/data/inat2018"),
        Path("/media/RAID5Array/backup_home/tindd4/Alo_SimGCD/iNaturalist 2018"),
        Path(r"D:\data\iNaturalist 2018"),
    ]
    for c in candidates:
        if not c.is_dir():
            continue
        json_path = c / "train2018.json"
        if json_path.is_file():
            return json_path, c
    raise FileNotFoundError(
        f"Could not find train2018.json in iNaturalist 2018 candidates: "
        + ", ".join(str(c) for c in candidates if c.exists())
    )


def partition_inat2018_on_the_fly(
    targets: Sequence[int] | np.ndarray,
    n_classes: int = N_CLASS,
    known_class_count: int = N_KNOWN,
    prop_train_labels: float = 0.5,
    seed: int = 416,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically partition iNaturalist 2018 on-the-fly (Zero .pt files needed).

    Protocol:
    - Known classes (0 .. known_class_count - 1):
      prop_train_labels (50%) -> labeled_indices (l_k)
      remaining (50%)         -> unlabeled_known_indices (unl_k)
    - Novel classes (known_class_count .. n_classes - 1):
      100% samples            -> unlabeled_novel_indices (unl_unk)
    """
    targets = np.asarray(targets, dtype=np.int64)
    cls_to_pos_list = [[] for _ in range(n_classes)]
    for idx, c in enumerate(targets):
        if 0 <= c < n_classes:
            cls_to_pos_list[c].append(idx)
    cls_to_pos = {c: np.asarray(cls_to_pos_list[c], dtype=np.int64) for c in range(n_classes)}

    rng = np.random.RandomState(seed)
    l_k, unl_k, unl_unk = [], [], []

    for c in range(n_classes):
        pos = cls_to_pos[c]
        if len(pos) == 0:
            continue

        if c < known_class_count:
            n_l = int(round(len(pos) * prop_train_labels))
            if len(pos) >= 2 and n_l == 0:
                n_l = 1
            if len(pos) >= 2 and n_l == len(pos):
                n_l = len(pos) - 1
            if len(pos) == 1:
                n_l = 1

            take = rng.choice(pos, size=n_l, replace=False)
            rest = np.setdiff1d(pos, take)
            l_k.extend(take.tolist())
            unl_k.extend(rest.tolist())
        else:
            unl_unk.extend(pos.tolist())

    l_k = np.sort(np.asarray(l_k, dtype=np.int64))
    unl_k = np.sort(np.asarray(unl_k, dtype=np.int64))
    unl_unk = np.sort(np.asarray(unl_unk, dtype=np.int64))

    return l_k, unl_k, unl_unk


def _load_split_tensor(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required split file is missing: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Split file {path} did not contain a torch tensor")
    return tensor.numpy().astype(np.int64)


class INat2018Dataset(Dataset):
    """iNaturalist 2018 dataset loading directly from official COCO-style JSON annotations."""

    def __init__(self, root: Path, transform: Any = None, json_file: str = "train2018.json"):
        self.root = Path(root).expanduser().resolve()
        json_path = self.root / json_file
        if not json_path.is_file():
            resolved_json, resolved_root = resolve_inat2018_paths(self.root)
            json_path = resolved_root / json_file
            self.root = resolved_root

        self.transform = transform

        with open(json_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        images = meta["images"]
        annotations = meta.get("annotations", [])
        ann_by_image_id = {a["image_id"]: a["category_id"] for a in annotations}

        self.samples = [img["file_name"] for img in images]
        self.targets = [ann_by_image_id.get(img["id"], -1) for img in images]
        self.uq_indices = np.arange(len(self.samples), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        rel_path = self.samples[index]
        path = self.root / rel_path
        if not path.is_file():
            path = self.root / "train_val2018" / rel_path

        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index]), int(self.uq_indices[index])


class _MultiViewTransform:
    def __init__(self, base_transform: Any, view_count: int) -> None:
        self.base_transform = base_transform
        self.view_count = view_count

    def __call__(self, image: Any) -> Sequence[Any]:
        return [self.base_transform(image) for _ in range(self.view_count)]


class _INatSubset(Dataset):
    def __init__(
        self,
        dataset: INat2018Dataset,
        indices: Sequence[int],
        class_to_id: Mapping[int, int],
        *,
        labeled: bool,
    ):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_to_id = class_to_id
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        source_index = int(self.indices[index])
        image, original_target, sample_id = self.dataset[source_index]
        class_id = (
            self.class_to_id[original_target] if self.labeled else UNKNOWN_TARGET
        )
        return image, class_id, sample_id, self.labeled


class _MergedINatTrainingSet(Dataset):
    def __init__(self, labeled: Dataset, unlabeled: Dataset) -> None:
        self.labeled = labeled
        self.unlabeled = unlabeled

    def __len__(self) -> int:
        return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        if index < len(self.labeled):
            return self.labeled[index]
        return self.unlabeled[index - len(self.labeled)]


class _INatDistributionSet(Dataset):
    def __init__(
        self,
        dataset: INat2018Dataset,
        labeled_indices: Sequence[int],
        unlabeled_indices: Sequence[int],
        class_to_id: Mapping[int, int],
    ):
        self.dataset = dataset
        self.labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
        self.unlabeled_indices = np.asarray(unlabeled_indices, dtype=np.int64)
        self.labeled_id_set = set(self.labeled_indices.tolist())
        self.indices = np.concatenate((self.labeled_indices, self.unlabeled_indices))
        self.class_to_id = class_to_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        source_index = int(self.indices[index])
        image, original_target, sample_id = self.dataset[source_index]
        is_labeled = source_index in self.labeled_id_set
        target = self.class_to_id[original_target] if is_labeled else UNKNOWN_TARGET
        return image, target, sample_id, is_labeled


def _collate_training(samples: Sequence[tuple[Any, int, int, bool]]) -> TrainingBatch:
    view_count = len(samples[0][0])
    views = tuple(
        torch.stack([sample[0][view_index] for sample in samples])
        for view_index in range(view_count)
    )
    return TrainingBatch(
        views=views,
        known_targets=torch.tensor([sample[1] for sample in samples], dtype=torch.long),
        sample_ids=torch.tensor([sample[2] for sample in samples], dtype=torch.long),
        labeled_mask=torch.tensor([sample[3] for sample in samples], dtype=torch.bool),
    )


def _collate_distribution(
    samples: Sequence[tuple[Any, int, int, bool]]
) -> DistributionBatch:
    return DistributionBatch(
        images=torch.stack([sample[0] for sample in samples]),
        known_targets=torch.tensor([sample[1] for sample in samples], dtype=torch.long),
        sample_ids=torch.tensor([sample[2] for sample in samples], dtype=torch.long),
        labeled_mask=torch.tensor([sample[3] for sample in samples], dtype=torch.bool),
    )


def _make_transforms(image_size: int, crop_pct: float):
    interpolation = transforms.InterpolationMode.BICUBIC
    resize_size = int(round(image_size / crop_pct))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, test_transform


class INat2018LongTailAdapter:
    """iNaturalist 2018 adapter with ON-THE-FLY partitioning (No .pt files required)."""

    name = "inat2018"

    def build(self, config: Mapping[str, Any]) -> DatasetBundle:
        data_config = config["data"]
        optimization = config["optimization"]
        protocol_config = data_config.get("protocol", {})

        known_class_count = int(protocol_config.get("known_class_count", N_KNOWN))
        prop_train_labels = float(protocol_config.get("prop_train_labels", 0.5))
        seed = int(config.get("seed", 416))
        image_size = int(data_config.get("image_size", 224))
        crop_pct = float(data_config.get("crop_pct", 0.875))

        train_transform, test_transform = _make_transforms(image_size, crop_pct)
        view_count = int(optimization.get("num_views", 2))
        root = Path(data_config.get("root", r"/media/RAID5Array/backup_home/tindd4/Alo_SimGCD/iNaturalist 2018"))

        # Load train dataset directly from train2018.json
        train_dataset = INat2018Dataset(
            root, transform=_MultiViewTransform(train_transform, view_count)
        )
        eval_dataset = INat2018Dataset(root, transform=test_transform)

        # Check if precomputed .pt split is explicitly requested, else compute on-the-fly (Default)
        use_precomputed = data_config.get("use_precomputed_splits", False)
        splits_root = Path(data_config.get("splits_root", "data_uq_idxs"))
        split_name = f"inat2018_k{known_class_count}_natural"
        split_directory = splits_root / split_name

        if use_precomputed and (split_directory / "l_k_uq_idxs.pt").is_file():
            print(f"[INAT2018] Loading precomputed .pt splits from: {split_directory}")
            labeled_indices = _load_split_tensor(split_directory / "l_k_uq_idxs.pt")
            unlabeled_known_indices = _load_split_tensor(split_directory / "unl_k_uq_idxs.pt")
            unlabeled_novel_indices = _load_split_tensor(split_directory / "unl_unk_uq_idxs.pt")
        else:
            print("[INAT2018] Computing GCD split ON-THE-FLY directly from train2018.json (No .pt files required)...")
            labeled_indices, unlabeled_known_indices, unlabeled_novel_indices = partition_inat2018_on_the_fly(
                train_dataset.targets,
                n_classes=len(set(train_dataset.targets)),
                known_class_count=known_class_count,
                prop_train_labels=prop_train_labels,
                seed=seed,
            )

        print("=" * 65)
        print(f"[DATASET PROTOCOL] Dataset: iNaturalist 2018 (Original Natural Distribution)")
        print(f" - Partition mode:          {'Precomputed .pt' if use_precomputed else 'On-the-fly (No .pt required)'}")
        print(f" - Labeled Samples (l_k):   {len(labeled_indices):,}")
        print(f" - Unlabeled Known (unl_k): {len(unlabeled_known_indices):,}")
        print(f" - Unlabeled Novel (unl_u): {len(unlabeled_novel_indices):,}")
        print(
            f" - Total Train Samples:     "
            f"{len(labeled_indices) + len(unlabeled_known_indices) + len(unlabeled_novel_indices):,}"
        )
        print("=" * 65)

        all_unlabeled_indices = np.concatenate(
            (unlabeled_known_indices, unlabeled_novel_indices)
        )
        all_indices = np.concatenate((labeled_indices, all_unlabeled_indices))

        train_targets = np.array(train_dataset.targets, dtype=np.int64)
        known_classes = sorted(set(train_targets[labeled_indices].tolist()))
        novel_classes = sorted(set(train_targets[unlabeled_novel_indices].tolist()))

        if len(known_classes) != known_class_count:
            raise ValueError(
                f"Expected {known_class_count} known iNat classes, found {len(known_classes)}"
            )
        if set(known_classes) & set(novel_classes):
            raise ValueError("iNaturalist 2018 known and novel class splits overlap")

        class_to_id = {
            original_id: mapped_id
            for mapped_id, original_id in enumerate(known_classes + novel_classes)
        }

        labeled_view = _INatSubset(
            train_dataset, labeled_indices, class_to_id, labeled=True
        )
        unlabeled_view = _INatSubset(
            train_dataset, all_unlabeled_indices, class_to_id, labeled=False
        )
        merged_train = _MergedINatTrainingSet(labeled_view, unlabeled_view)

        labeled_count = len(labeled_view)
        unlabeled_count = len(unlabeled_view)
        sample_weights = torch.ones(len(merged_train), dtype=torch.double)
        sample_weights[labeled_count:] = labeled_count / unlabeled_count

        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(merged_train), replacement=True
        )

        train_loader = DataLoader(
            merged_train,
            batch_size=int(optimization["batch_size"]),
            sampler=sampler,
            num_workers=int(optimization.get("num_workers", 4)),
            drop_last=True,
            pin_memory=True,
            collate_fn=_collate_training,
        )

        unlabeled_eval = _INatSubset(
            eval_dataset, all_unlabeled_indices, class_to_id, labeled=True
        )
        distribution_view = _INatDistributionSet(
            eval_dataset, labeled_indices, all_unlabeled_indices, class_to_id
        )

        mapped_train_targets = np.asarray(
            [class_to_id[int(train_targets[idx])] for idx in all_indices],
            dtype=np.int64,
        )
        class_counts = np.bincount(mapped_train_targets, minlength=len(class_to_id))

        eval_batch_size = int(data_config.get("eval_batch_size", 256))
        evaluation_loaders = {
            "test": DataLoader(
                unlabeled_eval,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=int(optimization.get("num_workers", 4)),
            ),
            "unlabeled_train": DataLoader(
                unlabeled_eval,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=int(optimization.get("num_workers", 4)),
            ),
            "distribution": DataLoader(
                distribution_view,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=int(optimization.get("num_workers", 4)),
                collate_fn=_collate_distribution,
            ),
        }

        # Check if validation set is available
        val_json = root / "val2018.json"
        if val_json.is_file():
            val_dataset = INat2018Dataset(root, transform=test_transform, json_file="val2018.json")
            val_indices = np.arange(len(val_dataset))
            val_subset = _INatSubset(val_dataset, val_indices, class_to_id, labeled=True)
            evaluation_loaders["val"] = DataLoader(
                val_subset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=int(optimization.get("num_workers", 4)),
            )

        return DatasetBundle(
            train_loader=train_loader,
            evaluation_loaders=evaluation_loaders,
            num_classes=len(class_to_id),
            known_class_ids=tuple(range(known_class_count)),
            class_counts=tuple(int(c) for c in class_counts),
        )


INat2018Adapter = INat2018LongTailAdapter
