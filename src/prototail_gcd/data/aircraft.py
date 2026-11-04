"""FGVC-Aircraft GCD data adapter with precomputed long-tail splits."""

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import FGVCAircraft

from prototail_gcd.contracts import (
    DistributionBatch,
    TrainingBatch,
    UNKNOWN_TARGET,
)
from prototail_gcd.data.protocol import DatasetBundle


def _load_split_tensor(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required Aircraft split file is missing: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Split file {path} did not contain a torch tensor")
    return tensor.numpy().astype(np.int64)


class AircraftDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, transform: Any = None):
        self.root = Path(root).expanduser()
        self.train = train
        self.transform = transform
        
        # Resolve root so torchvision FGVCAircraft finds fgvc-aircraft-2013b
        tv_root = self.root
        if (self.root / "data" / "images").is_dir():
            tv_root = self.root.parent
        elif not (self.root / "fgvc-aircraft-2013b").is_dir():
            for match in self.root.rglob("fgvc-aircraft-2013b"):
                if match.is_dir():
                    tv_root = match.parent
                    break

        split = "trainval" if train else "test"
        base = FGVCAircraft(root=str(tv_root), split=split, annotation_level="variant", transform=None, download=False)
        self.samples = [(str(p), int(t)) for p, t in zip(base._image_files, base._labels)]
        self.uq_indices = np.arange(len(self.samples), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        path, target = self.samples[index]
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target), int(self.uq_indices[index])


class _MultiViewTransform:
    def __init__(self, base_transform: Any, view_count: int) -> None:
        self.base_transform = base_transform
        self.view_count = view_count

    def __call__(self, image: Any) -> Sequence[Any]:
        return [self.base_transform(image) for _ in range(self.view_count)]


class _AircraftSubset(Dataset):
    def __init__(self, dataset: AircraftDataset, indices: Sequence[int], class_to_id: Mapping[int, int], *, labeled: bool):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_to_id = class_to_id
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        source_index = int(self.indices[index])
        image, original_target, sample_id = self.dataset[source_index]
        class_id = self.class_to_id[original_target] if self.labeled else UNKNOWN_TARGET
        return image, class_id, sample_id, self.labeled


class _MergedAircraftTrainingSet(Dataset):
    def __init__(self, labeled: Dataset, unlabeled: Dataset) -> None:
        self.labeled = labeled
        self.unlabeled = unlabeled

    def __len__(self) -> int:
        return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        if index < len(self.labeled):
            return self.labeled[index]
        return self.unlabeled[index - len(self.labeled)]


class _AircraftDistributionSet(Dataset):
    def __init__(self, dataset: AircraftDataset, labeled_indices: Sequence[int], unlabeled_indices: Sequence[int], class_to_id: Mapping[int, int]):
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
        known_targets=torch.tensor([sample[1] for sample in samples]),
        sample_ids=torch.tensor([sample[2] for sample in samples]),
        labeled_mask=torch.tensor([sample[3] for sample in samples]),
    )


def _collate_distribution(samples: Sequence[tuple[Any, int, int, bool]]) -> DistributionBatch:
    return DistributionBatch(
        images=torch.stack([sample[0] for sample in samples]),
        known_targets=torch.tensor([sample[1] for sample in samples]),
        sample_ids=torch.tensor([sample[2] for sample in samples]),
        labeled_mask=torch.tensor([sample[3] for sample in samples]),
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


class AircraftLongTailAdapter:
    name = "aircraft"

    def build(self, config: Mapping[str, Any]) -> DatasetBundle:
        data_config = config["data"]
        optimization = config["optimization"]
        protocol_config = data_config.get("protocol", {})
        imbalance_ratio = int(protocol_config.get("imbalance_ratio", 20))
        known_class_count = int(protocol_config.get("known_class_count", 80))
        image_size = int(data_config.get("image_size", 224))
        crop_pct = float(data_config.get("crop_pct", 0.875))

        split_directory = Path(data_config["splits_root"]) / f"aircraft_k{known_class_count}_imb{imbalance_ratio}"
        labeled_indices = _load_split_tensor(split_directory / "l_k_uq_idxs.pt")
        unlabeled_known_indices = _load_split_tensor(split_directory / "unl_k_uq_idxs.pt")
        unlabeled_novel_indices = _load_split_tensor(split_directory / "unl_unk_uq_idxs.pt")
        print("=" * 65)
        print(f"[DATASET PROTOCOL] Dataset: FGVC-Aircraft | Split: {split_directory.name}")
        print(f" - Splits Directory: {split_directory}")
        print(f" - Labeled Samples:         {len(labeled_indices)}")
        print(f" - Unlabeled Known Samples: {len(unlabeled_known_indices)}")
        print(f" - Unlabeled Novel Samples: {len(unlabeled_novel_indices)}")
        print(f" - Total Train Samples:     {len(labeled_indices) + len(unlabeled_known_indices) + len(unlabeled_novel_indices)}")
        print("=" * 65)

        train_transform, test_transform = _make_transforms(image_size, crop_pct)
        view_count = int(optimization.get("num_views", 2))
        root = Path(data_config["root"])
        train_dataset = AircraftDataset(root, train=True, transform=_MultiViewTransform(train_transform, view_count))
        test_dataset = AircraftDataset(root, train=False, transform=test_transform)

        all_unlabeled_indices = np.concatenate((unlabeled_known_indices, unlabeled_novel_indices))
        all_indices = np.concatenate((labeled_indices, all_unlabeled_indices))

        train_targets = np.array([t for _, t in train_dataset.samples], dtype=np.int64)
        known_classes = sorted(set(train_targets[labeled_indices].tolist()))
        novel_classes = sorted(set(train_targets[unlabeled_novel_indices].tolist()))
        if len(known_classes) != known_class_count:
            raise ValueError(f"Expected {known_class_count} known Aircraft classes, found {len(known_classes)}")
        if set(known_classes) & set(novel_classes):
            raise ValueError("Aircraft known and novel class splits overlap")

        class_to_id = {original_id: mapped_id for mapped_id, original_id in enumerate(known_classes + novel_classes)}
        labeled_view = _AircraftSubset(train_dataset, labeled_indices, class_to_id, labeled=True)
        unlabeled_view = _AircraftSubset(train_dataset, all_unlabeled_indices, class_to_id, labeled=False)
        merged_train = _MergedAircraftTrainingSet(labeled_view, unlabeled_view)

        labeled_count = len(labeled_view)
        unlabeled_count = len(unlabeled_view)
        sample_weights = torch.ones(len(merged_train), dtype=torch.double)
        sample_weights[labeled_count:] = labeled_count / unlabeled_count
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(merged_train), replacement=True)

        train_loader = DataLoader(
            merged_train,
            batch_size=int(optimization["batch_size"]),
            sampler=sampler,
            num_workers=int(optimization.get("num_workers", 2)),
            drop_last=True,
            pin_memory=True,
            collate_fn=_collate_training,
        )

        mapped_test = _AircraftSubset(test_dataset, np.arange(len(test_dataset), dtype=np.int64), class_to_id, labeled=True)
        eval_transform = test_transform
        distribution_dataset = AircraftDataset(root, train=True, transform=eval_transform)
        unlabeled_eval = _AircraftSubset(distribution_dataset, all_unlabeled_indices, class_to_id, labeled=True)
        distribution_view = _AircraftDistributionSet(distribution_dataset, labeled_indices, all_unlabeled_indices, class_to_id)

        mapped_train_targets = np.asarray([class_to_id[int(train_targets[idx])] for idx in all_indices], dtype=np.int64)
        class_counts = np.bincount(mapped_train_targets, minlength=len(class_to_id))

        eval_batch_size = int(data_config.get("eval_batch_size", 256))
        evaluation_loaders = {
            "test": DataLoader(mapped_test, batch_size=eval_batch_size, shuffle=False, num_workers=int(optimization.get("num_workers", 2))),
            "unlabeled_train": DataLoader(unlabeled_eval, batch_size=eval_batch_size, shuffle=False, num_workers=int(optimization.get("num_workers", 2))),
            "distribution": DataLoader(distribution_view, batch_size=eval_batch_size, shuffle=False, num_workers=int(optimization.get("num_workers", 2)), collate_fn=_collate_distribution),
        }

        return DatasetBundle(
            train_loader=train_loader,
            evaluation_loaders=evaluation_loaders,
            num_classes=len(class_to_id),
            known_class_ids=tuple(range(known_class_count)),
            class_counts=tuple(int(c) for c in class_counts),
        )
