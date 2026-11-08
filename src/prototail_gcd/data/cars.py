"""Stanford Cars GCD data adapter with precomputed long-tail splits."""

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
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

N_TRAIN, N_TEST, N_CLASS = 8144, 8041, 196


def _load_split_tensor(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required Cars split file is missing: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Split file {path} did not contain a torch tensor")
    return tensor.numpy().astype(np.int64)


def _load_devkit_mat(mat_path):
    from scipy.io import loadmat
    mat = loadmat(mat_path, squeeze_me=True)
    annos = mat["annotations"]
    items = []
    for a in np.atleast_1d(annos):
        fname = str(a["fname"])
        cls = int(a["class"]) - 1  # 1-based -> 0-based
        items.append((fname, cls))
    items.sort(key=lambda x: x[0])
    return items


def _load_kaggle_csv(root: Path, train: bool):
    csv_path = root / ("anno_train.csv" if train else "anno_test.csv")
    if not csv_path.is_file():
        # Check subdirectories
        for match in root.rglob("anno_train.csv" if train else "anno_test.csv"):
            csv_path = match
            break
    df = pd.read_csv(csv_path, header=None)
    want = "train" if train else "test"
    path_by_name = {}
    search_root = csv_path.parent
    for dirpath, _, filenames in os.walk(search_root):
        if want not in set(p.lower() for p in dirpath.split(os.sep)):
            continue
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                path_by_name.setdefault(fn, os.path.join(dirpath, fn))
    items = []
    for fname, cls in zip(df[0].astype(str), df[5].astype(int)):
        fname = os.path.basename(fname)
        if fname not in path_by_name:
            raise FileNotFoundError(f"{fname} ({csv_path}) not found under {search_root}.")
        items.append((path_by_name[fname], int(cls) - 1))
    items.sort(key=lambda x: os.path.basename(x[0]))
    return items


class CarsDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, transform: Any = None):
        self.root = Path(root).expanduser()
        self.train = train
        self.transform = transform
        
        # Check devkit or csv
        mat_path = self.root / "devkit" / ("cars_train_annos.mat" if train else "cars_test_annos_withlabels.mat")
        img_dir = self.root / ("cars_train" if train else "cars_test")
        has_csv = (self.root / ("anno_train.csv" if train else "anno_test.csv")).is_file() or any(self.root.rglob("anno_train.csv"))

        if mat_path.is_file() and img_dir.is_dir():
            items = _load_devkit_mat(str(mat_path))
            self.samples = [(str(img_dir / f), c) for f, c in items]
        elif has_csv:
            self.samples = _load_kaggle_csv(self.root, train)
        else:
            raise FileNotFoundError(
                f"No Cars data found under {self.root}. Expected official layout "
                "(cars_train/, devkit/) or Kaggle dump (anno_train.csv, anno_test.csv)."
            )
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


class _CarsSubset(Dataset):
    def __init__(self, dataset: CarsDataset, indices: Sequence[int], class_to_id: Mapping[int, int], *, labeled: bool):
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


class _MergedCarsTrainingSet(Dataset):
    def __init__(self, labeled: Dataset, unlabeled: Dataset) -> None:
        self.labeled = labeled
        self.unlabeled = unlabeled

    def __len__(self) -> int:
        return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        if index < len(self.labeled):
            return self.labeled[index]
        return self.unlabeled[index - len(self.labeled)]


class _CarsDistributionSet(Dataset):
    def __init__(self, dataset: CarsDataset, labeled_indices: Sequence[int], unlabeled_indices: Sequence[int], class_to_id: Mapping[int, int]):
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
        transforms.ColorJitter(),  # match official SimGCD ImageNet train transform
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


class CarsLongTailAdapter:
    name = "cars"

    def build(self, config: Mapping[str, Any]) -> DatasetBundle:
        data_config = config["data"]
        optimization = config["optimization"]
        protocol_config = data_config.get("protocol", {})
        imbalance_ratio = int(protocol_config.get("imbalance_ratio", 20))
        known_class_count = int(protocol_config.get("known_class_count", 98))
        image_size = int(data_config.get("image_size", 224))
        crop_pct = float(data_config.get("crop_pct", 0.875))

        split_directory = Path(data_config["splits_root"]) / f"cars196_k{known_class_count}_imb{imbalance_ratio}"
        labeled_indices = _load_split_tensor(split_directory / "l_k_uq_idxs.pt")
        unlabeled_known_indices = _load_split_tensor(split_directory / "unl_k_uq_idxs.pt")
        unlabeled_novel_indices = _load_split_tensor(split_directory / "unl_unk_uq_idxs.pt")
        print("=" * 65)
        print(f"[DATASET PROTOCOL] Dataset: Stanford Cars | Split: {split_directory.name}")
        print(f" - Splits Directory: {split_directory}")
        print(f" - Labeled Samples:         {len(labeled_indices)}")
        print(f" - Unlabeled Known Samples: {len(unlabeled_known_indices)}")
        print(f" - Unlabeled Novel Samples: {len(unlabeled_novel_indices)}")
        print(f" - Total Train Samples:     {len(labeled_indices) + len(unlabeled_known_indices) + len(unlabeled_novel_indices)}")
        print("=" * 65)

        train_transform, test_transform = _make_transforms(image_size, crop_pct)
        view_count = int(optimization.get("num_views", 2))
        root = Path(data_config["root"])
        train_dataset = CarsDataset(root, train=True, transform=_MultiViewTransform(train_transform, view_count))
        test_dataset = CarsDataset(root, train=False, transform=test_transform)

        all_unlabeled_indices = np.concatenate((unlabeled_known_indices, unlabeled_novel_indices))
        all_indices = np.concatenate((labeled_indices, all_unlabeled_indices))

        train_targets = np.array([t for _, t in train_dataset.samples], dtype=np.int64)
        known_classes = sorted(set(train_targets[labeled_indices].tolist()))
        novel_classes = sorted(set(train_targets[unlabeled_novel_indices].tolist()))
        if len(known_classes) != known_class_count:
            raise ValueError(f"Expected {known_class_count} known Cars classes, found {len(known_classes)}")
        if set(known_classes) & set(novel_classes):
            raise ValueError("Cars known and novel class splits overlap")

        class_to_id = {original_id: mapped_id for mapped_id, original_id in enumerate(known_classes + novel_classes)}
        labeled_view = _CarsSubset(train_dataset, labeled_indices, class_to_id, labeled=True)
        unlabeled_view = _CarsSubset(train_dataset, all_unlabeled_indices, class_to_id, labeled=False)
        merged_train = _MergedCarsTrainingSet(labeled_view, unlabeled_view)

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

        mapped_test = _CarsSubset(test_dataset, np.arange(len(test_dataset), dtype=np.int64), class_to_id, labeled=True)
        eval_transform = test_transform
        distribution_dataset = CarsDataset(root, train=True, transform=eval_transform)
        unlabeled_eval = _CarsSubset(distribution_dataset, all_unlabeled_indices, class_to_id, labeled=True)
        distribution_view = _CarsDistributionSet(distribution_dataset, labeled_indices, all_unlabeled_indices, class_to_id)

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
