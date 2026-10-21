"""CUB-200 GCD data adapter with optional precomputed long-tail splits."""

import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

from prototail_gcd.contracts import (
    DistributionBatch,
    TrainingBatch,
    UNKNOWN_TARGET,
)
from prototail_gcd.data.protocol import DatasetBundle


class _CUBImages(Dataset):
    base_folder = Path("CUB_200_2011/images")
    archive_url = (
        "http://www.vision.caltech.edu/visipedia-data/"
        "CUB-200-2011/CUB_200_2011.tgz"
    )
    archive_name = "CUB_200_2011.tgz"
    archive_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: Path,
        *,
        train: bool,
        transform: Any,
        download: bool,
    ) -> None:
        self.root = root.expanduser()
        self.train = train
        self.transform = transform
        if download:
            self._download()
        self.data = self._load_metadata()
        self.uq_indices = np.arange(len(self.data), dtype=np.int64)

    def _load_metadata(self) -> pd.DataFrame:
        dataset_dir = self.root / "CUB_200_2011"
        images = pd.read_csv(
            dataset_dir / "images.txt",
            sep=r"\s+",
            names=["image_id", "filepath"],
            engine="python",
        )
        labels = pd.read_csv(
            dataset_dir / "image_class_labels.txt",
            sep=r"\s+",
            names=["image_id", "target"],
            engine="python",
        )
        split = pd.read_csv(
            dataset_dir / "train_test_split.txt",
            sep=r"\s+",
            names=["image_id", "is_train"],
            engine="python",
        )
        metadata = images.merge(labels, on="image_id").merge(
            split,
            on="image_id",
        )
        train_value = 1 if self.train else 0
        return metadata.loc[metadata["is_train"] == train_value].reset_index(
            drop=True
        )

    def _download(self) -> None:
        dataset_dir = self.root / "CUB_200_2011"
        metadata_files = (
            dataset_dir / "images.txt",
            dataset_dir / "image_class_labels.txt",
            dataset_dir / "train_test_split.txt",
        )
        if all(path.is_file() for path in metadata_files):
            return

        self.root.mkdir(parents=True, exist_ok=True)
        download_url(
            self.archive_url,
            str(self.root),
            self.archive_name,
            self.archive_md5,
        )
        with tarfile.open(self.root / self.archive_name, "r:gz") as archive:
            archive.extractall(path=self.root)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        row = self.data.iloc[index]
        image_path = self.root / self.base_folder / row.filepath
        image = default_loader(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row.target) - 1, int(self.uq_indices[index])


class _CUBSubset(Dataset):
    def __init__(
        self,
        dataset: _CUBImages,
        indices: Sequence[int],
        class_to_id: Mapping[int, int],
        *,
        labeled: bool,
    ) -> None:
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
            self.class_to_id[original_target]
            if self.labeled
            else UNKNOWN_TARGET
        )
        return image, class_id, sample_id, self.labeled


class _MergedCUBTrainingSet(Dataset):
    def __init__(self, labeled: Dataset, unlabeled: Dataset) -> None:
        self.labeled = labeled
        self.unlabeled = unlabeled

    def __len__(self) -> int:
        return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        if index < len(self.labeled):
            return self.labeled[index]
        return self.unlabeled[index - len(self.labeled)]


class _CUBDistributionSet(Dataset):
    def __init__(
        self,
        dataset: _CUBImages,
        labeled_indices: Sequence[int],
        unlabeled_indices: Sequence[int],
        class_to_id: Mapping[int, int],
    ) -> None:
        self.dataset = dataset
        self.labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
        self.unlabeled_indices = np.asarray(unlabeled_indices, dtype=np.int64)
        self.labeled_id_set = set(self.labeled_indices.tolist())
        self.indices = np.concatenate(
            (self.labeled_indices, self.unlabeled_indices)
        )
        self.class_to_id = class_to_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, int, int, bool]:
        source_index = int(self.indices[index])
        image, original_target, sample_id = self.dataset[source_index]
        is_labeled = source_index in self.labeled_id_set
        target = (
            self.class_to_id[original_target]
            if is_labeled
            else UNKNOWN_TARGET
        )
        return image, target, sample_id, is_labeled


def _collate_training(
    samples: Sequence[tuple[Any, int, int, bool]],
) -> TrainingBatch:
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


def _collate_distribution(
    samples: Sequence[tuple[Any, int, int, bool]],
) -> DistributionBatch:
    images, targets, sample_ids, labeled = zip(*samples)
    return DistributionBatch(
        images=torch.stack(list(images)),
        known_targets=torch.tensor(targets),
        labeled_mask=torch.tensor(labeled),
        sample_ids=torch.tensor(sample_ids),
    )


def _load_split_tensor(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required CUB split file is missing: {path}")
    try:
        values = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        values = torch.load(path, map_location="cpu")
    return np.asarray(values, dtype=np.int64)


def _make_transforms(image_size: int, crop_pct: float) -> tuple[Any, Any]:
    interpolation = transforms.InterpolationMode.BICUBIC
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = int(image_size / crop_pct)
    train_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, test_transform


class CUBLongTailAdapter:
    name = "cub"

    def build(self, config: Mapping[str, Any]) -> DatasetBundle:
        data_config = config["data"]
        optimization = config["optimization"]
        protocol_config = data_config["protocol"]
        imbalance_ratio = int(protocol_config["imbalance_ratio"])
        known_class_count = int(protocol_config["known_class_count"])
        image_size = int(data_config.get("image_size", 224))
        crop_pct = float(data_config.get("crop_pct", 0.875))

        split_directory = Path(data_config["splits_root"]) / (
            f"cub200_k{known_class_count}_imb{imbalance_ratio}"
        )
        labeled_indices = _load_split_tensor(
            split_directory / "l_k_uq_idxs.pt"
        )
        unlabeled_known_indices = _load_split_tensor(
            split_directory / "unl_k_uq_idxs.pt"
        )
        unlabeled_novel_indices = _load_split_tensor(
            split_directory / "unl_unk_uq_idxs.pt"
        )

        train_transform, test_transform = _make_transforms(
            image_size,
            crop_pct,
        )
        view_count = int(optimization["num_views"])
        train_dataset = _CUBImages(
            Path(data_config["root"]),
            train=True,
            transform=_MultiViewTransform(train_transform, view_count),
            download=bool(data_config.get("download", False)),
        )
        test_dataset = _CUBImages(
            Path(data_config["root"]),
            train=False,
            transform=test_transform,
            download=False,
        )

        all_unlabeled_indices = np.concatenate(
            (unlabeled_known_indices, unlabeled_novel_indices)
        )
        all_indices = np.concatenate((labeled_indices, all_unlabeled_indices))
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("CUB labeled/unlabeled split indices overlap")
        if all_indices.size and (
            all_indices.min() < 0 or all_indices.max() >= len(train_dataset)
        ):
            raise IndexError("CUB split index is outside the training dataset")

        train_targets = (
            train_dataset.data["target"].to_numpy(dtype=np.int64) - 1
        )
        known_classes = sorted(
            set(train_targets[labeled_indices].tolist())
            | set(train_targets[unlabeled_known_indices].tolist())
        )
        novel_classes = sorted(
            set(train_targets[unlabeled_novel_indices].tolist())
        )
        if len(known_classes) != known_class_count:
            raise ValueError(
                f"Expected {known_class_count} known CUB classes, "
                f"found {len(known_classes)}"
            )
        if set(known_classes) & set(novel_classes):
            raise ValueError("CUB known and novel class splits overlap")
        class_to_id = {
            original_id: mapped_id
            for mapped_id, original_id in enumerate(
                known_classes + novel_classes
            )
        }

        labeled_view = _CUBSubset(
            train_dataset,
            labeled_indices,
            class_to_id,
            labeled=True,
        )
        unlabeled_view = _CUBSubset(
            train_dataset,
            all_unlabeled_indices,
            class_to_id,
            labeled=False,
        )
        merged_train = _MergedCUBTrainingSet(labeled_view, unlabeled_view)

        labeled_count = len(labeled_view)
        unlabeled_count = len(unlabeled_view)
        if not labeled_count or not unlabeled_count:
            raise ValueError(
                "CUB GCD requires non-empty labeled and unlabeled sets"
            )
        sample_weights = torch.ones(len(merged_train), dtype=torch.double)
        sample_weights[labeled_count:] = labeled_count / unlabeled_count
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(merged_train),
            replacement=True,
        )
        train_loader = DataLoader(
            merged_train,
            batch_size=int(optimization["batch_size"]),
            sampler=sampler,
            num_workers=int(optimization["num_workers"]),
            drop_last=True,
            pin_memory=True,
            collate_fn=_collate_training,
        )

        mapped_test = _CUBSubset(
            test_dataset,
            np.arange(len(test_dataset), dtype=np.int64),
            class_to_id,
            labeled=True,
        )
        distribution_dataset = _CUBImages(
            Path(data_config["root"]),
            train=True,
            transform=test_transform,
            download=False,
        )
        unlabeled_eval = _CUBSubset(
            distribution_dataset,
            all_unlabeled_indices,
            class_to_id,
            labeled=True,
        )
        distribution_view = _CUBDistributionSet(
            distribution_dataset,
            labeled_indices,
            all_unlabeled_indices,
            class_to_id,
        )
        mapped_train_targets = np.asarray(
            [class_to_id[int(train_targets[index])] for index in all_indices],
            dtype=np.int64,
        )
        class_counts = np.bincount(
            mapped_train_targets,
            minlength=len(class_to_id),
        )
        eval_batch_size = int(data_config.get("eval_batch_size", 256))
        evaluation_loaders = {
            "test": DataLoader(
                mapped_test,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=int(optimization["num_workers"]),
            ),
            "unlabeled_train": DataLoader(
                unlabeled_eval,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=int(optimization["num_workers"]),
            ),
            "distribution": DataLoader(
                distribution_view,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=int(optimization["num_workers"]),
                collate_fn=_collate_distribution,
            ),
        }

        return DatasetBundle(
            train_loader=train_loader,
            evaluation_loaders=evaluation_loaders,
            num_classes=len(class_to_id),
            known_class_ids=tuple(range(len(known_classes))),
            class_counts=tuple(int(count) for count in class_counts),
        )


class _MultiViewTransform:
    def __init__(self, transform: Any, num_views: int) -> None:
        self.transform = transform
        self.num_views = num_views

    def __call__(self, image: Any) -> list[Any]:
        return [self.transform(image) for _ in range(self.num_views)]
