"""Permutation-invariant GCD evaluation metrics."""

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def clustering_metrics(
    targets: Any,
    predictions: Any,
    known_class_count: int,
    class_counts: Any | None = None,
    accuracy_scale: float = 1.0,
    empty_group_value: float | None = None,
) -> dict[str, float | None]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError("targets and predictions must be matching 1-D arrays")
    if targets.size == 0:
        raise ValueError("cannot evaluate an empty prediction set")

    dimension = int(max(targets.max(), predictions.max()) + 1)
    confusion = np.zeros((dimension, dimension), dtype=np.int64)
    np.add.at(confusion, (predictions, targets), 1)
    predicted_ids, target_ids = linear_sum_assignment(
        confusion.max() - confusion
    )
    mapping = dict(zip(predicted_ids.tolist(), target_ids.tolist()))
    mapped_predictions = np.asarray(
        [mapping.get(int(prediction), -1) for prediction in predictions],
        dtype=np.int64,
    )

    known_mask = targets < known_class_count
    novel_mask = ~known_mask
    metrics = {
        "all_acc": float(np.mean(mapped_predictions == targets))
        * accuracy_scale,
        "known_acc": _masked_accuracy(
            mapped_predictions,
            targets,
            known_mask,
            accuracy_scale,
            empty_group_value,
        ),
        "novel_acc": _masked_accuracy(
            mapped_predictions,
            targets,
            novel_mask,
            accuracy_scale,
            empty_group_value,
        ),
    }
    metrics["all"] = metrics["all_acc"]
    metrics["old"] = metrics["known_acc"]
    metrics["new"] = metrics["novel_acc"]
    if class_counts is not None:
        counts = np.asarray(class_counts, dtype=np.int64)
        if counts.ndim != 1 or counts.size <= int(targets.max()):
            raise ValueError("class_counts must cover every evaluation class")
        _add_shot_metrics(
            metrics,
            mapped_predictions,
            targets,
            counts,
            known_class_count,
            accuracy_scale,
            empty_group_value,
        )
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )

        metrics["nmi"] = float(
            normalized_mutual_info_score(targets, predictions)
        )
        metrics["ari"] = float(adjusted_rand_score(targets, predictions))
    except ImportError:
        pass
    return metrics


def _add_shot_metrics(
    metrics: dict[str, float | None],
    predictions: np.ndarray,
    targets: np.ndarray,
    class_counts: np.ndarray,
    known_class_count: int,
    accuracy_scale: float,
    empty_group_value: float | None,
) -> None:
    class_ids = np.arange(class_counts.size)
    known_ids = class_ids[class_ids < known_class_count]
    novel_ids = class_ids[class_ids >= known_class_count]
    for prefix, ids in (("k", known_ids), ("u", novel_ids)):
        if not ids.size:
            many_threshold = few_threshold = 0
        elif known_class_count == 5:
            sorted_counts = np.sort(class_counts[ids])[::-1]
            many_threshold = sorted_counts[1]
            few_threshold = sorted_counts[2]
        else:
            sorted_counts = np.sort(class_counts[ids])[::-1]
            many_threshold = sorted_counts[len(ids) // 3]
            few_threshold = sorted_counts[(2 * len(ids)) // 3]
        per_class_counts = class_counts[ids]
        groups = {
            "many": ids[per_class_counts > many_threshold],
            "med": ids[
                (per_class_counts <= many_threshold)
                & (per_class_counts >= few_threshold)
            ],
            "few": ids[per_class_counts < few_threshold],
        }
        for group_name, group_ids in groups.items():
            group_mask = np.isin(targets, group_ids)
            value = _masked_accuracy(
                predictions,
                targets,
                group_mask,
                accuracy_scale,
                empty_group_value,
            )
            metrics[f"{prefix}_{group_name}"] = value
            metrics[
                f"{'known' if prefix == 'k' else 'novel'}_{group_name}_acc"
            ] = value


def _masked_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    accuracy_scale: float = 1.0,
    empty_value: float | None = None,
) -> float | None:
    if not mask.any():
        return empty_value
    return float(np.mean(predictions[mask] == targets[mask])) * accuracy_scale
