"""BaCon-native losses and distribution estimation."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import sklearn.cluster
from sklearn.cluster import KMeans
from prototail_gcd.utils.kmeans_backend import fit_predict_kmeans

_ORIG_KMEANS = sklearn.cluster.KMeans

from prototail_gcd.contracts import DistributionBatch


class SemiConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        soft_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must be [batch, views, dimensions]")
        batch_size, view_count = features.shape[:2]
        contrast_features = torch.cat(features.unbind(dim=1), dim=0)
        logits = contrast_features @ contrast_features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        if soft_mask.shape[0] != logits.shape[0]:
            soft_mask = soft_mask.repeat(view_count, view_count)
        logits_mask = torch.ones_like(soft_mask)
        diagonal = torch.arange(
            batch_size * view_count,
            device=features.device,
        )
        logits_mask.scatter_(1, diagonal[:, None], 0)
        soft_mask = soft_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        positive_mass = soft_mask.sum(dim=1).clamp_min(1e-12)
        mean_log_prob = (soft_mask * log_prob).sum(dim=1) / positive_mass
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        return loss.view(view_count, batch_size).mean()


def _collect_logits_and_labels(
    model: Any,
    loader: Any,
    *,
    num_classes: int,
    num_known_classes: int,
    finetune_from_block: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward the distribution loader once; return (logits, targets, labeled).

    Works for both SimGCD (backbone + projector -> logits) and BaCon
    (classification_backbone + classification_head -> logits) without KMeans.
    """
    from prototail_gcd.models.dino_backbone import forward_global_and_patches

    logits_batches: list[np.ndarray] = []
    targets_batches: list[np.ndarray] = []
    labeled_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, DistributionBatch):
                images = batch.images
                batch_targets = batch.known_targets
                batch_labeled = batch.labeled_mask
            else:
                images, batch_targets, _sample_ids, batch_labeled = batch

            images = images.to(device, non_blocking=True)
            backbone = getattr(model, "classification_backbone", None)
            head = getattr(model, "classification_head", None)
            if backbone is None or head is None:
                backbone = getattr(model, "backbone", None)
                head = getattr(model, "projector", None)
            if backbone is None or head is None:
                raise ValueError("Model has no parametric classification path")

            ce_features, _ = forward_global_and_patches(
                backbone,
                images,
                finetune_from_block=finetune_from_block,
            )
            logits = head(ce_features)
            if isinstance(logits, tuple):
                logits = logits[-1]  # DINOHead -> (projection, logits)
            logits_batches.append(logits.float().cpu().numpy())
            targets_batches.append(
                torch.as_tensor(batch_targets).cpu().numpy().reshape(-1)
            )
            labeled_batches.append(
                torch.as_tensor(batch_labeled).bool().cpu().numpy().reshape(-1)
            )
    return (
        np.concatenate(logits_batches, axis=0),
        np.concatenate(targets_batches).astype(np.int64),
        np.concatenate(labeled_batches).astype(bool),
    )


def estimate_class_frequencies_parametric(
    logits: np.ndarray,
    targets: np.ndarray,
    labeled: np.ndarray,
    *,
    num_classes: int,
    num_known_classes: int,
) -> np.ndarray:
    """Estimate per-class sample frequencies from parametric logits.

    1. Predict argmax class for every sample (known AND novel heads share the
       same classifier, so novel classes are covered too).
    2. Align predicted-class IDs to true class IDs via Hungarian matching on
       the labeled subset only — no novel ground truth is used.
    3. Count mapped predictions over ALL samples.

    Unlike the KMeans cluster-size estimator this does not flatten the
    distribution: the classifier already separates known/novel classes that
    KMeans merges, so head/tail ratios survive.
    """
    predictions = logits.argmax(axis=1).astype(np.int64)
    labeled_predictions = predictions[labeled]
    labeled_targets = targets[labeled]
    if not np.any(labeled):
        raise ValueError("parametric frequency estimation requires labeled samples")
    if np.any(labeled_targets < 0):
        raise ValueError("labeled targets must be non-negative class IDs")

    contingency = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(contingency, (labeled_predictions, labeled_targets), 1)
    pred_rows, class_cols = linear_sum_assignment(-contingency)
    prediction_to_class = {
        int(pred): int(cls) for pred, cls in zip(pred_rows, class_cols)
    }

    estimate = np.zeros(num_classes, dtype=np.float64)
    counts = np.bincount(predictions, minlength=num_classes)
    for pred, cls in prediction_to_class.items():
        estimate[cls] = counts[pred]
    # Predicted classes that received no labeled anchor keep zero mass;
    # floor at 1 so capacity targets and logit adjustment stay finite.
    estimate = np.maximum(estimate, 1.0)
    return estimate


def estimate_class_frequencies_labeled(
    loader: Any,
    *,
    num_classes: int,
    num_known_classes: int,
) -> "torch.Tensor":
    """Exact per-class support for KNOWN classes, from labeled GT only.

    Known classes: exact bincount over labeled targets of the distribution
    loader (labels are available by protocol). Novel classes: uniform
    estimate at the mean labeled-known support — novel counts are unknowable
    without ground truth, and uniformity is the standard GCD assumption.

    Unlike the KMeans/parametric estimators this is deterministic and exact
    for the known side: the head/tail structure matches the true imbalance
    and no longer drifts across re-estimation rounds, so gate warm-start and
    capacity targets stop chasing a moving target.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for batch in loader:
        targets = getattr(batch, "known_targets", None)
        if targets is None:
            targets = batch["known_targets"]
        mask = getattr(batch, "labeled_mask", None)
        if mask is None:
            mask = batch["labeled_mask"]
        labeled_targets = torch.as_tensor(targets).long()[
            torch.as_tensor(mask).bool()
        ]
        valid = labeled_targets >= 0
        counts += torch.bincount(
            labeled_targets[valid], minlength=num_classes
        ).float()
    known_counts = counts[:num_known_classes]
    if not known_counts.numel() or known_counts.sum() <= 0:
        raise ValueError(
            "labeled-bincount estimation requires labeled known samples"
        )
    frequencies = counts.clone()
    frequencies[:num_known_classes] = known_counts.clamp_min(1.0)
    # Novel: uniform at the floored mean labeled-known support — empty known
    # classes still count as existing (>=1) so the novel scale stays honest.
    uniform_novel = float(frequencies[:num_known_classes].mean())
    frequencies[num_known_classes:] = uniform_novel
    return frequencies


def estimate_class_frequencies(
    model: Any,
    loader: Any,
    *,
    num_classes: int,
    num_known_classes: int,
    finetune_from_block: int,
    random_state: int,
    device: torch.device,
    kmeans_backend: str = "faiss_gpu",
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    known_frequency_mapping: str = "bacon_rank",
) -> torch.Tensor:
    from prototail_gcd.models.dino_backbone import forward_global_and_patches

    if kmeans_backend == "faiss_gpu" and device.type == "cpu":
        kmeans_backend = "sklearn_cpu"

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labeled_masks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, DistributionBatch):
                images = batch.images
                batch_targets = batch.known_targets
                batch_labeled = batch.labeled_mask
            else:
                images, batch_targets, _sample_ids, batch_labeled = batch

            images = images.to(device, non_blocking=True)
            batch_targets = torch.as_tensor(batch_targets).cpu().numpy()
            batch_labeled = torch.as_tensor(batch_labeled).bool().cpu().numpy()
            backbone = getattr(model, "contrastive_backbone", getattr(model, "backbone", None))
            if backbone is None:
                raise ValueError("Model has no backbone for feature extraction")
            global_features, _ = forward_global_and_patches(
                backbone,
                images,
                finetune_from_block=finetune_from_block,
            )
            features.append(F.normalize(global_features, dim=-1).cpu().numpy())
            targets.append(batch_targets.reshape(-1))
            labeled_masks.append(batch_labeled.reshape(-1))

    all_features = np.concatenate(features, axis=0)
    all_targets = np.concatenate(targets).astype(np.int64)
    all_labeled = np.concatenate(labeled_masks).astype(bool)
    if not all_labeled.any():
        raise ValueError("distribution estimation requires labeled samples")
    if np.any(all_targets[all_labeled] < 0):
        raise ValueError("labeled targets must be non-negative class IDs")

    if KMeans is not _ORIG_KMEANS:
        assignments = KMeans(
            n_clusters=num_classes,
            random_state=random_state,
            n_init=kmeans_n_init,
        ).fit_predict(all_features)
    else:
        assignments = fit_predict_kmeans(
            all_features,
            num_classes,
            seed=random_state,
            backend=kmeans_backend,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
        )
    cluster_counts = np.bincount(assignments, minlength=num_classes)
    contingency = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(
        contingency,
        (assignments[all_labeled], all_targets[all_labeled]),
        1,
    )
    normalized_contingency = np.divide(
        contingency,
        cluster_counts[:, None],
        out=np.zeros_like(contingency, dtype=np.float64),
        where=cluster_counts[:, None] > 0,
    )
    cluster_rows, class_columns = linear_sum_assignment(
        normalized_contingency.max() - normalized_contingency
    )
    cluster_for_class = {
        int(class_id): int(cluster_id)
        for cluster_id, class_id in zip(cluster_rows, class_columns)
    }

    known_cluster_ids = [
        cluster_for_class[class_id]
        for class_id in range(num_known_classes)
        if class_id in cluster_for_class
    ]
    estimate = np.ones(num_classes, dtype=np.float32)
    if len(known_cluster_ids) != num_known_classes:
        raise RuntimeError("known-class frequency alignment is incomplete")
    if known_frequency_mapping == "direct_hungarian":
        for class_id in range(num_known_classes):
            estimate[class_id] = cluster_counts[cluster_for_class[class_id]]
    elif known_frequency_mapping == "bacon_rank":
        known_labeled_support = np.bincount(
            all_targets[all_labeled],
            minlength=num_known_classes,
        )[:num_known_classes]
        known_by_support = np.argsort(known_labeled_support, kind="stable")
        sorted_known_cluster_sizes = np.sort(cluster_counts[known_cluster_ids])
        estimate[known_by_support] = sorted_known_cluster_sizes
    else:
        raise ValueError(f"Unsupported known_frequency_mapping: {known_frequency_mapping}")

    novel_cluster_ids = sorted(set(range(num_classes)) - set(known_cluster_ids))
    novel_cluster_counts = np.sort(cluster_counts[novel_cluster_ids])[::-1]
    novel_count = num_classes - num_known_classes
    estimate[num_known_classes:] = novel_cluster_counts[:novel_count]
    return torch.as_tensor(estimate, dtype=torch.float32, device=device)


def distribution_logit_adjustment(
    frequencies: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    probabilities = frequencies / frequencies.sum().clamp_min(1e-12)
    return torch.log(probabilities.pow(strength) + 1e-12)


def distribution_regularizer(
    logits: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    temperature: float,
    power: float,
) -> torch.Tensor:
    average = (logits / temperature).softmax(dim=1).mean(dim=0)
    adjusted = average / frequencies.clamp_min(1e-12).pow(power)
    adjusted = adjusted / adjusted.sum().clamp_min(1e-12)
    return -torch.sum(torch.log(adjusted.pow(-adjusted))) + np.log(
        float(adjusted.numel())
    )


def soft_pseudo_label_contrastive_loss(
    *,
    logits: torch.Tensor,
    contrastive_features: torch.Tensor,
    labeled_targets: torch.Tensor,
    labeled_mask: torch.Tensor,
    estimated_frequencies: torch.Tensor,
    logit_adjustment: torch.Tensor,
    num_views: int,
    temperature: float,
    known_sampling_power: float,
    novel_sampling_power: float,
) -> torch.Tensor:
    batch_size = labeled_mask.numel()
    logits_by_view = logits.chunk(num_views)
    probabilities = torch.stack(
        [
            (view_logits / temperature - logit_adjustment).softmax(dim=1)
            for view_logits in logits_by_view
        ],
        dim=0,
    ).mean(dim=0)

    labeled_probabilities = probabilities[labeled_mask]
    unlabeled_probabilities = probabilities[~labeled_mask]
    confidence, predicted_classes = unlabeled_probabilities.max(dim=1)
    minimum_frequency = estimated_frequencies.min()
    frequency_ratio = minimum_frequency / estimated_frequencies.clamp_min(1e-12)
    sampling_rates = frequency_ratio.pow(novel_sampling_power)
    known_classes_in_batch = labeled_targets[labeled_mask].unique()
    known_rates = frequency_ratio.pow(known_sampling_power)
    sampling_rates[known_classes_in_batch] = known_rates[known_classes_in_batch]

    selected_global_indices: list[torch.Tensor] = []
    unlabeled_global_indices = torch.arange(
        batch_size,
        device=logits.device,
    )[~labeled_mask]
    for class_id in predicted_classes.unique():
        class_positions = torch.nonzero(
            predicted_classes == class_id,
            as_tuple=False,
        ).flatten()
        draw_count = torch.distributions.Binomial(
            total_count=float(class_positions.numel()),
            probs=sampling_rates[class_id].clamp(0.0, 1.0),
        ).sample().to(torch.long)
        keep_count = min(int(draw_count.item()), class_positions.numel())
        if keep_count:
            class_confidence = confidence[class_positions]
            top_positions = class_positions[
                torch.topk(class_confidence, k=keep_count).indices
            ]
            selected_global_indices.append(
                unlabeled_global_indices[top_positions]
            )

    normalized_features = F.normalize(contrastive_features, dim=-1)
    feature_views = normalized_features.chunk(num_views)
    labeled_features = torch.stack(
        [view_features[labeled_mask] for view_features in feature_views],
        dim=1,
    )
    pseudo_features = normalized_features.new_empty(
        (0, num_views, normalized_features.shape[-1])
    )
    selected_probabilities = probabilities.new_empty((0, probabilities.shape[1]))
    if selected_global_indices:
        selected_indices = torch.cat(selected_global_indices)
        pseudo_features = torch.stack(
            [view_features[selected_indices] for view_features in feature_views],
            dim=1,
        )
        selected_probabilities = probabilities[selected_indices]

    all_features = torch.cat((labeled_features, pseudo_features), dim=0)
    soft_targets = torch.cat(
        (labeled_probabilities, selected_probabilities),
        dim=0,
    )
    soft_mask = soft_targets @ soft_targets.T
    soft_mask.fill_diagonal_(1.0)
    return SemiConLoss()(all_features, soft_mask.detach())