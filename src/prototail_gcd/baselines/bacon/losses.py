"""BaCon-native losses and distribution estimation."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

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


def align_novel_cluster_counts(
    assignments: np.ndarray,
    novel_cluster_ids: list[int],
    classifier_probabilities: np.ndarray,
    num_known_classes: int,
) -> np.ndarray:
    """One-to-one KMeans-cluster -> NOVEL classifier-dimension alignment (no novel GT).

    When the association is uninformative early in training, return a neutral
    uniform capacity support rather than assigning a random cluster rank.
    """
    ids = np.asarray(novel_cluster_ids, dtype=np.int64)
    probs = np.asarray(classifier_probabilities, dtype=np.float64)
    assigned = np.asarray(assignments, dtype=np.int64)
    if probs.ndim != 2 or assigned.shape != (len(probs),):
        raise ValueError("Invalid probabilities / assignments")
    n = probs.shape[1] - num_known_classes
    if len(ids) != n or n <= 0:
        raise ValueError("Novel clusters and output dimensions do not match")
    scores = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float32)
    for i, cluster in enumerate(ids):
        members = assigned == cluster
        counts[i] = int(members.sum())
        if members.any():
            row = probs[members, num_known_classes:].mean(axis=0)
            scores[i] = row / max(float(row.sum()), 1e-12)
    if not np.isfinite(scores).all():
        raise RuntimeError("Invalid novel posterior")
    if n > 1 and float(np.max(scores, axis=1).mean()) < max(0.03, 2.0 / n):
        # Intentionally conservative, no rank-based made-up class identity.
        return np.full(n, max(float(counts.mean()), 1.0), dtype=np.float32)
    rows, cols = linear_sum_assignment(-scores)
    result = np.zeros(n, dtype=np.float32)
    result[cols] = counts[rows]
    return np.maximum(result, 1.0)


def estimate_class_frequencies(
    model: Any,
    loader: Any,
    *,
    num_classes: int,
    num_known_classes: int,
    finetune_from_block: int,
    random_state: int,
    device: torch.device,
    known_frequency_mapping: str = "bacon_rank",
    novel_frequency_mapping: str = "bacon_rank",
    classifier_temperature: float = 0.1,
) -> torch.Tensor:
    from prototail_gcd.models.dino_backbone import forward_global_and_patches

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labeled_masks: list[np.ndarray] = []
    classifier_probs: list[np.ndarray] = []
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
            if novel_frequency_mapping == "output_aligned":
                if not hasattr(model, "projector"):
                    raise RuntimeError("output_aligned support requires SimGCD model")
                output = model(images)
                class_logits = output[2]  # SimGCDNetwork: CLS, projection, logits, patches
                classifier_probs.append(
                    (class_logits.float() / classifier_temperature)
                    .softmax(-1)
                    .cpu()
                    .numpy()
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

    assignments = KMeans(
        n_clusters=num_classes,
        random_state=random_state,
        n_init=10,
    ).fit_predict(all_features)
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
    if len(known_cluster_ids) != num_known_classes:
        raise RuntimeError("Known-class Hungarian alignment is incomplete")
    estimate = np.ones(num_classes, dtype=np.float32)
    if known_frequency_mapping == "direct_hungarian":
        for c in range(num_known_classes):
            estimate[c] = cluster_counts[cluster_for_class[c]]
    elif known_frequency_mapping == "bacon_rank":
        known_support = np.bincount(
            all_targets[all_labeled],
            minlength=num_known_classes,
        )[:num_known_classes]
        known_by_support = np.argsort(known_support, kind="stable")
        estimate[known_by_support] = np.sort(cluster_counts[known_cluster_ids])
    else:
        raise ValueError(f"Unknown known frequency mapping: {known_frequency_mapping}")

    novel_cluster_ids = sorted(set(range(num_classes)) - set(known_cluster_ids))
    novel_count = num_classes - num_known_classes
    if novel_frequency_mapping == "output_aligned":
        if not classifier_probs:
            raise RuntimeError("Output alignment requires classifier predictions")
        estimate[num_known_classes:] = align_novel_cluster_counts(
            assignments,
            novel_cluster_ids,
            np.concatenate(classifier_probs, axis=0),
            num_known_classes,
        )
    elif novel_frequency_mapping == "bacon_rank":
        novel_cluster_counts = np.sort(cluster_counts[novel_cluster_ids])[::-1]
        estimate[num_known_classes:] = novel_cluster_counts[:novel_count]
    else:
        raise ValueError(f"Unknown novel frequency mapping: {novel_frequency_mapping}")
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