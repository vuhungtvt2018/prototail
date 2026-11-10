"""Paper D1/D2 persistent selection, without any novel ground-truth labels.

Known rounds: one most reliable known category; top 80%, max 500.
Novel rounds: two seeded KMeans partitions, stability, silhouette, agreement,
minimum size, DBI deterioration guard and no-remap; max 200/cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances_argmin


@dataclass(frozen=True)
class Snapshot:
    sample_ids: np.ndarray
    features: np.ndarray
    probs: np.ndarray
    known_targets: np.ndarray
    originally_labeled: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        if not (len(self.features) == len(self.probs) == len(self.known_targets) == len(self.originally_labeled) == n):
            raise ValueError("Snapshot arrays have inconsistent lengths")
        if len(set(self.sample_ids.tolist())) != n:
            raise ValueError("Stable sample_ids must be unique")
        if np.any(self.known_targets[~self.originally_labeled] != -1):
            raise RuntimeError("Novel/unlabeled GT cannot enter paper selection")


class PaperDualSelector:
    """Explicit schedule; 20 known rounds and 6 novel rounds at most.

    Known round epochs in the manuscript are not specified; the default here
    (30..190 every 10 epochs) is an implementation choice, not paper provenance.
    """
    def __init__(
        self,
        n_known: int,
        n_classes: int,
        *,
        known_epochs: tuple[int, ...] = tuple(range(30, 200, 10)),
        novel_epochs: tuple[int, ...] = (50, 60, 70, 80, 90, 100),
        seed: int = 0,
        class_temperature: float = 0.1,
    ) -> None:
        if not (0 < n_known < n_classes):
            raise ValueError("Selector requires both known and novel classes")
        if len(set(known_epochs)) != len(known_epochs) or len(known_epochs) > 20:
            raise ValueError("Known selection permits at most 20 distinct rounds")
        if len(set(novel_epochs)) != len(novel_epochs) or len(novel_epochs) > 6:
            raise ValueError("Novel selection permits at most 6 distinct rounds")
        if any(epoch < 50 or epoch > 100 for epoch in novel_epochs):
            raise ValueError("Paper novel selection occurs only at epochs 50..100")
        self.n_known = n_known
        self.n_classes = n_classes
        self.known_epochs = frozenset(known_epochs)
        self.novel_epochs = frozenset(novel_epochs)
        self.seed = seed
        self.class_temperature = class_temperature
        self.previous_accepted_dbi: float | None = None
        self.executed_known: set[int] = set()
        self.executed_novel: set[int] = set()

    def scheduled(self, epoch: int) -> bool:
        return epoch in self.known_epochs or epoch in self.novel_epochs

    @torch.no_grad()
    def collect(self, model: Any, adapter: Any, loader: Any, device: torch.device) -> Snapshot:
        was_training = model.training
        model.eval()
        sid: list[np.ndarray] = []
        feat: list[np.ndarray] = []
        prob: list[np.ndarray] = []
        known: list[np.ndarray] = []
        labeled: list[np.ndarray] = []
        try:
            for batch in loader:
                # Original labels of unlabeled samples are deliberately unavailable.
                if hasattr(batch, "known_targets"):
                    raw_targets = batch.known_targets.detach().cpu().numpy()
                    raw_mask = batch.labeled_mask.detach().cpu().numpy().astype(bool)
                    batch_images = batch.images if hasattr(batch, "images") else batch.views[0]
                    batch_ids = batch.sample_ids.detach().cpu().numpy()
                else:
                    # Tuple loader: (images, targets, sample_ids, is_labeled)
                    batch_images = batch[0]
                    raw_targets = batch[1].detach().cpu().numpy() if isinstance(batch[1], torch.Tensor) else np.asarray(batch[1])
                    batch_ids = batch[2].detach().cpu().numpy() if isinstance(batch[2], torch.Tensor) else np.asarray(batch[2])
                    raw_mask = batch[3].detach().cpu().numpy().astype(bool) if isinstance(batch[3], torch.Tensor) else np.asarray(batch[3], dtype=bool)

                if np.any(raw_targets[~raw_mask] != -1):
                    raise RuntimeError("Unlabeled GT leak in distribution loader")
                output = adapter.forward_eval(model, batch_images.to(device))
                if "classification" in output.branches:
                    cls_branch = output.branches["classification"]
                    feature_branch = output.branches.get("contrastive", cls_branch)
                else:
                    cls_branch = output.branches["student"]
                    feature_branch = cls_branch
                if cls_branch.logits is None or feature_branch.global_features is None:
                    raise RuntimeError("Selector requires logits and CLS features")
                logits = cls_branch.logits.float() / self.class_temperature
                feat.append(torch.nn.functional.normalize(
                    feature_branch.global_features.float(), dim=-1
                ).cpu().numpy())
                prob.append(logits.softmax(-1).cpu().numpy())
                known.append(raw_targets)
                labeled.append(raw_mask)
                sid.append(batch_ids)
        finally:
            model.train(was_training)
        return Snapshot(
            np.concatenate(sid).astype(np.int64),
            np.concatenate(feat).astype(np.float32),
            np.concatenate(prob).astype(np.float32),
            np.concatenate(known).astype(np.int64),
            np.concatenate(labeled).astype(bool),
        )

    def select_known(self, snap: Snapshot, store: Any) -> dict[str, Any]:
        promoted = store.export_state()
        # D1 class reliability MUST use original known GT only, not its own
        # accepted pseudo targets; the latter can make an unreliable class
        # appear 100% accurate (feedback loop especially in LT20).
        targets = snap.known_targets
        pool = snap.originally_labeled
        preds = snap.probs.argmax(axis=1)
        accuracies = np.full(self.n_known, -1.0, dtype=np.float32)
        for c in range(self.n_known):
            which = pool & (targets == c)
            if np.any(which):
                n = int(which.sum())
                correct = int((preds[which] == c).sum())
                accuracies[c] = float((correct + 1) / (n + 2))
        if np.all(accuracies < 0):
            return {"accepted": 0, "reason": "no_original_supervised_known_classes"}
        # Do not reaccept original labeled or previously promoted IDs.
        already_used = np.isin(snap.sample_ids, np.asarray(list(promoted), dtype=np.int64))
        eligible = (~pool) & (~already_used)
        for c in np.argsort(-accuracies, kind="stable"):
            if accuracies[c] < 0:
                continue
            candidate_indices = np.flatnonzero(eligible & (preds == c))
            if not len(candidate_indices):
                continue
            n_keep = min(500, int(np.ceil(0.8 * len(candidate_indices))))
            sorted_indices = candidate_indices[
                np.argsort(-snap.probs[candidate_indices, c], kind="stable")
            ][:n_keep]
            pairs = [(int(snap.sample_ids[i]), int(c)) for i in sorted_indices]
            accepted = store.add_accepted(pairs)
            return {"accepted": accepted, "class_id": int(c), "supervised_accuracy": float(accuracies[c])}
        return {"accepted": 0, "reason": "no_eligible_known_predictions"}

    def _partition(self, features: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(features)
        if n <= self.n_classes:
            raise ValueError("Need more samples than classes for paper clustering")
        rng = np.random.default_rng(seed)
        fit_ids = rng.choice(n, min(n, 15000), replace=False)
        estimator = KMeans(n_clusters=self.n_classes, n_init=10, random_state=seed)
        estimator.fit(features[fit_ids])
        labels = pairwise_distances_argmin(features, estimator.cluster_centers_)
        return labels.astype(np.int64), estimator.cluster_centers_

    def select_novel(self, snap: Snapshot, store: Any, epoch: int) -> dict[str, Any]:
        # There must not be any reference to targets on unlabeled samples.
        ids, features, probs = snap.sample_ids, snap.features, snap.probs
        # Paper: FIT on unlabeled samples only, then assign labeled known
        # examples to the learned centroids for known-cluster alignment.
        unlabeled = ~snap.originally_labeled
        if int(unlabeled.sum()) <= self.n_classes:
            return {"accepted": 0, "reason": "insufficient_unlabeled_samples"}
        labels_a_u, centers_a = self._partition(features[unlabeled], self.seed + epoch * 2)
        labels_b_u, centers_b = self._partition(features[unlabeled], self.seed + epoch * 2 + 1)
        labels_a = pairwise_distances_argmin(features, centers_a).astype(np.int64)
        labels_b = pairwise_distances_argmin(features, centers_b).astype(np.int64)
        k = self.n_classes
        size_a = np.bincount(labels_a_u, minlength=k)
        size_b = np.bincount(labels_b_u, minlength=k)
        cross = np.zeros((k, k), dtype=np.int64)
        np.add.at(cross, (labels_a_u, labels_b_u), 1)
        union = size_a[:, None] + size_b[None, :] - cross
        jaccard = np.divide(cross, union, out=np.zeros_like(cross, dtype=np.float64), where=union > 0)
        stability = np.max(jaccard, axis=1)
        # Match known category labels to clusters using labeled training samples only.
        contingency = np.zeros((k, self.n_known), dtype=np.int64)
        labeled = snap.originally_labeled
        if not np.any(labeled):
            raise ValueError("Known-cluster alignment requires labeled known data")
        if np.any((snap.known_targets[labeled] < 0) | (snap.known_targets[labeled] >= self.n_known)):
            raise RuntimeError("Invalid label found in known supervised pool")
        np.add.at(contingency, (labels_a[labeled], snap.known_targets[labeled]), 1)
        known_cluster_rows, known_cols = linear_sum_assignment(-contingency)
        if len(known_cols) != self.n_known:
            raise RuntimeError("Known-cluster matching incomplete")
        known_clusters = set(int(x) for x in known_cluster_rows)
        # Silhouette is defined on a random subset; fail closed when undefined.
        subset_rng = np.random.default_rng(self.seed + 7 + epoch)
        unlabeled_ids = np.flatnonzero(unlabeled)
        sub = subset_rng.choice(unlabeled_ids, min(len(unlabeled_ids), 2000), replace=False)
        sub_clusters = labels_a[sub]
        uniq = len(np.unique(sub_clusters))
        if uniq < 2 or uniq >= len(sub):
            return {"accepted": 0, "reason": "undefined_silhouette"}
        sil_values = silhouette_samples(features[sub], sub_clusters)
        sil_means = np.full(k, -np.inf, dtype=np.float64)
        for c in np.unique(sub_clusters):
            sil_means[c] = float(sil_values[sub_clusters == c].mean())
        median_silhouette = float(np.median(sil_means[np.isfinite(sil_means)]))
        if len(np.unique(labels_a_u)) < 2:
            return {"accepted": 0, "reason": "undefined_dbi"}
        dbi = float(davies_bouldin_score(features[unlabeled], labels_a_u))
        if self.previous_accepted_dbi is not None and dbi > 1.05 * self.previous_accepted_dbi:
            return {"accepted": 0, "reason": "dbi_deteriorated", "dbi": dbi}
        accepted_total, accepted_clusters = 0, 0
        predictions = probs.argmax(axis=1)
        existing = store.export_state()
        for cluster in range(k):
            if cluster in known_clusters or size_a[cluster] < 10:
                continue
            if stability[cluster] < 0.6 or sil_means[cluster] < median_silhouette:
                continue
            members = np.flatnonzero((labels_a == cluster) & unlabeled)
            member_predictions = predictions[members]
            novel_preds = member_predictions[member_predictions >= self.n_known]
            if len(novel_preds) == 0:
                continue
            majority_count = np.bincount(novel_preds, minlength=k)
            novel_id = int(np.argmax(majority_count))
            agreement = float(majority_count[novel_id] / len(members))
            if agreement < 0.7:
                continue
            prior_ids = {int(sample_id) for sample_id, old_class in existing.items() if old_class == novel_id}
            if prior_ids:
                member_ids = {int(ids[idx]) for idx in members}
                overlap = len(prior_ids & member_ids) / len(prior_ids | member_ids)
                if overlap < 0.3:
                    continue
            remaining = np.asarray([i for i in members if int(ids[i]) not in existing], dtype=np.int64)
            if len(remaining) == 0:
                continue
            # Rank by probability assigned to the accepted majority output.
            best = remaining[np.argsort(-probs[remaining, novel_id], kind="stable")][:200]
            pairs = [(int(ids[i]), novel_id) for i in best]
            accepted_now = store.add_accepted(pairs)
            if accepted_now:
                accepted_clusters += 1
                accepted_total += accepted_now
                existing.update(dict(pairs))
        if accepted_total:
            self.previous_accepted_dbi = dbi
        return {"accepted": accepted_total, "clusters": accepted_clusters, "dbi": dbi}

    def run(self, epoch: int, model: Any, adapter: Any, loader: Any, device: torch.device, store: Any) -> dict[str, Any]:
        if not self.scheduled(epoch):
            return {}
        snap = self.collect(model, adapter, loader, device)
        event: dict[str, Any] = {"epoch": epoch, "store_before": len(store.export_state())}
        if epoch in self.known_epochs and epoch not in self.executed_known:
            event["known"] = self.select_known(snap, store)
            self.executed_known.add(epoch)
        if epoch in self.novel_epochs and epoch not in self.executed_novel:
            event["novel"] = self.select_novel(snap, store, epoch)
            self.executed_novel.add(epoch)
        event["store_after"] = len(store.export_state())
        return event
