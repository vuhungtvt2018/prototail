"""Stable accepted pseudo-labels. No ground-truth novel IDs are read here."""

from __future__ import annotations

from collections.abc import Iterable
import torch

from prototail_gcd.contracts import TrainingBatch


class PromotionStore:
    def __init__(self, total_classes: int, known_classes: int) -> None:
        self.total_classes = int(total_classes)
        self.known_classes = int(known_classes)
        self._labels: dict[int, int] = {}

    def add_accepted(self, pairs: Iterable[tuple[int, int]]) -> int:
        """Called ONLY by a selection round that passed the paper criteria."""
        n_added = 0
        for raw_sample_id, raw_class_id in pairs:
            sample_id = int(raw_sample_id)
            class_id = int(raw_class_id)
            if not 0 <= class_id < self.total_classes:
                raise ValueError("Promoted class ID is outside the classifier")
            previous = self._labels.get(sample_id)
            if previous is not None and previous != class_id:
                raise ValueError("Cannot remap an already-promoted sample")
            if previous is None:
                self._labels[sample_id] = class_id
                n_added += 1
        return n_added

    def lookup(self, batch: TrainingBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return accepted pseudo targets separately from ground-truth labels."""
        ids = batch.sample_ids.detach().cpu().tolist()
        targets = torch.full_like(batch.known_targets, -1)
        mask = torch.zeros_like(batch.labeled_mask, dtype=torch.bool)
        for i, sample_id in enumerate(ids):
            if bool(batch.labeled_mask[i]):
                continue
            promoted = self._labels.get(int(sample_id))
            if promoted is not None:
                targets[i] = promoted
                mask[i] = True
        return targets, mask

    def inject(self, batch: TrainingBatch) -> TrainingBatch:
        """Make accepted unlabeled samples supervised whenever sampled."""
        ids = batch.sample_ids.detach().cpu().tolist()
        targets = batch.known_targets.clone()
        mask = batch.labeled_mask.clone()
        for i, sample_id in enumerate(ids):
            if bool(mask[i]):
                continue
            promoted = self._labels.get(int(sample_id))
            if promoted is not None:
                targets[i] = promoted
                mask[i] = True
        return TrainingBatch(
            views=batch.views,
            known_targets=targets,
            labeled_mask=mask,
            sample_ids=batch.sample_ids,
        )

    def export_state(self) -> dict[int, int]:
        return dict(self._labels)

    def load_state(self, state: dict[int, int]) -> None:
        self._labels = {int(k): int(v) for k, v in state.items()}
