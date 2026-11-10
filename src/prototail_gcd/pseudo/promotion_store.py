"""Persistent D1/D2 sample-ID mapping; no ground-truth Novel labels."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence


class PromotionStore:
    def __init__(self, num_classes: int, num_known_classes: int) -> None:
        if not 0 < num_known_classes < num_classes:
            raise ValueError("Invalid known/novel class partition")
        self.num_classes = int(num_classes)
        self.num_known_classes = int(num_known_classes)
        self._labels: dict[int, int] = {}

    def add_accepted(self, pairs: Iterable[tuple[int, int]]) -> int:
        added = 0
        for sid, label in pairs:
            sid, label = int(sid), int(label)
            if not 0 <= label < self.num_classes:
                raise ValueError(f"Invalid promoted class ID {label}")
            if sid in self._labels:
                # Reject relabeling of an already accepted sample.
                continue
            self._labels[sid] = label
            added += 1
        return added

    def export_state(self) -> dict[int, int]:
        return dict(self._labels)

    def load_state(self, labels: Mapping[int | str, int]) -> None:
        self._labels.clear()
        self.add_accepted((int(k), int(v)) for k, v in labels.items())

    def lookup(self, sample_ids: Sequence[int]) -> dict[int, int]:
        return {
            int(i): self._labels[int(i)]
            for i in sample_ids
            if int(i) in self._labels
        }

    def __len__(self) -> int:
        return len(self._labels)
