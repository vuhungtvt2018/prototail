"""Framework-neutral records exchanged by data, baseline, and method
plugins.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


UNKNOWN_TARGET = -1


@dataclass(frozen=True)
class TrainingBatch:
    """A normalized batch; unlabeled targets must use ``UNKNOWN_TARGET``."""

    views: Sequence[Any]
    known_targets: Any
    labeled_mask: Any
    sample_ids: Any

    def to(self, device: Any) -> "TrainingBatch":
        def move(value: Any) -> Any:
            if hasattr(value, "to"):
                return value.to(device, non_blocking=True)
            return value

        return TrainingBatch(
            views=tuple(move(view) for view in self.views),
            known_targets=move(self.known_targets),
            labeled_mask=move(self.labeled_mask),
            sample_ids=move(self.sample_ids),
        )


@dataclass(frozen=True)
class DistributionBatch:
    """Single-view batch for frequency estimation without novel GT labels."""

    images: Any
    known_targets: Any
    labeled_mask: Any
    sample_ids: Any

    def to(self, device: Any) -> "DistributionBatch":
        def move(value: Any) -> Any:
            if hasattr(value, "to"):
                return value.to(device, non_blocking=True)
            return value

        return DistributionBatch(
            images=move(self.images),
            known_targets=move(self.known_targets),
            labeled_mask=move(self.labeled_mask),
            sample_ids=move(self.sample_ids),
        )


@dataclass(frozen=True)
class BranchOutput:
    """Outputs exposed by one baseline branch (e.g. SimGCD or BaCon)."""

    global_features: Any
    patch_tokens: Any | None = None
    projection: Any | None = None
    logits: Any | None = None


@dataclass(frozen=True)
class ModelOutput:
    """Named branches keep single- and multi-branch baselines representable."""

    branches: Mapping[str, BranchOutput]
    auxiliary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionOutput:
    """Predictions or features emitted by a baseline's inference protocol."""

    class_ids: Any
    scores: Any | None = None
    features: Any | None = None


@dataclass(frozen=True)
class PseudoLabelCandidate:
    """A selected pseudo-label, keyed by stable dataset identity."""

    sample_id: int | str
    class_id: int
    confidence: float
    source: str
