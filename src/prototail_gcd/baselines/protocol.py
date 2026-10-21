"""Extension contract for baseline-specific model and training behavior."""

from typing import Any, Mapping, Protocol, Sequence

from prototail_gcd.contracts import (
    ModelOutput,
    PredictionOutput,
    TrainingBatch,
)


class BaselineAdapter(Protocol):
    """A baseline owns its forward pass, losses, optimizer, and inference."""

    name: str
    capabilities: frozenset[str]

    def create_training_context(
        self,
        config: Mapping[str, Any],
        *,
        num_views: int,
        total_epochs: int,
    ) -> Mapping[str, Any]:
        """Create baseline-owned criteria, teachers, and scheduled state."""
        ...

    def build_model(
        self,
        model_config: Mapping[str, Any],
        baseline_config: Mapping[str, Any],
        num_classes: int,
    ) -> Any:
        """Build a model from separate model and baseline settings."""
        ...

    def forward(self, model: Any, batch: TrainingBatch) -> ModelOutput:
        """Return named feature, projection, and logit outputs per branch."""
        ...

    def forward_eval(self, model: Any, images: Any) -> ModelOutput:
        """Return named outputs for one-view evaluation images."""
        ...

    def compute_losses(
        self,
        output: ModelOutput,
        batch: TrainingBatch,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Compute native baseline objectives, excluding plugin losses."""
        ...

    def optimizer_groups(
        self,
        model: Any,
        config: Mapping[str, Any],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Return named, disjoint optimizer parameter groups."""
        ...

    def predict(self, output: ModelOutput) -> PredictionOutput:
        """Apply the baseline's native inference protocol."""
        ...

    def before_epoch(
        self,
        model: Any,
        epoch: int,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Refresh scheduled baseline state before the epoch begins."""
        ...

    def after_optimizer_step(
        self,
        model: Any,
        batch: TrainingBatch,
        output: ModelOutput,
        epoch: int,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        """Update teacher or other baseline state after optimizers step."""
        ...

    def after_epoch(
        self,
        model: Any,
        epoch: int,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Perform baseline-specific epoch-boundary updates."""
        ...
