"""Extension contract for optional methods layered over a GCD baseline."""

from typing import Any, Mapping, Protocol, Sequence

from prototail_gcd.contracts import (
    ModelOutput,
    PredictionOutput,
    TrainingBatch,
)


class MethodPlugin(Protocol):
    """A method adds outputs/losses without replacing baseline semantics."""

    name: str
    required_capabilities: frozenset[str]

    def attach(
        self,
        model: Any,
        config: Mapping[str, Any],
        num_classes: int,
        feature_dim: int,
    ) -> None:
        """Attach method-owned trainable modules to the baseline model."""
        ...

    def augment_output(
        self,
        output: ModelOutput,
    ) -> ModelOutput:
        """Compute method-specific outputs from baseline-exposed features."""
        ...

    def compute_losses(
        self,
        output: ModelOutput,
        batch: TrainingBatch,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return named auxiliary losses with config-owned weights."""
        ...

    def optimizer_groups(
        self,
        model: Any,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Return named optimizer groups, including intentional shared params."""
        ...

    def shared_parameters(self) -> Sequence[Any]:
        """Return baseline parameters also stepped by a method optimizer."""
        ...

    def after_optimizer_step(
        self,
        output: ModelOutput,
        batch: TrainingBatch,
        context: Mapping[str, Any],
    ) -> None:
        """Update EMA prototypes or other non-gradient state after a step."""
        ...

    def predict(self, output: ModelOutput) -> PredictionOutput:
        """Use method-specific inference when the plugin is enabled."""
        ...
