"""Dataset adapter contract shared by GCD baselines."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class DatasetBundle:
    """Loaders and class metadata produced from one GCD data protocol."""

    train_loader: Any
    evaluation_loaders: Mapping[str, Any]
    num_classes: int
    known_class_ids: tuple[int, ...]
    class_counts: tuple[int, ...] | None = None


class DatasetAdapter(Protocol):
    """Build data without coupling a benchmark to a particular baseline."""

    name: str

    def build(self, config: Mapping[str, Any]) -> DatasetBundle:
        """Create splits/loaders and return stable class/split metadata."""
        ...
