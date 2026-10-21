"""Named baseline registry; adapters remain independent implementations."""

from prototail_gcd.baselines.bacon import BaConAdapter
from prototail_gcd.baselines.simgcd import SimGCDAdapter


BASELINE_REGISTRY = {
    "bacon": BaConAdapter,
    "simgcd": SimGCDAdapter,
}


def build_baseline(name: str):
    try:
        adapter_type = BASELINE_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(
            f"Unknown baseline {name!r}; available baselines: {available}"
        ) from error
    return adapter_type()
