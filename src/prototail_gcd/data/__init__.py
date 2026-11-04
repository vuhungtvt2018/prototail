"""Dataset protocols and benchmark adapters."""

from prototail_gcd.data.cub import CUBLongTailAdapter
from prototail_gcd.data.cars import CarsLongTailAdapter
from prototail_gcd.data.aircraft import AircraftLongTailAdapter

ADAPTERS = {
    "cub": CUBLongTailAdapter,
    "cub200": CUBLongTailAdapter,
    "cars": CarsLongTailAdapter,
    "stanford_cars": CarsLongTailAdapter,
    "scars": CarsLongTailAdapter,
    "aircraft": AircraftLongTailAdapter,
    "fgvc_aircraft": AircraftLongTailAdapter,
}


def get_dataset_adapter(name: str):
    norm = name.strip().lower()
    if norm not in ADAPTERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported datasets: {sorted(set(ADAPTERS.keys()))}"
        )
    return ADAPTERS[norm]()


__all__ = [
    "CUBLongTailAdapter",
    "CarsLongTailAdapter",
    "AircraftLongTailAdapter",
    "get_dataset_adapter",
]
