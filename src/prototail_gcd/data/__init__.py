"""Dataset protocols and benchmark adapters."""

from prototail_gcd.data.cub import CUBLongTailAdapter
from prototail_gcd.data.cars import CarsLongTailAdapter
from prototail_gcd.data.aircraft import AircraftLongTailAdapter
from prototail_gcd.data.herbarium19 import HerbariumLongTailAdapter

ADAPTERS = {
    "cub": CUBLongTailAdapter,
    "cub200": CUBLongTailAdapter,
    "cars": CarsLongTailAdapter,
    "stanford_cars": CarsLongTailAdapter,
    "scars": CarsLongTailAdapter,
    "aircraft": AircraftLongTailAdapter,
    "fgvc_aircraft": AircraftLongTailAdapter,
    "herbarium": HerbariumLongTailAdapter,
    "herbarium19": HerbariumLongTailAdapter,
    "herb19": HerbariumLongTailAdapter,
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
    "HerbariumLongTailAdapter",
    "get_dataset_adapter",
]
