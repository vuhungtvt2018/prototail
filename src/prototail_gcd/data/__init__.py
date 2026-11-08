"""Dataset protocols and benchmark adapters."""

from prototail_gcd.data.cub import CUBLongTailAdapter
from prototail_gcd.data.cars import CarsLongTailAdapter
from prototail_gcd.data.aircraft import AircraftLongTailAdapter
from prototail_gcd.data.imagenet100_lt import ImageNet100LTAdapter
from prototail_gcd.data.herbarium19 import Herbarium19Adapter
from prototail_gcd.data.inat2018 import INat2018Adapter

ADAPTERS = {
    "cub": CUBLongTailAdapter,
    "cub200": CUBLongTailAdapter,
    "cars": CarsLongTailAdapter,
    "stanford_cars": CarsLongTailAdapter,
    "scars": CarsLongTailAdapter,
    "aircraft": AircraftLongTailAdapter,
    "fgvc_aircraft": AircraftLongTailAdapter,
    "imagenet100_lt": ImageNet100LTAdapter,
    "herbarium19": Herbarium19Adapter,
    "inat2018": INat2018Adapter,
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
    "ImageNet100LTAdapter",
    "Herbarium19Adapter",
    "INat2018Adapter",
    "get_dataset_adapter",
]
