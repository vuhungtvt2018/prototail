"""Dataset protocols and benchmark adapters."""

from prototail_gcd.data.cub import CUBLongTailAdapter
from prototail_gcd.data.cars import CarsLongTailAdapter
from prototail_gcd.data.aircraft import AircraftLongTailAdapter
from prototail_gcd.data.herbarium19 import HerbariumLongTailAdapter
from prototail_gcd.data.inat2018 import INat2018LongTailAdapter

ADAPTERS = {
    "cub": CUBLongTailAdapter,
    "cub200": CUBLongTailAdapter,
    "cars": CarsLongTailAdapter,
    "car": CarsLongTailAdapter,
    "stanford_cars": CarsLongTailAdapter,
    "scars": CarsLongTailAdapter,
    "aircraft": AircraftLongTailAdapter,
    "air": AircraftLongTailAdapter,
    "fgvc_aircraft": AircraftLongTailAdapter,
    "herbarium": HerbariumLongTailAdapter,
    "herbarium19": HerbariumLongTailAdapter,
    "herb19": HerbariumLongTailAdapter,
    "inat": INat2018LongTailAdapter,
    "inat2018": INat2018LongTailAdapter,
    "inaturalist": INat2018LongTailAdapter,
    "inaturalist2018": INat2018LongTailAdapter,
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
    "INat2018LongTailAdapter",
    "get_dataset_adapter",
]
