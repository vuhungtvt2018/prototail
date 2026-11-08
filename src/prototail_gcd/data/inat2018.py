"""iNaturalist 2018 full-scale exploratory GCD split (not an official protocol)."""

from prototail_gcd.data.manifest_gcd import ManifestGCDAdapter


class INat2018Adapter(ManifestGCDAdapter):
    def __init__(self) -> None:
        super().__init__("inat2018", n_classes=8142, n_known=4071)
