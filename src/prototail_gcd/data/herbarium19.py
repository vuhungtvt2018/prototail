"""Natural long-tail Herbarium19 (683 total, 341 known) dataset adapter."""

from prototail_gcd.data.manifest_gcd import ManifestGCDAdapter


class Herbarium19Adapter(ManifestGCDAdapter):
    def __init__(self) -> None:
        super().__init__("herbarium19", n_classes=683, n_known=341)
