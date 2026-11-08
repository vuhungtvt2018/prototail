"""BaCon ImageNet-100-LT (100 total, 50 known) dataset adapter."""

from prototail_gcd.data.manifest_gcd import ManifestGCDAdapter


class ImageNet100LTAdapter(ManifestGCDAdapter):
    def __init__(self) -> None:
        super().__init__("imagenet100_lt", n_classes=100, n_known=50)
