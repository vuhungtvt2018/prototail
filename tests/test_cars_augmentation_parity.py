"""Protocol regression: Cars train transform matches the legacy run default (no ColorJitter)."""
from prototail_gcd.data.cars import _make_transforms


def test_cars_train_uses_color_jitter():
    train, test = _make_transforms(224, 0.875)
    train_names = [type(x).__name__ for x in train.transforms]
    test_names = [type(x).__name__ for x in test.transforms]
    assert "ColorJitter" not in train_names
    assert "ColorJitter" not in test_names
