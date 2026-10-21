import importlib.util
import unittest
from importlib import import_module
from pathlib import Path
import sys


REQUIRED_MODULES = ("numpy", "pandas", "torch", "torchvision")
HAS_CUB_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in REQUIRED_MODULES
)


@unittest.skipUnless(HAS_CUB_DEPS, "Install the SimGCD optional dependencies")
class CUBTrainingPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        cls.cub_module = import_module("prototail_gcd.data.cub")
        cls.contracts = import_module("prototail_gcd.contracts")

    def test_unlabeled_view_never_returns_its_ground_truth(self):
        class DatasetStub:
            def __getitem__(self, index):
                return "image", 73, 9001

            def __len__(self):
                return 1

        view = self.cub_module._CUBSubset(
            DatasetStub(),
            [0],
            {73: 4},
            labeled=False,
        )

        _image, target, sample_id, is_labeled = view[0]

        self.assertEqual(target, self.contracts.UNKNOWN_TARGET)
        self.assertEqual(sample_id, 9001)
        self.assertFalse(is_labeled)


if __name__ == "__main__":
    unittest.main()
