import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prototail_gcd.utils.kmeans_backend import (
    fit_predict_kmeans,
    require_kmeans_backend,
)


class KMeansBackendTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [[0., 0.], [0., 0.2], [0.1, 0.],
             [10., 10.], [10., 10.2], [10.1, 10.]],
            dtype=np.float32,
        )

    def test_cpu_matches_existing_sklearn_exactly(self):
        reference = KMeans(
            n_clusters=2, random_state=0, n_init=10, max_iter=300
        ).fit_predict(self.x)
        actual = fit_predict_kmeans(
            self.x, 2, seed=0, backend="sklearn_cpu",
            n_init=10, max_iter=300,
        )
        np.testing.assert_array_equal(actual, reference)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            fit_predict_kmeans(self.x, 7)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            require_kmeans_backend("fake")

    def test_gpu_missing_must_not_fallback(self):
        with patch.dict("sys.modules", {"faiss": None}):
            with self.assertRaises(RuntimeError):
                require_kmeans_backend("faiss_gpu")


if __name__ == "__main__":
    unittest.main()
