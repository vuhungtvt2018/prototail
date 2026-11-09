"""Explicit CPU / GPU KMeans backends used by eval and frequency estimation."""

from __future__ import annotations

import math
import time

import numpy as np
from sklearn.cluster import KMeans


def require_kmeans_backend(backend: str) -> None:
    if backend == "sklearn_cpu":
        return
    if backend != "faiss_gpu":
        raise ValueError(f"Unknown KMeans backend: {backend}")
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss_gpu requires a GPU-enabled FAISS build; "
            "install it in a compatible separate environment"
        ) from exc
    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
        raise RuntimeError(
            "faiss_gpu was selected but no usable FAISS GPU was detected"
        )


def fit_predict_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 0,
    backend: str = "faiss_gpu",
    n_init: int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    """Return integer cluster assignments; GPU mode must never silently fallback."""
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[0] < n_clusters:
        raise ValueError("Expected a 2D array with N >= n_clusters")
    if n_clusters <= 0 or n_init <= 0 or max_iter <= 0:
        raise ValueError("KMeans counts and iterations must be positive")
    if not np.isfinite(x).all():
        raise ValueError("KMeans features contain NaN or Inf")

    started = time.perf_counter()
    if backend == "sklearn_cpu":
        assignments = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
        ).fit_predict(x)
    elif backend == "faiss_gpu":
        require_kmeans_backend(backend)
        import faiss

        x32 = np.ascontiguousarray(x, dtype=np.float32)
        n, d = x32.shape
        clustering = faiss.Kmeans(
            d=d,
            k=n_clusters,
            niter=max_iter,
            nredo=n_init,
            seed=seed,
            gpu=True,
            spherical=False,
            min_points_per_centroid=1,
            max_points_per_centroid=max(256, math.ceil(n / n_clusters)),
            verbose=False,
        )
        clustering.train(x32)
        _, nearest = clustering.index.search(x32, 1)
        assignments = nearest.reshape(-1)
    else:
        raise ValueError(f"Unknown KMeans backend: {backend}")

    result = np.asarray(assignments, dtype=np.int64)
    if result.shape != (x.shape[0],):
        raise RuntimeError("KMeans returned invalid assignment shape")
    if result.min() < 0 or result.max() >= n_clusters:
        raise RuntimeError("KMeans returned invalid cluster IDs")
    elapsed = time.perf_counter() - started
    gpu_info = ""
    if backend == "faiss_gpu":
        try:
            import faiss

            gpu_info = f" gpus={getattr(faiss, 'get_num_gpus', lambda: 1)()}"
        except Exception:
            gpu_info = ""
    print(
        f"[KMEANS_PROFILE] backend={backend}{gpu_info} n={x.shape[0]} "
        f"k={n_clusters} d={x.shape[1]} n_init={n_init} "
        f"max_iter={max_iter} seconds={elapsed:.3f}",
        flush=True,
    )
    return result
