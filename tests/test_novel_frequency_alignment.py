"""Novel KMeans support alignment must follow output IDs, never support rank."""
import numpy as np
import pytest
from prototail_gcd.baselines.bacon.losses import align_novel_cluster_counts


def test_alignment_follows_classifier_outputs_not_sorted_cluster_sizes():
    # 1 known + 2 novel clusters: sizes 7 and 3.
    # Cluster 2 (size 7) maps to output 2; cluster 3 (size 3) maps to output 1.
    assignments = np.array([1] * 2 + [2] * 7 + [3] * 3)
    probabilities = np.zeros((12, 3), dtype=np.float32)
    probabilities[:2] = [0.95, 0.03, 0.02]
    probabilities[2:9] = [0.03, 0.02, 0.95]
    probabilities[9:] = [0.02, 0.96, 0.02]
    estimate = align_novel_cluster_counts(assignments, [2, 3], probabilities, 1)
    np.testing.assert_array_equal(estimate, [3, 7])


def test_rejects_ambiguous_unidentifiable_alignment():
    assignments = np.array([2, 2, 3, 3])
    probabilities = np.full((4, 3), 1.0 / 3.0)
    with pytest.raises(RuntimeError, match="unidentifiable"):
        align_novel_cluster_counts(assignments, [2, 3], probabilities, 1)


def test_rejects_missing_classifier_output_dimension():
    assignments = np.array([1, 2, 3])
    probabilities = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="must match"):
        align_novel_cluster_counts(assignments, [2, 3], probabilities, 1)
