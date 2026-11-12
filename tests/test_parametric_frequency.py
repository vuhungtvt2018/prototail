"""Regression tests for the parametric Hungarian frequency estimator.

Catches the tail=1 collapse observed in the 2025-11 cars/cub runs: novel
classes with no labeled anchor were floored at 1, so nearly every class
looked like tail and the tail-aware gate targets inverted.
"""
import numpy as np
import pytest

from prototail_gcd.baselines.bacon.losses import (
    estimate_class_frequencies_parametric,
)


def _make_logits_and_targets(seed: int = 0):
    rng = np.random.default_rng(seed)
    num_classes = 6
    num_known = 3
    # True counts: known head/tail vary, novel classes sized between.
    true_counts = np.array([500, 100, 50, 200, 80, 20], dtype=np.int64)
    class_order = np.repeat(np.arange(num_classes), true_counts)
    rng.shuffle(class_order)
    targets = class_order.copy()
    # Perfect classifier logits: argmax == true class.
    logits = np.full((len(targets), num_classes), -10.0, dtype=np.float64)
    logits[np.arange(len(targets)), targets] = 10.0
    # Label only the first 200 samples (mixed classes).
    labeled = np.zeros(len(targets), dtype=bool)
    labeled[:200] = True
    return logits, targets, labeled, num_classes, num_known, true_counts


def test_known_counts_recovered_exactly():
    logits, targets, labeled, num_classes, num_known, true_counts = (
        _make_logits_and_targets()
    )
    est = estimate_class_frequencies_parametric(
        logits, targets, labeled,
        num_classes=num_classes, num_known_classes=num_known,
    )
    for c in range(num_known):
        assert est[c] == pytest.approx(true_counts[c])


def test_novel_counts_use_real_mass_not_floor():
    """Novel classes must get the unmatched predicted-class masses (sorted
    largest first), not the floor of 1."""
    logits, targets, labeled, num_classes, num_known, true_counts = (
        _make_logits_and_targets()
    )
    est = estimate_class_frequencies_parametric(
        logits, targets, labeled,
        num_classes=num_classes, num_known_classes=num_known,
    )
    novel_true = np.sort(true_counts[num_known:])[::-1]
    novel_est = np.sort(est[num_known:])[::-1]
    assert np.allclose(novel_est, novel_true)


def test_novel_never_collapses_to_one_when_mass_exists():
    logits, targets, labeled, num_classes, num_known, _ = _make_logits_and_targets()
    est = estimate_class_frequencies_parametric(
        logits, targets, labeled,
        num_classes=num_classes, num_known_classes=num_known,
    )
    # Every novel class must receive real mass well above the floor.
    assert np.all(est[num_known:] > 10.0)


def test_noisy_classifier_keeps_ratio():
    """With 70% argmax accuracy the ratio should stay near the true ratio,
    unlike the old estimator whose floor drove ratio -> num_known count."""
    logits, targets, labeled, num_classes, num_known, true_counts = (
        _make_logits_and_targets(seed=1)
    )
    rng = np.random.default_rng(2)
    flip = rng.random(len(targets)) < 0.3
    random_classes = rng.integers(0, num_classes, size=len(targets))
    noisy = np.where(flip, random_classes, targets)
    logits_noisy = np.full((len(targets), num_classes), -10.0, dtype=np.float64)
    logits_noisy[np.arange(len(targets)), noisy] = 10.0
    est = estimate_class_frequencies_parametric(
        logits_noisy, targets, labeled,
        num_classes=num_classes, num_known_classes=num_known,
    )
    true_ratio = true_counts.max() / true_counts.min()
    est_ratio = est.max() / est.min()
    assert est_ratio <= true_ratio * 1.5
    assert est.min() >= 1.0
