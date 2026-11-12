from importlib import import_module
from pathlib import Path
import sys
import unittest

import torch


def load_modules():
    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    return (
        import_module("prototail_gcd.baselines.bacon.losses"),
        import_module("prototail_gcd.contracts"),
    )


losses, contracts = load_modules()


class _FakeLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


def _batch(labeled_targets, unlabeled_count, num_classes):
    """DistributionBatch with `unlabeled_count` masked targets (UNKNOWN)."""
    known = torch.tensor(labeled_targets, dtype=torch.long)
    unknown = torch.full((unlabeled_count,), contracts.UNKNOWN_TARGET, dtype=torch.long)
    targets = torch.cat([known, unknown])
    mask = torch.cat(
        [torch.ones(len(known), dtype=torch.bool), torch.zeros(unlabeled_count, dtype=torch.bool)]
    )
    return contracts.DistributionBatch(
        images=targets,  # placeholder
        known_targets=targets,
        labeled_mask=mask,
        sample_ids=torch.arange(len(targets)),
    )


class EstimateClassFrequenciesLabeledTests(unittest.TestCase):
    def test_exact_known_counts_and_uniform_novel(self):
        num_classes, num_known = 6, 3
        loader = _FakeLoader(
            [
                _batch([0, 0, 1, 2], 4, num_classes),
                _batch([1, 2, 2], 3, num_classes),
            ]
        )
        freq = losses.estimate_class_frequencies_labeled(
            loader, num_classes=num_classes, num_known_classes=num_known
        )
        self.assertIsInstance(freq, torch.Tensor)
        self.assertEqual(freq.shape, (num_classes,))
        # Labeled bincount: class0=2, class1=2, class2=3 (exact, not estimated)
        self.assertEqual(freq[0].item(), 2.0)
        self.assertEqual(freq[1].item(), 2.0)
        self.assertEqual(freq[2].item(), 3.0)
        # Novel: uniform at mean labeled-known support = (2+2+3)/3 = 7/3
        self.assertAlmostEqual(freq[3].item(), 7.0 / 3.0, places=4)
        self.assertAlmostEqual(freq[5].item(), 7.0 / 3.0, places=4)

    def test_empty_known_class_floored_at_one(self):
        num_classes, num_known = 4, 2
        loader = _FakeLoader([_batch([0, 0], 5, num_classes)])
        freq = losses.estimate_class_frequencies_labeled(
            loader, num_classes=num_classes, num_known_classes=num_known
        )
        self.assertEqual(freq[0].item(), 2.0)
        self.assertEqual(freq[1].item(), 1.0, "empty known class must floor at 1")
        self.assertAlmostEqual(freq[2].item(), 1.5)
        self.assertAlmostEqual(freq[3].item(), 1.5)

    def test_deterministic_across_calls(self):
        num_classes, num_known = 5, 2
        loader = _FakeLoader([_batch([0, 1, 1], 7, num_classes)])
        a = losses.estimate_class_frequencies_labeled(
            loader, num_classes=num_classes, num_known_classes=num_known
        )
        b = losses.estimate_class_frequencies_labeled(
            loader, num_classes=num_classes, num_known_classes=num_known
        )
        self.assertTrue(torch.equal(a, b), "must be deterministic (no k-means noise)")

    def test_requires_labeled_samples(self):
        num_classes, num_known = 4, 2
        loader = _FakeLoader([_batch([], 10, num_classes)])
        with self.assertRaises(ValueError):
            losses.estimate_class_frequencies_labeled(
                loader, num_classes=num_classes, num_known_classes=num_known
            )

    def test_head_tail_ratio_matches_true_structure(self):
        """Imbalanced labeled counts must survive: ratio preserved, not flattened."""
        num_classes, num_known = 6, 3
        # head class 0: 20 labeled, tail class 2: 1 labeled -> ratio 20
        loader = _FakeLoader(
            [_batch([0] * 20 + [1] * 5 + [2], 50, num_classes)]
        )
        freq = losses.estimate_class_frequencies_labeled(
            loader, num_classes=num_classes, num_known_classes=num_known
        )
        ratio = freq.max().item() / freq.min().item()
        self.assertEqual(freq[0].item(), 20.0)
        self.assertEqual(freq[2].item(), 1.0)
        self.assertAlmostEqual(ratio, 20.0, places=5)


if __name__ == "__main__":
    unittest.main()
