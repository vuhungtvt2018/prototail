import unittest
import torch

from prototail_gcd.contracts import TrainingBatch
from prototail_gcd.pseudo.promotion_store import PromotionStore


class PromotionStoreTests(unittest.TestCase):
    def test_promotion_store_accepts_and_injects_samples(self):
        store = PromotionStore(total_classes=10, known_classes=5)
        # Add sample 101 promoted to class 6 (novel)
        n = store.add_accepted([(101, 6)])
        self.assertEqual(n, 1)

        batch = TrainingBatch(
            views=(torch.zeros(2, 4),),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([100, 101]),
        )

        injected = store.inject(batch)
        self.assertTrue(injected.labeled_mask[0].item())
        self.assertEqual(injected.known_targets[0].item(), 0)
        # Sample 101 was unlabeled, now supervised class 6
        self.assertTrue(injected.labeled_mask[1].item())
        self.assertEqual(injected.known_targets[1].item(), 6)

    def test_promotion_store_rejects_remapping_and_out_of_bounds(self):
        store = PromotionStore(total_classes=10, known_classes=5)
        store.add_accepted([(101, 6)])
        # Cannot remap already-promoted sample to a different class
        with self.assertRaises(ValueError):
            store.add_accepted([(101, 7)])

        # Out of bounds class ID raises
        with self.assertRaises(ValueError):
            store.add_accepted([(102, 10)])


if __name__ == "__main__":
    unittest.main()
