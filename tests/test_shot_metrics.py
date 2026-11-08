import unittest
import numpy as np

from prototail_gcd.evaluation.clustering import _add_shot_metrics, clustering_metrics


class ShotMetricsTests(unittest.TestCase):
    def test_cub_100_known_100_novel_split_sizes(self):
        # 100 known, 100 novel classes with tied and long-tailed frequencies
        class_counts = np.ones(200, dtype=np.int64)
        class_counts[:50] = 10
        class_counts[50:100] = 5
        class_counts[100:150] = 10
        class_counts[150:200] = 5

        metrics = {}
        targets = np.repeat(np.arange(200), 2)
        predictions = targets.copy()

        _add_shot_metrics(
            metrics,
            predictions,
            targets,
            class_counts,
            known_class_count=100,
            accuracy_scale=100.0,
            empty_group_value=None,
        )

        # In 100 classes: cut1 = 33, cut2 = 66 -> many=33, med=33, few=34
        for prefix in ("k", "u"):
            self.assertIn(f"{prefix}_many", metrics)
            self.assertIn(f"{prefix}_med", metrics)
            self.assertIn(f"{prefix}_few", metrics)
            self.assertEqual(metrics[f"{prefix}_many"], 100.0)
            self.assertEqual(metrics[f"{prefix}_med"], 100.0)
            self.assertEqual(metrics[f"{prefix}_few"], 100.0)

    def test_cars_98_known_98_novel_split_sizes(self):
        # 98 known, 98 novel classes -> 32 / 33 / 33
        class_counts = np.ones(196, dtype=np.int64)
        known_ids = np.arange(98)
        ordered = known_ids[np.lexsort((known_ids, -class_counts[known_ids]))]
        cut1, cut2 = len(ordered) // 3, (2 * len(ordered)) // 3
        many = ordered[:cut1]
        med = ordered[cut1:cut2]
        few = ordered[cut2:]

        self.assertEqual(len(many), 32)
        self.assertEqual(len(med), 33)
        self.assertEqual(len(few), 33)
        # Verify partition: disjoint and covers all 98 classes
        all_partitioned = np.sort(np.concatenate([many, med, few]))
        np.testing.assert_array_equal(all_partitioned, known_ids)

    def test_stable_tie_breaking(self):
        # When frequencies tie, lexsort breaks ties deterministically by class ID
        class_counts = np.array([5, 5, 5, 5, 5, 5], dtype=np.int64)
        known_ids = np.arange(6)
        ordered = known_ids[np.lexsort((known_ids, -class_counts[known_ids]))]
        # Since -class_counts is constant, secondary sort by ids ensures [0, 1, 2, 3, 4, 5]
        np.testing.assert_array_equal(ordered, np.array([0, 1, 2, 3, 4, 5]))

    def test_clustering_metrics_overall_accuracies_are_invariant_to_shot_counts(self):
        # Targets and predictions
        targets = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        predictions = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        class_counts1 = np.array([10, 5, 2, 1])
        class_counts2 = np.array([1, 2, 5, 10])

        res1 = clustering_metrics(targets, predictions, known_class_count=2, class_counts=class_counts1)
        res2 = clustering_metrics(targets, predictions, known_class_count=2, class_counts=class_counts2)

        self.assertEqual(res1["all_acc"], res2["all_acc"])
        self.assertEqual(res1["known_acc"], res2["known_acc"])
        self.assertEqual(res1["novel_acc"], res2["novel_acc"])
        self.assertEqual(res1["h_score"], res2["h_score"])


if __name__ == "__main__":
    unittest.main()
