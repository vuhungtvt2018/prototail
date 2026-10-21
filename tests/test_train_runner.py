import importlib.util
import sys
import unittest
from importlib import import_module
from pathlib import Path


REQUIRED_MODULES = ("numpy", "scipy", "sklearn", "torch")
HAS_RUNNER_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in REQUIRED_MODULES
)


@unittest.skipUnless(HAS_RUNNER_DEPS, "Install the training dependencies")
class UnifiedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        cls.np = import_module("numpy")
        cls.torch = import_module("torch")
        cls.contracts = import_module("prototail_gcd.contracts")
        cls.runner = import_module("prototail_gcd.train")

    def test_evaluate_uses_class_ids_from_parametric_baseline(self):
        torch = self.torch
        contracts = self.contracts

        class Adapter:
            def forward_eval(self, _model, images):
                return images

            def predict(self, output):
                return contracts.PredictionOutput(
                    class_ids=output,
                    scores=None,
                )

        loader = [
            (
                torch.tensor([[0], [1], [0], [1]]),
                torch.tensor([0, 1, 0, 1]),
                torch.arange(4),
                torch.ones(4, dtype=torch.bool),
            )
        ]

        metrics = self.runner.evaluate(
            model=object(),
            loader=loader,
            adapter=Adapter(),
            device=torch.device("cpu"),
            known_class_count=1,
            num_classes=2,
            random_state=0,
        )

        self.assertEqual(metrics["all_acc"], 1.0)

    def test_evaluate_clusters_features_from_non_parametric_baseline(self):
        torch = self.torch
        np = self.np
        contracts = self.contracts

        class Adapter:
            def forward_eval(self, _model, images):
                return contracts.ModelOutput(
                    branches={
                        "contrastive": contracts.BranchOutput(
                            global_features=images,
                        )
                    }
                )

            def predict(self, output):
                return contracts.PredictionOutput(
                    class_ids=None,
                    features=output.branches["contrastive"].global_features,
                )

        features = torch.tensor(
            [[-2.0, 0.0], [-1.8, 0.1], [-2.2, -0.1],
             [2.0, 0.0], [1.8, 0.1], [2.2, -0.1]]
        )
        loader = [
            (
                features,
                torch.tensor([0, 0, 0, 1, 1, 1]),
                torch.arange(6),
                torch.ones(6, dtype=torch.bool),
            )
        ]

        metrics = self.runner.evaluate(
            model=object(),
            loader=loader,
            adapter=Adapter(),
            device=torch.device("cpu"),
            known_class_count=1,
            num_classes=2,
            random_state=0,
        )

        self.assertEqual(metrics["all_acc"], 1.0)
        self.assertTrue(np.isfinite(metrics["ari"]))

    def test_clustering_metrics_keep_reference_names_and_shot_groups(self):
        metrics_module = import_module(
            "prototail_gcd.evaluation.clustering"
        )
        targets = self.np.arange(12, dtype=self.np.int64)
        class_counts = (10, 9, 8, 7, 5, 1, 10, 9, 8, 7, 5, 1)

        metrics = metrics_module.clustering_metrics(
            targets=targets,
            predictions=targets,
            known_class_count=6,
            class_counts=class_counts,
        )

        self.assertEqual(metrics["all"], 1.0)
        self.assertEqual(metrics["old"], 1.0)
        self.assertEqual(metrics["new"], 1.0)
        self.assertEqual(metrics["k_many"], 1.0)
        self.assertEqual(metrics["u_few"], 1.0)

    def test_bacon_metric_scale_and_empty_shot_groups_match_reference(self):
        metrics_module = import_module(
            "prototail_gcd.evaluation.clustering"
        )
        targets = self.np.asarray([0, 1, 0, 1], dtype=self.np.int64)

        metrics = metrics_module.clustering_metrics(
            targets=targets,
            predictions=targets,
            known_class_count=1,
            class_counts=(2, 2),
            accuracy_scale=100.0,
            empty_group_value=-1.0,
        )

        self.assertEqual(metrics["all"], 100.0)
        self.assertEqual(metrics["old"], 100.0)
        self.assertEqual(metrics["new"], 100.0)
        self.assertEqual(metrics["k_many"], -1.0)
        self.assertEqual(metrics["k_few"], -1.0)
        self.assertEqual(metrics["u_many"], -1.0)
        self.assertEqual(metrics["u_few"], -1.0)
        self.assertEqual(metrics["nmi"], 1.0)


if __name__ == "__main__":
    unittest.main()
