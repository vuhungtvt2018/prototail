import importlib.util
import sys
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch


REQUIRED_MODULES = ("numpy", "torch", "scipy", "sklearn")
HAS_BACON_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in REQUIRED_MODULES
)


@unittest.skipUnless(HAS_BACON_DEPS, "Install the BaCon dependencies")
class BaConAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        cls.np = import_module("numpy")
        cls.torch = import_module("torch")
        cls.contracts = import_module("prototail_gcd.contracts")
        cls.losses = import_module("prototail_gcd.baselines.bacon.losses")

    def test_frequency_alignment_normalizes_by_cluster_size(self):
        torch = self.torch
        np = self.np
        cluster_assignments = np.repeat((0, 1, 2), (200, 100, 9))
        targets = np.full(cluster_assignments.size, -1, dtype=np.int64)
        labeled_mask = np.zeros(cluster_assignments.size, dtype=bool)

        targets[:100] = 0
        targets[100:200] = 1
        targets[200:210] = 0
        targets[300:] = 1
        labeled_mask[:210] = True
        labeled_mask[300:] = True

        images = torch.zeros(cluster_assignments.size, 3)
        target_tensor = torch.from_numpy(targets)
        mask_tensor = torch.from_numpy(labeled_mask)
        loader = [
            (
                images,
                target_tensor,
                torch.arange(cluster_assignments.size),
                mask_tensor,
            )
        ]

        class Model:
            contrastive_backbone = object()
            finetune_from_block = 0

            def eval(self):
                return self

        class FixedKMeans:
            def __init__(self, **_kwargs):
                pass

            def fit_predict(self, _features):
                return cluster_assignments

        global_features = torch.zeros(cluster_assignments.size, 4)
        with (
            patch.object(self.losses, "KMeans", FixedKMeans),
            patch(
                "prototail_gcd.models.dino_backbone.forward_global_and_patches",
                return_value=(global_features, None),
            ),
        ):
            frequencies = self.losses.estimate_class_frequencies(
                Model(),
                loader,
                num_classes=3,
                num_known_classes=2,
                finetune_from_block=0,
                random_state=0,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(
            frequencies,
            torch.tensor([200.0, 9.0, 100.0]),
        )

    def test_optimizer_groups_are_disjoint_and_cover_trainable_parameters(self):
        torch = self.torch
        adapter_module = import_module("prototail_gcd.baselines.bacon.adapter")
        model_module = import_module("prototail_gcd.baselines.bacon.model")
        model = model_module.BaConNetwork(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 3),
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 5),
            finetune_from_block=0,
        )

        parameter_groups = adapter_module.BaConAdapter().optimizer_groups(
            model,
            {},
        )
        grouped_parameter_ids = [
            id(parameter)
            for groups in parameter_groups.values()
            for group in groups
            for parameter in group["params"]
        ]
        trainable_parameter_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }

        self.assertEqual(len(grouped_parameter_ids), len(set(grouped_parameter_ids)))
        self.assertEqual(set(grouped_parameter_ids), trainable_parameter_ids)


if __name__ == "__main__":
    unittest.main()
