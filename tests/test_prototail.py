import importlib.util
import sys
import unittest
from importlib import import_module
from pathlib import Path


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Install the PyTorch project dependencies")
class ProtoTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source_root))
        cls.torch = import_module("torch")
        cls.contracts = import_module("prototail_gcd.contracts")
        cls.plugin_type = import_module(
            "prototail_gcd.methods.prototail"
        ).ProtoTailPlugin

    def test_plugin_losses_propagate_to_part_and_gate_parameters(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 2}, 3, 4)

        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=torch.randn(4, 4),
                    patch_tokens=torch.randn(4, 5, 4),
                    logits=torch.randn(4, 3, requires_grad=True),
                )
            }
        )
        batch = self.contracts.TrainingBatch(
            views=(torch.randn(2, 4), torch.randn(2, 4)),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([10, 11]),
        )

        losses = plugin.compute_losses(
            plugin.augment_output(output),
            batch,
            {"num_known_classes": 1, "epoch": 0},
        )
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertIsNotNone(plugin.components.parts.queries.grad)
        self.assertIsNotNone(plugin.components.prototypes.gate_logits.grad)
        torch.testing.assert_close(
            plugin.components.prototypes.gate_logits,
            torch.full_like(
                plugin.components.prototypes.gate_logits,
                -1.0,
            ),
        )

    def test_simgcd_part_loss_uses_labeled_samples_from_both_views(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 2}, 3, 4)
        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=torch.randn(4, 4),
                    patch_tokens=torch.randn(4, 5, 4),
                    logits=torch.randn(4, 3),
                )
            }
        )
        batch = self.contracts.TrainingBatch(
            views=(torch.randn(2, 4), torch.randn(2, 4)),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([10, 11]),
        )
        augmented = plugin.augment_output(output)

        losses = plugin.compute_losses(
            augmented,
            batch,
            {"num_known_classes": 1, "epoch": 0},
        )
        values = augmented.auxiliary["prototail"]
        expected = torch.nn.functional.cross_entropy(
            (
                values["global_logits"]
                + 0.5 * values["part_scores"]
            ).view(2, 2, 3)[:, 0, :] / 0.1,
            torch.tensor([0, 0]),
        )

        torch.testing.assert_close(losses["part_classification"], expected)

    def test_simgcd_unlabeled_pseudo_labels_can_update_known_prototypes(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(
            model,
            {"num_slots": 2, "prototype_warmup_epochs": 0},
            2,
            4,
        )
        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 5, 4),
                    logits=torch.tensor([[0.0, 0.0], [20.0, -20.0]]),
                )
            }
        )
        batch = self.contracts.TrainingBatch(
            views=(torch.randn(2, 4),),
            known_targets=torch.tensor([1, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([10, 11]),
        )
        before = plugin.components.prototypes.prototypes[0].clone()

        plugin.after_optimizer_step(
            plugin.augment_output(output),
            batch,
            {
                "epoch": 0,
                "total_epochs": 1,
                "num_known_classes": 2,
            },
        )

        self.assertFalse(
            torch.equal(before, plugin.components.prototypes.prototypes[0])
        )

    def test_capacity_loss_requires_a_real_distribution_estimate(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(
            model,
            {
                "num_slots": 2,
                "capacity_regularization_weight": 0.05,
            },
            3,
            4,
        )
        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 5, 4),
                    logits=torch.randn(2, 3),
                )
            }
        )
        batch = self.contracts.TrainingBatch(
            views=(torch.randn(2, 4),),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([10, 11]),
        )

        with self.assertRaisesRegex(RuntimeError, "estimated_frequencies"):
            plugin.compute_losses(
                plugin.augment_output(output),
                batch,
                {"num_known_classes": 1, "epoch": 0},
            )

    def test_capacity_loss_uses_estimated_class_frequencies(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(
            model,
            {
                "num_slots": 2,
                "capacity_regularization_weight": 0.05,
            },
            3,
            4,
        )
        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 5, 4),
                    logits=torch.randn(2, 3),
                )
            }
        )
        batch = self.contracts.TrainingBatch(
            views=(torch.randn(2, 4),),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([10, 11]),
        )

        losses = plugin.compute_losses(
            plugin.augment_output(output),
            batch,
            {
                "num_known_classes": 1,
                "epoch": 0,
                "estimated_frequencies": torch.tensor([10.0, 5.0, 2.0]),
            },
        )

        frequencies = torch.tensor([10.0, 5.0, 2.0])
        log_frequencies = torch.log(frequencies + 1e-8)
        target_capacity = 1.0 + (
            (2 - 1)
            * (
                log_frequencies - torch.log(frequencies.min() + 1e-8)
            )
            / (
                torch.log(frequencies.max() + 1e-8)
                - torch.log(frequencies.min() + 1e-8)
                + 1e-8
            )
        )
        expected = torch.nn.functional.mse_loss(
            plugin.components.prototypes.gates().sum(dim=-1),
            target_capacity,
        )
        torch.testing.assert_close(losses["capacity_regularization"], expected)

    def test_method_optimizer_keeps_reference_shared_backbone_step(self):
        torch = self.torch
        plugin = self.plugin_type()

        class TinyNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.blocks = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(12)]
                )
                self.finetune_from_block = 11

        model = TinyNetwork()
        plugin.attach(model, {"num_slots": 2}, 3, 2)
        groups = plugin.optimizer_groups(model)
        part_parameter_ids = {
            id(parameter)
            for group in groups["part"]
            for parameter in group["params"]
        }

        self.assertIn(
            id(model.backbone.blocks[11].weight),
            part_parameter_ids,
        )
        self.assertIn(
            id(model.backbone.blocks[11].weight),
            {id(parameter) for parameter in plugin.shared_parameters()},
        )
        self.assertNotIn(
            id(model.backbone.blocks[10].weight),
            part_parameter_ids,
        )

    def test_bacon_inference_returns_part_augmented_features(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 2}, 3, 4)
        output = self.contracts.ModelOutput(
            branches={
                "classification": self.contracts.BranchOutput(
                    global_features=torch.randn(2, 4),
                    logits=torch.randn(2, 3),
                ),
                "contrastive": self.contracts.BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 5, 4),
                ),
            }
        )

        prediction = plugin.predict(plugin.augment_output(output))

        self.assertIsNone(prediction.class_ids)
        self.assertEqual(tuple(prediction.features.shape), (2, 8))

    def test_simgcd_inference_uses_kmeans_part_features_and_diagnostics(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 2}, 3, 4)
        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 5, 4),
                    logits=torch.randn(2, 3),
                )
            }
        )
        augmented = plugin.augment_output(output)

        prediction = plugin.predict(augmented)
        diagnostics = plugin.evaluation_diagnostics(augmented)

        self.assertIsNone(prediction.class_ids)
        self.assertEqual(tuple(prediction.features.shape), (2, 8))
        self.assertEqual(set(diagnostics), {"fused", "global"})


if __name__ == "__main__":
    unittest.main()
