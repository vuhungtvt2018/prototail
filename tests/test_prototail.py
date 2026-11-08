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

        with self.assertRaisesRegex(RuntimeError, "label-free estimated_frequencies"):
            plugin.compute_losses(
                plugin.augment_output(output),
                batch,
                {"num_known_classes": 1, "epoch": 0},
            )

        with self.assertRaisesRegex(RuntimeError, "label-free estimated_frequencies"):
            plugin.compute_losses(
                plugin.augment_output(output),
                batch,
                {
                    "num_known_classes": 1,
                    "epoch": 0,
                    "class_frequencies": torch.tensor([10.0, 5.0, 2.0]),
                },
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

    def test_capacity_loss_supports_uniform_mode_and_validates_mode(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(
            model,
            {
                "num_slots": 2,
                "capacity_regularization_weight": 0.05,
                "capacity_target_mode": "uniform",
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

        frequencies = torch.tensor([10.0, 5.0, 2.0])
        losses = plugin.compute_losses(
            plugin.augment_output(output),
            batch,
            {
                "num_known_classes": 1,
                "epoch": 0,
                "estimated_frequencies": frequencies,
            },
        )

        freq_targets = plugin.components.prototypes.target_capacities(frequencies)
        uniform_target = freq_targets.mean().expand_as(freq_targets)
        self.assertEqual(float(uniform_target.var()), 0.0)
        self.assertAlmostEqual(float(uniform_target.mean()), float(freq_targets.mean()), places=5)

        expected = torch.nn.functional.mse_loss(
            plugin.components.prototypes.gates().sum(dim=-1),
            uniform_target,
        )
        torch.testing.assert_close(losses["capacity_regularization"], expected)

        # Invalid mode raises ValueError
        plugin.config["capacity_target_mode"] = "invalid_mode"
        with self.assertRaises(ValueError):
            plugin.compute_losses(
                plugin.augment_output(output),
                batch,
                {
                    "num_known_classes": 1,
                    "epoch": 0,
                    "estimated_frequencies": frequencies,
                },
            )

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

        self.assertNotIn(
            id(model.backbone.blocks[11].weight),
            part_parameter_ids,
        )
        self.assertEqual(tuple(plugin.shared_parameters()), ())
        self.assertNotIn(
            id(model.backbone.blocks[10].weight),
            part_parameter_ids,
        )

    def test_known_prototype_ema_continues_past_novel_end_epoch(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(
            model,
            {
                "num_slots": 2,
                "prototype_warmup_epochs": 0,
                "novel_prototype_end_epoch": 60,
                "temperature": 0.1,
            },
            3,
            4,
        )
        # 3 classes: class 0 (known), class 1 (novel), class 2 (novel)
        # Sample 0 is labeled class 0 (known)
        # Sample 1 is unlabeled, predicts class 1 with high confidence
        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 5, 4),
                    logits=torch.tensor([
                        [10.0, -10.0, -10.0],
                        [-10.0, 20.0, -10.0],
                    ]),
                )
            }
        )
        batch = self.contracts.TrainingBatch(
            views=(torch.randn(2, 4),),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([10, 11]),
        )
        before_known = plugin.components.prototypes.prototypes[0].clone()
        before_novel = plugin.components.prototypes.prototypes[1].clone()

        plugin.after_optimizer_step(
            plugin.augment_output(output),
            batch,
            {
                "epoch": 61,
                "total_epochs": 200,
                "num_known_classes": 1,
            },
        )

        after_known = plugin.components.prototypes.prototypes[0]
        after_novel = plugin.components.prototypes.prototypes[1]

        # Known prototype MUST update at epoch 61
        self.assertFalse(torch.equal(before_known, after_known))
        # Novel prototype MUST NOT update after epoch 60
        torch.testing.assert_close(before_novel, after_novel)

    def test_target_capacities_computation(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 3}, 3, 4)
        bank = plugin.components.prototypes

        freqs = torch.tensor([1.0, 10.0, 100.0])
        targets = bank.target_capacities(freqs)
        torch.testing.assert_close(targets, torch.tensor([1.0, 2.0, 3.0]))

        equal_freqs = torch.tensor([5.0, 5.0, 5.0])
        equal_targets = bank.target_capacities(equal_freqs)
        torch.testing.assert_close(equal_targets, torch.tensor([3.0, 3.0, 3.0]))

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

    def test_warm_start_from_estimated_support(self):
        torch = self.torch
        plugin = self.plugin_type()
        model = torch.nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 3}, 3, 4)
        bank = plugin.components.prototypes

        freqs = torch.tensor([1.0, 10.0, 100.0])
        bank.warm_start_from_estimated_support(freqs)
        active_slots = bank.gates().sum(dim=-1)
        self.assertAlmostEqual(active_slots[0].item(), 1.0, places=2)
        self.assertAlmostEqual(active_slots[1].item(), 2.0, places=2)
        self.assertAlmostEqual(active_slots[2].item(), 2.94, places=1)

    def test_eval_feature_mode_balanced_vs_legacy(self):
        torch = self.torch
        plugin_legacy = self.plugin_type()
        plugin_balanced = self.plugin_type()
        model1 = torch.nn.Linear(4, 3)
        model2 = torch.nn.Linear(4, 3)
        plugin_legacy.attach(model1, {"num_slots": 2, "eval_feature_mode": "legacy_concat"}, 3, 4)
        plugin_balanced.attach(model2, {"num_slots": 2, "eval_feature_mode": "balanced_concat"}, 3, 4)

        global_features = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        output = self.contracts.ModelOutput(
            branches={
                "student": self.contracts.BranchOutput(
                    global_features=global_features,
                    patch_tokens=torch.randn(1, 5, 4),
                    logits=torch.randn(1, 3),
                )
            }
        )
        pred_legacy = plugin_legacy.predict(plugin_legacy.augment_output(output))
        pred_balanced = plugin_balanced.predict(plugin_balanced.augment_output(output))

        self.assertFalse(torch.allclose(pred_legacy.features, pred_balanced.features))


if __name__ == "__main__":
    unittest.main()
