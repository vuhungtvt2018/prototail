"""Verification of all planfix.md Section 7 requirements for tamper-proof parity."""
from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prototail_gcd.contracts import BranchOutput, ModelOutput, PredictionOutput, TrainingBatch
from prototail_gcd.methods.prototail import ProtoTailPlugin
from prototail_gcd.pseudo import PaperDualSelector, PromotionStore, Snapshot
from prototail_gcd.training.engine import TrainingEngine
from prototail_gcd.training.momentum_teacher import GlobalEMATeacher


class TestTamParityLT20(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)

    def test_optimizer_disjointness_and_late_backbone_gradient(self) -> None:
        """Each parameter in exactly 1 optimizer; late backbone receives gradient from part loss."""
        class DummyBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(12)])
                self.finetune_from_block = 11

            def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                for b in self.blocks:
                    x = b(x)
                return x, x.unsqueeze(1).repeat(1, 4, 1)

        class DummyProjector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return F.normalize(x, dim=-1), self.head(x)

        class DummySimGCD(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = DummyBackbone()
                self.projector = DummyProjector()
                self.finetune_from_block = 11

            def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                feats, patches = self.backbone(images)
                proj, logits = self.projector(feats)
                return feats, proj, logits, patches

        model = DummySimGCD()
        plugin = ProtoTailPlugin()
        plugin.attach(model, {"num_slots": 2, "part_loss_weight": 1.0, "part_weight": 0.5}, num_classes=4, feature_dim=8)

        # Baseline parameters
        baseline_params = list(model.backbone.parameters()) + list(model.projector.parameters())
        method_groups = plugin.optimizer_groups(model)
        optimizers = {
            "main": torch.optim.SGD(baseline_params, lr=0.01),
            "prototail_part": torch.optim.SGD(method_groups["part"][0]["params"], lr=0.05),
            "prototail_gate": torch.optim.SGD(method_groups["gate"][0]["params"], lr=0.01),
        }

        # Verify completely disjoint parameter sets
        all_ids: list[int] = []
        for opt in optimizers.values():
            for g in opt.param_groups:
                for p in g["params"]:
                    all_ids.append(id(p))
        self.assertEqual(len(all_ids), len(set(all_ids)), "Optimizers must not share parameters")
        self.assertEqual(len(plugin.shared_parameters()), 0)

        # Forward and backward test: part loss must propagate to backbone block 11
        batch = TrainingBatch(
            views=(torch.randn(2, 8), torch.randn(2, 8)),
            known_targets=torch.tensor([0, 1]),
            labeled_mask=torch.tensor([True, True]),
            sample_ids=torch.tensor([1, 2]),
        )
        images = torch.cat(tuple(batch.views), dim=0)
        feats, proj, logits, patches = model(images)
        out = ModelOutput(
            branches={"student": BranchOutput(global_features=feats, projection=proj, logits=logits, patch_tokens=patches)}
        )
        aug_out = plugin.augment_output(out)
        losses = plugin.compute_losses(aug_out, batch, {"num_known_classes": 2, "epoch": 0})
        losses["part_classification"].backward()

        self.assertIsNotNone(model.backbone.blocks[11].weight.grad)
        self.assertGreater(float(model.backbone.blocks[11].weight.grad.abs().sum()), 0.0)

    def test_data_privacy_estimator_fail_closed(self) -> None:
        """Estimator without distribution loader must fail closed; never fallback to GT class counts."""
        from prototail_gcd.baselines.simgcd.adapter import SimGCDAdapter

        adapter = SimGCDAdapter()
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(4, 4)
                self.finetune_from_block = 0
        model = DummyModel()

        context = {
            "class_frequencies": torch.tensor([10.0, 5.0, 2.0]),
            "num_classes": 3,
            "num_known_classes": 2,
        }
        with self.assertRaisesRegex(RuntimeError, "Distribution estimator needs a training distribution_loader"):
            adapter.before_epoch(model, epoch=0, context=context, config={})

    def test_capacity_loss_fails_without_estimated_frequencies(self) -> None:
        """Capacity regularization requires label-free estimated_frequencies and forbids GT class_counts."""
        plugin = ProtoTailPlugin()
        model = nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 2, "capacity_regularization_weight": 0.05}, num_classes=3, feature_dim=4)
        out = ModelOutput(
            branches={"student": BranchOutput(global_features=torch.randn(2, 4), patch_tokens=torch.randn(2, 3, 4), logits=torch.randn(2, 3))}
        )
        batch = TrainingBatch(views=(torch.randn(2, 4),), known_targets=torch.tensor([0, -1]), labeled_mask=torch.tensor([True, False]), sample_ids=torch.tensor([1, 2]))
        aug = plugin.augment_output(out)

        # Context has class_frequencies (oracle) but not estimated_frequencies
        context = {"class_frequencies": torch.tensor([10.0, 5.0, 2.0]), "num_known_classes": 2, "epoch": 0}
        with self.assertRaisesRegex(RuntimeError, "label-free estimated_frequencies"):
            plugin.compute_losses(aug, batch, context)

    def test_ema_bank_update_schedule_and_novel_isolation(self) -> None:
        """Known prototypes update at e61; raw Novel does not at e61; Novel guess of known ID does not contaminate Known bank."""
        plugin = ProtoTailPlugin()
        model = nn.Linear(4, 3)
        plugin.attach(
            model,
            {"num_slots": 2, "prototype_warmup_epochs": 0, "novel_prototype_end_epoch": 60, "prototype_momentum": 0.5},
            num_classes=3,
            feature_dim=4,
        )
        bank = plugin.components.prototypes
        initial_known = bank.prototypes[0].clone()
        initial_novel = bank.prototypes[2].clone()

        # Batch at epoch 61: sample 0 is labeled class 0, sample 1 is unlabeled predicting class 2 (novel)
        output = ModelOutput(
            branches={
                "student": BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 5, 4),
                    logits=torch.tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]]),
                )
            }
        )
        batch = TrainingBatch(
            views=(torch.randn(2, 4),),
            known_targets=torch.tensor([0, -1]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([1, 2]),
        )
        aug = plugin.augment_output(output)

        plugin.after_optimizer_step(aug, batch, {"epoch": 61, "total_epochs": 200, "num_known_classes": 2})

        # Known prototype WAS updated at epoch 61
        self.assertFalse(torch.equal(initial_known, bank.prototypes[0]), "Known prototype should continue EMA at e61")
        # Raw novel prototype was NOT updated at epoch 61 (ended at e60)
        self.assertTrue(torch.equal(initial_novel, bank.prototypes[2]), "Raw novel pseudo update should stop after e60")

        # Now test unlabeled predicting known class 0 at e35: must NOT contaminate known bank
        initial_known_e35 = bank.prototypes[0].clone()
        output_known_guess = ModelOutput(
            branches={
                "student": BranchOutput(
                    global_features=torch.randn(1, 4),
                    patch_tokens=torch.randn(1, 5, 4),
                    logits=torch.tensor([[20.0, -20.0, -20.0]]),
                )
            }
        )
        batch_unlabeled = TrainingBatch(
            views=(torch.randn(1, 4),),
            known_targets=torch.tensor([-1]),
            labeled_mask=torch.tensor([False]),
            sample_ids=torch.tensor([3]),
        )
        aug_known_guess = plugin.augment_output(output_known_guess)
        plugin.after_optimizer_step(aug_known_guess, batch_unlabeled, {"epoch": 35, "total_epochs": 200, "num_known_classes": 2})
        self.assertTrue(
            torch.equal(initial_known_e35, bank.prototypes[0]),
            "Unlabeled sample predicting known class must never update known prototypes",
        )

    def test_global_ema_teacher_lifecycle(self) -> None:
        """GlobalEMATeacher: m0=0.996, cosine momentum, requires_grad=False, state_dict restore."""
        bb = nn.Linear(4, 4)
        head = nn.Linear(4, 2)
        teacher = GlobalEMATeacher(bb, head, m0=0.996)

        for p in list(teacher.backbone.parameters()) + list(teacher.head.parameters()):
            self.assertFalse(p.requires_grad)

        orig_w = teacher.backbone.weight.clone()
        # Simulate student parameter changes
        with torch.no_grad():
            bb.weight.add_(1.0)
        teacher.update(bb, head, epoch=0, total_epochs=200)

        self.assertEqual(teacher.steps, 1)
        self.assertFalse(torch.equal(orig_w, teacher.backbone.weight))

        state = teacher.state_dict()
        new_teacher = GlobalEMATeacher(bb, head, m0=0.996)
        new_teacher.load_state_dict(state)
        self.assertEqual(new_teacher.steps, 1)
        torch.testing.assert_close(new_teacher.backbone.weight, teacher.backbone.weight)

        # WeightNorm compatibility test: DINOHead with norm_last_layer=True
        from prototail_gcd.baselines.simgcd.components import DINOHead
        dino_head = DINOHead(in_dim=4, out_dim=2, use_bn=False, norm_last_layer=True, num_layers=2, hidden_dim=8, bottleneck_dim=4)
        teacher_dino = GlobalEMATeacher(bb, dino_head, m0=0.996)
        self.assertIsNotNone(teacher_dino)

    def test_promotion_store_and_paper_dual_selector(self) -> None:
        """PromotionStore and PaperDualSelector without any Novel GT leakage."""
        store = PromotionStore(num_classes=4, num_known_classes=2)
        added = store.add_accepted([(101, 1), (102, 2)])
        self.assertEqual(added, 2)
        self.assertEqual(len(store), 2)
        # Duplicate addition is rejected
        self.assertEqual(store.add_accepted([(101, 1)]), 0)
        self.assertEqual(store.lookup([101, 103]), {101: 1})

        # PaperDualSelector scheduled rounds
        selector = PaperDualSelector(
            n_known=2,
            n_classes=4,
            known_epochs=(30, 40),
            novel_epochs=(50, 60),
            seed=0,
        )
        self.assertTrue(selector.scheduled(30))
        self.assertTrue(selector.scheduled(50))
        self.assertFalse(selector.scheduled(35))

        # Snapshot checks: fail closed if unlabeled sample has target != -1
        with self.assertRaisesRegex(RuntimeError, "Novel/unlabeled GT cannot enter paper selection"):
            Snapshot(
                sample_ids=np.array([1, 2]),
                features=np.zeros((2, 4), dtype=np.float32),
                probs=np.zeros((2, 4), dtype=np.float32),
                known_targets=np.array([0, 3]),  # target 3 on unlabeled sample!
                originally_labeled=np.array([True, False]),
            )

    def test_eval_feature_modes(self) -> None:
        """ProtoTail eval feature modes: legacy_concat vs original_parts_concat."""
        plugin = ProtoTailPlugin()
        model = nn.Linear(4, 3)
        plugin.attach(model, {"num_slots": 2, "eval_feature_mode": "original_parts_concat"}, num_classes=3, feature_dim=4)

        out = ModelOutput(
            branches={
                "student": BranchOutput(
                    global_features=torch.randn(2, 4),
                    patch_tokens=torch.randn(2, 3, 4),
                    logits=torch.randn(2, 3),
                )
            }
        )
        aug = plugin.augment_output(out)
        pred_norm = plugin.predict(aug)
        self.assertIsNone(pred_norm.class_ids)
        self.assertEqual(tuple(pred_norm.features.shape), (2, 8))
        # Features should be normalized (norm close to 1)
        norms = pred_norm.features.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

        # Test legacy_concat
        plugin.config = {"num_slots": 2, "eval_feature_mode": "legacy_concat"}
        pred_legacy = plugin.predict(aug)
        self.assertEqual(tuple(pred_legacy.features.shape), (2, 8))


if __name__ == "__main__":
    unittest.main()
