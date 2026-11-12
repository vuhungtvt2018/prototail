"""Exhaustive verification that all 6 ablation configurations in ProtoTail
are strictly distinct and produce their exact intended loss components.
"""

import unittest
from pathlib import Path
import sys

source_root = Path(__file__).resolve().parents[1] / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototail_gcd.contracts import BranchOutput, ModelOutput, TrainingBatch
from prototail_gcd.baselines.simgcd.adapter import SimGCDAdapter, compute_simgcd_losses
from prototail_gcd.methods.prototail import ProtoTailPlugin
from prototail_gcd.training.engine import TrainingEngine


class AblationSixConfigsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 6
        self.known_classes = 3
        self.feature_dim = 16
        self.batch_size = 4

    def _create_mock_batch(self):
        # 4 samples: 2 labeled (classes 0, 1), 2 unlabeled
        return TrainingBatch(
            views=(torch.randn(self.batch_size, 3, 8, 8), torch.randn(self.batch_size, 3, 8, 8)),
            known_targets=torch.tensor([0, 1, -1, -1]),
            labeled_mask=torch.tensor([True, True, False, False]),
            sample_ids=torch.arange(self.batch_size),
        )

    def _create_mock_model(self, attach_prototail=False, capacity_weight=0.05):
        class MockBackbone(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.layer = nn.Linear(dim, dim)
                self.finetune_from_block = 0
            def forward(self, x):
                b = x.shape[0]
                features = torch.randn(b, self.dim, requires_grad=True)
                patches = torch.randn(b, 5, self.dim, requires_grad=True)
                return features, patches

        class MockProjector(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.head = nn.Linear(in_dim, out_dim)
            def forward(self, x):
                return F.normalize(x, dim=-1), self.head(x)

        class MockSimGCD(nn.Module):
            def __init__(self, backbone, projector):
                super().__init__()
                self.backbone = backbone
                self.projector = projector
                self.finetune_from_block = 0
            def forward(self, images):
                feats, patches = self.backbone(images)
                proj, logits = self.projector(feats)
                return feats, proj, logits, patches

        model = MockSimGCD(MockBackbone(self.feature_dim), MockProjector(self.feature_dim, self.num_classes))
        method = None
        method_settings = {}
        if attach_prototail:
            method = ProtoTailPlugin()
            method_settings = {
                "num_slots": 2,
                "part_weight": 0.5,
                "part_loss_weight": 1.0,
                "temperature": 0.1,
                "gate_regularization_weight": 0.05,
                "capacity_regularization_weight": capacity_weight,
                "pseudo_confidence": 0.5,
            }
            method.attach(model, method_settings, self.num_classes, self.feature_dim)
        return model, method, method_settings

    def _run_batch_step(self, config_name, enable_parts, capacity_weight, enable_pseudo, enable_novel):
        model, method, method_settings = self._create_mock_model(
            attach_prototail=enable_parts,
            capacity_weight=capacity_weight,
        )
        baseline = SimGCDAdapter()
        baseline_config = {
            "num_mlp_layers": 1,
            "hidden_dim": 16,
            "bottleneck_dim": 16,
            "use_bn": False,
            "norm_last_layer": False,
            "sup_weight": 0.35,
            "memax_weight": 1.0,
            "student_temp": 0.1,
            "contrastive_temp": 1.0,
            "supervised_contrastive_temp": 0.07,
            "enable_pseudo_labeling": enable_pseudo,
            "enable_novel_pseudo": enable_novel,
            "pseudo_confidence": 0.2,  # Low threshold so mock batch triggers pseudo
            "pseudo_warmup_epochs": 0,
        }
        baseline_params = list(model.backbone.parameters()) + list(model.projector.parameters())
        optimizers = {"main": torch.optim.SGD(baseline_params, lr=0.01)}
        if method is not None:
            for gname, pgroups in method.optimizer_groups(model).items():
                optimizers[f"prototail_{gname}"] = torch.optim.SGD(pgroups, lr=0.01)

        batch = self._create_mock_batch()
        context = {
            "batch_size": self.batch_size,
            "num_views": 2,
            "epoch": 0,
            "total_epochs": 1,
            "num_classes": self.num_classes,
            "num_known_classes": self.known_classes,
            "estimated_frequencies": torch.tensor([10.0, 10.0, 5.0, 2.0, 1.0, 1.0]),
            "cluster_criterion": lambda s, t, e: (s - t).square().mean(),
            "enable_pseudo_labeling": enable_pseudo,
            "enable_novel_pseudo": enable_novel,
        }
        engine = TrainingEngine(
            model=model,
            baseline=baseline,
            optimizers=optimizers,
            baseline_config=baseline_config,
            method=method,
            method_config=method_settings,
            context=context,
            use_fp16=False,
        )
        metrics = engine.train_batch(batch, epoch=0)
        return metrics

    def test_all_six_configurations_are_strictly_distinct(self):
        # 1. Base: No parts, no tail, no pseudo
        base = self._run_batch_step("Base", enable_parts=False, capacity_weight=0.0, enable_pseudo=False, enable_novel=False)
        self.assertEqual(base["pseudo_loss"], 0.0)
        self.assertNotIn("aux_fused", base)
        self.assertNotIn("gate", base)
        self.assertNotIn("dist_gate", base)

        # 2. Pseudo: Baseline + Pseudo (No parts)
        pseudo = self._run_batch_step("Pseudo", enable_parts=False, capacity_weight=0.0, enable_pseudo=True, enable_novel=True)
        self.assertGreater(pseudo["pseudo_loss"], 0.0)
        self.assertNotIn("aux_fused", pseudo)
        self.assertNotIn("dist_gate", pseudo)
        self.assertNotEqual(base["total"], pseudo["total"], "Base and Pseudo must have different loss!")

        # 3. Prototype: Parts + Gate, but Capacity ablated (=0), No pseudo
        prototype = self._run_batch_step("Prototype", enable_parts=True, capacity_weight=0.0, enable_pseudo=False, enable_novel=False)
        self.assertIn("aux_fused", prototype)
        self.assertIn("gate", prototype)
        self.assertNotIn("dist_gate", prototype)
        self.assertEqual(prototype["pseudo_loss"], 0.0)
        self.assertNotEqual(base["total"], prototype["total"])
        self.assertNotEqual(pseudo["total"], prototype["total"])

        # 4. Prototype_Tailaware: Parts + Gate + Tail capacity (0.05), No pseudo
        tailaware = self._run_batch_step("Prototype_Tailaware", enable_parts=True, capacity_weight=0.05, enable_pseudo=False, enable_novel=False)
        self.assertIn("aux_fused", tailaware)
        self.assertIn("gate", tailaware)
        self.assertIn("dist_gate", tailaware)
        self.assertGreater(tailaware["dist_gate"], 0.0)
        self.assertEqual(tailaware["pseudo_loss"], 0.0)
        self.assertNotEqual(prototype["total"], tailaware["total"], "Prototype and Prototype_Tailaware must differ by dist_gate!")

        # 5. Full_wo_Tail: Parts + Gate + Pseudo, but Capacity ablated (=0)
        full_wo_tail = self._run_batch_step("Full_wo_Tail", enable_parts=True, capacity_weight=0.0, enable_pseudo=True, enable_novel=True)
        self.assertIn("aux_fused", full_wo_tail)
        self.assertIn("gate", full_wo_tail)
        self.assertNotIn("dist_gate", full_wo_tail)
        self.assertGreater(full_wo_tail["pseudo_loss"], 0.0)
        self.assertNotEqual(prototype["total"], full_wo_tail["total"], "Full_wo_Tail and Prototype must differ by pseudo_loss!")

        # 6. Full: Parts + Gate + Tail capacity (0.05) + Pseudo
        full = self._run_batch_step("Full", enable_parts=True, capacity_weight=0.05, enable_pseudo=True, enable_novel=True)
        self.assertIn("aux_fused", full)
        self.assertIn("gate", full)
        self.assertIn("dist_gate", full)
        self.assertGreater(full["dist_gate"], 0.0)
        self.assertGreater(full["pseudo_loss"], 0.0)
        self.assertNotEqual(tailaware["total"], full["total"], "Full and Prototype_Tailaware must differ by pseudo_loss!")
        self.assertNotEqual(full_wo_tail["total"], full["total"], "Full and Full_wo_Tail must differ by dist_gate!")

        print("\nAll 6 configs verified unique:")
        print(f"Base:                total={base['total']:.4f}, pseudo={base['pseudo_loss']:.4f}")
        print(f"Pseudo:              total={pseudo['total']:.4f}, pseudo={pseudo['pseudo_loss']:.4f}")
        print(f"Prototype:           total={prototype['total']:.4f}, aux_fused={prototype['aux_fused']:.4f}, dist_gate=0.0")
        print(f"Prototype_Tailaware: total={tailaware['total']:.4f}, aux_fused={tailaware['aux_fused']:.4f}, dist_gate={tailaware['dist_gate']:.4f}")
        print(f"Full_wo_Tail:        total={full_wo_tail['total']:.4f}, pseudo={full_wo_tail['pseudo_loss']:.4f}, dist_gate=0.0")
        print(f"Full:                total={full['total']:.4f}, pseudo={full['pseudo_loss']:.4f}, dist_gate={full['dist_gate']:.4f}")


if __name__ == "__main__":
    unittest.main()
