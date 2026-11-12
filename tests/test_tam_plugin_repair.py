"""Correctness and end-to-end integration tests for the tam plug-and-play repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from prototail_gcd.baselines.bacon.losses import (
    align_novel_cluster_counts,
    estimate_class_frequencies,
)
from prototail_gcd.baselines.simgcd.adapter import SimGCDAdapter
from prototail_gcd.contracts import (
    DistributionBatch,
    TrainingBatch,
    UNKNOWN_TARGET,
)
from prototail_gcd.data.protocol import DatasetBundle
from prototail_gcd.methods.prototail import PartPrototypeBank, ProtoTailPlugin
from prototail_gcd.pseudo.paper_selector import PaperDualSelector
from prototail_gcd.pseudo.promotion_store import PromotionStore
from prototail_gcd.training.engine import TrainingEngine


class _TinyViTBackbone(nn.Module):
    def __init__(self, feature_dim: int = 8, num_blocks: int = 12) -> None:
        super().__init__()
        self.proj = nn.Linear(4, feature_dim)
        self.blocks = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(feature_dim)

    def prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        base = self.proj(images.reshape(images.shape[0], -1)[:, :4])
        # CLS token + 4 patch tokens
        offsets = torch.linspace(
            -0.2, 0.2, 5, device=images.device, dtype=base.dtype
        ).view(1, 5, 1)
        return base.unsqueeze(1) + offsets


def _make_tiny_dataset_bundle(
    *,
    num_classes: int = 4,
    known_classes: int = 2,
    samples_per_class: int = 6,
    with_class_counts: bool = True,
) -> tuple[DatasetBundle, np.ndarray]:
    """Deterministic synthetic dataset with masked unlabeled targets."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    images_list = []
    targets_list = []
    labeled_list = []
    sample_ids_list = []
    true_targets_list = []

    sid = 0
    for c in range(num_classes):
        center = torch.zeros(4)
        center[c % 4] = 2.5 if c < 2 else -2.5
        if c >= 2:
            center[(c + 1) % 4] = 2.0
        for j in range(samples_per_class):
            img = center + 0.1 * torch.randn(4)
            is_labeled = (c < known_classes) and (j < samples_per_class // 2)
            images_list.append(img)
            true_targets_list.append(c)
            targets_list.append(c if is_labeled else UNKNOWN_TARGET)
            labeled_list.append(is_labeled)
            sample_ids_list.append(sid)
            sid += 1

    all_images = torch.stack(images_list)
    all_targets = torch.tensor(targets_list, dtype=torch.long)
    all_labeled = torch.tensor(labeled_list, dtype=torch.bool)
    all_ids = torch.tensor(sample_ids_list, dtype=torch.long)
    all_true = np.asarray(true_targets_list, dtype=np.int64)

    n = len(all_ids)
    # Single-batch or 2-batch deterministic loaders
    half = n // 2
    train_batches = [
        TrainingBatch(
            views=(
                all_images[:half],
                all_images[:half] + 0.02 * torch.randn_like(all_images[:half]),
            ),
            known_targets=all_targets[:half],
            labeled_mask=all_labeled[:half],
            sample_ids=all_ids[:half],
        ),
        TrainingBatch(
            views=(
                all_images[half:],
                all_images[half:] + 0.02 * torch.randn_like(all_images[half:]),
            ),
            known_targets=all_targets[half:],
            labeled_mask=all_labeled[half:],
            sample_ids=all_ids[half:],
        ),
    ]
    dist_batches = [
        DistributionBatch(
            images=all_images,
            known_targets=all_targets,
            labeled_mask=all_labeled,
            sample_ids=all_ids,
        )
    ]
    unlabeled_mask_np = ~all_labeled.numpy()
    unlabeled_eval_batches = [
        (
            all_images[unlabeled_mask_np],
            torch.from_numpy(all_true[unlabeled_mask_np]),
            all_ids[unlabeled_mask_np],
            torch.ones(int(unlabeled_mask_np.sum()), dtype=torch.bool),
        )
    ]
    test_batches = [
        (
            all_images,
            torch.from_numpy(all_true),
            all_ids,
            torch.ones(n, dtype=torch.bool),
        )
    ]
    bundle = DatasetBundle(
        train_loader=train_batches,
        evaluation_loaders={
            "distribution": dist_batches,
            "unlabeled_train": unlabeled_eval_batches,
            "test": test_batches,
        },
        num_classes=num_classes,
        known_class_ids=tuple(range(known_classes)),
        class_counts=(
            tuple(int((all_true == c).sum()) for c in range(num_classes))
            if with_class_counts
            else None
        ),
    )
    return bundle, rng.permutation(n)


def test_novel_pseudo_cannot_write_known_prototypes():
    bank = PartPrototypeBank(num_classes=4, num_slots=2, feature_dim=4)
    a = torch.nn.functional.normalize(torch.randn(1, 2, 4), dim=-1)
    bank.update_known(a, torch.tensor([0]), 0.99)
    known_before = bank.prototypes[0].clone()
    bank.update_novel(
        a.expand(2, -1, -1),
        torch.tensor([0, 2]),
        torch.ones(2),
        threshold=0.7,
        momentum=0.99,
        min_novel_class_id=2,
    )
    torch.testing.assert_close(bank.prototypes[0], known_before)
    assert bank.update_counts[0] == 1
    assert bank.update_counts[2] == 1
    assert bank.update_counts[1] == 0


def test_novel_support_is_mapped_to_classifier_dimension_not_size_rank():
    ids = np.array([0] * 20 + [1] * 3)
    probs = np.zeros((23, 4), dtype=np.float64)
    probs[:20, 3] = 0.99
    probs[20:, 2] = 0.99
    counts = align_novel_cluster_counts(ids, [0, 1], probs, num_known_classes=2)
    np.testing.assert_array_equal(counts, [3, 20])


def test_uniform_logits_give_neutral_novel_support():
    ids = np.array([0] * 20 + [1] * 3)
    probs = np.ones((23, 4), dtype=np.float64) / 4
    counts = align_novel_cluster_counts(ids, [0, 1], probs, num_known_classes=2)
    np.testing.assert_allclose(counts, [11.5, 11.5])


def test_shared_backbone_only_in_baseline_optimizer():
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(4, 4)
    model.finetune_from_block = 0
    method = ProtoTailPlugin()
    method.attach(model, {"num_slots": 2, "temperature": 0.1}, 3, 4)
    groups = method.optimizer_groups(model)
    extra_ids = {
        id(p) for g in groups.values() for block in g for p in block["params"]
    }
    base_ids = {id(p) for p in model.backbone.parameters()}
    assert not extra_ids.intersection(base_ids)
    assert tuple(method.shared_parameters()) == ()


def test_first_centroid_initializes_directly():
    bank = PartPrototypeBank(num_classes=4, num_slots=2, feature_dim=4)
    parts = torch.nn.functional.normalize(torch.randn(1, 2, 4), dim=-1)
    bank.update_known(parts, torch.tensor([1]), momentum=0.99)
    torch.testing.assert_close(bank.prototypes[1], parts.squeeze(0))
    assert bank.update_counts[1] == 1


def test_training_engine_rejects_overlapping_optimizers():
    model = nn.Linear(4, 2)
    baseline = SimGCDAdapter()
    opt1 = torch.optim.SGD(model.parameters(), lr=0.01)
    opt2 = torch.optim.SGD([model.weight], lr=0.01)
    with pytest.raises(ValueError, match="overlaps earlier optimizer"):
        TrainingEngine(
            model=model,
            baseline=baseline,
            optimizers={"main": opt1, "dup": opt2},
            baseline_config={},
        )


def test_method_none_preserves_native_simgcd_losses_gradients_and_skips_estimator():
    torch.manual_seed(7)
    adapter = SimGCDAdapter()
    baseline_cfg = {
        "num_mlp_layers": 1,
        "hidden_dim": 8,
        "bottleneck_dim": 8,
        "use_bn": False,
        "norm_last_layer": True,
        "sup_weight": 0.35,
        "memax_weight": 2.0,
        "student_temp": 0.1,
        "contrastive_temp": 1.0,
        "supervised_contrastive_temp": 0.07,
        "warmup_teacher_temp": 0.07,
        "teacher_temp": 0.04,
        "warmup_teacher_temp_epochs": 1,
    }
    backbone = _TinyViTBackbone(feature_dim=8, num_blocks=12)
    from prototail_gcd.baselines.simgcd.components import DINOHead
    from prototail_gcd.baselines.simgcd.adapter import SimGCDNetwork
    from prototail_gcd.models.dino_backbone import set_finetune_blocks

    set_finetune_blocks(backbone, 11)
    head = DINOHead(
        in_dim=8,
        out_dim=4,
        use_bn=False,
        norm_last_layer=True,
        num_layers=1,
        hidden_dim=8,
        bottleneck_dim=8,
    )
    model = SimGCDNetwork(backbone, head, finetune_from_block=11)
    batch = TrainingBatch(
        views=(torch.randn(4, 4), torch.randn(4, 4)),
        known_targets=torch.tensor([0, 1, -1, -1]),
        labeled_mask=torch.tensor([True, True, False, False]),
        sample_ids=torch.tensor([0, 1, 2, 3]),
    )
    ctx = adapter.create_training_context(
        baseline_cfg, num_views=2, total_epochs=2
    )
    ctx["num_classes"] = 4
    ctx["num_known_classes"] = 2
    ctx["needs_frequency_estimate"] = False
    # Should not run estimator or raise even if distribution_loader is None
    assert adapter.before_epoch(model, 0, ctx, baseline_cfg) is None

    # Compare TrainingEngine with method=None vs direct baseline forward + compute_simgcd_losses
    out_direct = adapter.forward(model, batch)
    assert out_direct.branches["student"].patch_tokens is None
    losses_direct = adapter.compute_losses(
        out_direct, batch, {**ctx, "epoch": 0}, baseline_cfg
    )
    losses_direct["total"].backward()
    grad_head_direct = model.projector.last_layer.weight_v.grad.detach().clone()
    grad_blk_direct = model.backbone.blocks[11].weight.grad.detach().clone()
    logits_direct = out_direct.branches["student"].logits.detach().clone()

    opt = torch.optim.SGD(
        [
            p
            for g in adapter.optimizer_groups(model, baseline_cfg)["main"]
            for p in g["params"]
        ],
        lr=0.0,
    )
    engine = TrainingEngine(
        model=model,
        baseline=adapter,
        optimizers={"main": opt},
        baseline_config=baseline_cfg,
        method=None,
        context=ctx,
    )
    metrics = engine.train_batch(batch, epoch=0)
    assert pytest.approx(metrics["total"], rel=1e-5) == float(
        losses_direct["total"].item()
    )
    torch.testing.assert_close(
        model.projector.last_layer.weight_v.grad, grad_head_direct
    )
    torch.testing.assert_close(model.backbone.blocks[11].weight.grad, grad_blk_direct)
    with torch.no_grad():
        logits_after = adapter.forward(model, batch).branches["student"].logits
    torch.testing.assert_close(logits_after, logits_direct)


def test_simgcd_before_epoch_fails_loudly_without_fallback_to_gt():
    adapter = SimGCDAdapter()
    backbone = _TinyViTBackbone(feature_dim=8, num_blocks=2)
    from prototail_gcd.baselines.simgcd.components import DINOHead
    from prototail_gcd.baselines.simgcd.adapter import SimGCDNetwork

    model = SimGCDNetwork(
        backbone,
        DINOHead(
            in_dim=8,
            out_dim=4,
            use_bn=False,
            norm_last_layer=True,
            num_layers=1,
            hidden_dim=8,
            bottleneck_dim=8,
        ),
        finetune_from_block=0,
    )
    ctx = {
        "needs_frequency_estimate": True,
        "num_classes": 4,
        "num_known_classes": 2,
        "class_frequencies": torch.tensor([10.0, 10.0, 5.0, 1.0]),
        "distribution_loader": None,
    }
    with pytest.raises(RuntimeError, match="masked distribution_loader"):
        adapter.before_epoch(model, 0, ctx, {})

    # Empty/failing distribution loader must also fail loudly instead of using class_frequencies
    ctx["distribution_loader"] = [
        DistributionBatch(
            images=torch.randn(2, 4),
            known_targets=torch.tensor([-1, -1]),
            labeled_mask=torch.tensor([False, False]),
            sample_ids=torch.tensor([0, 1]),
        )
    ]
    with pytest.raises(ValueError, match="requires labeled samples"):
        adapter.before_epoch(model, 0, ctx, {})


def test_known_ema_continues_after_epoch_60_and_uninitialized_scores_masked():
    plugin = ProtoTailPlugin()
    model = nn.Linear(4, 4)
    plugin.attach(
        model,
        {
            "num_slots": 2,
            "known_prototype_end_epoch": 60,
            "novel_prototype_end_epoch": 60,
            "prototype_momentum": 0.5,
        },
        4,
        4,
    )
    bank = plugin.components.prototypes
    parts = torch.nn.functional.normalize(torch.randn(2, 2, 4), dim=-1)
    # Before any update, scores must be strictly masked to 0.0
    scores_init, _, _ = bank(parts)
    torch.testing.assert_close(scores_init, torch.zeros_like(scores_init))

    from prototail_gcd.contracts import BranchOutput, ModelOutput

    output = ModelOutput(
        branches={
            "student": BranchOutput(
                global_features=torch.randn(2, 4),
                patch_tokens=torch.randn(2, 3, 4),
                logits=torch.tensor(
                    [[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]]
                ),
            )
        }
    )
    batch = TrainingBatch(
        views=(torch.randn(2, 4),),
        known_targets=torch.tensor([0, -1]),
        labeled_mask=torch.tensor([True, False]),
        sample_ids=torch.tensor([1, 2]),
    )
    plugin.after_optimizer_step(
        plugin.augment_output(output),
        batch,
        {"epoch": 75, "total_epochs": 200, "num_known_classes": 2},
    )
    # Known class 0 updated even at epoch 75 (> 60); Novel class 2 blocked (> 60 without D2 store)
    assert bank.update_counts[0] == 1
    assert bank.update_counts[2] == 0
    scores_after, _, _ = bank(parts)
    assert not torch.allclose(scores_after[:, 0], torch.tensor(0.0))
    torch.testing.assert_close(scores_after[:, 1:], torch.zeros_like(scores_after[:, 1:]))


def test_promoted_novel_enters_native_ce_supcon_part_ce_and_updates_bank_after_e60():
    torch.manual_seed(3)
    adapter = SimGCDAdapter()
    baseline_cfg = {
        "num_mlp_layers": 1,
        "hidden_dim": 8,
        "bottleneck_dim": 8,
        "use_bn": False,
        "norm_last_layer": True,
        "sup_weight": 0.35,
        "memax_weight": 1.0,
        "student_temp": 0.1,
        "contrastive_temp": 1.0,
        "supervised_contrastive_temp": 0.07,
        "warmup_teacher_temp": 0.07,
        "teacher_temp": 0.04,
        "warmup_teacher_temp_epochs": 1,
        "enable_pseudo_labeling": False,
        "enable_novel_pseudo": False,
    }
    backbone = _TinyViTBackbone(feature_dim=8, num_blocks=12)
    from prototail_gcd.baselines.simgcd.adapter import SimGCDNetwork
    from prototail_gcd.baselines.simgcd.components import DINOHead
    from prototail_gcd.models.dino_backbone import set_finetune_blocks

    set_finetune_blocks(backbone, 11)
    model = SimGCDNetwork(
        backbone,
        DINOHead(
            in_dim=8,
            out_dim=4,
            use_bn=False,
            norm_last_layer=True,
            num_layers=1,
            hidden_dim=8,
            bottleneck_dim=8,
        ),
        finetune_from_block=11,
    )
    plugin = ProtoTailPlugin()
    method_cfg = {
        "num_slots": 2,
        "part_weight": 0.5,
        "part_loss_weight": 0.5,
        "temperature": 0.1,
        "gate_regularization_weight": 0.05,
        "capacity_regularization_weight": 0.0,
        "part_ce_start_epoch": 30,
        "pseudo_confidence": 0.5,
        "prototype_momentum": 0.5,
        "novel_prototype_end_epoch": 60,
    }
    plugin.attach(model, method_cfg, 4, 8)
    store = PromotionStore(total_classes=4, known_classes=2)
    # Sample 101 is unlabeled novel promoted to novel class 3; sample 102 is unlabeled promoted to known class 0.
    store.add_accepted([(101, 3), (102, 0)])

    batch = TrainingBatch(
        views=(torch.randn(4, 4), torch.randn(4, 4)),
        known_targets=torch.tensor([1, -1, -1, -1]),
        labeled_mask=torch.tensor([True, False, False, False]),
        sample_ids=torch.tensor([100, 101, 102, 103]),
    )
    ctx = adapter.create_training_context(
        baseline_cfg, num_views=2, total_epochs=100
    )
    ctx.update(
        {
            "num_classes": 4,
            "num_known_classes": 2,
            "persistent_promotion_store": store,
        }
    )

    # First check: promoted samples enter native CE, SupCon, and part CE without changing original_batch
    injected = store.inject(batch)
    assert injected.labeled_mask.tolist() == [True, True, True, False]
    assert injected.known_targets.tolist() == [1, 3, 0, -1]
    assert batch.labeled_mask.tolist() == [True, False, False, False]

    # Verify explicitly on identical model forward that PromotionStore changes native CE, SupCon, and part CE
    with torch.no_grad():
        out_raw = adapter.forward(model, batch)
        out_inj = plugin.augment_output(adapter.forward(model, injected))
        out_raw_aug = plugin.augment_output(out_raw)
        loss_raw = adapter.compute_losses(out_raw, batch, {**ctx, "epoch": 75}, baseline_cfg)
        loss_inj = adapter.compute_losses(out_inj, injected, {**ctx, "epoch": 75}, baseline_cfg)
        part_raw = plugin.compute_losses(out_raw_aug, batch, {**ctx, "epoch": 75})
        part_inj = plugin.compute_losses(out_inj, injected, {**ctx, "epoch": 75})
        assert not torch.allclose(loss_raw["classification"], loss_inj["classification"])
        assert not torch.allclose(
            loss_raw["supervised_contrastive"], loss_inj["supervised_contrastive"]
        )
        assert not torch.allclose(
            part_raw["part_classification"], part_inj["part_classification"]
        )

    optimizers = {
        "main": torch.optim.SGD(
            [
                p
                for g in adapter.optimizer_groups(model, baseline_cfg)["main"]
                for p in g["params"]
            ],
            lr=0.01,
        ),
        **{
            f"prototail_{k}": torch.optim.SGD(v, lr=0.01)
            for k, v in plugin.optimizer_groups(model).items()
        },
    }
    engine = TrainingEngine(
        model=model,
        baseline=adapter,
        optimizers=optimizers,
        baseline_config=baseline_cfg,
        method=plugin,
        method_config=method_cfg,
        context=ctx,
    )

    metrics_with_store = engine.train_batch(batch, epoch=75)
    assert metrics_with_store["pseudo_loss"] == 0.0  # No double batch pseudo CE
    assert metrics_with_store["aux_fused"] > 0.0
    # Known bank: ONLY original GT sample 100 (class 1) updated Known; sample 102 (promoted to Known 0) must NOT update Known 0!
    assert plugin.components.prototypes.update_counts[1] == 1
    assert plugin.components.prototypes.update_counts[0] == 0

    # Verify explicitly that accepted D2 novel sample reinforces Novel prototype 3 at epoch 75 (> 60)
    from prototail_gcd.contracts import BranchOutput, ModelOutput

    forced_out = ModelOutput(
        branches={
            "student": BranchOutput(
                global_features=torch.randn(8, 8),
                patch_tokens=torch.randn(8, 5, 8),
                logits=torch.tensor(
                    [
                        [0.0, 5.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 5.0],
                        [5.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 5.0, 0.0],
                    ]
                    * 2
                ),
            )
        }
    )
    plugin.after_optimizer_step(
        plugin.augment_output(forced_out),
        batch,  # original_batch passed by TrainingEngine
        {**ctx, "epoch": 75, "total_epochs": 200},
    )
    assert plugin.components.prototypes.update_counts[3] == 1
    assert plugin.components.prototypes.update_counts[0] == 0
    assert plugin.components.prototypes.update_counts[2] == 0


def test_paper_selector_rejects_hidden_gt_and_runs_d1_d2_with_simgcd_adapter():
    bundle, _ = _make_tiny_dataset_bundle(
        num_classes=4, known_classes=2, samples_per_class=8
    )
    adapter = SimGCDAdapter()
    backbone = _TinyViTBackbone(feature_dim=8, num_blocks=2)
    from prototail_gcd.baselines.simgcd.adapter import SimGCDNetwork
    from prototail_gcd.baselines.simgcd.components import DINOHead

    model = SimGCDNetwork(
        backbone,
        DINOHead(
            in_dim=8,
            out_dim=4,
            use_bn=False,
            norm_last_layer=True,
            num_layers=1,
            hidden_dim=8,
            bottleneck_dim=8,
        ),
        finetune_from_block=0,
    )
    selector = PaperDualSelector(
        n_known=2,
        n_classes=4,
        known_epochs=(30,),
        novel_epochs=(50,),
        min_novel_cluster_size=3,
    )
    store = PromotionStore(total_classes=4, known_classes=2)
    ev30 = selector.run(
        30, model, adapter, bundle.evaluation_loaders["distribution"], torch.device("cpu"), store
    )
    ev50 = selector.run(
        50, model, adapter, bundle.evaluation_loaders["distribution"], torch.device("cpu"), store
    )
    assert ev30["epoch"] == 30 and "known" in ev30
    assert ev50["epoch"] == 50 and "novel" in ev50
    json.dumps([ev30, ev50], allow_nan=False)

    # Leaking an unlabeled sample's GT target must fail loudly
    leaked_loader = [
        DistributionBatch(
            images=torch.randn(2, 4),
            known_targets=torch.tensor([0, 3]),
            labeled_mask=torch.tensor([True, False]),
            sample_ids=torch.tensor([0, 1]),
        )
    ]
    with pytest.raises(RuntimeError, match="Unlabeled GT leak"):
        selector.collect(model, adapter, leaked_loader, torch.device("cpu"))


def test_lcap_has_gate_gradients_and_uniform_target_mode_shares_budget():
    from prototail_gcd.contracts import BranchOutput, ModelOutput

    freqs = torch.tensor([100.0, 50.0, 10.0, 1.0])
    batch = TrainingBatch(
        views=(torch.randn(2, 4), torch.randn(2, 4)),
        known_targets=torch.tensor([0, -1]),
        labeled_mask=torch.tensor([True, False]),
        sample_ids=torch.tensor([0, 1]),
    )
    out = ModelOutput(
        branches={
            "student": BranchOutput(
                global_features=torch.randn(4, 4),
                patch_tokens=torch.randn(4, 5, 4),
                logits=torch.randn(4, 4),
            )
        }
    )

    plugin_freq = ProtoTailPlugin()
    plugin_freq.attach(
        nn.Linear(4, 4),
        {
            "num_slots": 3,
            "capacity_regularization_weight": 0.1,
            "capacity_target_mode": "frequency",
        },
        4,
        4,
    )
    losses_freq = plugin_freq.compute_losses(
        plugin_freq.augment_output(out),
        batch,
        {"num_known_classes": 2, "epoch": 30, "estimated_frequencies": freqs},
    )
    losses_freq["total"].backward()
    assert plugin_freq.components.prototypes.gate_logits.grad is not None
    assert float(plugin_freq.components.prototypes.gate_logits.grad.abs().sum()) > 0.0

    plugin_uni = ProtoTailPlugin()
    plugin_uni.attach(
        nn.Linear(4, 4),
        {
            "num_slots": 3,
            "capacity_regularization_weight": 0.1,
            "capacity_target_mode": "uniform",
        },
        4,
        4,
    )
    losses_uni = plugin_uni.compute_losses(
        plugin_uni.augment_output(out),
        batch,
        {"num_known_classes": 2, "epoch": 30, "estimated_frequencies": freqs},
    )
    assert float(losses_uni["dist_gate"].detach()) != float(
        losses_freq["dist_gate"].detach()
    )


def test_eval_modes_balanced_concat_mean_gate_vs_legacy():
    from prototail_gcd.contracts import BranchOutput, ModelOutput

    plugin = ProtoTailPlugin()
    plugin.attach(nn.Linear(4, 4), {"num_slots": 2}, 4, 4)
    # Give different classes very different gate patterns
    with torch.no_grad():
        plugin.components.prototypes.gate_logits[0].fill_(3.0)
        plugin.components.prototypes.gate_logits[1].fill_(-3.0)
    # Raw global features with huge scale vs unit parts
    out = ModelOutput(
        branches={
            "student": BranchOutput(
                global_features=torch.tensor([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0]]),
                patch_tokens=torch.randn(2, 3, 4),
                logits=torch.randn(2, 4),
            )
        }
    )
    aug = plugin.augment_output(out)
    pred_bal = plugin.predict(aug)
    assert pred_bal.features.shape == (2, 8)
    # First half and second half should have equal L2 norm in balanced_concat
    torch.testing.assert_close(
        pred_bal.features[:, :4].norm(dim=-1),
        pred_bal.features[:, 4:].norm(dim=-1),
    )

    plugin.config = {
        **plugin.config,
        "eval_feature_mode": "legacy_concat",
        "eval_pool_mode": "argmax_class_gate",
    }
    pred_leg = plugin.predict(aug)
    assert float(pred_leg.features[:, :4].norm(dim=-1).mean().detach()) > float(
        pred_leg.features[:, 4:].norm(dim=-1).mean().detach()
    )


def test_tiny_train_main_end_to_end_cpu_two_epochs_fixed_checkpoint_and_paper_selection(
    monkeypatch, tmp_path: Path
):
    import yaml
    from prototail_gcd import train as train_mod
    from prototail_gcd.baselines.simgcd import adapter as simgcd_adapter_mod
    from prototail_gcd.pseudo import paper_selector as selector_mod

    bundle, _ = _make_tiny_dataset_bundle(
        num_classes=4, known_classes=2, samples_per_class=6
    )

    class _FakeAdapter:
        def build(self, _cfg):
            return bundle

    monkeypatch.setattr(
        "prototail_gcd.data.get_dataset_adapter", lambda _name: _FakeAdapter()
    )
    monkeypatch.setattr(
        simgcd_adapter_mod,
        "load_backbone",
        lambda _name, pretrained=True: _TinyViTBackbone(feature_dim=8, num_blocks=12),
    )
    monkeypatch.setattr(
        simgcd_adapter_mod,
        "backbone_spec",
        lambda _name: {"feature_dim": 8},
    )
    monkeypatch.setattr(
        train_mod,
        "backbone_spec",
        lambda _name: {"feature_dim": 8},
    )

    # Ensure selector triggers on epoch 1 & 2 during the 2-epoch test
    orig_selector = selector_mod.PaperDualSelector

    class _TwoEpochSelector(orig_selector):
        def __init__(self, n_known, n_classes, **kwargs):
            kwargs["known_epochs"] = (1, 2)
            kwargs["novel_epochs"] = ()
            super().__init__(n_known, n_classes, **kwargs)

    monkeypatch.setattr(selector_mod, "PaperDualSelector", _TwoEpochSelector)

    # Make TEST accuracy artificially peak at epoch 1 and drop at epoch 2 to prove .best.pt does NOT use TEST GT
    orig_evaluate = train_mod.evaluate

    def _wrapped_evaluate(*args, **kwargs):
        metrics = orig_evaluate(*args, **kwargs)
        loader = args[1]
        if loader is bundle.evaluation_loaders["test"]:
            # Determine call count via attribute on wrapper
            _wrapped_evaluate.test_calls += 1
            metrics["all_acc"] = 99.0 if _wrapped_evaluate.test_calls == 1 else 12.0
        return metrics

    _wrapped_evaluate.test_calls = 0
    monkeypatch.setattr(train_mod, "evaluate", _wrapped_evaluate)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (repo_root / "configs" / "experiment_simgcd.yaml").read_text(encoding="utf-8")
    )
    cfg["baseline"]["configs"]["simgcd"]["warmup_teacher_temp_epochs"] = 1
    cfg["baseline"]["configs"]["simgcd"]["hidden_dim"] = 16
    cfg["baseline"]["configs"]["simgcd"]["bottleneck_dim"] = 8
    cfg_path = tmp_path / "tiny_simgcd.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 1. Full run on cars (tests cars memax=1, D1/D2, Lcap estimator, fixed-epoch checkpoint, unlabeled_train)
    full_dir = tmp_path / "full_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(cfg_path),
            "--dataset-name",
            "cars",
            "--imb-ratio",
            "20",
            "--use-parts",
            "--enable-pseudo-labeling",
            "--enable-novel-pseudo",
            "--pseudo-mode",
            "paper",
            "--d2-min-cluster-size",
            "3",
            "--simgcd-eval-head",
            "kmeans",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--no-fp16",
            "--output-dir",
            str(full_dir),
            "--run-name",
            "tiny_full",
        ],
    )
    train_mod.main()

    full_metrics = [
        json.loads(line)
        for line in (full_dir / "tiny_full.metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(full_metrics) == 2
    assert "unlabeled_train" in full_metrics[1]
    assert "paper_selection" in full_metrics[0]
    assert full_metrics[0]["train"]["pseudo_loss"] == 0.0
    saved_cfg = yaml.safe_load(
        (full_dir / "tiny_full.config.yaml").read_text(encoding="utf-8")
    )
    assert saved_cfg["baseline"]["resolved"]["memax_weight"] == 1.0
    best_ckpt = torch.load(
        full_dir / "tiny_full.best.pt", map_location="cpu", weights_only=False
    )
    assert best_ckpt["epoch"] == 2
    assert best_ckpt["selection_protocol"] == "fixed_final_epoch_unlabeled_train"

    # 2. Base run on cub (tests method=None, no estimator overhead, unlabeled_train KMeans present)
    _wrapped_evaluate.test_calls = 0
    base_dir = tmp_path / "base_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(cfg_path),
            "--dataset-name",
            "cub",
            "--imb-ratio",
            "20",
            "--simgcd-eval-head",
            "kmeans",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--no-fp16",
            "--output-dir",
            str(base_dir),
            "--run-name",
            "tiny_base",
        ],
    )
    train_mod.main()
    base_metrics = [
        json.loads(line)
        for line in (base_dir / "tiny_base.metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(base_metrics) == 2
    assert "unlabeled_train" in base_metrics[1]
    assert "paper_selection" not in base_metrics[0]
    base_saved_cfg = yaml.safe_load(
        (base_dir / "tiny_base.config.yaml").read_text(encoding="utf-8")
    )
    assert base_saved_cfg["baseline"]["resolved"]["memax_weight"] == 2.0
    base_ckpt = torch.load(
        base_dir / "tiny_base.best.pt", map_location="cpu", weights_only=False
    )
    assert base_ckpt["epoch"] == 2
    assert base_ckpt["selection_protocol"] == "fixed_final_epoch_unlabeled_train"
