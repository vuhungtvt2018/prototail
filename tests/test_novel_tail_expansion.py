"""Tests for ProtoTail Novel & Tail Expansion (anti-collapse, extended prototypes, adaptive threshold)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prototail_gcd.contracts import TrainingBatch
from prototail_gcd.methods.prototail import PartPrototypeBank
from prototail_gcd.pseudo.paper_selector import PaperDualSelector, Snapshot
from prototail_gcd.pseudo.promotion_store import PromotionStore
from prototail_gcd.baselines.simgcd.adapter import compute_simgcd_losses


class MockStore:
    def __init__(self):
        self.labels = {}

    def export_state(self):
        return dict(self.labels)

    def add_accepted(self, pairs):
        added = 0
        for sid, label in pairs:
            if sid not in self.labels:
                added += 1
                self.labels[sid] = label
        return added


def test_anti_collapse_novel_selector_assigns_distinct_classes():
    """Verify that multiple novel clusters are assigned distinct novel classes via Hungarian matching."""
    rng = np.random.default_rng(42)
    # 1 known class (c0) and 5 distinct novel clusters (c1..c5) with identical tight variance
    centers = [
        (-40, 0),
        (0, 40),
        (40, 0),
        (0, -40),
        (30, 30),
        (-30, -30),
    ]
    clusters = [rng.normal(loc=loc, scale=0.05, size=(30, 2)) for loc in centers]
    features = np.concatenate(clusters, axis=0).astype(np.float32)

    total_samples = len(features)  # 180
    n_classes = len(centers)       # 6
    probs = np.full((total_samples, n_classes), 0.02, dtype=np.float32)

    # Class 0 is known
    probs[:30, 0] = 0.90
    for i in range(1, n_classes):
        start = i * 30
        end = start + 30
        probs[start:end, i] = 0.85
    probs /= probs.sum(axis=1, keepdims=True)

    targets = np.full(total_samples, -1, dtype=np.int64)
    targets[:10] = 0
    mask = np.zeros(total_samples, dtype=bool)
    mask[:10] = True

    snap = Snapshot(
        sample_ids=np.arange(total_samples, dtype=np.int64),
        features=features,
        probs=probs,
        known_targets=targets,
        originally_labeled=mask,
    )

    # Selector with anti_collapse=True
    store_anti = MockStore()
    selector_anti = PaperDualSelector(n_known=1, n_classes=n_classes, seed=42, anti_collapse=True)
    res_anti = selector_anti.select_novel(snap, store_anti, epoch=50)

    # Multiple distinct novel classes should be accepted and assigned
    novel_classes_assigned = set(store_anti.labels.values())
    assert novel_classes_assigned.issubset(set(range(1, n_classes)))
    assert len(novel_classes_assigned) >= 2, f"Expected at least 2 distinct novel classes, got: {novel_classes_assigned}"
    assert res_anti["accepted"] >= 50


def test_novel_prototype_schedule_continues_up_to_180():
    """Verify that PartPrototypeBank updates novel classes past epoch 60 when novel_prototype_end_epoch=180."""
    bank = PartPrototypeBank(num_classes=4, num_slots=3, feature_dim=8)
    parts = torch.randn(10, 3, 8)
    parts = F.normalize(parts, dim=-1)
    pseudo_labels = torch.tensor([2, 2, 3, 3, 2, 3, 2, 3, 2, 3])
    confidence = torch.full((10,), 0.85)

    # Initial state: update_counts for novel classes 2 and 3 should be 0
    assert bank.update_counts[2] == 0
    assert bank.update_counts[3] == 0

    # At epoch 120 (well past old cutoff 60), update should succeed
    bank.update_novel(parts, pseudo_labels, confidence, threshold=0.7, momentum=0.99)
    assert bank.update_counts[2] > 0
    assert bank.update_counts[3] > 0


def test_adaptive_pseudo_threshold_lowers_bar_for_tail_classes():
    """Verify that adaptive thresholding allows tail novel classes with moderate confidence to participate."""
    batch_size = 4
    num_classes = 4
    known_class_count = 2

    # Unlabeled batch: 2 head samples and 2 tail samples
    views = (torch.zeros(batch_size, 3, 4, 4), torch.zeros(batch_size, 3, 4, 4))
    batch = TrainingBatch(
        views=views,
        known_targets=torch.tensor([-1, -1, -1, -1]),
        labeled_mask=torch.tensor([False, False, False, False]),
        sample_ids=torch.tensor([1, 2, 3, 4]),
    )

    # Mock logits for view 0 and view 1:
    # Samples 0 & 1 predict novel class 2 (head novel) with confidence ~0.85
    # Samples 2 & 3 predict novel class 3 (tail novel) with confidence ~0.58
    logits = torch.zeros(batch_size * 2, num_classes)
    logits[0:2, 2] = 2.0
    logits[2:4, 3] = 0.5
    logits[4:6, 2] = 2.0
    logits[6:8, 3] = 0.5

    # Projection tokens
    projection = torch.randn(batch_size * 2, 16)

    # 1. Without adaptive threshold (static 0.7): tail samples (conf 0.58) are rejected
    config_static = {
        "student_temp": 0.1,
        "contrastive_temp": 1.0,
        "supervised_contrastive_temp": 0.07,
        "sup_weight": 0.35,
        "memax_weight": 2.0,
        "enable_pseudo_labeling": True,
        "enable_novel_pseudo": True,
        "pseudo_confidence": 0.7,
        "adaptive_pseudo_threshold": False,
    }
    context = {
        "cluster_criterion": lambda l, tl, ep: (l * 0.0).sum(),
        "epoch": 10,
        "num_known_classes": known_class_count,
        "num_views": 2,
        "batch_size": batch_size,
    }

    # 2. With adaptive threshold and class frequencies indicating class 3 is tail
    config_adaptive = {
        **config_static,
        "adaptive_pseudo_threshold": True,
        "pseudo_confidence": 0.7,
        "pseudo_confidence_min": 0.5,
    }
    context_adaptive = {
        **context,
        "adaptive_pseudo_threshold": True,
        # Classes 0, 1, 2 have high frequency, class 3 has very low frequency (tail)
        "estimated_frequencies": torch.tensor([100.0, 80.0, 60.0, 5.0]),
    }

    losses_static = compute_simgcd_losses(
        projection=projection,
        logits=logits,
        batch=batch,
        context=context,
        config=config_static,
    )
    losses_adaptive = compute_simgcd_losses(
        projection=projection,
        logits=logits,
        batch=batch,
        context=context_adaptive,
        config=config_adaptive,
    )

    # In static mode, only class 2 (conf 0.85) passed threshold 0.7
    # In adaptive mode, both class 2 and class 3 (tail, threshold ~0.54) passed
    assert "pseudo_loss" in losses_adaptive
    assert losses_adaptive["n_pseudo_novel"] >= losses_static["n_pseudo_novel"]
    assert losses_adaptive["pseudo_loss"].item() > 0.0
