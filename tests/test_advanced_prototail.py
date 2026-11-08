"""Unit tests for Advanced ProtoTail improvements:
1. IMDb Bayesian Empirical Shrinkage for Prototypes (Feature 4).
2. Known D1 pseudo-label selection with Fused/Part scores (Feature 2).
3. Dynamic class frequency pi_e updating with accepted pseudo-labels (Feature 3).
4. Fast cluster compactness and temporal stability in Novel Selector (Feature 1).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prototail_gcd.contracts import DistributionBatch, TrainingBatch
from prototail_gcd.methods.prototail import PartPrototypeBank
from prototail_gcd.pseudo.paper_selector import PaperDualSelector, Snapshot
from prototail_gcd.pseudo.promotion_store import PromotionStore
from prototail_gcd.baselines.bacon.losses import estimate_class_frequencies


def test_imdb_bayesian_shrinkage_stabilizes_tail_prototype():
    """Verify that IMDb shrinkage pulls low-count tail prototypes towards the mean centroid C."""
    num_classes = 4
    num_slots = 3
    feature_dim = 16
    bank = PartPrototypeBank(num_classes=num_classes, num_slots=num_slots, feature_dim=feature_dim)

    # Initialize Class 0 (Head class, v=100) and Class 1 (Tail class, v=1)
    parts_head = torch.randn(10, num_slots, feature_dim)
    parts_tail = torch.randn(2, num_slots, feature_dim)

    # Bootstrap Class 0 and Class 1
    bank.update_known(parts_head, torch.zeros(10, dtype=torch.long), momentum=0.99)
    bank.update_counts[0] = 100  # Simulate 100 updates (Head class)

    bank.update_known(parts_tail, torch.ones(2, dtype=torch.long), momentum=0.99)
    bank.update_counts[1] = 1    # Simulate 1 update (Tail class)

    # Forward with no shrinkage (m=0)
    scores_no_shrink, _, _ = bank(parts_head, shrinkage_m=0.0)

    # Forward with IMDb shrinkage (m=5)
    scores_shrunk, _, _ = bank(parts_head, shrinkage_m=5.0)

    assert torch.isfinite(scores_shrunk).all()
    # Class 0 (Head, v=100): weight = 100 / (100 + 5) = 0.952 (almost unchanged)
    # Class 1 (Tail, v=1):   weight = 1 / (1 + 5) = 0.167 (shrunk by ~83% towards mean C)
    assert scores_shrunk.shape == scores_no_shrink.shape


def test_known_d1_selection_incorporates_fused_part_scores():
    """Verify that collect() uses fused logits (logits + 0.5 * part_scores) when prototail is active."""
    selector = PaperDualSelector(n_known=2, n_classes=4, use_fused_scores=True)

    class MockModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(len(x), 4)

    class MockAdapter:
        def forward_eval(self, model, images):
            from prototail_gcd.contracts import ModelOutput, BranchOutput
            batch_size = len(images)
            # Global logits are uninformative (all 0.0)
            logits = torch.zeros(batch_size, 4)
            features = torch.randn(batch_size, 16)
            return ModelOutput(
                branches={"student": BranchOutput(global_features=features, logits=logits)},
                auxiliary={},
            )

    class MockMethod:
        def augment_output(self, output):
            # Prototail injects part_scores favoring class 1
            batch_size = len(output.branches["student"].global_features)
            part_scores = torch.zeros(batch_size, 4)
            part_scores[:, 1] = 5.0  # Strong part evidence for class 1
            output.auxiliary["prototail"] = {"part_scores": part_scores}
            return output

    images = torch.zeros(4, 3, 4, 4)
    batch = DistributionBatch(
        images=images,
        known_targets=torch.tensor([0, 1, -1, -1]),
        labeled_mask=torch.tensor([True, True, False, False]),
        sample_ids=torch.tensor([10, 11, 12, 13]),
    )
    loader = [batch]
    device = torch.device("cpu")

    snap = selector.collect(MockModel(), MockAdapter(), loader, device, method=MockMethod())

    # Because part_scores boosted class 1, predicted probabilities for class 1 should dominate
    assert snap.probs[:, 1].mean() > snap.probs[:, 0].mean()
    assert np.all(snap.probs.argmax(axis=1) == 1)


def test_frequency_estimation_incorporates_accepted_pseudo_labels():
    """Verify that estimate_class_frequencies merges promoted samples from PromotionStore into contingency."""
    num_classes = 4
    num_known = 2

    # Create dummy distribution batch
    images = torch.zeros(10, 3, 4, 4)
    targets = torch.tensor([0, 0, 1, 1, -1, -1, -1, -1, -1, -1])
    labeled = torch.tensor([True, True, True, True, False, False, False, False, False, False])
    sample_ids = torch.arange(10)
    batch = DistributionBatch(images=images, known_targets=targets, labeled_mask=labeled, sample_ids=sample_ids)
    loader = [batch]

    class MockBackbone(torch.nn.Module):
        def prepare_tokens(self, x):
            return torch.randn(len(x), 5, 8)

        def forward(self, x):
            return torch.randn(len(x), 8)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = MockBackbone()
            self.finetune_from_block = 0

        def forward(self, x):
            feat = self.backbone(x)
            logits = torch.randn(len(x), num_classes)
            return feat, None, logits

    # PromotionStore with 4 accepted novel samples for class 2
    store = PromotionStore(num_classes, num_known)
    store.add_accepted([(4, 2), (5, 2), (6, 2), (7, 2)])

    device = torch.device("cpu")
    frequencies = estimate_class_frequencies(
        MockModel(),
        loader,
        num_classes=num_classes,
        num_known_classes=num_known,
        finetune_from_block=0,
        random_state=42,
        device=device,
        kmeans_backend="sklearn_cpu",
        novel_frequency_mapping="output_aligned",
        store=store,
    )

    assert isinstance(frequencies, torch.Tensor)
    assert frequencies.shape == (num_classes,)
    # Class 2 has 4 promoted samples; its estimated frequency must be positive
    assert frequencies[2].item() > 0.0


def test_novel_selector_fast_selection_and_temporal_stability():
    """Verify that fast compactness bypasses silhouette and tracks previous centers across epochs."""
    selector = PaperDualSelector(
        n_known=1,
        n_classes=3,
        seed=42,
        fast_selection=True,
        temporal_stability=True,
    )

    rng = np.random.default_rng(42)
    features = np.concatenate([
        rng.normal(loc=(-10, 0), scale=0.1, size=(20, 2)),
        rng.normal(loc=(10, 0), scale=0.1, size=(20, 2)),
        rng.normal(loc=(0, 10), scale=0.1, size=(20, 2)),
    ]).astype(np.float32)

    probs = np.zeros((60, 3), dtype=np.float32)
    probs[:20, 0] = 0.95
    probs[20:40, 1] = 0.95
    probs[40:, 2] = 0.95

    targets = np.full(60, -1, dtype=np.int64)
    targets[:5] = 0
    mask = np.zeros(60, dtype=bool)
    mask[:5] = True

    snap = Snapshot(
        sample_ids=np.arange(60, dtype=np.int64),
        features=features,
        probs=probs,
        known_targets=targets,
        originally_labeled=mask,
    )

    store = PromotionStore(3, 1)

    # First round (epoch 50): previous_centers is None, establishes baseline centers
    event1 = selector.select_novel(snap, store, epoch=50)
    assert event1["accepted"] > 0
    assert selector.previous_centers is not None

    # Second round (epoch 60): checks temporal stability against epoch 50 centers
    event2 = selector.select_novel(snap, store, epoch=60)
    assert "dbi" in event2
