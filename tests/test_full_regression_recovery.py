"""CPU regression guards for Full ProtoTail repairs (no GT unlabeled)."""

import numpy as np
import torch

from prototail_gcd.contracts import TrainingBatch
from prototail_gcd.methods.prototail import PartPrototypeBank
from prototail_gcd.pseudo.paper_selector import Snapshot, PaperDualSelector
from prototail_gcd.pseudo.promotion_store import PromotionStore
from prototail_gcd.training.engine import promoted_consistency_loss


def _batch():
    return TrainingBatch(
        views=(torch.zeros(3, 3, 4, 4), torch.zeros(3, 3, 4, 4)),
        known_targets=torch.tensor([0, -1, -1]),
        labeled_mask=torch.tensor([True, False, False]),
        sample_ids=torch.tensor([11, 12, 13]),
    )


def test_promotions_never_enter_ground_truth_mask():
    batch = _batch()
    store = PromotionStore(4, 2)
    store.add_accepted([(12, 2), (13, 3)])
    targets, mask = store.lookup(batch)
    assert torch.equal(mask, torch.tensor([False, True, True]))
    assert targets.tolist() == [-1, 2, 3]
    assert batch.known_targets.tolist() == [0, -1, -1]
    assert batch.labeled_mask.tolist() == [True, False, False]


def test_stale_promoted_labels_have_zero_gradient():
    batch = _batch()
    store = PromotionStore(4, 2)
    store.add_accepted([(12, 2)])
    logits = torch.zeros(6, 4, requires_grad=True)
    # Disagree with promoted class 2 on both views.
    with torch.no_grad():
        logits[1, 0] = 20
        logits[4, 0] = 20
    loss, kept = promoted_consistency_loss(
        logits, batch, store, temperature=0.1, min_confidence=0.7
    )
    assert kept == 0
    assert loss.item() == 0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_stable_promoted_label_has_gradient():
    batch = _batch()
    store = PromotionStore(4, 2)
    store.add_accepted([(12, 2)])
    logits = torch.zeros(6, 4, requires_grad=True)
    with torch.no_grad():
        logits[1, 2] = 2.0
        logits[4, 2] = 2.0
    loss, kept = promoted_consistency_loss(
        logits, batch, store, temperature=0.1, min_confidence=0.7
    )
    assert kept == 1
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() > 0


def test_unobserved_prototypes_never_contribute_evidence():
    bank = PartPrototypeBank(num_classes=4, num_slots=3, feature_dim=16)
    parts = torch.randn(2, 3, 16)
    scores, _, _ = bank(torch.nn.functional.normalize(parts, dim=-1))
    assert torch.count_nonzero(scores) == 0
    bank.update_known(
        torch.nn.functional.normalize(parts[:1], dim=-1),
        torch.tensor([0]), 0.99,
    )
    scores, _, _ = bank(torch.nn.functional.normalize(parts, dim=-1))
    assert torch.count_nonzero(scores[:, 1:]) == 0
    assert bank.update_counts[0] == 1


def test_aux_fused_ce_does_not_duplicate_global_head_gradient():
    logits = torch.randn(5, 4, requires_grad=True)
    parts = torch.randn(5, 4, requires_grad=True)
    labels = torch.arange(5) % 4
    aux = torch.nn.functional.cross_entropy(
        (logits.detach() + 0.5 * parts) / 0.1, labels
    )
    aux.backward()
    assert logits.grad is None
    assert parts.grad is not None and parts.grad.abs().sum() > 0


def test_d1_does_not_treat_previous_pseudo_as_gt():
    # Each known category has a different number of original labeled examples.
    selector = PaperDualSelector(n_known=2, n_classes=4)
    store = PromotionStore(4, 2)
    store.add_accepted([(40, 0)])
    snap = Snapshot(
        sample_ids=np.array([10, 11, 20, 21, 22, 40, 41]),
        features=np.eye(7, dtype=np.float32),
        probs=np.array([
            [.9,.1,0,0], [.9,.1,0,0], [.1,.9,0,0],
            [.1,.9,0,0], [.1,.9,0,0], [.95,.05,0,0], [.1,.9,0,0],
        ], dtype=np.float32),
        known_targets=np.array([0,0,1,1,1,-1,-1]),
        originally_labeled=np.array([1,1,1,1,1,0,0], dtype=bool),
    )
    event = selector.select_known(snap, store)
    # Smoothed score n=3 favors known class 1, unlike arbitrary class id 0.
    assert event['class_id'] == 1
    assert event['accepted'] >= 1
    assert selector.selected_known_classes == {1}
