import numpy as np
import pytest
from prototail_gcd.pseudo.paper_selector import PaperDualSelector, Snapshot


class Store:
    def __init__(self):
        self.labels = {}

    def export_state(self):
        return dict(self.labels)

    def add_accepted(self, pairs):
        added = 0
        for sid, label in pairs:
            if sid in self.labels and self.labels[sid] != label:
                raise RuntimeError("remap")
            if sid not in self.labels:
                added += 1
                self.labels[sid] = label
        return added


def make_snapshot():
    rng = np.random.default_rng(1)
    # Three well-separated clusters; 1 known and 2 novel.
    features = np.concatenate([
        rng.normal(loc=(-10, 0), scale=.2, size=(25, 2)),
        rng.normal(loc=(10, 0), scale=.2, size=(25, 2)),
        rng.normal(loc=(0, 10), scale=.2, size=(25, 2)),
    ])
    probs = np.zeros((75, 3), dtype=np.float32)
    probs[:25, 0] = .95
    probs[25:50, 1] = .96
    probs[50:, 2] = .96
    probs += .02
    probs /= probs.sum(axis=1, keepdims=True)
    targets = np.full(75, -1, dtype=np.int64)
    targets[:6] = 0
    mask = np.zeros(75, dtype=bool)
    mask[:6] = True
    return Snapshot(np.arange(75, dtype=np.int64), features.astype(np.float32), probs, targets, mask)


def test_rejects_oracle_gt():
    s = make_snapshot()
    s.known_targets[25] = 1
    with pytest.raises(RuntimeError, match="Novel/unlabeled GT"):
        Snapshot(s.sample_ids, s.features, s.probs, s.known_targets, s.originally_labeled)


def test_known_selection_promotes_only_predicted_known():
    s = make_snapshot()
    store = Store()
    selector = PaperDualSelector(n_known=1, n_classes=3)
    event = selector.select_known(s, store)
    assert event["accepted"] == 16  # ceil(.8 * (25 - 6))
    assert all(sid < 25 and sid >= 6 and label == 0 for sid, label in store.labels.items())
    assert selector.select_known(s, store)["accepted"] == 3  # ceil(.8 * 3)


def test_novel_selection_passes_paper_gates():
    s = make_snapshot()
    selector = PaperDualSelector(n_known=1, n_classes=3, seed=21)
    store = Store()
    event = selector.select_novel(s, store, epoch=50)
    assert event["accepted"] >= 20, event
    assert set(store.labels.values()).issubset({1, 2})
    assert len(set(store.labels)) == len(store.labels)


def test_selector_does_not_run_novel_before_epoch50():
    selector = PaperDualSelector(n_known=1, n_classes=3)
    assert not selector.scheduled(40) or 40 in selector.known_epochs
    assert 40 not in selector.novel_epochs
    with pytest.raises(ValueError):
        PaperDualSelector(n_known=1, n_classes=3, novel_epochs=(30,))


def test_same_ids_are_never_repromoted():
    s = make_snapshot()
    store = Store()
    selector = PaperDualSelector(n_known=1, n_classes=3, seed=21)
    first = selector.select_novel(s, store, epoch=50)
    second = selector.select_novel(s, store, epoch=60)
    assert first["accepted"] >= 20
    assert second["accepted"] == 0


# Paper D2 fits clusters on unlabeled only, then assigns labeled known to centroids.
def test_partitions_only_unlabeled_samples():
    snap = make_snapshot()
    selector = PaperDualSelector(n_known=1, n_classes=3, seed=21)
    lengths = []
    real_partition = selector._partition
    def track(features, seed):
        lengths.append(len(features))
        return real_partition(features, seed)
    selector._partition = track
    selector.select_novel(snap, Store(), epoch=50)
    assert lengths == [69, 69]
