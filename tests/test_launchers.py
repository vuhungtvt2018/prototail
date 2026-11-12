"""Unit tests for Kaggle and Modal launcher flag forwarding (Stage 2+ controls)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_root_module(name: str):
    path = REPO_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_train_forwards_stage2_flags_and_normalizes_dataset(
    monkeypatch, tmp_path: Path
):
    kaggle_train = _load_root_module("kaggle_train")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        kaggle_train,
        "_prepare_gcd_dataset",
        lambda **_kwargs: (str(tmp_path / "data"), str(tmp_path / "splits")),
    )
    monkeypatch.setattr(
        kaggle_train.subprocess,
        "run",
        lambda cmd, env=None: captured.update({"cmd": cmd, "env": env}) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kaggle_train.py",
            "--baseline",
            "simgcd",
            "--dataset-name",
            "car",
            "--imb-ratio",
            "20",
            "--use-parts",
            "--enable-pseudo-labeling",
            "--enable-novel-pseudo",
            "--pseudo-mode",
            "legacy_ce",
            "--d2-min-cluster-size",
            "3",
            "--simgcd-eval-head",
            "kmeans",
            "--seed",
            "1",
            "--epochs",
            "200",
            "--eval-interval",
            "5",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    kaggle_train.main()
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--dataset-name") + 1] == "cars"
    assert cmd[cmd.index("--imb-ratio") + 1] == "20"
    assert cmd[cmd.index("--pseudo-mode") + 1] == "legacy_ce"
    assert cmd[cmd.index("--d2-min-cluster-size") + 1] == "3"
    assert cmd[cmd.index("--seed") + 1] == "1"
    assert "--use-parts" in cmd
    assert "--enable-pseudo-labeling" in cmd
    assert "--enable-novel-pseudo" in cmd


def test_kaggle_push_generates_unique_slug_and_forwards_pseudo_mode(
    monkeypatch, tmp_path: Path
):
    kaggle_push = _load_root_module("kaggle_push")
    fake_repo = tmp_path / "repo"
    (fake_repo / "src" / "prototail_gcd").mkdir(parents=True)
    (fake_repo / "src" / "prototail_gcd" / "train.py").write_text("# stub\n", encoding="utf-8")
    (fake_repo / "configs").mkdir()
    (fake_repo / "configs" / "experiment_simgcd.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (fake_repo / "kaggle_train.py").write_text("# stub\n", encoding="utf-8")
    (fake_repo / "kaggle_push.py").write_text("# stub\n", encoding="utf-8")

    monkeypatch.setattr(kaggle_push, "__file__", str(fake_repo / "kaggle_push.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kaggle_push.py",
            "--baseline",
            "simgcd",
            "--dataset-name",
            "cub",
            "--imb-ratio",
            "20",
            "--config",
            "Full",
            "--pseudo-mode",
            "legacy_ce",
            "--d2-min-cluster-size",
            "3",
            "--seed",
            "2",
            "--kaggle-user",
            "testuser",
            "--dry-run",
        ],
    )
    kaggle_push.main()
    staging = fake_repo / "build_kaggle_kernel"
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "testuser/prototail-simgcd-cub-full-legacy-ce-min3-imb20-s2"
    script_text = (staging / "run_script.py").read_text(encoding="utf-8")
    assert "'--pseudo-mode', 'legacy_ce'" in script_text
    assert "'--d2-min-cluster-size', '3'" in script_text
    assert "'--seed', '2'" in script_text


def test_modal_train_helpers_and_command_forwarding(monkeypatch, tmp_path: Path):
    modal_train = _load_root_module("modal_train")
    assert modal_train._normalize_dataset_name("car") == "cars"
    assert modal_train._normalize_dataset_name("scars") == "cars"
    assert modal_train._normalize_gpu("a100") == "A100"

    cfg, use_parts, ablate, pseudo, novel = modal_train._resolve_preset_flags(
        "Full",
        use_parts=False,
        ablate_adaptive_capacity=False,
        enable_pseudo_labeling=False,
        enable_novel_pseudo=False,
        enable_prototail=False,
    )
    assert cfg == "Full"
    assert (use_parts, ablate, pseudo, novel) == (True, False, True, True)

    captured: dict[str, object] = {}
    monkeypatch.setattr(modal_train, "PROJECT_DIR", str(REPO_ROOT))
    monkeypatch.setattr(modal_train, "OUTPUT_DIR", str(tmp_path / "modal_out"))
    monkeypatch.setattr(modal_train.storage_volume, "reload", lambda: None)
    monkeypatch.setattr(
        modal_train,
        "_run_training",
        lambda cmd, log_path: captured.update({"cmd": cmd, "log_path": log_path}),
    )

    # Invoke underlying function bypassing Modal Function wrapper
    modal_train.train_remote.get_raw_f()(
        baseline="simgcd",
        dataset_name="cars",
        imb_ratio=20,
        config_name="Full",
        pseudo_mode="legacy_ce",
        d2_min_cluster_size=3,
        seed=0,
        epochs=200,
        eval_interval=5,
        data_root_override=str(tmp_path / "cars_data"),
        splits_root_override=str(REPO_ROOT / "data_uq_idxs"),
    )
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--dataset-name") + 1] == "cars"
    assert cmd[cmd.index("--imb-ratio") + 1] == "20"
    assert cmd[cmd.index("--pseudo-mode") + 1] == "legacy_ce"
    assert cmd[cmd.index("--d2-min-cluster-size") + 1] == "3"
    assert "--use-parts" in cmd
    assert "--enable-pseudo-labeling" in cmd
    assert "--enable-novel-pseudo" in cmd
