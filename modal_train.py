"""Modal launcher for ProtoTail GCD experiments (SimGCD and BaCon baselines).

Designed for seamless execution on Modal with persistent storage volume ('bacon-storage'),
supporting dataset/imbalance selection, Stage 2 ablation presets, persistent D1/D2 vs legacy_ce,
dynamic GPU selection, periodic volume commits, and full experiment logging.

Usage examples:
  # 1. Run SimGCD Base LT20 on CUB (200 epochs on A10G)
  modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Base

  # 2. Run SimGCD Full LT20 (paper D1/D2 min10) on Cars
  modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 20 --config Full --pseudo-mode paper --d2-min-cluster-size 10

  # 3. Run SimGCD legacy_ce control on CUB
  modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full --pseudo-mode legacy_ce
"""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "prototail-train"
LOCAL_REPO = Path(__file__).resolve().parent
PROJECT_DIR = "/root/prototail"
STORAGE_DIR = "/bacon-storage"
DATA_DIR = f"{STORAGE_DIR}/data"
OUTPUT_DIR = f"{STORAGE_DIR}/outputs/prototail"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyyaml",
        "tqdm",
        "pillow",
        "matplotlib",
        "torch",
        "torchvision",
    )
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path=PROJECT_DIR,
        copy=True,
        ignore=[
            "outputs",
            "build_kaggle_kernel",
            "__pycache__",
            "venv",
            ".venv",
            ".git",
            ".pytest_cache",
            "*.pt",
        ],
    )
    .run_commands(f"pip install -e {PROJECT_DIR}")
)

storage_volume = modal.Volume.from_name("bacon-storage", create_if_missing=True)

_DATASET_ALIASES = {
    "cub": "cub",
    "cub200": "cub",
    "cars": "cars",
    "car": "cars",
    "scars": "cars",
    "stanford_cars": "cars",
    "aircraft": "aircraft",
    "air": "aircraft",
    "fgvc_aircraft": "aircraft",
}


def _normalize_dataset_name(dataset_name: str) -> str:
    norm = dataset_name.lower().strip()
    if norm not in _DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset_name {dataset_name!r}; expected one of cub, cars, aircraft"
        )
    return _DATASET_ALIASES[norm]


def _normalize_gpu(gpu_name: str) -> str:
    normalized = gpu_name.lower().strip()
    mapping = {
        "a10g": "A10G",
        "a10": "A10G",
        "a100": "A100",
        "a100-40gb": "A100",
        "a100-80gb": "A100-80GB",
        "l4": "L4",
        "t4": "T4",
        "h100": "H100",
        "h200": "H200",
    }
    return mapping.get(normalized, "A10G")


def _resolve_preset_flags(
    config_name: str,
    *,
    use_parts: bool,
    ablate_adaptive_capacity: bool,
    enable_pseudo_labeling: bool,
    enable_novel_pseudo: bool,
    enable_prototail: bool,
) -> tuple[str, bool, bool, bool, bool]:
    cfg = config_name.strip()
    valid = {
        "Base",
        "Pseudo",
        "Prototype",
        "Prototype_Tailaware",
        "Full_wo_Tail",
        "Full",
        "Custom",
    }
    if cfg not in valid:
        raise ValueError(f"Unknown config preset {config_name!r}; choose from {sorted(valid)}")
    if cfg == "Pseudo":
        enable_pseudo_labeling = True
        enable_novel_pseudo = True
    elif cfg == "Prototype":
        use_parts = True
        ablate_adaptive_capacity = True
    elif cfg == "Prototype_Tailaware":
        use_parts = True
    elif cfg == "Full_wo_Tail":
        use_parts = True
        ablate_adaptive_capacity = True
        enable_pseudo_labeling = True
        enable_novel_pseudo = True
    elif cfg == "Full":
        use_parts = True
        enable_pseudo_labeling = True
        enable_novel_pseudo = True
    elif enable_prototail:
        use_parts = True
    return cfg, use_parts, ablate_adaptive_capacity, enable_pseudo_labeling, enable_novel_pseudo


def _resolve_data_root(dataset_name: str) -> str:
    norm = _normalize_dataset_name(dataset_name)
    if norm == "cub":
        candidates = [
            f"{DATA_DIR}/cub",
            f"{DATA_DIR}/cub/CUB_200_2011",
            f"{DATA_DIR}/CUB_200_2011",
            f"{STORAGE_DIR}/CUB_200_2011",
        ]
        for c in candidates:
            p = Path(c)
            if (p / "CUB_200_2011" / "images.txt").is_file():
                return str(p)
            if (p / "images.txt").is_file():
                return str(p.parent if p.name == "CUB_200_2011" else p)
        return f"{DATA_DIR}/cub"

    cand_names = {
        "cars": ["car", "cars", "stanford_cars", "scars"],
        "aircraft": ["aircraft", "fgvc_aircraft", "fgvc-aircraft-2013b"],
    }[norm]
    for sdir in (Path(DATA_DIR), Path(STORAGE_DIR)):
        if not sdir.is_dir():
            continue
        for name in cand_names:
            p = sdir / name
            if p.is_dir():
                return str(p)
    return f"{DATA_DIR}/{norm}"


def _resolve_splits_root() -> str:
    candidates = [
        f"{PROJECT_DIR}/data_uq_idxs",
        f"{STORAGE_DIR}/data_uq_idxs_bacon",
        f"{STORAGE_DIR}/data_uq_idxs",
    ]
    for c in candidates:
        p = Path(c)
        if (
            (p / "cub200_k100_imb10").is_dir()
            or (p / "cub200_k100_imb20").is_dir()
            or (p / "cars196_k98_imb20").is_dir()
        ):
            return str(p)
    return f"{PROJECT_DIR}/data_uq_idxs"


def _run_training(cmd: list[str], log_file_path: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    stop_commit = threading.Event()

    def _periodic_commit() -> None:
        while not stop_commit.wait(600):  # Commit every 10 minutes
            try:
                storage_volume.commit()
                print("[Volume] Periodic commit successful.", flush=True)
            except Exception as exc:
                print(f"[Volume] Periodic commit warning: {exc}", flush=True)

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{PROJECT_DIR}/src"

    try:
        with log_file_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            tail_lines = []
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                tail_lines.append(line)
                if len(tail_lines) > 200:
                    tail_lines.pop(0)

            return_code = process.wait()
            if return_code != 0:
                print("\n[Error] Training failed! Last 200 lines:\n", flush=True)
                print("".join(tail_lines), flush=True)
                raise subprocess.CalledProcessError(return_code, cmd)
    finally:
        stop_commit.set()
        try:
            storage_volume.commit()
            print("[Volume] Final commit successful.", flush=True)
        except Exception as exc:
            print(f"[Volume] Final commit warning: {exc}", flush=True)


@app.function(
    image=image,
    volumes={STORAGE_DIR: storage_volume},
    gpu="A10G",
    timeout=86400,  # 24 hours
)
def train_remote(
    baseline: str = "simgcd",
    dataset_name: str = "cub",
    imb_ratio: int = 20,
    config_name: str = "Base",
    enable_prototail: bool = False,
    config_file: Optional[str] = None,
    epochs: int = 200,
    batch_size: int = 128,
    lr: Optional[float] = None,
    seed: int = 0,
    eval_interval: int = 5,
    num_workers: int = 4,
    simgcd_eval_head: str = "kmeans",
    pseudo_mode: str = "paper",
    d2_min_cluster_size: int = 10,
    use_parts: bool = False,
    ablate_adaptive_capacity: bool = False,
    enable_pseudo_labeling: bool = False,
    enable_novel_pseudo: bool = False,
    fp16: bool = True,
    pseudo_warmup_epochs: Optional[int] = None,
    pseudo_confidence: Optional[float] = None,
    exp_name_suffix: str = "",
    run_name: Optional[str] = None,
    data_root_override: Optional[str] = None,
    splits_root_override: Optional[str] = None,
) -> None:
    storage_volume.reload()

    baseline_clean = baseline.lower().strip()
    if baseline_clean not in ("simgcd", "bacon"):
        raise ValueError(f"Unknown baseline '{baseline}'. Choose 'simgcd' or 'bacon'.")
    dataset_clean = _normalize_dataset_name(dataset_name)
    if pseudo_mode not in ("paper", "legacy_ce"):
        raise ValueError(f"Unknown pseudo_mode {pseudo_mode!r}; choose 'paper' or 'legacy_ce'")

    (
        cfg_clean,
        use_parts,
        ablate_adaptive_capacity,
        enable_pseudo_labeling,
        enable_novel_pseudo,
    ) = _resolve_preset_flags(
        config_name,
        use_parts=use_parts,
        ablate_adaptive_capacity=ablate_adaptive_capacity,
        enable_pseudo_labeling=enable_pseudo_labeling,
        enable_novel_pseudo=enable_novel_pseudo,
        enable_prototail=enable_prototail,
    )

    if config_file:
        config_path = Path(PROJECT_DIR) / config_file
    else:
        config_path = Path(PROJECT_DIR) / f"configs/experiment_{baseline_clean}.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        resolved_run_name = run_name
    elif exp_name_suffix:
        resolved_run_name = f"{exp_name_suffix}-{timestamp}"
    else:
        parts = [baseline_clean, dataset_clean, cfg_clean]
        if pseudo_mode != "paper":
            parts.append(pseudo_mode)
        if d2_min_cluster_size != 10:
            parts.append(f"min{d2_min_cluster_size}")
        parts.append(f"imb{imb_ratio}")
        parts.append(f"s{seed}")
        resolved_run_name = f"{'_'.join(parts)}-{timestamp}"

    exp_output_dir = Path(OUTPUT_DIR) / resolved_run_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_output_dir / "train.log"

    data_root = data_root_override or _resolve_data_root(dataset_clean)
    splits_root = splits_root_override or _resolve_splits_root()

    print("=== Starting ProtoTail Training Run ===", flush=True)
    print(f"Baseline:         {baseline_clean}", flush=True)
    print(f"Dataset / Ratio:  {dataset_clean} / imb={imb_ratio}", flush=True)
    print(f"Config Preset:    {cfg_clean}", flush=True)
    print(f"Pseudo Mode / D2: {pseudo_mode} / min_cluster_size={d2_min_cluster_size}", flush=True)
    print(f"Eval Head / Seed: {simgcd_eval_head} / seed={seed}", flush=True)
    print(f"Run name:         {resolved_run_name}", flush=True)
    print(f"Config YAML:      {config_path}", flush=True)
    print(f"Data root:        {data_root}", flush=True)
    print(f"Splits root:      {splits_root}", flush=True)
    print(f"Output directory: {exp_output_dir}", flush=True)
    print("=======================================", flush=True)

    cmd = [
        "python",
        f"{PROJECT_DIR}/src/prototail_gcd/train.py",
        "--config",
        str(config_path),
        "--dataset-name",
        dataset_clean,
        "--imb-ratio",
        str(imb_ratio),
        "--run-name",
        resolved_run_name,
        "--data-root",
        data_root,
        "--splits-root",
        splits_root,
        "--output-dir",
        str(exp_output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--eval-interval",
        str(eval_interval),
        "--num-workers",
        str(num_workers),
        "--simgcd-eval-head",
        simgcd_eval_head,
        "--pseudo-mode",
        pseudo_mode,
        "--d2-min-cluster-size",
        str(d2_min_cluster_size),
        "--fp16" if fp16 else "--no-fp16",
    ]

    if use_parts:
        cmd.append("--use-parts")
    elif enable_prototail:
        cmd.append("--enable-prototail")
    if ablate_adaptive_capacity:
        cmd.append("--ablate-adaptive-capacity")
    if enable_pseudo_labeling:
        cmd.append("--enable-pseudo-labeling")
    if enable_novel_pseudo:
        cmd.append("--enable-novel-pseudo")
    if pseudo_warmup_epochs is not None:
        cmd.extend(["--pseudo-warmup-epochs", str(pseudo_warmup_epochs)])
    if pseudo_confidence is not None:
        cmd.extend(["--pseudo-confidence", str(pseudo_confidence)])
    if lr is not None:
        cmd.extend(["--lr", str(lr)])

    _run_training(cmd, log_path)


@app.local_entrypoint()
def launch(
    baseline: str = "simgcd",
    dataset_name: str = "cub",
    imb_ratio: int = 20,
    config: str = "Base",
    enable_prototail: bool = False,
    config_file: Optional[str] = None,
    gpu: str = "a10g",
    epochs: int = 200,
    batch_size: int = 128,
    lr: Optional[float] = None,
    seed: int = 0,
    eval_interval: int = 5,
    num_workers: int = 4,
    simgcd_eval_head: str = "kmeans",
    pseudo_mode: str = "paper",
    d2_min_cluster_size: int = 10,
    use_parts: bool = False,
    ablate_adaptive_capacity: bool = False,
    enable_pseudo_labeling: bool = False,
    enable_novel_pseudo: bool = False,
    fp16: bool = True,
    pseudo_warmup_epochs: Optional[int] = None,
    pseudo_confidence: Optional[float] = None,
    exp_name_suffix: str = "",
    run_name: Optional[str] = None,
    data_root: Optional[str] = None,
    splits_root: Optional[str] = None,
) -> None:
    """CLI launcher for local users to submit runs to Modal."""
    gpu_spec = _normalize_gpu(gpu)
    cfg_clean, use_parts, ablate_adaptive_capacity, enable_pseudo_labeling, enable_novel_pseudo = (
        _resolve_preset_flags(
            config,
            use_parts=use_parts,
            ablate_adaptive_capacity=ablate_adaptive_capacity,
            enable_pseudo_labeling=enable_pseudo_labeling,
            enable_novel_pseudo=enable_novel_pseudo,
            enable_prototail=enable_prototail,
        )
    )
    dataset_clean = _normalize_dataset_name(dataset_name)
    print(
        f"Submitting {baseline}/{dataset_clean}/imb{imb_ratio}/{cfg_clean} "
        f"(pseudo_mode={pseudo_mode}, d2_min={d2_min_cluster_size}, seed={seed}) "
        f"to Modal on GPU: {gpu_spec}..."
    )
    train_remote.with_options(gpu=gpu_spec).remote(
        baseline=baseline,
        dataset_name=dataset_clean,
        imb_ratio=imb_ratio,
        config_name=cfg_clean,
        enable_prototail=enable_prototail,
        config_file=config_file,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        eval_interval=eval_interval,
        num_workers=num_workers,
        simgcd_eval_head=simgcd_eval_head,
        pseudo_mode=pseudo_mode,
        d2_min_cluster_size=d2_min_cluster_size,
        use_parts=use_parts,
        ablate_adaptive_capacity=ablate_adaptive_capacity,
        enable_pseudo_labeling=enable_pseudo_labeling,
        enable_novel_pseudo=enable_novel_pseudo,
        fp16=fp16,
        pseudo_warmup_epochs=pseudo_warmup_epochs,
        pseudo_confidence=pseudo_confidence,
        exp_name_suffix=exp_name_suffix,
        run_name=run_name,
        data_root_override=data_root,
        splits_root_override=splits_root,
    )
