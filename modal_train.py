"""Modal launcher for ProtoTail GCD experiments (SimGCD and BaCon baselines).

Designed for seamless execution on Modal with persistent storage volume ('bacon-storage'),
supporting dynamic GPU selection, periodic volume commits, and full experiment logging.

Usage examples:
  # 1. Run SimGCD baseline (200 epochs on A10G)
  modal run modal_train.py::launch --baseline simgcd

  # 2. Run SimGCD + ProtoTail
  modal run modal_train.py::launch --baseline simgcd --enable-prototail

  # 3. Run BaCon baseline
  modal run modal_train.py::launch --baseline bacon

  # 4. Run BaCon + ProtoTail
  modal run modal_train.py::launch --baseline bacon --enable-prototail

  # 5. Run with custom GPU, epochs, or detached mode
  modal run -d modal_train.py::launch --baseline bacon --enable-prototail --gpu a10g --epochs 100
"""

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
        ignore=[
            "outputs",
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


def _resolve_cub_root() -> str:
    candidates = [
        f"{DATA_DIR}/cub",
        f"{DATA_DIR}/cub/CUB_200_2011",
        f"{STORAGE_DIR}/CUB_200_2011",
    ]
    for c in candidates:
        p = Path(c)
        if (p / "CUB_200_2011" / "images.txt").is_file() or (p / "images.txt").is_file():
            # If path directly points to CUB_200_2011, return its parent because CUBLongTailAdapter joins CUB_200_2011
            if p.name == "CUB_200_2011":
                return str(p.parent)
            return str(p)
    return f"{DATA_DIR}/cub"


def _resolve_splits_root() -> str:
    candidates = [
        f"{PROJECT_DIR}/data_uq_idxs",
        f"{STORAGE_DIR}/data_uq_idxs_bacon",
        f"{STORAGE_DIR}/data_uq_idxs",
    ]
    for c in candidates:
        p = Path(c)
        if (p / "cub200_k100_imb10").is_dir():
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


def _get_gpu_spec(gpu_name: str) -> modal.gpu.GPU:
    normalized = gpu_name.lower().strip()
    if normalized in ("a10g", "a10"):
        return modal.gpu.A10G()
    elif normalized in ("a100", "a100-40gb", "a100-80gb"):
        return modal.gpu.A100()
    elif normalized in ("l4",):
        return modal.gpu.L4()
    elif normalized in ("t4",):
        return modal.gpu.T4()
    return modal.gpu.A10G()


@app.function(
    image=image,
    volumes={STORAGE_DIR: storage_volume},
    gpu="A10G",
    timeout=86400,  # 24 hours
)
def train_remote(
    baseline: str,
    enable_prototail: bool = False,
    config_file: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
    exp_name_suffix: str = "",
    data_root_override: Optional[str] = None,
    splits_root_override: Optional[str] = None,
) -> None:
    storage_volume.reload()

    baseline_clean = baseline.lower().strip()
    if baseline_clean not in ("simgcd", "bacon"):
        raise ValueError(f"Unknown baseline '{baseline}'. Choose 'simgcd' or 'bacon'.")

    # Select config file
    if config_file:
        config_path = Path(PROJECT_DIR) / config_file
    else:
        config_path = Path(PROJECT_DIR) / f"configs/experiment_{baseline_clean}.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = "prototail" if enable_prototail else "baseline"
    suffix = f"-{exp_name_suffix}" if exp_name_suffix else ""
    run_name = f"{baseline_clean}-{method_tag}{suffix}-{timestamp}"

    exp_output_dir = Path(OUTPUT_DIR) / run_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_output_dir / "train.log"

    data_root = data_root_override or _resolve_cub_root()
    splits_root = splits_root_override or _resolve_splits_root()

    print(f"=== Starting ProtoTail Training Run ===")
    print(f"Baseline:         {baseline_clean}")
    print(f"Method:           {'ProtoTail (Enabled)' if enable_prototail else 'Baseline (Pure)'}")
    print(f"Run name:         {run_name}")
    print(f"Config:           {config_path}")
    print(f"Data root:        {data_root}")
    print(f"Splits root:      {splits_root}")
    print(f"Output directory: {exp_output_dir}")
    print(f"=======================================")

    cmd = [
        "python",
        f"{PROJECT_DIR}/src/prototail_gcd/train.py",
        "--config",
        str(config_path),
        "--run-name",
        run_name,
        "--data-root",
        data_root,
        "--splits-root",
        splits_root,
        "--output-dir",
        str(exp_output_dir),
    ]

    if enable_prototail:
        cmd.append("--enable-prototail")
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    _run_training(cmd, log_path)


@app.local_entrypoint()
def launch(
    baseline: str = "simgcd",
    enable_prototail: bool = False,
    config: Optional[str] = None,
    gpu: str = "a10g",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
    exp_name_suffix: str = "",
    data_root: Optional[str] = None,
    splits_root: Optional[str] = None,
) -> None:
    """CLI launcher for local users to submit runs to Modal."""
    print(f"Submitting job to Modal on GPU: {gpu.upper()}...")
    train_remote.remote(
        baseline=baseline,
        enable_prototail=enable_prototail,
        config_file=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        exp_name_suffix=exp_name_suffix,
        data_root_override=data_root,
        splits_root_override=splits_root,
    )
