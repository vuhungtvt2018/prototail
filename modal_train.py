"""Modal launcher for ProtoTail GCD experiments (SimGCD and BaCon baselines).

Designed for seamless execution on Modal with persistent storage volume ('bacon-storage'),
supporting dynamic GPU selection, periodic volume commits, and full experiment logging.

Usage examples:
  # 1. Run SimGCD baseline (200 epochs on A10G)
  modal run modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Base

  # 2. Run SimGCD Full (Warm-start + CPU KMeans)
  modal run modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full

  # 3. Run BaCon Full
  modal run modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 20 --config Full

  # 4. Run with detached mode
  modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Full
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
        "kagglehub",
    )
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path=PROJECT_DIR,
        copy=True,
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


def _resolve_data_root(dataset_name: str) -> str:
    norm = dataset_name.lower().strip()
    cand_names = {
        "cub": ["cub", "CUB_200_2011", "cub200"],
        "cars": ["car", "cars", "stanford_cars", "scars"],
        "car": ["car", "cars", "stanford_cars", "scars"],
        "aircraft": ["aircraft", "fgvc_aircraft", "fgvc-aircraft-2013b"],
        "air": ["aircraft", "fgvc_aircraft", "fgvc-aircraft-2013b"],
        "herbarium": ["Herbarium 19", "herbarium", "herbarium19", "herb19"],
        "herbarium19": ["Herbarium 19", "herbarium", "herbarium19", "herb19"],
        "herb19": ["Herbarium 19", "herbarium", "herbarium19", "herb19"],
        "inat2018": ["iNaturalist 2018", "inat2018", "inat", "inaturalist"],
        "inat": ["iNaturalist 2018", "inat2018", "inat", "inaturalist"],
        "inaturalist": ["iNaturalist 2018", "inat2018", "inat", "inaturalist"],
    }.get(norm, [norm])

    search_dirs = [Path(DATA_DIR), Path(STORAGE_DIR)]
    for sdir in search_dirs:
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
        if (p / "cub200_k100_imb10").is_dir() or (p / "cub200_k100_imb20").is_dir():
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
    imb_ratio: Optional[str] = None,
    config_name: str = "Base",
    epochs: int = 200,
    batch_size: int = 128,
    lr: Optional[float] = None,
    seed: int = 0,
    use_parts: bool = False,
    ablate_adaptive_capacity: bool = False,
    enable_pseudo_labeling: bool = False,
    enable_novel_pseudo: bool = False,
    gate_warm_start_from_support: bool = False,
    kmeans_backend: str = "sklearn_cpu",
    exp_name_suffix: str = "",
    data_root_override: Optional[str] = None,
    splits_root_override: Optional[str] = None,
) -> None:
    storage_volume.reload()

    baseline_clean = baseline.lower().strip()
    is_natural = dataset_name.lower() in ("inat", "inat2018", "inaturalist", "inaturalist2018") and (imb_ratio is None or str(imb_ratio).lower() in ("natural", "none", "0", ""))
    if is_natural:
        run_name = f"{baseline_clean}_{dataset_name}_{config_name}"
    else:
        actual_imb = imb_ratio if imb_ratio is not None else "20"
        run_name = f"{baseline_clean}_{dataset_name}_{config_name}_imb{actual_imb}"
    if exp_name_suffix:
        run_name = exp_name_suffix

    exp_output_dir = Path(OUTPUT_DIR) / run_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_output_dir / "train.log"

    data_root = data_root_override or _resolve_data_root(dataset_name)
    splits_root = splits_root_override or _resolve_splits_root()

    print(f"=======================================")
    print(f"       PROTOTAIL MODAL GPU RUNNER       ")
    print(f"=======================================")
    print(f"Baseline:         {baseline_clean.upper()}")
    print(f"Dataset:          {dataset_name} (imb={imb_ratio})")
    print(f"Config:           {config_name}")
    print(f"KMeans Backend:   {kmeans_backend}")
    print(f"Run name:         {run_name}")
    print(f"Data root:        {data_root}")
    print(f"Splits root:      {splits_root}")
    print(f"Output directory: {exp_output_dir}")
    print(f"=======================================")

    cmd = [
        "python",
        f"{PROJECT_DIR}/train.py",
        "--baseline", baseline_clean,
        "--dataset-name", dataset_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--kmeans-backend", kmeans_backend,
        "--run-name", run_name,
        "--data-root", data_root,
        "--splits-root", splits_root,
        "--output-dir", str(exp_output_dir),
    ]

    if not is_natural and imb_ratio is not None:
        cmd.extend(["--imb-ratio", str(imb_ratio)])
    if lr is not None:
        cmd.extend(["--lr", str(lr)])

    if use_parts:
        cmd.append("--use-parts")
    if ablate_adaptive_capacity:
        cmd.append("--ablate-adaptive-capacity")
    if enable_pseudo_labeling:
        cmd.append("--enable-pseudo-labeling")
    if enable_novel_pseudo:
        cmd.append("--enable-novel-pseudo")
    is_tailaware = use_parts and not ablate_adaptive_capacity
    if gate_warm_start_from_support or is_tailaware:
        cmd.append("--gate-warm-start-from-support")

    _run_training(cmd, log_path)


@app.local_entrypoint()
def launch(
    baseline: str = "simgcd",
    dataset_name: str = "cub",
    imb_ratio: Optional[str] = None,
    config: str = "Base",
    epochs: int = 200,
    batch_size: int = 128,
    lr: Optional[float] = None,
    seed: int = 0,
    gpu: str = "a10g",
    use_parts: bool = False,
    ablate_adaptive_capacity: bool = False,
    enable_pseudo_labeling: bool = False,
    enable_novel_pseudo: bool = False,
    gate_warm_start_from_support: bool = False,
    kmeans_backend: str = "sklearn_cpu",
    exp_name_suffix: str = "",
    data_root: Optional[str] = None,
    splits_root: Optional[str] = None,
    backbone: str = "dinov2-vitb14",  # Backward compatibility
) -> None:
    """CLI launcher for local users to submit runs to Modal."""
    # Resolve config preset flags
    cfg_clean = config.strip()
    if cfg_clean == "Pseudo":
        enable_pseudo_labeling = True
        enable_novel_pseudo = True
    elif cfg_clean == "Prototype":
        use_parts = True
        ablate_adaptive_capacity = True
    elif cfg_clean == "Prototype_Tailaware":
        use_parts = True
        gate_warm_start_from_support = True
    elif cfg_clean == "Full_wo_Tail":
        use_parts = True
        ablate_adaptive_capacity = True
        enable_pseudo_labeling = True
        enable_novel_pseudo = True
    elif cfg_clean == "Full":
        use_parts = True
        enable_pseudo_labeling = True
        enable_novel_pseudo = True
        gate_warm_start_from_support = True

    print(f"Submitting job to Modal on GPU: {gpu.upper()}...")
    train_remote.remote(
        baseline=baseline,
        dataset_name=dataset_name,
        imb_ratio=str(imb_ratio),
        config_name=cfg_clean,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        use_parts=use_parts,
        ablate_adaptive_capacity=ablate_adaptive_capacity,
        enable_pseudo_labeling=enable_pseudo_labeling,
        enable_novel_pseudo=enable_novel_pseudo,
        gate_warm_start_from_support=gate_warm_start_from_support,
        kmeans_backend=kmeans_backend,
        exp_name_suffix=exp_name_suffix,
        data_root_override=data_root,
        splits_root_override=splits_root,
    )
