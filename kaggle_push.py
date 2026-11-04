"""Kaggle Kernel Push Launcher.

Automatically packages prototail code as a self-contained bundle,
creates kernel-metadata.json, and pushes to Kaggle to trigger GPU T4
background training (Save & Run All).
You can shut down your PC immediately after running this script.
"""

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def _get_active_kaggle_username() -> str:
    """Read active Kaggle username from kaggle CLI / credentials."""
    try:
        import kaggle
        cfg = getattr(kaggle.api, "config_values", {}) or {}
        username = cfg.get("username")
        if username:
            return str(username)
    except Exception:
        pass
    for k in ("KAGGLE_USERNAME", "KAGGLE_USER_NAME", "KAGGLE_USER"):
        v = os.environ.get(k)
        if v:
            return v
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "username" in data:
                    return str(data["username"])
        except Exception:
            pass
    return "minhtmnguyntrn"


def _build_code_bundle_b64(repo_root: Path) -> str:
    """Zip kaggle_train.py, src/, and configs/ into an in-memory base64 payload."""
    buf = io.BytesIO()
    # Resolve prototail root (in case we are running from BaCon or prototail)
    prototail_candidates = [
        repo_root,
        Path(r"C:\Users\tam\Documents\GitHub\gcd-hosts\prototail"),
        repo_root.parent / "gcd-hosts" / "prototail",
        repo_root.parent / "prototail",
    ]
    p_root = None
    for c in prototail_candidates:
        if (c / "src/prototail_gcd/train.py").is_file():
            p_root = c
            break
    if p_root is None:
        raise FileNotFoundError("Could not find prototail source root directory")

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. kaggle_train.py
        z.write(p_root / "kaggle_train.py", "kaggle_train.py")
        # 2. src/
        for root, dirs, files in os.walk(p_root / "src"):
            if "__pycache__" in root:
                continue
            for f in files:
                p = Path(root) / f
                z.write(p, p.relative_to(p_root))
        # 3. configs/
        for root, dirs, files in os.walk(p_root / "configs"):
            if "__pycache__" in root:
                continue
            for f in files:
                p = Path(root) / f
                z.write(p, p.relative_to(p_root))

        # 4. data_uq_idxs/ (precomputed long-tail splits for CUB, Cars, Aircraft, etc.)
        splits_candidates = [
            p_root / "data_uq_idxs",
            repo_root / "data_uq_idxs",
            Path(r"C:\Users\tam\Documents\GitHub\gcd-hosts\prototail\data_uq_idxs"),
            Path(r"C:\Users\tam\Documents\GitHub\BaCon\data_uq_idxs"),
        ]
        s_dir = None
        for cand in splits_candidates:
            if cand.is_dir():
                s_dir = cand
                break
        if s_dir is not None:
            for root, dirs, files in os.walk(s_dir):
                if "__pycache__" in root:
                    continue
                for f in files:
                    p = Path(root) / f
                    arcname = "data_uq_idxs/" + p.relative_to(s_dir).as_posix()
                    z.write(p, arcname)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Push ProtoTail training job to Kaggle GPU")
    parser.add_argument("--baseline", choices=["simgcd", "bacon"], default="simgcd")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", choices=["cub", "cub200", "cars", "car", "scars", "aircraft", "air", "fgvc_aircraft"], default="cub")
    parser.add_argument("--imb-ratio", "--imb_ratio", dest="imb_ratio", type=int, choices=[10, 20, 50, 100], default=20)
    parser.add_argument(
        "--config",
        choices=["Base", "Pseudo", "Prototype", "Prototype_Tailaware", "Full_wo_Tail", "Full"],
        default="Base",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", "--eval_interval", dest="eval_interval", type=int, default=5)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--kaggle-user", "--kaggle_user", dest="kaggle_user", type=str, default=None)
    parser.add_argument("--dataset-slug", "--dataset_slug", dest="dataset_slug", type=str, default="nguyntrnminhtm/gcd-data")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Build kernel locally without pushing")
    args = parser.parse_args()
    name_map = {
        "car": "cars",
        "scars": "cars",
        "air": "aircraft",
        "fgvc_aircraft": "aircraft",
        "cub200": "cub",
    }
    args.dataset_name = name_map.get(args.dataset_name.lower(), args.dataset_name.lower())

    repo_root = Path(__file__).resolve().parent
    kaggle_user = args.kaggle_user or _get_active_kaggle_username()

    # Determine flags based on config
    flags = []
    if args.config == "Base":
        pass
    elif args.config == "Pseudo":
        flags.extend(["--enable-pseudo-labeling", "--enable-novel-pseudo"])
    elif args.config == "Prototype":
        flags.extend(["--use-parts", "--ablate-adaptive-capacity"])
    elif args.config == "Prototype_Tailaware":
        flags.append("--use-parts")
    elif args.config == "Full_wo_Tail":
        flags.extend(["--use-parts", "--ablate-adaptive-capacity", "--enable-pseudo-labeling", "--enable-novel-pseudo"])
    elif args.config == "Full":
        flags.extend(["--use-parts", "--enable-pseudo-labeling", "--enable-novel-pseudo"])

    exp_name = f"{args.baseline}_{args.dataset_name}_{args.config}_imb{args.imb_ratio}"
    kernel_slug = f"prototail-{args.baseline}-{args.dataset_name}-{args.config.lower()}-imb{args.imb_ratio}".replace("_", "-")

    # Staging folder for kernel push
    staging_dir = repo_root / "build_kaggle_kernel"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Build self-contained code bundle
    print("Packaging ProtoTail codebase into self-contained bundle...")
    b64_bundle = _build_code_bundle_b64(repo_root)

    # Generate self-unpacking runner script for Kaggle
    flag_str = " ".join(flags)
    run_script_content = f'''# Self-contained ProtoTail runner for Kaggle GPU
import base64
import io
import os
import subprocess
import sys
import zipfile

# 1. Unpack code bundle directly into current working directory
BUNDLE_B64 = """{b64_bundle}"""
target_dir = os.getcwd()
print(f"[Kaggle Bootstrap] Unpacking ProtoTail code into {{target_dir}}...")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64.encode("ascii")))) as z:
    z.extractall(target_dir)

print("[Kaggle Bootstrap] Code unpacked successfully. Available files:")
for item in os.listdir(target_dir):
    print(" -", item)
sys.stdout.flush()

# 2. Execute training
cmd = [
    sys.executable,
    os.path.join(target_dir, "kaggle_train.py"),
    "--baseline", "{args.baseline}",
    "--dataset-slug", "{args.dataset_slug}",
    "--dataset-name", "{args.dataset_name}",
    "--imb-ratio", "{args.imb_ratio}",
    "--epochs", "{args.epochs}",
    "--eval-interval", "{args.eval_interval}",
    "--batch-size", "{args.batch_size}",
    "--run-name", "{exp_name}",
]
extra_flags = {flags}
cmd.extend(extra_flags)

print("\\n[Kaggle Runner] Launching command:\\n" + " ".join(cmd) + "\\n")
sys.stdout.flush()

process = subprocess.run(cmd)
sys.exit(process.returncode)
'''
    with open(staging_dir / "run_script.py", "w", encoding="utf-8") as f:
        f.write(run_script_content)

    # Create kernel-metadata.json
    metadata = {
        "id": f"{kaggle_user}/{kernel_slug}",
        "title": f"ProtoTail {exp_name}",
        "code_file": "run_script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [args.dataset_slug],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 65)
    print("        PROTOTAIL KAGGLE CLI PUSH LAUNCHER")
    print("=" * 65)
    print(f"Active Account:  {kaggle_user}")
    print(f"Kernel ID:       {kaggle_user}/{kernel_slug}")
    print(f"Configuration:   {exp_name}")
    print(f"Baseline:        {args.baseline.upper()}")
    print(f"Dataset:         {args.dataset_name.upper()} (imb={args.imb_ratio})")
    print(f"Config Ablation: {args.config}")
    print(f"Flags:           {flag_str or '(none - pure baseline)'}")
    print(f"GPU:             NVIDIA T4 (enable_gpu: true)")
    print(f"Code Payload:    {len(b64_bundle)} chars (~{len(b64_bundle)//1024} KB self-extracting)")
    print("=" * 65)
    if args.dry_run:
        print("\n[DRY RUN] Kernel packaged successfully at:", staging_dir)
        return
    print("\nPushing kernel to Kaggle...")
    sys.stdout.flush()

    cmd_push = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    res = subprocess.run(cmd_push)

    if res.returncode == 0:
        print("\n" + "=" * 65)
        print("  >>> THÀNH CÔNG! Kernel đã được đẩy lên Kaggle <<<")
        print(f"  Mô hình đang được train ngầm trên GPU T4 của account: {kaggle_user}")
        print(f"  Kiểm tra trạng thái: kaggle kernels status {kaggle_user}/{kernel_slug}")
        print("=" * 65)
    else:
        print(f"\n[Error] Push thất bại (code {res.returncode}). Kiểm tra lại kết nối hoặc account.")


if __name__ == "__main__":
    main()
