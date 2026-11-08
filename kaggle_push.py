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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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
    parser.add_argument("--kmeans-backend", choices=["sklearn_cpu", "faiss_gpu"], default="faiss_gpu")
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--kaggle-user", "--kaggle_user", dest="kaggle_user", type=str, default=None)
    parser.add_argument("--dataset-slug", "--dataset_slug", dest="dataset_slug", type=str, default="nguyntrnminhtm/gcd-data")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="NvidiaTeslaT4",
        help="Hardware accelerator to request on Kaggle ('NvidiaTeslaT4' allocates Dual NVIDIA T4 GPUs)",
    )
    parser.add_argument(
        "--eval-feature-mode",
        choices=["legacy_concat", "balanced_concat"],
        default="legacy_concat",
        help="Evaluation feature concatenation mode ('legacy_concat' or 'balanced_concat')",
    )
    parser.add_argument(
        "--gate-warm-start-from-support",
        action="store_true",
        default=False,
        help="Init gates from estimated training supports at epoch 0 (LT-02 fix)",
    )
    parser.add_argument(
        "--known-frequency-mapping",
        choices=["direct_hungarian", "bacon_rank"],
        default="direct_hungarian",
        help="Known frequency mapping mode ('direct_hungarian' or 'bacon_rank')",
    )
    parser.add_argument(
        "--pseudo-mode", choices=["paper", "legacy_ce"], default="paper",
        help="Paper persistent D1/D2 mode is default for Full and Pseudo",
    )
    parser.add_argument(
        "--novel-frequency-mapping",
        choices=["auto", "bacon_rank", "output_aligned"],
        default="auto",
    )
    parser.add_argument(
        "--novel-prototype-end-epoch",
        type=int,
        default=180,
        help="Epoch cutoff for updating novel prototypes (default: 180)",
    )
    parser.add_argument(
        "--anti-collapse-novel-selector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Hungarian bipartite 1-to-1 matching for D2 novel clusters",
    )
    parser.add_argument(
        "--adaptive-pseudo-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable class-adaptive confidence thresholding for long-tail self-training",
    )
    parser.add_argument(
        "--pseudo-confidence-min",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for tail classes under adaptive thresholding",
    )
    parser.add_argument(
        "--require-paper-dual-selection",
        action="store_true",
        default=False,
        help="Fail unless the persistent paper D1/D2 selector is integrated",
    )
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

    if args.eval_feature_mode != "legacy_concat":
        flags.extend(["--eval-feature-mode", args.eval_feature_mode])
    if args.gate_warm_start_from_support:
        flags.append("--gate-warm-start-from-support")
    if args.known_frequency_mapping != "direct_hungarian":
        flags.extend(["--known-frequency-mapping", args.known_frequency_mapping])
    if args.novel_frequency_mapping != "auto":
        flags.extend(["--novel-frequency-mapping", args.novel_frequency_mapping])
    if args.require_paper_dual_selection:
        flags.append("--require-paper-dual-selection")
    if getattr(args, "novel_prototype_end_epoch", None) is not None:
        flags.extend(["--novel-prototype-end-epoch", str(args.novel_prototype_end_epoch)])
    if getattr(args, "anti_collapse_novel_selector", True):
        flags.append("--anti-collapse-novel-selector")
    else:
        flags.append("--no-anti-collapse-novel-selector")
    if getattr(args, "adaptive_pseudo_threshold", True):
        flags.append("--adaptive-pseudo-threshold")
    else:
        flags.append("--no-adaptive-pseudo-threshold")
    if getattr(args, "pseudo_confidence_min", None) is not None:
        flags.extend(["--pseudo-confidence-min", str(args.pseudo_confidence_min)])
    if args.config in ("Full", "Full_wo_Tail", "Pseudo"):
        flags.extend(["--pseudo-mode", args.pseudo_mode])

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
    "--kmeans-backend", "{args.kmeans_backend}",
    "--kmeans-n-init", "{args.kmeans_n_init}",
    "--kmeans-max-iter", "{args.kmeans_max_iter}",
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
        "accelerator": args.accelerator,
        "machine_shape": args.accelerator,
        "dataset_sources": [args.dataset_slug],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    acc_desc = "2x NVIDIA Tesla T4 (Dual GPU)" if args.accelerator == "NvidiaTeslaT4" else args.accelerator
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
    print(f"Hardware Acc:    {acc_desc} (--accelerator {args.accelerator})")
    print(f"Code Payload:    {len(b64_bundle)} chars (~{len(b64_bundle)//1024} KB self-extracting)")
    print("=" * 65)
    if args.dry_run:
        print("\n[DRY RUN] Kernel packaged successfully at:", staging_dir)
        return
    print("\nPushing kernel to Kaggle...")
    sys.stdout.flush()

    cmd_push = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(staging_dir),
        "--accelerator",
        args.accelerator,
    ]
    res = subprocess.run(cmd_push, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout, end="")
    if res.stderr:
        print(res.stderr, end="")
    sys.stdout.flush()

    stdout_str = res.stdout or ""
    is_success = (
        res.returncode == 0
        and "successfully pushed" in stdout_str.lower()
        and "kernel push error" not in stdout_str.lower()
    )

    if is_success:
        print("\n" + "=" * 65)
        print("  >>> THÀNH CÔNG! Kernel đã được đẩy lên Kaggle <<<")
        print(f"  Mô hình đang được train trên {acc_desc} của account: {kaggle_user}")
        print(f"  Kiểm tra trạng thái: kaggle kernels status {kaggle_user}/{kernel_slug}")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print("  [ERROR] Đẩy Kernel lên Kaggle THẤT BÀI!")
        if "maximum batch gpu session count" in stdout_str.lower():
            print("  LÝ DO: Đã đạt giới hạn số lượng GPU session chạy đồng thời (tối đa 2 GPU sessions trên Kaggle).")
            print("  Vì bạn đang dùng 2x T4 (Dual GPU), 1 kernel đã chiếm trọn cả 2 GPU slots của account.")
            print("  Cách khắc phục:")
            print("    1. Kiểm tra kernel đang chạy: kaggle kernels status <user>/<running-kernel>")
            print("    2. Đợi kernel đó chạy xong, hoặc hủy trên Web Kaggle (https://www.kaggle.com)")
            print("    3. Sau khi GPU được giải phóng, hãy chạy lại lệnh push này.")
        print("=" * 65)


if __name__ == "__main__":
    main()
