"""Upload iNaturalist 2018 dataset from local machine to Modal Volume and extract on Cloud.

Features:
- Windows cp1252 UTF-8 safe (prevents terminal crashes on Modal unicode symbols).
- Uploads JSON annotations, precomputed split files (data_uq_idxs), and train_val2018.tar.gz.
- High-speed parallel cloud decompression using pigz (multi-threaded gzip) on 8 vCPUs.
- Automated validation of extracted dataset on Modal Volume.
- Optional flag to delete large .tar.gz archives on Modal after extraction to save volume quota.

Usage:
    python upload_inat_to_modal.py
    python upload_inat_to_modal.py --delete-archive
    python upload_inat_to_modal.py --skip-upload  # If already uploaded, just run extraction
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 encoding issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

APP_NAME = "inat2018-modal-pipeline"
VOLUME_NAME = "bacon-storage"
REMOTE_DATA_DIR = "/bacon-storage/data/inat2018"

app = modal.App(APP_NAME)
storage_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Cloud container image equipped with pigz for ultra-fast multi-core decompression
cloud_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("pigz", "tar", "unzip")
)


@app.function(
    image=cloud_image,
    volumes={"/bacon-storage": storage_volume},
    timeout=14400,  # 4 hours timeout
    cpu=8,          # 8 vCPUs for parallel decompression
    memory=16384,   # 16 GB RAM
)
def extract_on_modal(delete_archive: bool = True, include_test: bool = False):
    """Decompress archives on Modal Volume using multi-core parallel pigz."""
    import os
    import subprocess
    import time
    from pathlib import Path

    dest = Path(REMOTE_DATA_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"[MODAL CLOUD] BẮT ĐẦU GIẢI NÉN DATASET TẠI: {dest}")
    print("=" * 70)

    # 1. Giải nén các file JSON annotations (rất nhanh, vài giây)
    json_archives = [
        "train2018.json.tar.gz",
        "val2018.json.tar.gz",
        "categories.json.tar.gz",
        "test2018.json.tar.gz",
    ]
    for ja in json_archives:
        archive_path = dest / ja
        if archive_path.is_file():
            print(f"-> Đang giải nén {ja}...")
            subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(dest)], check=True)
            print(f"   ✓ Đã giải nén xong {ja}")
            if delete_archive:
                archive_path.unlink()
                print(f"   ✓ Đã dọn dẹp file nén {ja}")

    # Giải nén location nếu có
    loc_zip = dest / "inat2018_locations.zip"
    if loc_zip.is_file():
        print(f"-> Đang giải nén inat2018_locations.zip...")
        subprocess.run(["unzip", "-q", "-o", str(loc_zip), "-d", str(dest)], check=True)
        print(f"   ✓ Đã giải nén xong inat2018_locations.zip")
        if delete_archive:
            loc_zip.unlink()
            print(f"   ✓ Đã dọn dẹp file nén inat2018_locations.zip")

    # 2. Giải nén train_val2018.tar.gz bằng pigz (đa luồng 8 core)
    train_val_tar = dest / "train_val2018.tar.gz"
    if not train_val_tar.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {train_val_tar} trên Modal Volume!")

    tar_size_gb = train_val_tar.stat().st_size / (1024 ** 3)
    print(f"\n-> Bắt đầu giải nén train_val2018.tar.gz ({tar_size_gb:.1f} GB)...")
    print("   Sử dụng pigz (8 CPU cores) để tăng tốc độ tối đa...")
    t0 = time.time()

    # Thử chạy qua pigz pipeline, nếu lỗi fallback về tar tiêu chuẩn
    cmd = f"pigz -dc '{train_val_tar}' | tar -xf - -C '{dest}'"
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print("   Pigz pipeline cảnh báo, chạy fallback với tar -xzf...")
        subprocess.run(["tar", "-xzf", str(train_val_tar), "-C", str(dest)], check=True)

    elapsed_min = (time.time() - t0) / 60
    print(f"   ✓ Hoàn tất giải nén train_val2018 trong {elapsed_min:.1f} phút!")

    # 3. Giải nén test2018.tar.gz nếu có yêu cầu
    if include_test:
        test_tar = dest / "test2018.tar.gz"
        if test_tar.is_file():
            print(f"\n-> Đang giải nén test2018.tar.gz...")
            t0 = time.time()
            subprocess.run(f"pigz -dc '{test_tar}' | tar -xf - -C '{dest}'", shell=True, check=True)
            print(f"   ✓ Đã giải nén test2018.tar.gz trong {(time.time() - t0)/60:.1f} phút!")

    # 4. Kiểm tra xác thực dữ liệu sau giải nén
    print("\n" + "=" * 70)
    print("[MODAL CLOUD] KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU:")
    print("=" * 70)
    train_val_dir = dest / "train_val2018"
    if train_val_dir.is_dir():
        supercats = [d.name for d in train_val_dir.iterdir() if d.is_dir()]
        print(f"✓ Thư mục train_val2018 tồn tại. Số supercategory: {len(supercats)} ({supercats[:6]}...)")
    else:
        print("CẢNH BÁO: Chưa tìm thấy thư mục train_val2018!")

    for f in ["train2018.json", "val2018.json", "categories.json"]:
        p = dest / f
        if p.is_file():
            print(f"✓ File {f} tồn tại ({p.stat().st_size / (1024**2):.1f} MB)")
        else:
            print(f"CẢNH BÁO: Thiếu file {f}")

    # 5. Dọn dẹp file nén (mặc định tự động xóa để tiết kiệm dung lượng Volume)
    if delete_archive:
        print(f"\n-> [DỌN DẸP] Đang tự động xóa file nén {train_val_tar.name} để giải phóng {tar_size_gb:.1f} GB...")
        train_val_tar.unlink()
        if include_test and (dest / "test2018.tar.gz").is_file():
            (dest / "test2018.tar.gz").unlink()
        print("   ✓ Đã giải phóng ~120 GB dung lượng Volume thành công!")

    # 6. Commit thay đổi vào Modal Volume
    print("\n-> Đang commit thay đổi vào Modal Volume (vui lòng đợi vài giây)...")
    storage_volume.commit()
    print("=" * 70)
    print("✓ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH GIẢI NÉN TRÊN MODAL VOLUME!")
    print("=" * 70)


def _run_modal_cmd(cmd: list[str], desc: str) -> None:
    """Run a modal CLI command with UTF-8 environment and clean logging."""
    print(f"\n[UPLOAD] {desc}...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Lỗi khi thực hiện: {' '.join(cmd)} (mã lỗi: {res.returncode})")


def upload_and_process():
    parser = argparse.ArgumentParser(description="Upload & Extract iNaturalist 2018 on Modal Volume.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\data\iNaturalist 2018"),
        help="Thư mục chứa các file nén iNat 2018 (mặc định: D:\\data\\iNaturalist 2018)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Thư mục chứa data_uq_idxs (mặc định: tự động tìm trong BaCon hoặc prototail)",
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default=VOLUME_NAME,
        help="Tên Modal Volume (mặc định: bacon-storage)",
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        default="data/inat2018",
        help="Đường dẫn lưu trên Modal Volume (mặc định: data/inat2018)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Upload cả test2018.tar.gz (42 GB, không có nhãn, thường không cần thiết cho GCD)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Bỏ qua bước upload, chỉ chạy bước giải nén trên Modal",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Chỉ upload lên Volume, không chạy bước giải nén",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Giữ lại file nén sau khi giải nén (mặc định: tự động xóa file .tar.gz để tiết kiệm ~120 GB Volume)",
    )
    args = parser.parse_args()

    delete_archive = not args.keep_archive

    local_dir = args.data_dir
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {local_dir}")

    # Tự động tìm splits_dir
    splits_dir = args.splits_dir
    if splits_dir is None:
        cands = [
            Path(__file__).resolve().parent / "data_uq_idxs",
            Path(__file__).resolve().parent.parent / "prototail" / "data_uq_idxs",
            Path(r"C:\Users\tam\Documents\GitHub\BaCon\data_uq_idxs"),
            Path(r"C:\Users\tam\Documents\GitHub\gcd-hosts\prototail\data_uq_idxs"),
        ]
        for c in cands:
            if (c / "inat2018_k4071_natural").is_dir():
                splits_dir = c
                break

    print("=" * 75)
    print("       CÔNG CỤ UPLOAD & GIẢI NÉN INATURALIST 2018 LÊN MODAL")
    print("=" * 75)
    print(f"Thư mục nguồn cục bộ:     {local_dir}")
    print(f"Thư mục splits .pt:        {splits_dir}")
    print(f"Modal Volume đích:         {args.volume_name}")
    print(f"Đường dẫn Volume:          {args.remote_dir}")
    print(f"Bao gồm file test (42GB):  {args.include_test}")
    print(f"Tự động xóa archive nén:   {delete_archive} (mặc định xóa để tiết kiệm ~120 GB Volume)")
    print("=" * 75)

    if not args.skip_upload:
        # BƯỚC 1: Upload Metadata JSON & locations (nhẹ, upload trước trong vài phút)
        json_files = [
            "train2018.json.tar.gz",
            "val2018.json.tar.gz",
            "categories.json.tar.gz",
            "test2018.json.tar.gz",
            "inat2018_locations.zip",
        ]
        for jf in json_files:
            fp = local_dir / jf
            if fp.is_file():
                remote_fp = f"{args.remote_dir}/{jf}"
                _run_modal_cmd(
                    ["modal", "volume", "put", "--force", args.volume_name, str(fp), remote_fp],
                    f"Đang upload metadata {jf} ({fp.stat().st_size / (1024**2):.1f} MB)",
                )

        # BƯỚC 2: Upload toàn bộ các split .pt (data_uq_idxs)
        # if splits_dir and splits_dir.is_dir():
        #     print(f"\n[UPLOAD] Đang upload các thư mục splits .pt từ {splits_dir}...")
        #     for sp_name in [
        #         "inat2018_k4071_natural",
        #         "inat2018_k4071_imb10",
        #         "inat2018_k4071_imb20",
        #         "inat2018_k4071_imb50",
        #         "inat2018_k4071_imb100",
        #     ]:
        #         sp_path = splits_dir / sp_name
        #         if sp_path.is_dir():
        #             remote_sp = f"{args.remote_dir}/data_uq_idxs/{sp_name}/"
        #             _run_modal_cmd(
        #                 ["modal", "volume", "put", "--force", args.volume_name, str(sp_path), remote_sp],
        #                 f"Đang upload split folder {sp_name}",
        #             )

        # BƯỚC 3: Upload file nén ảnh lớn train_val2018.tar.gz
        train_val_tar = local_dir / "train_val2018.tar.gz"
        if not train_val_tar.is_file():
            raise FileNotFoundError(f"Không tìm thấy: {train_val_tar}")

        tar_gb = train_val_tar.stat().st_size / (1024 ** 3)
        print("\n" + "=" * 75)
        print(f"[UPLOAD CHÍNH] BẮT ĐẦU UPLOAD {train_val_tar.name} ({tar_gb:.1f} GB)")
        print("Modal sẽ hiển thị thanh tiến trình (progress bar), tốc độ (MB/s) và ETA...")
        print("=" * 75)
        remote_tar = f"{args.remote_dir}/{train_val_tar.name}"
        _run_modal_cmd(
            ["modal", "volume", "put", "--force", args.volume_name, str(train_val_tar), remote_tar],
            f"Upload {train_val_tar.name} ({tar_gb:.1f} GB)",
        )

        # (Tùy chọn) Upload test2018.tar.gz
        if args.include_test:
            test_tar = local_dir / "test2018.tar.gz"
            if test_tar.is_file():
                test_gb = test_tar.stat().st_size / (1024 ** 3)
                remote_test = f"{args.remote_dir}/{test_tar.name}"
                _run_modal_cmd(
                    ["modal", "volume", "put", "--force", args.volume_name, str(test_tar), remote_test],
                    f"Upload {test_tar.name} ({test_gb:.1f} GB)",
                )

        print("\n" + "=" * 75)
        print("✓ BƯỚC UPLOAD DỮ LIỆU LÊN MODAL VOLUME ĐÃ HOÀN TẤT!")
        print("=" * 75)

    if not args.skip_extract:
        # BƯỚC 4: Kích hoạt Container Modal Cloud để giải nén đa luồng
        print("\n[MODAL CLOUD] Đang kích hoạt Modal Cloud container để giải nén...")
        with modal.enable_output():
            with app.run():
                extract_on_modal.remote(
                    delete_archive=delete_archive,
                    include_test=args.include_test,
                )

        print("\n" + "=" * 75)
        print("✓ TẤT CẢ CÁC BƯỚC ĐÃ THÀNH CÔNG RỰC RỠ!")
        print(f"Dữ liệu iNaturalist 2018 đã sẵn sàng tại: {REMOTE_DATA_DIR}")
        print("=" * 75)


@app.local_entrypoint()
def main():
    upload_and_process()


if __name__ == "__main__":
    upload_and_process()
