"""=============================================================================
PROTOTAIL BENCHMARK LOG PARSER & EXCEL EXPORTER (108 CẤU HÌNH)
=============================================================================
- Trích xuất kết quả đánh giá trên tập UNLABELED TRAIN (Epoch 200)
- Tự động quét và cập nhật các log mới nhất từ D:\\data\\log prototail (bỏ qua 'old log')
- Tự động thay thế bản cũ khi có log mới hơn mà không cần phải xóa file thủ công
- Xử lý mượt mà khi file Excel đang mở (tự lưu file dự phòng tránh lỗi Permission)
- Tự động highlight: IN ĐẬM (Max - Tốt nhất) và GẠCH CHÂN (2nd - Tốt nhì)
- Phân nhóm 108 cấu hình (2 Models x 3 Datasets x 3 Imb x 6 Configs) thành các bảng nhỏ trực quan
=============================================================================
"""

import os
import re
import json
import glob
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Thư mục log mặc định và đường dẫn xuất file
LOG_DIR = r"D:\data\log prototail"
OUTPUT_EXCEL = r"D:\data\log prototail\Bang_Ket_Qua_Thuc_Nghiem_ProtoTail.xlsx"

MODELS = [
    ("simgcd", "SimGCD"),
    ("bacon", "BaCon"),
]

DATASETS = [
    ("cub", "CUB-200-2011", ["cub", "cub200"]),
    ("car", "Stanford Cars", ["car", "cars", "scars", "stanford_cars"]),
    ("air", "FGVC Aircraft", ["air", "aircraft", "fgvc_aircraft"]),
]

DATASET_MAP = {
    "cub": "cub", "cub200": "cub",
    "car": "car", "cars": "car", "scars": "car", "stanford_cars": "car",
    "air": "air", "aircraft": "air", "fgvc_aircraft": "air",
}

IMB_RATIOS = ["imb20", "imb50", "imb100"]

CONFIGS = [
    ("Base", "Base", "Baseline chuẩn"),
    ("Pseudo", "Pseudo", "Baseline + Pseudo-labeling"),
    ("Prototype", "Prototype", "Baseline + Prototype Parts (Ablated)"),
    ("Prototype_Tailaware", "Prototype_Tailaware", "Baseline + Prototype + Tail-Aware"),
    ("Full_wo_Tail", "Full_wo_Tail", "Full Model w/o Tail-Aware"),
    ("Full", "Full", "ProtoTail Full Model"),
]

METRIC_KEYS = ["all", "old", "new", "h_score", "k_many", "k_med", "k_few", "u_many", "u_med", "u_few"]


def extract_timestamp(fname, mtime):
    """Trích xuất mốc thời gian từ tên file log (dạng ISO) hoặc lấy mtime của hệ thống."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
        except Exception:
            pass
    return datetime.fromtimestamp(mtime)


def normalize_canonical_key(run_name, fname):
    """Chuẩn hoá Run Name thành key chuẩn mực: {model}_{dataset}_{config}_{imb}."""
    combined = (run_name + " " + fname).lower().replace("-", "_")
    
    # 1. Model
    m = "simgcd" if "simgcd" in combined else ("bacon" if "bacon" in combined else "")
    
    # 2. Dataset
    d = ""
    for k, v in DATASET_MAP.items():
        if f"_{k}_" in combined or f"_{k}" in combined or f"{k}_" in combined or f"-{k}-" in combined:
            d = v
            break

    # 3. Imbalance Ratio
    imb = ""
    for i in ["imb20", "imb50", "imb100"]:
        if i in combined:
            imb = i
            break

    # 4. Config
    cfg = ""
    if "prototype_tailaware" in combined:
        cfg = "prototype_tailaware"
    elif "prototype" in combined:
        cfg = "prototype"
    elif "full_wo_tail" in combined:
        cfg = "full_wo_tail"
    elif "full" in combined:
        cfg = "full"
    elif "pseudo" in combined:
        cfg = "pseudo"
    elif "base" in combined:
        cfg = "base"

    if m and d and cfg and imb:
        return f"{m}_{d}_{cfg}_{imb}"
    return combined.strip()


def parse_all_logs(log_dir):
    """Quét và trích xuất chỉ số UNLABELED TRAIN từ toàn bộ log (tự động lấy file mới nhất)."""
    pattern = os.path.join(log_dir, "*.log")
    files = [f for f in glob.glob(pattern) if "old log" not in f.lower()]

    candidates = {}

    for filepath in files:
        fname = os.path.basename(filepath)
        stat = os.stat(filepath)
        file_dt = extract_timestamp(fname, stat.st_mtime)

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[!] Lỗi khi đọc file {fname}: {e}")
            continue

        content = "".join(lines)

        # 1. Nhận diện nơi chạy (Kaggle Account / Server / Modal)
        account = ""
        acc_m = re.search(r"Kaggle Account:\s*(\S+)", content)
        if acc_m:
            account = f"Kaggle ({acc_m.group(1)})"
        elif "modal run" in content.lower() or "modal app" in content.lower():
            account = "Modal Cloud"
        elif "cuda_visible_devices" in content.lower() or "nohup" in content.lower():
            account = "Server Linux"
        else:
            prefix_m = re.match(r"(tam\d+)_", fname)
            if prefix_m:
                account = f"Kaggle ({prefix_m.group(1)})"
            else:
                account = "Server / Local"

        # 2. Nhận diện Run Name & Chuẩn hoá key
        rn_m = re.search(r"Run Name:\s*(\S+)", content)
        run_name = rn_m.group(1) if rn_m else fname
        canon_key = normalize_canonical_key(run_name, fname)

        # 3. Trích xuất chỉ số UNLABELED TRAIN tại Epoch cuối (Paper protocol Epoch 200)
        last_unl = None
        for line in lines:
            if "unlabeled_train" in line and "epoch" in line:
                try:
                    data = json.loads(line)
                    if "unlabeled_train" in data and isinstance(data["unlabeled_train"], dict):
                        last_unl = (data.get("epoch", None), data["unlabeled_train"])
                except Exception:
                    pass

        # Fallback qua regex text UNLABELED-TRAIN EVAL nếu json không có
        if not last_unl:
            text_m = re.findall(
                r"\[UNLABELED-TRAIN EVAL\] Epoch (\d+): All ([\d\.]+) \| Known ([\d\.]+) \| Novel ([\d\.]+) \| H-Score ([\d\.]+)",
                content,
            )
            if text_m:
                last = text_m[-1]
                ep = int(last[0])
                all_v, old_v, new_v, h_v = float(last[1]), float(last[2]), float(last[3]), float(last[4])
                sb_m = re.findall(
                    r"Shot Breakdown \(Train\):\s*Known \[Many: ([\d\.-]+), Med: ([\d\.-]+), Few: ([\d\.-]+)\]\s*\|\s*Novel \[Many: ([\d\.-]+), Med: ([\d\.-]+), Few: ([\d\.-]+)\]",
                    content,
                )
                sb = sb_m[-1] if sb_m else (-1, -1, -1, -1, -1, -1)
                last_unl = (
                    ep,
                    {
                        "all": all_v, "old": old_v, "new": new_v, "h_score": h_v,
                        "k_many": float(sb[0]), "k_med": float(sb[1]), "k_few": float(sb[2]),
                        "u_many": float(sb[3]), "u_med": float(sb[4]), "u_few": float(sb[5]),
                    },
                )

        metrics_dict = None
        epoch = None
        if last_unl:
            epoch, t = last_unl
            metrics_dict = {
                "all": float(t.get("all", t.get("all_acc", 0.0))),
                "old": float(t.get("old", t.get("known_acc", 0.0))),
                "new": float(t.get("new", t.get("novel_acc", 0.0))),
                "h_score": float(t.get("h_score", 0.0)),
                "k_many": float(t.get("k_many", t.get("known_many_acc", -1.0))),
                "k_med": float(t.get("k_med", t.get("known_med_acc", -1.0))),
                "k_few": float(t.get("k_few", t.get("known_few_acc", -1.0))),
                "u_many": float(t.get("u_many", t.get("novel_many_acc", -1.0))),
                "u_med": float(t.get("u_med", t.get("novel_med_acc", -1.0))),
                "u_few": float(t.get("u_few", t.get("novel_few_acc", -1.0))),
            }

        item = {
            "run_name": run_name,
            "canon_key": canon_key,
            "file_name": fname,
            "account": account,
            "epoch": epoch,
            "metrics": metrics_dict,
            "dt": file_dt,
            "status": "Hoàn thành" if metrics_dict and epoch == 200 else ("Đang chạy" if metrics_dict else "Chưa có kết quả"),
        }

        # 4. Logic tự động cập nhật: Luôn ưu tiên run hoàn thành và có thời gian mới hơn
        if canon_key not in candidates:
            candidates[canon_key] = item
        else:
            prev = candidates[canon_key]
            prev_done = (prev["metrics"] is not None and prev["epoch"] == 200)
            curr_done = (metrics_dict is not None and epoch == 200)

            if curr_done and not prev_done:
                candidates[canon_key] = item
            elif curr_done and prev_done:
                if file_dt > prev["dt"]:
                    candidates[canon_key] = item
            elif not prev_done and file_dt > prev["dt"]:
                candidates[canon_key] = item

    return candidates


def find_log_item(logs_data, model_key, d_key, cfg_key, imb):
    """Tìm log chính xác theo cấu hình chuẩn."""
    expected_run = f"{model_key}_{d_key}_{cfg_key}_{imb}".lower()
    return logs_data.get(expected_run)


def build_excel_report(logs_data, output_path):
    """Tạo workbook Excel hoàn chỉnh với 4 sheet chuyên biệt (dữ liệu Unlabeled Train)."""
    wb = Workbook()
    default_sheet = wb.active

    # Bảng màu Pastel chuẩn mực
    BLUE_HEADER = "2563EB"     # Royal Blue (SimGCD)
    BLUE_LIGHT = "EFF6FF"
    EMERALD_HEADER = "047857"  # Emerald (BaCon)
    EMERALD_LIGHT = "ECFDF5"
    AMBER_LIGHT = "FFFBEB"
    PURPLE_LIGHT = "FAF5FF"
    GRAY_HEADER = "475569"
    GRAY_LIGHT = "F8FAFC"
    BORDER_COLOR = "CBD5E1"
    
    # Highlight
    FILL_MAX = "DCFCE7"        # Xanh bạc hà nhạt cho giá trị lớn nhất (Bold)
    FILL_2ND = "E0F2FE"        # Xanh da trời nhạt cho giá trị lớn nhì (Underline)
    FILL_ZEBRA = "F8FAFC"      # Nền xen kẽ rất nhẹ

    font_family = "Segoe UI"
    
    thin_border = Border(
        left=Side(style="thin", color=BORDER_COLOR),
        right=Side(style="thin", color=BORDER_COLOR),
        top=Side(style="thin", color=BORDER_COLOR),
        bottom=Side(style="thin", color=BORDER_COLOR),
    )
    
    thick_bottom_border = Border(
        left=Side(style="thin", color=BORDER_COLOR),
        right=Side(style="thin", color=BORDER_COLOR),
        top=Side(style="thin", color=BORDER_COLOR),
        bottom=Side(style="medium", color="64748B"),
    )

    # Helper render bảng nhỏ (6 cấu hình)
    def render_mini_table(ws, start_row, model_key, model_label, d_key, d_label, imb):
        title = f"BẢNG: {model_label} — {d_label} (Tỷ lệ mất cân bằng: {imb.upper()})"
        
        # 1. Title Banner của bảng nhỏ
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=14)
        title_cell = ws.cell(row=start_row, column=1, value=title)
        title_cell.font = Font(name=font_family, size=11, bold=True, color="FFFFFF")
        title_cell.fill = PatternFill(start_color=BLUE_HEADER if model_key == "simgcd" else EMERALD_HEADER, fill_type="solid")
        title_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[start_row].height = 26

        # 2. Super Headers
        sr = start_row + 1
        ws.row_dimensions[sr].height = 20

        # Cấu hình
        c_cfg = ws.cell(row=sr, column=1, value="CẤU HÌNH MÔ HÌNH")
        c_cfg.font = Font(name=font_family, size=9, bold=True, color="FFFFFF")
        c_cfg.fill = PatternFill(start_color=GRAY_HEADER, fill_type="solid")
        c_cfg.alignment = Alignment(horizontal="center", vertical="center")
        c_cfg.border = thin_border

        # Chỉ số chung (Unlabeled Train)
        ws.merge_cells(start_row=sr, start_column=2, end_row=sr, end_column=5)
        c_main = ws.cell(row=sr, column=2, value="CHỈ SỐ CHUNG (UNLABELED TRAIN)")
        c_main.font = Font(name=font_family, size=9, bold=True, color="1E3A8A")
        c_main.fill = PatternFill(start_color=BLUE_LIGHT, fill_type="solid")
        c_main.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(2, 6):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=BLUE_LIGHT, fill_type="solid")

        # Known Classes (Old)
        ws.merge_cells(start_row=sr, start_column=6, end_row=sr, end_column=8)
        c_old = ws.cell(row=sr, column=6, value="KNOWN CLASSES (OLD)")
        c_old.font = Font(name=font_family, size=9, bold=True, color="92400E")
        c_old.fill = PatternFill(start_color=AMBER_LIGHT, fill_type="solid")
        c_old.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(6, 9):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=AMBER_LIGHT, fill_type="solid")

        # Novel Classes (New)
        ws.merge_cells(start_row=sr, start_column=9, end_row=sr, end_column=11)
        c_new = ws.cell(row=sr, column=9, value="NOVEL CLASSES (NEW)")
        c_new.font = Font(name=font_family, size=9, bold=True, color="6B21A8")
        c_new.fill = PatternFill(start_color=PURPLE_LIGHT, fill_type="solid")
        c_new.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(9, 12):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=PURPLE_LIGHT, fill_type="solid")

        # Thông tin thực nghiệm
        ws.merge_cells(start_row=sr, start_column=12, end_row=sr, end_column=14)
        c_info = ws.cell(row=sr, column=12, value="THÔNG TIN THỰC NGHIỆM")
        c_info.font = Font(name=font_family, size=9, bold=True, color="334155")
        c_info.fill = PatternFill(start_color=GRAY_LIGHT, fill_type="solid")
        c_info.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(12, 15):
            ws.cell(row=sr, column=col).border = thin_border
            ws.cell(row=sr, column=col).fill = PatternFill(start_color=GRAY_LIGHT, fill_type="solid")

        # 3. Column Sub-headers
        sub_headers = [
            ("Ablation Setting", 1),
            ("All (%)", 2),
            ("Old (%)", 3),
            ("New (%)", 4),
            ("H-Score", 5),
            ("Many (%)", 6),
            ("Med (%)", 7),
            ("Few (%)", 8),
            ("Many (%)", 9),
            ("Med (%)", 10),
            ("Few (%)", 11),
            ("Ghi chú nơi chạy", 12),
            ("Tên File Log", 13),
            ("Trạng thái", 14),
        ]
        shr = start_row + 2
        ws.row_dimensions[shr].height = 22
        for text, col_idx in sub_headers:
            cell = ws.cell(row=shr, column=col_idx, value=text)
            cell.font = Font(name=font_family, size=9, bold=True, color="334155")
            cell.fill = PatternFill(start_color="F1F5F9", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thick_bottom_border

        # 4. Gom dữ liệu 6 cấu hình
        rows_data = []
        column_values = {k: [] for k in METRIC_KEYS}

        for cfg_key, cfg_label, cfg_desc in CONFIGS:
            log_item = find_log_item(logs_data, model_key, d_key, cfg_key, imb)
            metrics = log_item["metrics"] if log_item else None
            note = log_item["account"] if (log_item and log_item["account"]) else "Chưa chạy"
            logfile = log_item["file_name"] if log_item else "-"
            status = log_item["status"] if log_item else "Chưa chạy"

            rows_data.append({
                "config_label": cfg_label,
                "metrics": metrics,
                "note": note,
                "logfile": logfile,
                "status": status,
            })

            if metrics:
                for k in METRIC_KEYS:
                    val = metrics.get(k)
                    if val is not None and val >= 0:
                        column_values[k].append(val)

        # Tính thứ hạng lớn nhất (Max) và lớn nhì (2nd) cho từng cột metric
        ranks = {}
        for k in METRIC_KEYS:
            vals = sorted(list(set(column_values[k])), reverse=True)
            first = vals[0] if len(vals) > 0 else None
            second = vals[1] if len(vals) > 1 else None
            ranks[k] = (first, second)

        # 5. Đổ dữ liệu 6 hàng
        for i, rdata in enumerate(rows_data):
            curr_row = start_row + 3 + i
            ws.row_dimensions[curr_row].height = 20
            is_zebra = (i % 2 == 1)
            row_fill = PatternFill(start_color=FILL_ZEBRA, fill_type="solid") if is_zebra else PatternFill(fill_type=None)

            # Cấu hình
            c_cfg = ws.cell(row=curr_row, column=1, value=rdata["config_label"])
            c_cfg.font = Font(name=font_family, size=9.5, bold=True if "Full" in rdata["config_label"] else False, color="0F172A")
            c_cfg.alignment = Alignment(horizontal="left", vertical="center", indent=1)
            c_cfg.border = thin_border
            if is_zebra:
                c_cfg.fill = row_fill

            # Các cột chỉ số (Cols 2 đến 11)
            metrics = rdata["metrics"]
            for col_offset, mkey in enumerate(METRIC_KEYS):
                col_idx = 2 + col_offset
                cell = ws.cell(row=curr_row, column=col_idx)
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="right", vertical="center")

                if metrics and mkey in metrics and metrics[mkey] is not None and metrics[mkey] >= 0:
                    val = metrics[mkey]
                    cell.value = val
                    cell.number_format = "0.00"

                    first_val, second_val = ranks[mkey]
                    # MAX: IN ĐẬM + NỀN XANH LÁ NHẠT
                    if first_val is not None and abs(val - first_val) < 1e-5:
                        cell.font = Font(name=font_family, size=9.5, bold=True, color="14532D")
                        cell.fill = PatternFill(start_color=FILL_MAX, fill_type="solid")
                    # 2ND: GẠCH CHÂN + NỀN XANH DƯƠNG NHẠT
                    elif second_val is not None and abs(val - second_val) < 1e-5:
                        cell.font = Font(name=font_family, size=9.5, bold=False, underline="single", color="1E3A8A")
                        cell.fill = PatternFill(start_color=FILL_2ND, fill_type="solid")
                    else:
                        cell.font = Font(name=font_family, size=9.5, color="334155")
                        if is_zebra:
                            cell.fill = row_fill
                else:
                    cell.value = "-"
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.font = Font(name=font_family, size=9.5, italic=True, color="94A3B8")
                    if is_zebra:
                        cell.fill = row_fill

            # Ghi chú nơi chạy
            c_note = ws.cell(row=curr_row, column=12, value=rdata["note"])
            c_note.font = Font(name=font_family, size=9, color="1E293B" if rdata["note"] != "Chưa chạy" else "94A3B8")
            c_note.alignment = Alignment(horizontal="left", vertical="center", indent=1)
            c_note.border = thin_border
            if is_zebra:
                c_note.fill = row_fill

            # Tên File Log
            c_log = ws.cell(row=curr_row, column=13, value=rdata["logfile"])
            c_log.font = Font(name="Consolas", size=8.5, color="475569" if rdata["logfile"] != "-" else "94A3B8")
            c_log.alignment = Alignment(horizontal="left", vertical="center")
            c_log.border = thin_border
            if is_zebra:
                c_log.fill = row_fill

            # Trạng thái
            c_status = ws.cell(row=curr_row, column=14, value=rdata["status"])
            if rdata["status"] == "Hoàn thành":
                c_status.font = Font(name=font_family, size=9, bold=True, color="15803D")
                c_status.fill = PatternFill(start_color="DCFCE7", fill_type="solid")
            elif rdata["status"] == "Đang chạy":
                c_status.font = Font(name=font_family, size=9, bold=True, color="B45309")
                c_status.fill = PatternFill(start_color="FEF3C7", fill_type="solid")
            else:
                c_status.font = Font(name=font_family, size=9, color="94A3B8", italic=True)
                if is_zebra:
                    c_status.fill = row_fill
            c_status.alignment = Alignment(horizontal="center", vertical="center")
            c_status.border = thin_border

        return start_row + 3 + len(CONFIGS) + 2

    # ----------------------------------------------------
    # Sheet 1: SimGCD Benchmark
    # ----------------------------------------------------
    ws_simgcd = wb.create_sheet(title="SimGCD Benchmark")
    ws_simgcd.views.sheetView[0].showGridLines = True
    
    ws_simgcd.merge_cells("A1:N1")
    s_banner = ws_simgcd.cell(row=1, column=1, value="BẢNG TỔNG HỢP KẾT QUẢ UNLABELED TRAIN — MÔ HÌNH SIMGCD (27 CẤU HÌNH x 3 DATASETS)")
    s_banner.font = Font(name=font_family, size=13, bold=True, color="FFFFFF")
    s_banner.fill = PatternFill(start_color="1E3A8A", fill_type="solid")
    s_banner.alignment = Alignment(horizontal="center", vertical="center")
    ws_simgcd.row_dimensions[1].height = 36

    ws_simgcd.merge_cells("A2:N2")
    s_sub = ws_simgcd.cell(row=2, column=1, value="Quy ước trực quan (Unlabeled Train): IN ĐẬM (nền xanh lá nhạt) = Tốt nhất (1st) | GẠCH CHÂN (nền xanh dương nhạt) = Tốt nhì (2nd) | Dấu '-' = Chưa chạy / N/A")
    s_sub.font = Font(name=font_family, size=9.5, italic=True, color="475569")
    s_sub.alignment = Alignment(horizontal="center", vertical="center")
    ws_simgcd.row_dimensions[2].height = 20

    current_row = 4
    for d_key, d_label, _ in DATASETS:
        ws_simgcd.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=14)
        d_cell = ws_simgcd.cell(row=current_row, column=1, value=f"DATASET: {d_label.upper()}")
        d_cell.font = Font(name=font_family, size=11, bold=True, color="1E3A8A")
        d_cell.fill = PatternFill(start_color="DBEAFE", fill_type="solid")
        d_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws_simgcd.row_dimensions[current_row].height = 26
        current_row += 1

        for imb in IMB_RATIOS:
            current_row = render_mini_table(ws_simgcd, current_row, "simgcd", "SimGCD", d_key, d_label, imb)

    # ----------------------------------------------------
    # Sheet 2: BaCon Benchmark
    # ----------------------------------------------------
    ws_bacon = wb.create_sheet(title="BaCon Benchmark")
    ws_bacon.views.sheetView[0].showGridLines = True

    ws_bacon.merge_cells("A1:N1")
    b_banner = ws_bacon.cell(row=1, column=1, value="BẢNG TỔNG HỢP KẾT QUẢ UNLABELED TRAIN — MÔ HÌNH BACON (27 CẤU HÌNH x 3 DATASETS)")
    b_banner.font = Font(name=font_family, size=13, bold=True, color="FFFFFF")
    b_banner.fill = PatternFill(start_color="065F46", fill_type="solid")
    b_banner.alignment = Alignment(horizontal="center", vertical="center")
    ws_bacon.row_dimensions[1].height = 36

    ws_bacon.merge_cells("A2:N2")
    b_sub = ws_bacon.cell(row=2, column=1, value="Quy ước trực quan (Unlabeled Train): IN ĐẬM (nền xanh lá nhạt) = Tốt nhất (1st) | GẠCH CHÂN (nền xanh dương nhạt) = Tốt nhì (2nd) | Dấu '-' = Chưa chạy / N/A")
    b_sub.font = Font(name=font_family, size=9.5, italic=True, color="475569")
    b_sub.alignment = Alignment(horizontal="center", vertical="center")
    ws_bacon.row_dimensions[2].height = 20

    current_row = 4
    for d_key, d_label, _ in DATASETS:
        ws_bacon.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=14)
        d_cell = ws_bacon.cell(row=current_row, column=1, value=f"DATASET: {d_label.upper()}")
        d_cell.font = Font(name=font_family, size=11, bold=True, color="065F46")
        d_cell.fill = PatternFill(start_color="D1FAE5", fill_type="solid")
        d_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws_bacon.row_dimensions[current_row].height = 26
        current_row += 1

        for imb in IMB_RATIOS:
            current_row = render_mini_table(ws_bacon, current_row, "bacon", "BaCon", d_key, d_label, imb)

    # ----------------------------------------------------
    # Sheet 3: Tổng Quan & Tiến Độ
    # ----------------------------------------------------
    ws_dash = wb.create_sheet(title="Tổng Quan & Tiến Độ", index=0)
    ws_dash.views.sheetView[0].showGridLines = True
    
    ws_dash.merge_cells("A1:K1")
    dash_title = ws_dash.cell(row=1, column=1, value="BẢNG ĐIỀU KHIỂN TIẾN ĐỘ THỰC NGHIỆM PROTOTAIL (108 CẤU HÌNH)")
    dash_title.font = Font(name=font_family, size=14, bold=True, color="FFFFFF")
    dash_title.fill = PatternFill(start_color="0F172A", fill_type="solid")
    dash_title.alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[1].height = 40

    ws_dash.merge_cells("A2:K2")
    dash_time = ws_dash.cell(row=2, column=1, value=f"Dữ liệu: Unlabeled Train (Epoch 200) | Cập nhật lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Thư mục: {LOG_DIR}")
    dash_time.font = Font(name=font_family, size=9.5, italic=True, color="64748B")
    dash_time.alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[2].height = 22

    total_configs = len(MODELS) * len(DATASETS) * len(IMB_RATIOS) * len(CONFIGS)
    completed_runs = sum(1 for v in logs_data.values() if v["status"] == "Hoàn thành")
    running_runs = sum(1 for v in logs_data.values() if v["status"] == "Đang chạy")
    pending_runs = total_configs - completed_runs - running_runs

    kpi_cards = [
        ("TỔNG SỐ RUN", total_configs, "1E3A8A", "DBEAFE", 2),
        ("ĐÃ HOÀN THÀNH", completed_runs, "14532D", "DCFCE7", 4),
        ("ĐANG CHẠY", running_runs, "78350F", "FEF3C7", 6),
        ("CHƯA CHẠY", pending_runs, "334155", "F1F5F9", 8),
    ]

    for title, val, text_c, bg_c, col in kpi_cards:
        ws_dash.merge_cells(start_row=4, start_column=col, end_row=4, end_column=col+1)
        ws_dash.merge_cells(start_row=5, start_column=col, end_row=5, end_column=col+1)
        
        c1 = ws_dash.cell(row=4, column=col, value=title)
        c1.font = Font(name=font_family, size=9, bold=True, color=text_c)
        c1.fill = PatternFill(start_color=bg_c, fill_type="solid")
        c1.alignment = Alignment(horizontal="center", vertical="center")
        
        c2 = ws_dash.cell(row=5, column=col, value=val)
        c2.font = Font(name=font_family, size=18, bold=True, color=text_c)
        c2.fill = PatternFill(start_color=bg_c, fill_type="solid")
        c2.alignment = Alignment(horizontal="center", vertical="center")
        
        ws_dash.cell(row=4, column=col).border = thin_border
        ws_dash.cell(row=4, column=col+1).border = thin_border
        ws_dash.cell(row=5, column=col).border = thin_border
        ws_dash.cell(row=5, column=col+1).border = thin_border

    ws_dash.row_dimensions[4].height = 20
    ws_dash.row_dimensions[5].height = 32

    ws_dash.cell(row=7, column=2, value="MA TRẬN TIẾN ĐỘ THỰC NGHIỆM (UNLABELED TRAIN - THEO DATASET & RATIO)").font = Font(name=font_family, size=11, bold=True, color="0F172A")
    
    matrix_headers = ["Mô hình", "Dataset", "Tỷ lệ Imb", "Số run xong", "Tiến độ", "Full All (%)", "Full Old (%)", "Full New (%)", "Nơi chạy Full", "Trạng thái bảng"]
    for c_i, h in enumerate(matrix_headers, start=2):
        cell = ws_dash.cell(row=8, column=c_i, value=h)
        cell.font = Font(name=font_family, size=9.5, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="334155", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws_dash.row_dimensions[8].height = 24

    m_row = 9
    for m_key, m_label in MODELS:
        for d_key, d_label, _ in DATASETS:
            for imb in IMB_RATIOS:
                done_count = 0
                full_metrics = None
                full_account = "-"
                for cfg_key, _, _ in CONFIGS:
                    item = find_log_item(logs_data, m_key, d_key, cfg_key, imb)
                    if item and item["status"] == "Hoàn thành":
                        done_count += 1
                    if cfg_key == "Full" and item and item["metrics"]:
                        full_metrics = item["metrics"]
                        full_account = item["account"]

                pct_done = done_count / 6.0
                ws_dash.cell(row=m_row, column=2, value=m_label).alignment = Alignment(horizontal="center", vertical="center")
                ws_dash.cell(row=m_row, column=3, value=d_label).alignment = Alignment(horizontal="left", vertical="center", indent=1)
                ws_dash.cell(row=m_row, column=4, value=imb.upper()).alignment = Alignment(horizontal="center", vertical="center")
                ws_dash.cell(row=m_row, column=5, value=f"{done_count} / 6").alignment = Alignment(horizontal="center", vertical="center")
                
                c_pct = ws_dash.cell(row=m_row, column=6, value=pct_done)
                c_pct.number_format = "0.0%"
                c_pct.alignment = Alignment(horizontal="center", vertical="center")
                
                if full_metrics:
                    c_fa = ws_dash.cell(row=m_row, column=7, value=full_metrics["all"])
                    c_fa.number_format = "0.00"
                    c_fa.font = Font(name=font_family, size=9.5, bold=True, color="14532D")
                    c_fo = ws_dash.cell(row=m_row, column=8, value=full_metrics["old"])
                    c_fo.number_format = "0.00"
                    c_fn = ws_dash.cell(row=m_row, column=9, value=full_metrics["new"])
                    c_fn.number_format = "0.00"
                else:
                    for ci in [7, 8, 9]:
                        ws_dash.cell(row=m_row, column=ci, value="-").alignment = Alignment(horizontal="center", vertical="center")
                
                ws_dash.cell(row=m_row, column=10, value=full_account).alignment = Alignment(horizontal="center", vertical="center")
                
                c_st = ws_dash.cell(row=m_row, column=11, value="Hoàn thành" if done_count == 6 else ("Đang có kết quả" if done_count > 0 else "Chưa có log"))
                c_st.alignment = Alignment(horizontal="center", vertical="center")
                if done_count == 6:
                    c_st.fill = PatternFill(start_color="DCFCE7", fill_type="solid")
                    c_st.font = Font(name=font_family, size=9, bold=True, color="15803D")
                elif done_count > 0:
                    c_st.fill = PatternFill(start_color="FEF3C7", fill_type="solid")
                    c_st.font = Font(name=font_family, size=9, bold=True, color="B45309")
                else:
                    c_st.font = Font(name=font_family, size=9, color="94A3B8")

                for ci in range(2, 12):
                    ws_dash.cell(row=m_row, column=ci).border = thin_border
                ws_dash.row_dimensions[m_row].height = 20
                m_row += 1

    # ----------------------------------------------------
    # Sheet 4: Master List (108 Runs Flat Table)
    # ----------------------------------------------------
    ws_flat = wb.create_sheet(title="Master List (108 Runs)")
    ws_flat.views.sheetView[0].showGridLines = True
    
    flat_headers = [
        "STT", "Mô hình", "Dataset", "Imbalance", "Cấu hình (Setting)", "Tên Run Chuẩn", 
        "All (%)", "Old (%)", "New (%)", "H-Score", 
        "Old Many", "Old Med", "Old Few", "New Many", "New Med", "New Few", 
        "Nơi chạy / Account", "Tên File Log", "Trạng thái", "Thời gian file"
    ]
    
    ws_flat.merge_cells("A1:T1")
    f_title = ws_flat.cell(row=1, column=1, value="DANH SÁCH CHI TIẾT 108 THỰC NGHIỆM (DỮ LIỆU UNLABELED TRAIN — HỖ TRỢ LỌC NHANH)")
    f_title.font = Font(name=font_family, size=12, bold=True, color="FFFFFF")
    f_title.fill = PatternFill(start_color="1E293B", fill_type="solid")
    f_title.alignment = Alignment(horizontal="center", vertical="center")
    ws_flat.row_dimensions[1].height = 32

    ws_flat.row_dimensions[2].height = 24
    for c_i, h in enumerate(flat_headers, start=1):
        cell = ws_flat.cell(row=2, column=c_i, value=h)
        cell.font = Font(name=font_family, size=9.5, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="475569", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    idx = 1
    f_row = 3
    for m_key, m_label in MODELS:
        for d_key, d_label, _ in DATASETS:
            for imb in IMB_RATIOS:
                for cfg_key, cfg_label, _ in CONFIGS:
                    exp_name = f"{m_key}_{d_key}_{cfg_key}_{imb}"
                    item = find_log_item(logs_data, m_key, d_key, cfg_key, imb)
                    metrics = item["metrics"] if item else None
                    
                    ws_flat.cell(row=f_row, column=1, value=idx).alignment = Alignment(horizontal="center", vertical="center")
                    ws_flat.cell(row=f_row, column=2, value=m_label).alignment = Alignment(horizontal="center", vertical="center")
                    ws_flat.cell(row=f_row, column=3, value=d_label).alignment = Alignment(horizontal="left", vertical="center")
                    ws_flat.cell(row=f_row, column=4, value=imb.upper()).alignment = Alignment(horizontal="center", vertical="center")
                    ws_flat.cell(row=f_row, column=5, value=cfg_label).alignment = Alignment(horizontal="left", vertical="center")
                    ws_flat.cell(row=f_row, column=6, value=exp_name).alignment = Alignment(horizontal="left", vertical="center")
                    
                    if metrics:
                        for m_i, k in enumerate(["all", "old", "new", "h_score", "k_many", "k_med", "k_few", "u_many", "u_med", "u_few"]):
                            val = metrics.get(k)
                            c = ws_flat.cell(row=f_row, column=7 + m_i, value=val if val is not None and val >= 0 else "-")
                            if val is not None and val >= 0:
                                c.number_format = "0.00"
                                c.alignment = Alignment(horizontal="right", vertical="center")
                            else:
                                c.alignment = Alignment(horizontal="center", vertical="center")
                    else:
                        for m_i in range(10):
                            c = ws_flat.cell(row=f_row, column=7 + m_i, value="-")
                            c.alignment = Alignment(horizontal="center", vertical="center")
                            c.font = Font(name=font_family, size=9, color="94A3B8")

                    ws_flat.cell(row=f_row, column=17, value=item["account"] if item else "-").alignment = Alignment(horizontal="left", vertical="center")
                    ws_flat.cell(row=f_row, column=18, value=item["file_name"] if item else "-").alignment = Alignment(horizontal="left", vertical="center")
                    
                    st_cell = ws_flat.cell(row=f_row, column=19, value=item["status"] if item else "Chưa chạy")
                    st_cell.alignment = Alignment(horizontal="center", vertical="center")
                    if item and item["status"] == "Hoàn thành":
                        st_cell.fill = PatternFill(start_color="DCFCE7", fill_type="solid")
                        st_cell.font = Font(name=font_family, size=9, bold=True, color="15803D")
                    elif item and item["status"] == "Đang chạy":
                        st_cell.fill = PatternFill(start_color="FEF3C7", fill_type="solid")
                        st_cell.font = Font(name=font_family, size=9, bold=True, color="B45309")
                    else:
                        st_cell.font = Font(name=font_family, size=9, color="94A3B8")

                    ws_flat.cell(row=f_row, column=20, value=item["dt"].strftime("%Y-%m-%d %H:%M") if item else "-").alignment = Alignment(horizontal="center", vertical="center")

                    for ci in range(1, 21):
                        ws_flat.cell(row=f_row, column=ci).border = thin_border
                    ws_flat.row_dimensions[f_row].height = 20
                    idx += 1
                    f_row += 1

    ws_flat.auto_filter.ref = f"A2:T{f_row-1}"

    # Căn chỉnh độ rộng cột tự động
    for sheet in [ws_dash, ws_simgcd, ws_bacon, ws_flat]:
        for col in sheet.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.row in [1, 2, 4] and sheet in [ws_simgcd, ws_bacon, ws_dash, ws_flat]:
                    continue
                if cell.value:
                    val_str = str(cell.value)
                    if len(val_str) > max_len:
                        max_len = len(val_str)
            sheet.column_dimensions[col_letter].width = max(max_len + 3, 11)

    for ws in [ws_simgcd, ws_bacon]:
        ws.column_dimensions["A"].width = 22
        for cl in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]:
            ws.column_dimensions[cl].width = 11
        ws.column_dimensions["L"].width = 22
        ws.column_dimensions["M"].width = 46
        ws.column_dimensions["N"].width = 14

    if default_sheet.title in wb.sheetnames:
        wb.remove(default_sheet)

    # Lưu file với cơ chế bảo vệ khi Excel đang mở
    try:
        wb.save(output_path)
        print(f"[+] Báo cáo Excel đã được cập nhật thành công tại: {output_path}")
    except PermissionError:
        alt_path = output_path.replace(".xlsx", "_MoiNhat.xlsx")
        wb.save(alt_path)
        print(f"[!] CẢNH BÁO: File '{os.path.basename(output_path)}' đang được mở trong Excel!")
        print(f"    -> Đã tự động lưu phiên bản mới nhất ra file: {alt_path}")
        print(f"    -> (Hãy đóng file Excel chính và chạy lại nếu muốn ghi đè trực tiếp).")


if __name__ == "__main__":
    import sys
    import argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="ProtoTail GCD Benchmark Log Parser & Excel Exporter (Unlabeled Train)")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR, help="Thư mục chứa các file log")
    parser.add_argument("--output", type=str, default=OUTPUT_EXCEL, help="Đường dẫn file excel xuất ra")
    args = parser.parse_args()

    print(f"[*] Đang quét thư mục log: {args.log_dir}")
    logs = parse_all_logs(args.log_dir)
    print(f"[*] Đã nhận diện và chuẩn hoá thành công: {len(logs)} cấu hình thực nghiệm.")
    build_excel_report(logs, args.output)
    print(f"[+] Hoàn tất!")
