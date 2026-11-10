# KAGGLE RUN GUIDE — TOÀN BỘ CẤU HÌNH (KAGGLE CLI PUSH — TẮT MÁY ĐI NGỦ)

> **CẬP NHẬT CẤU HÌNH FIX V4 (CHUẨN HOÁ TỰ ĐỘNG)**:
> - **KMeans Backend**: Mặc định là `sklearn_cpu` (k-means++ chuẩn bài báo), đảm bảo Feature Space đánh giá chuẩn xác mốc 70%+, loại bỏ hoàn toàn lỗi sụt Novel của `faiss_gpu`.
> - **Tail-Aware Warm-Start**: Tự động kích hoạt Epoch 0 (`--gate-warm-start-from-support`) cho các cấu hình `Prototype_Tailaware` và `Full`. Cổng phân bổ mở tự nhiên ~1.96 slot/class thay vì bị đông cứng ở 0.82 slot.
> - **Bạn không cần gõ thêm flag nào cả**: Các lệnh ngắn gọn bên dưới đã tự động cấu hình tối ưu 100%.

> **HƯỚNG DẪN CHẠY VÀ TẮT MÁY ĐI NGỦ (KẾT HỢP APP ĐỔI TÀI KHOẢN)**:
> 1. **Bước 1**: Trên app đổi tài khoản của bạn, nhấp chọn tài khoản Kaggle bạn muốn sử dụng (ví dụ: `k1`, `k2`...).
> 2. **Bước 2**: Mở PowerShell trên máy tính (tại thư mục repo `prototail` hoặc `BaCon`), copy và chạy lệnh tương ứng bên dưới.
>    *Lưu ý*: Mỗi tài khoản Kaggle chạy tối đa **2 batch GPU cùng lúc** (theo cặp lệnh đã được nhóm sẵn `k1` -> `k12`).
> 3. **Bước 3**: Khi màn hình hiện thông báo `Kernel pushed successfully -> Đang chạy trên GPU Kaggle`, bạn có thể **TẮT MÁY TÍNH WINDOWS ĐI NGỦ** ngay lập tức.
>
> - *GPU T4 trên máy chủ Kaggle sẽ tự động chạy ngầm độc lập (tối đa 12 tiếng) và tự lưu kết quả khi hoàn thành.*
> - *Kiểm tra tiến độ từ xa: `kaggle kernels status <username>/<kernel_name>`*
> - *Tải log và kết quả khi xong: `kaggle kernels output <username>/<kernel_name> -p ./outputs`*

---

### BẢNG PHÂN BỔ 12 TÀI KHOẢN KAGGLE (WAVE 1: 24 RUNS ĐỒNG THỜI)

| Tài khoản | Model 1 (GPU Session 1) | Model 2 (GPU Session 2) |
| :--- | :--- | :--- |
| **k1** | `simgcd_cub_Prototype_imb20` | `simgcd_cub_Prototype_Tailaware_imb20` |
| **k2** | `simgcd_cub_Base_imb20` | `simgcd_cub_Pseudo_imb20` |
| **k3** | `simgcd_cub_Base_imb50` | `simgcd_cub_Pseudo_imb50` |
| **k4** | `simgcd_cub_Prototype_imb50` | `simgcd_cub_Prototype_Tailaware_imb50` |
| **k5** | `simgcd_cub_Full_wo_Tail_imb50` | `simgcd_cub_Full_imb50` |
| **k6** | `simgcd_cars_Base_imb20` | `simgcd_cars_Pseudo_imb20` |
| **k7** | `simgcd_cars_Prototype_imb20` | `simgcd_cars_Prototype_Tailaware_imb20` |
| **k8** | `simgcd_cars_Full_wo_Tail_imb20` | `simgcd_cars_Full_imb20` |
| **k9** | `simgcd_cars_Base_imb50` | `simgcd_cars_Pseudo_imb50` |
| **k10** | `simgcd_cars_Prototype_imb50` | `simgcd_cars_Prototype_Tailaware_imb50` |
| **k11** | `simgcd_cars_Full_wo_Tail_imb50` | `simgcd_cars_Full_imb50` |
| **k12** | `simgcd_cub_Full_wo_Tail_imb20` | `simgcd_cub_Full_imb20` |

---

# ===========================================================================
# MÔ HÌNH: SIMGCD
# ===========================================================================

## Dataset: CUB (`cub`)

### Imbalance Ratio: imb20

#### simgcd_cub_Base_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Base
```

#### simgcd_cub_Pseudo_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Pseudo
```

#### simgcd_cub_Prototype_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_cub_Base_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Base
```

#### simgcd_cub_Pseudo_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Pseudo
```

#### simgcd_cub_Prototype_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_cub_Base_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Base
```

#### simgcd_cub_Pseudo_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Pseudo
```

#### simgcd_cub_Prototype_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Full
```

## Dataset: CARS (`cars`)

### Imbalance Ratio: imb20

#### simgcd_cars_Base_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Base
```

#### simgcd_cars_Pseudo_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Pseudo
```

#### simgcd_cars_Prototype_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Prototype
```

#### simgcd_cars_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_cars_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_cars_Full_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_cars_Base_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Base
```

#### simgcd_cars_Pseudo_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Pseudo
```

#### simgcd_cars_Prototype_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Prototype
```

#### simgcd_cars_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_cars_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_cars_Full_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_cars_Base_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Base
```

#### simgcd_cars_Pseudo_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Pseudo
```

#### simgcd_cars_Prototype_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Prototype
```

#### simgcd_cars_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_cars_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_cars_Full_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Full
```

## Dataset: AIRCRAFT (`aircraft`)

### Imbalance Ratio: imb20

#### simgcd_aircraft_Base_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Base
```

#### simgcd_aircraft_Pseudo_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Pseudo
```

#### simgcd_aircraft_Prototype_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Prototype
```

#### simgcd_aircraft_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_aircraft_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_aircraft_Full_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_aircraft_Base_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Base
```

#### simgcd_aircraft_Pseudo_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Pseudo
```

#### simgcd_aircraft_Prototype_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Prototype
```

#### simgcd_aircraft_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_aircraft_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_aircraft_Full_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_aircraft_Base_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Base
```

#### simgcd_aircraft_Pseudo_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Pseudo
```

#### simgcd_aircraft_Prototype_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Prototype
```

#### simgcd_aircraft_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_aircraft_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_aircraft_Full_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Full
```

## Dataset: HERBARIUM 19 (`herb19`)

### Imbalance Ratio: imb20

#### simgcd_herb19_Base_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Base
```

#### simgcd_herb19_Pseudo_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Pseudo
```

#### simgcd_herb19_Prototype_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Prototype
```

#### simgcd_herb19_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_herb19_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_herb19_Full_imb20
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_herb19_Base_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Base
```

#### simgcd_herb19_Pseudo_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Pseudo
```

#### simgcd_herb19_Prototype_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Prototype
```

#### simgcd_herb19_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_herb19_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_herb19_Full_imb50
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_herb19_Base_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Base
```

#### simgcd_herb19_Pseudo_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Pseudo
```

#### simgcd_herb19_Prototype_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Prototype
```

#### simgcd_herb19_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_herb19_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_herb19_Full_imb100
```powershell
python kaggle_push.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Full
```

## Dataset: INATURALIST 2018 (`inat2018`)

#### simgcd_inat2018_Base
```powershell
python kaggle_push.py --baseline simgcd --dataset-name inat2018 --config Base
```

#### simgcd_inat2018_Pseudo
```powershell
python kaggle_push.py --baseline simgcd --dataset-name inat2018 --config Pseudo
```

#### simgcd_inat2018_Prototype
```powershell
python kaggle_push.py --baseline simgcd --dataset-name inat2018 --config Prototype
```

#### simgcd_inat2018_Prototype_Tailaware
```powershell
python kaggle_push.py --baseline simgcd --dataset-name inat2018 --config Prototype_Tailaware
```

#### simgcd_inat2018_Full_wo_Tail
```powershell
python kaggle_push.py --baseline simgcd --dataset-name inat2018 --config Full_wo_Tail
```

#### simgcd_inat2018_Full
```powershell
python kaggle_push.py --baseline simgcd --dataset-name inat2018 --config Full
```

# ===========================================================================
# MÔ HÌNH: BACON
# ===========================================================================



## Dataset: CUB (`cub`)

### Imbalance Ratio: imb20

#### bacon_cub_Base_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Base
```

#### bacon_cub_Pseudo_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Pseudo
```

#### bacon_cub_Prototype_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Prototype
```

#### bacon_cub_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_cub_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_cub_Full_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_cub_Base_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Base
```

#### bacon_cub_Pseudo_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Pseudo
```

#### bacon_cub_Prototype_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Prototype
```

#### bacon_cub_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_cub_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_cub_Full_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_cub_Base_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Base
```

#### bacon_cub_Pseudo_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Pseudo
```

#### bacon_cub_Prototype_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Prototype
```

#### bacon_cub_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_cub_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_cub_Full_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Full
```

## Dataset: CARS (`cars`)

### Imbalance Ratio: imb20

#### bacon_cars_Base_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Base
```

#### bacon_cars_Pseudo_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Pseudo
```

#### bacon_cars_Prototype_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Prototype
```

#### bacon_cars_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_cars_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_cars_Full_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_cars_Base_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Base
```

#### bacon_cars_Pseudo_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Pseudo
```

#### bacon_cars_Prototype_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Prototype
```

#### bacon_cars_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_cars_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_cars_Full_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_cars_Base_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Base
```

#### bacon_cars_Pseudo_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Pseudo
```

#### bacon_cars_Prototype_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Prototype
```

#### bacon_cars_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_cars_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_cars_Full_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Full
```

## Dataset: AIRCRAFT (`aircraft`)

### Imbalance Ratio: imb20

#### bacon_aircraft_Base_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Base
```

#### bacon_aircraft_Pseudo_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Pseudo
```

#### bacon_aircraft_Prototype_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Prototype
```

#### bacon_aircraft_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_aircraft_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_aircraft_Full_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_aircraft_Base_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Base
```

#### bacon_aircraft_Pseudo_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Pseudo
```

#### bacon_aircraft_Prototype_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Prototype
```

#### bacon_aircraft_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_aircraft_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_aircraft_Full_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_aircraft_Base_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Base
```

#### bacon_aircraft_Pseudo_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Pseudo
```

#### bacon_aircraft_Prototype_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Prototype
```

#### bacon_aircraft_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_aircraft_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_aircraft_Full_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Full
```

## Dataset: HERBARIUM 19 (`herb19`)

### Imbalance Ratio: imb20

#### bacon_herb19_Base_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Base
```

#### bacon_herb19_Pseudo_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Pseudo
```

#### bacon_herb19_Prototype_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Prototype
```

#### bacon_herb19_Prototype_Tailaware_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_herb19_Full_wo_Tail_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_herb19_Full_imb20
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_herb19_Base_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Base
```

#### bacon_herb19_Pseudo_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Pseudo
```

#### bacon_herb19_Prototype_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Prototype
```

#### bacon_herb19_Prototype_Tailaware_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_herb19_Full_wo_Tail_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_herb19_Full_imb50
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_herb19_Base_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Base
```

#### bacon_herb19_Pseudo_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Pseudo
```

#### bacon_herb19_Prototype_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Prototype
```

#### bacon_herb19_Prototype_Tailaware_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_herb19_Full_wo_Tail_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_herb19_Full_imb100
```powershell
python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Full
```

## Dataset: INATURALIST 2018 (`inat2018`)

#### bacon_inat2018_Base
```powershell
python kaggle_push.py --baseline bacon --dataset-name inat2018 --config Base
```

#### bacon_inat2018_Pseudo
```powershell
python kaggle_push.py --baseline bacon --dataset-name inat2018 --config Pseudo
```

#### bacon_inat2018_Prototype
```powershell
python kaggle_push.py --baseline bacon --dataset-name inat2018 --config Prototype
```

#### bacon_inat2018_Prototype_Tailaware
```powershell
python kaggle_push.py --baseline bacon --dataset-name inat2018 --config Prototype_Tailaware
```

#### bacon_inat2018_Full_wo_Tail
```powershell
python kaggle_push.py --baseline bacon --dataset-name inat2018 --config Full_wo_Tail
```

#### bacon_inat2018_Full
```powershell
python kaggle_push.py --baseline bacon --dataset-name inat2018 --config Full
```
