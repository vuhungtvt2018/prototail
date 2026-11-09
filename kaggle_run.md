# KAGGLE RUN GUIDE — 108 CẤU HÌNH (KAGGLE CLI PUSH — TẮT MÁY ĐI NGỦ)

> **CẬP NHẬT CẤU HÌNH FIX V4 (CHUẨN HOÁ TỰ ĐỘNG)**:
> - **KMeans Backend**: Mặc định là sklearn_cpu (k-means++ chuẩn bài báo), đảm bảo Feature Space đánh giá chuẩn xác mốc 70%+, loại bỏ hoàn toàn lỗi sụt Novel của aiss_gpu.
> - **Tail-Aware Warm-Start**: Tự động kích hoạt Epoch 0 (--gate-warm-start-from-support) cho các cấu hình Prototype_Tailaware và Full. Cổng phân bổ mở tự nhiên ~1.96 slot/class thay vì bị đông cứng ở 0.82 slot.
> - **Bạn không cần gõ thêm flag nào cả**: Các lệnh ngắn gọn bên dưới đã tự động cấu hình tối ưu 100%.

> **HƯỚNG DẪN CHẠY VÀ TẮT MÁY ĐI NGỦ (KẾT HỢP APP ĐỔI TÀI KHOẢN)**:
> 1. **Bước 1**: Trên app đổi tài khoản của bạn, nhấp chọn tài khoản Kaggle bạn muốn sử dụng (ví dụ: k1, k2...).
> 2. **Bước 2**: Mở PowerShell trên máy tính (tại thư mục repo prototail hoặc BaCon), copy và chạy lệnh tương ứng bên dưới.
>    *Lưu ý*: Mỗi tài khoản Kaggle chạy tối đa **2 batch GPU cùng lúc** (theo cặp lệnh đã được nhóm sẵn k1 -> k12).
> 3. **Bước 3**: Khi màn hình hiện thông báo Kernel pushed successfully -> Đang chạy trên GPU Kaggle, bạn có thể **TẮT MÁY TÍNH WINDOWS ĐI NGỦ** ngay lập tức.
>
> - *GPU T4 trên máy chủ Kaggle sẽ tự động chạy ngầm độc lập (tối đa 12 tiếng) và tự lưu kết quả khi hoàn thành.*
> - *Kiểm tra tiến độ từ xa: kaggle kernels status <username>/<kernel_name>*
> - *Tải log và kết quả khi xong: kaggle kernels output <username>/<kernel_name> -p ./outputs*

---

### BẢNG PHÂN BỔ 12 TÀI KHOẢN KAGGLE (WAVE 1: 24 RUNS ĐỒNG THỜI)

| Tài khoản | Model 1 (GPU Session 1) | Model 2 (GPU Session 2) |
| :--- | :--- | :--- |
| **k1** | simgcd_cub_Prototype_imb20 | simgcd_cub_Prototype_Tailaware_imb20 |
| **k2** | simgcd_cub_Base_imb20 | simgcd_cub_Pseudo_imb20 |
| **k3** | simgcd_cub_Base_imb50 | simgcd_cub_Pseudo_imb50 |
| **k4** | simgcd_cub_Prototype_imb50 | simgcd_cub_Prototype_Tailaware_imb50 |
| **k5** | simgcd_cub_Full_wo_Tail_imb50 | simgcd_cub_Full_imb50 |
| **k6** | simgcd_car_Base_imb20 | simgcd_car_Pseudo_imb20 |
| **k7** | simgcd_car_Prototype_imb20 | simgcd_car_Prototype_Tailaware_imb20 |
| **k8** | simgcd_car_Full_wo_Tail_imb20 | simgcd_car_Full_imb20 |
| **k9** | simgcd_car_Base_imb50 | simgcd_car_Pseudo_imb50 |
| **k10** | simgcd_car_Prototype_imb50 | simgcd_car_Prototype_Tailaware_imb50 |
| **k11** | simgcd_car_Full_wo_Tail_imb50 | simgcd_car_Full_imb50 |
| **k12** | simgcd_cub_Full_wo_Tail_imb20 | simgcd_cub_Full_imb20 |

---

# ===========================================================================

# MÔ HÌNH: SIMGCD

# ===========================================================================

## Dataset: CUB

### Imbalance Ratio: imb20

#### simgcd_cub_Base_imb20 k2

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Base
```

#### simgcd_cub_Pseudo_imb20 k2

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Pseudo
```

#### simgcd_cub_Prototype_imb20 k1

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb20 k1

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb20 k12

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb20 k12

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_cub_Base_imb50 k3

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Base
```

#### simgcd_cub_Pseudo_imb50 k3

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Pseudo
```

#### simgcd_cub_Prototype_imb50 k4

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb50 k4

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb50 k5

```powershell
python kaggle_push.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb50 k5

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

## Dataset: CAR

### Imbalance Ratio: imb20

#### simgcd_car_Base_imb20 k6

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 20 --config Base
```

#### simgcd_car_Pseudo_imb20 k6

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 20 --config Pseudo
```

#### simgcd_car_Prototype_imb20 k7

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 20 --config Prototype
```

#### simgcd_car_Prototype_Tailaware_imb20 k7

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_car_Full_wo_Tail_imb20 k8

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_car_Full_imb20 k8

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_car_Base_imb50 k9

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 50 --config Base
```

#### simgcd_car_Pseudo_imb50 k9

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 50 --config Pseudo
```

#### simgcd_car_Prototype_imb50 k10

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 50 --config Prototype
```

#### simgcd_car_Prototype_Tailaware_imb50 k10

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_car_Full_wo_Tail_imb50 k11

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_car_Full_imb50 k11

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_car_Base_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 100 --config Base
```

#### simgcd_car_Pseudo_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 100 --config Pseudo
```

#### simgcd_car_Prototype_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 100 --config Prototype
```

#### simgcd_car_Prototype_Tailaware_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_car_Full_wo_Tail_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_car_Full_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name car --imb-ratio 100 --config Full
```

## Dataset: AIR

### Imbalance Ratio: imb20

#### simgcd_air_Base_imb20

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 20 --config Base
```

#### simgcd_air_Pseudo_imb20

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 20 --config Pseudo
```

#### simgcd_air_Prototype_imb20

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 20 --config Prototype
```

#### simgcd_air_Prototype_Tailaware_imb20

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_air_Full_wo_Tail_imb20

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_air_Full_imb20

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_air_Base_imb50

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 50 --config Base
```

#### simgcd_air_Pseudo_imb50

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 50 --config Pseudo
```

#### simgcd_air_Prototype_imb50

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 50 --config Prototype
```

#### simgcd_air_Prototype_Tailaware_imb50

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_air_Full_wo_Tail_imb50

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_air_Full_imb50

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_air_Base_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 100 --config Base
```

#### simgcd_air_Pseudo_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 100 --config Pseudo
```

#### simgcd_air_Prototype_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 100 --config Prototype
```

#### simgcd_air_Prototype_Tailaware_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_air_Full_wo_Tail_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_air_Full_imb100

```powershell
python kaggle_push.py --baseline simgcd --dataset-name air --imb-ratio 100 --config Full
```

# ===========================================================================

# MÔ HÌNH: BACON

# ===========================================================================

## Dataset: CUB

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

## Dataset: CAR

### Imbalance Ratio: imb20

#### bacon_car_Base_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 20 --config Base
```

#### bacon_car_Pseudo_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 20 --config Pseudo
```

#### bacon_car_Prototype_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 20 --config Prototype
```

#### bacon_car_Prototype_Tailaware_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_car_Full_wo_Tail_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_car_Full_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_car_Base_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 50 --config Base
```

#### bacon_car_Pseudo_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 50 --config Pseudo
```

#### bacon_car_Prototype_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 50 --config Prototype
```

#### bacon_car_Prototype_Tailaware_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_car_Full_wo_Tail_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_car_Full_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_car_Base_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 100 --config Base
```

#### bacon_car_Pseudo_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 100 --config Pseudo
```

#### bacon_car_Prototype_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 100 --config Prototype
```

#### bacon_car_Prototype_Tailaware_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_car_Full_wo_Tail_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_car_Full_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name car --imb-ratio 100 --config Full
```

## Dataset: AIR

### Imbalance Ratio: imb20

#### bacon_air_Base_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 20 --config Base
```

#### bacon_air_Pseudo_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 20 --config Pseudo
```

#### bacon_air_Prototype_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 20 --config Prototype
```

#### bacon_air_Prototype_Tailaware_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_air_Full_wo_Tail_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_air_Full_imb20

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_air_Base_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 50 --config Base
```

#### bacon_air_Pseudo_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 50 --config Pseudo
```

#### bacon_air_Prototype_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 50 --config Prototype
```

#### bacon_air_Prototype_Tailaware_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_air_Full_wo_Tail_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_air_Full_imb50

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_air_Base_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 100 --config Base
```

#### bacon_air_Pseudo_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 100 --config Pseudo
```

#### bacon_air_Prototype_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 100 --config Prototype
```

#### bacon_air_Prototype_Tailaware_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_air_Full_wo_Tail_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_air_Full_imb100

```powershell
python kaggle_push.py --baseline bacon --dataset-name air --imb-ratio 100 --config Full
```
