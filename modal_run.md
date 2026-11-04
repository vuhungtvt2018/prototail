# MODAL RUN GUIDE — 108 CẤU HÌNH (TẮT MÁY ĐI NGỦ VẪN CHẠY)

> **Ý nghĩa của cú pháp `modal_train.py::launch`**:
> - `::launch` là tên hàm entrypoint (`def launch(...)`) được định nghĩa trong file `modal_train.py`.
> - Khi gọi `modal run -d modal_train.py::launch`, Modal sẽ gọi hàm này ở chế độ **Fire-and-forget** (`-d` / `--detach`).
> - Hàm `launch` sẽ submit job lên GPU Cloud (A10G/A100/H200) rồi ngắt kết nối ngay lập tức.
> - Bạn có thể **tắt terminal, tắt máy tính đi ngủ**, mô hình vẫn chạy độc lập trên Modal Cloud.
> - Kiểm tra trạng thái: `modal app list` hoặc xem log: `modal app logs bacon-train`.

---

# ===========================================================================
# MÔ HÌNH: SIMGCD
# ===========================================================================

## Dataset: CUB (cub)

### Imbalance Ratio: imb20

#### simgcd_cub_Base_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_cub_Base_imb20
```

#### simgcd_cub_Pseudo_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_cub_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_cub_Prototype_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_cub_Prototype_imb20 --use-parts --ablate-adaptive-capacity
```

#### simgcd_cub_Prototype_Tailaware_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_cub_Prototype_Tailaware_imb20 --use-parts
```

#### simgcd_cub_Full_wo_Tail_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_cub_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_cub_Full_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_cub_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb50

#### simgcd_cub_Base_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_cub_Base_imb50
```

#### simgcd_cub_Pseudo_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_cub_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_cub_Prototype_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_cub_Prototype_imb50 --use-parts --ablate-adaptive-capacity
```

#### simgcd_cub_Prototype_Tailaware_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_cub_Prototype_Tailaware_imb50 --use-parts
```

#### simgcd_cub_Full_wo_Tail_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_cub_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_cub_Full_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_cub_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb100

#### simgcd_cub_Base_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_cub_Base_imb100
```

#### simgcd_cub_Pseudo_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_cub_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_cub_Prototype_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_cub_Prototype_imb100 --use-parts --ablate-adaptive-capacity
```

#### simgcd_cub_Prototype_Tailaware_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_cub_Prototype_Tailaware_imb100 --use-parts
```

#### simgcd_cub_Full_wo_Tail_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_cub_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_cub_Full_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name cub --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_cub_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

## Dataset: CAR (scars)

### Imbalance Ratio: imb20

#### simgcd_car_Base_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_car_Base_imb20
```

#### simgcd_car_Pseudo_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_car_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_car_Prototype_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_car_Prototype_imb20 --use-parts --ablate-adaptive-capacity
```

#### simgcd_car_Prototype_Tailaware_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_car_Prototype_Tailaware_imb20 --use-parts
```

#### simgcd_car_Full_wo_Tail_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_car_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_car_Full_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_car_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb50

#### simgcd_car_Base_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_car_Base_imb50
```

#### simgcd_car_Pseudo_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_car_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_car_Prototype_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_car_Prototype_imb50 --use-parts --ablate-adaptive-capacity
```

#### simgcd_car_Prototype_Tailaware_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_car_Prototype_Tailaware_imb50 --use-parts
```

#### simgcd_car_Full_wo_Tail_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_car_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_car_Full_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_car_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb100

#### simgcd_car_Base_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_car_Base_imb100
```

#### simgcd_car_Pseudo_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_car_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_car_Prototype_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_car_Prototype_imb100 --use-parts --ablate-adaptive-capacity
```

#### simgcd_car_Prototype_Tailaware_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_car_Prototype_Tailaware_imb100 --use-parts
```

#### simgcd_car_Full_wo_Tail_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_car_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_car_Full_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name scars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_car_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

## Dataset: AIR (aircraft)

### Imbalance Ratio: imb20

#### simgcd_air_Base_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_air_Base_imb20
```

#### simgcd_air_Pseudo_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_air_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_air_Prototype_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_air_Prototype_imb20 --use-parts --ablate-adaptive-capacity
```

#### simgcd_air_Prototype_Tailaware_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_air_Prototype_Tailaware_imb20 --use-parts
```

#### simgcd_air_Full_wo_Tail_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_air_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_air_Full_imb20
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix simgcd_air_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb50

#### simgcd_air_Base_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_air_Base_imb50
```

#### simgcd_air_Pseudo_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_air_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_air_Prototype_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_air_Prototype_imb50 --use-parts --ablate-adaptive-capacity
```

#### simgcd_air_Prototype_Tailaware_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_air_Prototype_Tailaware_imb50 --use-parts
```

#### simgcd_air_Full_wo_Tail_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_air_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_air_Full_imb50
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix simgcd_air_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb100

#### simgcd_air_Base_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_air_Base_imb100
```

#### simgcd_air_Pseudo_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_air_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_air_Prototype_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_air_Prototype_imb100 --use-parts --ablate-adaptive-capacity
```

#### simgcd_air_Prototype_Tailaware_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_air_Prototype_Tailaware_imb100 --use-parts
```

#### simgcd_air_Full_wo_Tail_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_air_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### simgcd_air_Full_imb100
```bash
modal run -d modal_train_simgcd.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix simgcd_air_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

# ===========================================================================
# MÔ HÌNH: BACON
# ===========================================================================

## Dataset: CUB (cub200)

### Imbalance Ratio: imb20

#### bacon_cub_Base_imb20
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_cub_Base_imb20
```

#### bacon_cub_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_cub_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_cub_Prototype_imb20
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_cub_Prototype_imb20 --use-parts --ablate-adaptive-capacity
```

#### bacon_cub_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_cub_Prototype_Tailaware_imb20 --use-parts
```

#### bacon_cub_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_cub_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_cub_Full_imb20
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_cub_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb50

#### bacon_cub_Base_imb50
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_cub_Base_imb50
```

#### bacon_cub_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_cub_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_cub_Prototype_imb50
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_cub_Prototype_imb50 --use-parts --ablate-adaptive-capacity
```

#### bacon_cub_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_cub_Prototype_Tailaware_imb50 --use-parts
```

#### bacon_cub_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_cub_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_cub_Full_imb50
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_cub_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb100

#### bacon_cub_Base_imb100
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_cub_Base_imb100
```

#### bacon_cub_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_cub_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_cub_Prototype_imb100
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_cub_Prototype_imb100 --use-parts --ablate-adaptive-capacity
```

#### bacon_cub_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_cub_Prototype_Tailaware_imb100 --use-parts
```

#### bacon_cub_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_cub_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_cub_Full_imb100
```bash
modal run -d modal_train.py::launch --dataset-name cub200 --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_cub_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

## Dataset: CAR (stanford_cars)

### Imbalance Ratio: imb20

#### bacon_car_Base_imb20
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_car_Base_imb20
```

#### bacon_car_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_car_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_car_Prototype_imb20
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_car_Prototype_imb20 --use-parts --ablate-adaptive-capacity
```

#### bacon_car_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_car_Prototype_Tailaware_imb20 --use-parts
```

#### bacon_car_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_car_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_car_Full_imb20
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_car_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb50

#### bacon_car_Base_imb50
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_car_Base_imb50
```

#### bacon_car_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_car_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_car_Prototype_imb50
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_car_Prototype_imb50 --use-parts --ablate-adaptive-capacity
```

#### bacon_car_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_car_Prototype_Tailaware_imb50 --use-parts
```

#### bacon_car_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_car_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_car_Full_imb50
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_car_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb100

#### bacon_car_Base_imb100
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_car_Base_imb100
```

#### bacon_car_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_car_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_car_Prototype_imb100
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_car_Prototype_imb100 --use-parts --ablate-adaptive-capacity
```

#### bacon_car_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_car_Prototype_Tailaware_imb100 --use-parts
```

#### bacon_car_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_car_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_car_Full_imb100
```bash
modal run -d modal_train.py::launch --dataset-name stanford_cars --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_car_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

## Dataset: AIR (aircraft)

### Imbalance Ratio: imb20

#### bacon_air_Base_imb20
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_air_Base_imb20
```

#### bacon_air_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_air_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_air_Prototype_imb20
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_air_Prototype_imb20 --use-parts --ablate-adaptive-capacity
```

#### bacon_air_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_air_Prototype_Tailaware_imb20 --use-parts
```

#### bacon_air_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_air_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_air_Full_imb20
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 20 --exp-name-suffix bacon_air_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb50

#### bacon_air_Base_imb50
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_air_Base_imb50
```

#### bacon_air_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_air_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_air_Prototype_imb50
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_air_Prototype_imb50 --use-parts --ablate-adaptive-capacity
```

#### bacon_air_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_air_Prototype_Tailaware_imb50 --use-parts
```

#### bacon_air_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_air_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_air_Full_imb50
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 50 --exp-name-suffix bacon_air_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```

### Imbalance Ratio: imb100

#### bacon_air_Base_imb100
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_air_Base_imb100
```

#### bacon_air_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_air_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_air_Prototype_imb100
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_air_Prototype_imb100 --use-parts --ablate-adaptive-capacity
```

#### bacon_air_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_air_Prototype_Tailaware_imb100 --use-parts
```

#### bacon_air_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_air_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo
```

#### bacon_air_Full_imb100
```bash
modal run -d modal_train.py::launch --dataset-name aircraft --backbone dinov2-vitb14 --imb-ratio 100 --exp-name-suffix bacon_air_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo
```
