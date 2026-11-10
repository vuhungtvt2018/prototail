# SERVER RUN GUIDE — TOÀN BỘ CẤU HÌNH (LINUX / GPU SERVER / SSH)

> **CẬP NHẬT CẤU HÌNH FIX V4 CHUẨN KAGGLE (TỰ ĐỘNG)**:
> - **Entrypoint**: Gọi trực tiếp `train.py` tại thư mục repo.
> - **KMeans Backend**: Tự động đặt mặc định là `sklearn_cpu` (k-means++ chuẩn bài báo).
> - **Tail-Aware Warm-Start**: Tự động kích hoạt Epoch 0 cho `Prototype_Tailaware` và `Full`.
> - **Quy trình chạy ngầm**: Sử dụng `nohup python train.py ... > outputs/logs/<exp_name>.log 2>&1 &`.
>   Sau khi kích hoạt lệnh, bạn có thể **tắt SSH / tắt máy tính đi ngủ**, server vẫn chạy độc lập.
> - **Kiểm tra tiến độ**: `nvidia-smi` hoặc `tail -f outputs/logs/<exp_name>.log`.

---

# ===========================================================================
# MÔ HÌNH: SIMGCD
# ===========================================================================

## Dataset: CUB (`cub`)

### Imbalance Ratio: imb20

#### simgcd_cub_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Base > outputs/logs/simgcd_cub_Base_imb20.log 2>&1 &
```

#### simgcd_cub_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Pseudo > outputs/logs/simgcd_cub_Pseudo_imb20.log 2>&1 &
```

#### simgcd_cub_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype > outputs/logs/simgcd_cub_Prototype_imb20.log 2>&1 &
```

#### simgcd_cub_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/simgcd_cub_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_cub_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full_wo_Tail > outputs/logs/simgcd_cub_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_cub_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full > outputs/logs/simgcd_cub_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_cub_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Base > outputs/logs/simgcd_cub_Base_imb50.log 2>&1 &
```

#### simgcd_cub_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Pseudo > outputs/logs/simgcd_cub_Pseudo_imb50.log 2>&1 &
```

#### simgcd_cub_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype > outputs/logs/simgcd_cub_Prototype_imb50.log 2>&1 &
```

#### simgcd_cub_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/simgcd_cub_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_cub_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Full_wo_Tail > outputs/logs/simgcd_cub_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_cub_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 50 --config Full > outputs/logs/simgcd_cub_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_cub_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Base > outputs/logs/simgcd_cub_Base_imb100.log 2>&1 &
```

#### simgcd_cub_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Pseudo > outputs/logs/simgcd_cub_Pseudo_imb100.log 2>&1 &
```

#### simgcd_cub_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Prototype > outputs/logs/simgcd_cub_Prototype_imb100.log 2>&1 &
```

#### simgcd_cub_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/simgcd_cub_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_cub_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Full_wo_Tail > outputs/logs/simgcd_cub_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_cub_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cub --imb-ratio 100 --config Full > outputs/logs/simgcd_cub_Full_imb100.log 2>&1 &
```

## Dataset: CARS (`cars`)

### Imbalance Ratio: imb20

#### simgcd_cars_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Base > outputs/logs/simgcd_cars_Base_imb20.log 2>&1 &
```

#### simgcd_cars_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Pseudo > outputs/logs/simgcd_cars_Pseudo_imb20.log 2>&1 &
```

#### simgcd_cars_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Prototype > outputs/logs/simgcd_cars_Prototype_imb20.log 2>&1 &
```

#### simgcd_cars_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/simgcd_cars_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_cars_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Full_wo_Tail > outputs/logs/simgcd_cars_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_cars_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 20 --config Full > outputs/logs/simgcd_cars_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_cars_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Base > outputs/logs/simgcd_cars_Base_imb50.log 2>&1 &
```

#### simgcd_cars_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Pseudo > outputs/logs/simgcd_cars_Pseudo_imb50.log 2>&1 &
```

#### simgcd_cars_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Prototype > outputs/logs/simgcd_cars_Prototype_imb50.log 2>&1 &
```

#### simgcd_cars_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/simgcd_cars_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_cars_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Full_wo_Tail > outputs/logs/simgcd_cars_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_cars_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 50 --config Full > outputs/logs/simgcd_cars_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_cars_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Base > outputs/logs/simgcd_cars_Base_imb100.log 2>&1 &
```

#### simgcd_cars_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Pseudo > outputs/logs/simgcd_cars_Pseudo_imb100.log 2>&1 &
```

#### simgcd_cars_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Prototype > outputs/logs/simgcd_cars_Prototype_imb100.log 2>&1 &
```

#### simgcd_cars_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/simgcd_cars_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_cars_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Full_wo_Tail > outputs/logs/simgcd_cars_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_cars_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name cars --imb-ratio 100 --config Full > outputs/logs/simgcd_cars_Full_imb100.log 2>&1 &
```

## Dataset: AIRCRAFT (`aircraft`)

### Imbalance Ratio: imb20

#### simgcd_aircraft_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Base > outputs/logs/simgcd_aircraft_Base_imb20.log 2>&1 &
```

#### simgcd_aircraft_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Pseudo > outputs/logs/simgcd_aircraft_Pseudo_imb20.log 2>&1 &
```

#### simgcd_aircraft_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Prototype > outputs/logs/simgcd_aircraft_Prototype_imb20.log 2>&1 &
```

#### simgcd_aircraft_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/simgcd_aircraft_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_aircraft_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Full_wo_Tail > outputs/logs/simgcd_aircraft_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_aircraft_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Full > outputs/logs/simgcd_aircraft_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_aircraft_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Base > outputs/logs/simgcd_aircraft_Base_imb50.log 2>&1 &
```

#### simgcd_aircraft_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Pseudo > outputs/logs/simgcd_aircraft_Pseudo_imb50.log 2>&1 &
```

#### simgcd_aircraft_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Prototype > outputs/logs/simgcd_aircraft_Prototype_imb50.log 2>&1 &
```

#### simgcd_aircraft_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/simgcd_aircraft_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_aircraft_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Full_wo_Tail > outputs/logs/simgcd_aircraft_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_aircraft_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Full > outputs/logs/simgcd_aircraft_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_aircraft_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Base > outputs/logs/simgcd_aircraft_Base_imb100.log 2>&1 &
```

#### simgcd_aircraft_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Pseudo > outputs/logs/simgcd_aircraft_Pseudo_imb100.log 2>&1 &
```

#### simgcd_aircraft_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Prototype > outputs/logs/simgcd_aircraft_Prototype_imb100.log 2>&1 &
```

#### simgcd_aircraft_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/simgcd_aircraft_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_aircraft_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Full_wo_Tail > outputs/logs/simgcd_aircraft_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_aircraft_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Full > outputs/logs/simgcd_aircraft_Full_imb100.log 2>&1 &
```

## Dataset: HERBARIUM 19 (`herb19`)

### Imbalance Ratio: imb20

#### simgcd_herb19_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Base > outputs/logs/simgcd_herb19_Base_imb20.log 2>&1 &
```

#### simgcd_herb19_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Pseudo > outputs/logs/simgcd_herb19_Pseudo_imb20.log 2>&1 &
```

#### simgcd_herb19_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Prototype > outputs/logs/simgcd_herb19_Prototype_imb20.log 2>&1 &
```

#### simgcd_herb19_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/simgcd_herb19_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_herb19_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Full_wo_Tail > outputs/logs/simgcd_herb19_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_herb19_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Full > outputs/logs/simgcd_herb19_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_herb19_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Base > outputs/logs/simgcd_herb19_Base_imb50.log 2>&1 &
```

#### simgcd_herb19_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Pseudo > outputs/logs/simgcd_herb19_Pseudo_imb50.log 2>&1 &
```

#### simgcd_herb19_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Prototype > outputs/logs/simgcd_herb19_Prototype_imb50.log 2>&1 &
```

#### simgcd_herb19_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/simgcd_herb19_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_herb19_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Full_wo_Tail > outputs/logs/simgcd_herb19_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_herb19_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Full > outputs/logs/simgcd_herb19_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_herb19_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Base > outputs/logs/simgcd_herb19_Base_imb100.log 2>&1 &
```

#### simgcd_herb19_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Pseudo > outputs/logs/simgcd_herb19_Pseudo_imb100.log 2>&1 &
```

#### simgcd_herb19_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Prototype > outputs/logs/simgcd_herb19_Prototype_imb100.log 2>&1 &
```

#### simgcd_herb19_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/simgcd_herb19_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_herb19_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Full_wo_Tail > outputs/logs/simgcd_herb19_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_herb19_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Full > outputs/logs/simgcd_herb19_Full_imb100.log 2>&1 &
```

## Dataset: INATURALIST 2018 (`inat2018`)

### Imbalance Ratio: imb20

#### simgcd_inat2018_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 20 --config Base > outputs/logs/simgcd_inat2018_Base_imb20.log 2>&1 &
```

#### simgcd_inat2018_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 20 --config Pseudo > outputs/logs/simgcd_inat2018_Pseudo_imb20.log 2>&1 &
```

#### simgcd_inat2018_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 20 --config Prototype > outputs/logs/simgcd_inat2018_Prototype_imb20.log 2>&1 &
```

#### simgcd_inat2018_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/simgcd_inat2018_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_inat2018_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 20 --config Full_wo_Tail > outputs/logs/simgcd_inat2018_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_inat2018_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 20 --config Full > outputs/logs/simgcd_inat2018_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_inat2018_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 50 --config Base > outputs/logs/simgcd_inat2018_Base_imb50.log 2>&1 &
```

#### simgcd_inat2018_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 50 --config Pseudo > outputs/logs/simgcd_inat2018_Pseudo_imb50.log 2>&1 &
```

#### simgcd_inat2018_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 50 --config Prototype > outputs/logs/simgcd_inat2018_Prototype_imb50.log 2>&1 &
```

#### simgcd_inat2018_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/simgcd_inat2018_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_inat2018_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 50 --config Full_wo_Tail > outputs/logs/simgcd_inat2018_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_inat2018_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 50 --config Full > outputs/logs/simgcd_inat2018_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_inat2018_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 100 --config Base > outputs/logs/simgcd_inat2018_Base_imb100.log 2>&1 &
```

#### simgcd_inat2018_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 100 --config Pseudo > outputs/logs/simgcd_inat2018_Pseudo_imb100.log 2>&1 &
```

#### simgcd_inat2018_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 100 --config Prototype > outputs/logs/simgcd_inat2018_Prototype_imb100.log 2>&1 &
```

#### simgcd_inat2018_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/simgcd_inat2018_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_inat2018_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 100 --config Full_wo_Tail > outputs/logs/simgcd_inat2018_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_inat2018_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio 100 --config Full > outputs/logs/simgcd_inat2018_Full_imb100.log 2>&1 &
```

### Imbalance Ratio: natural

#### simgcd_inat2018_Base_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio natural --config Base > outputs/logs/simgcd_inat2018_Base_natural.log 2>&1 &
```

#### simgcd_inat2018_Pseudo_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio natural --config Pseudo > outputs/logs/simgcd_inat2018_Pseudo_natural.log 2>&1 &
```

#### simgcd_inat2018_Prototype_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio natural --config Prototype > outputs/logs/simgcd_inat2018_Prototype_natural.log 2>&1 &
```

#### simgcd_inat2018_Prototype_Tailaware_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio natural --config Prototype_Tailaware > outputs/logs/simgcd_inat2018_Prototype_Tailaware_natural.log 2>&1 &
```

#### simgcd_inat2018_Full_wo_Tail_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio natural --config Full_wo_Tail > outputs/logs/simgcd_inat2018_Full_wo_Tail_natural.log 2>&1 &
```

#### simgcd_inat2018_Full_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline simgcd --dataset-name inat2018 --imb-ratio natural --config Full > outputs/logs/simgcd_inat2018_Full_natural.log 2>&1 &
```

# ===========================================================================
# MÔ HÌNH: BACON
# ===========================================================================

## Dataset: CUB (`cub`)

### Imbalance Ratio: imb20

#### bacon_cub_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Base > outputs/logs/bacon_cub_Base_imb20.log 2>&1 &
```

#### bacon_cub_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Pseudo > outputs/logs/bacon_cub_Pseudo_imb20.log 2>&1 &
```

#### bacon_cub_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Prototype > outputs/logs/bacon_cub_Prototype_imb20.log 2>&1 &
```

#### bacon_cub_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/bacon_cub_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_cub_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Full_wo_Tail > outputs/logs/bacon_cub_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_cub_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 20 --config Full > outputs/logs/bacon_cub_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_cub_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Base > outputs/logs/bacon_cub_Base_imb50.log 2>&1 &
```

#### bacon_cub_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Pseudo > outputs/logs/bacon_cub_Pseudo_imb50.log 2>&1 &
```

#### bacon_cub_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Prototype > outputs/logs/bacon_cub_Prototype_imb50.log 2>&1 &
```

#### bacon_cub_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/bacon_cub_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_cub_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Full_wo_Tail > outputs/logs/bacon_cub_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_cub_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 50 --config Full > outputs/logs/bacon_cub_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_cub_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Base > outputs/logs/bacon_cub_Base_imb100.log 2>&1 &
```

#### bacon_cub_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Pseudo > outputs/logs/bacon_cub_Pseudo_imb100.log 2>&1 &
```

#### bacon_cub_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Prototype > outputs/logs/bacon_cub_Prototype_imb100.log 2>&1 &
```

#### bacon_cub_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/bacon_cub_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_cub_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Full_wo_Tail > outputs/logs/bacon_cub_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_cub_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cub --imb-ratio 100 --config Full > outputs/logs/bacon_cub_Full_imb100.log 2>&1 &
```

## Dataset: CARS (`cars`)

### Imbalance Ratio: imb20

#### bacon_cars_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Base > outputs/logs/bacon_cars_Base_imb20.log 2>&1 &
```

#### bacon_cars_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Pseudo > outputs/logs/bacon_cars_Pseudo_imb20.log 2>&1 &
```

#### bacon_cars_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Prototype > outputs/logs/bacon_cars_Prototype_imb20.log 2>&1 &
```

#### bacon_cars_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/bacon_cars_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_cars_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Full_wo_Tail > outputs/logs/bacon_cars_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_cars_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 20 --config Full > outputs/logs/bacon_cars_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_cars_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Base > outputs/logs/bacon_cars_Base_imb50.log 2>&1 &
```

#### bacon_cars_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Pseudo > outputs/logs/bacon_cars_Pseudo_imb50.log 2>&1 &
```

#### bacon_cars_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Prototype > outputs/logs/bacon_cars_Prototype_imb50.log 2>&1 &
```

#### bacon_cars_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/bacon_cars_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_cars_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Full_wo_Tail > outputs/logs/bacon_cars_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_cars_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Full > outputs/logs/bacon_cars_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_cars_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Base > outputs/logs/bacon_cars_Base_imb100.log 2>&1 &
```

#### bacon_cars_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Pseudo > outputs/logs/bacon_cars_Pseudo_imb100.log 2>&1 &
```

#### bacon_cars_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Prototype > outputs/logs/bacon_cars_Prototype_imb100.log 2>&1 &
```

#### bacon_cars_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/bacon_cars_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_cars_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Full_wo_Tail > outputs/logs/bacon_cars_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_cars_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name cars --imb-ratio 100 --config Full > outputs/logs/bacon_cars_Full_imb100.log 2>&1 &
```

## Dataset: AIRCRAFT (`aircraft`)

### Imbalance Ratio: imb20

#### bacon_aircraft_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Base > outputs/logs/bacon_aircraft_Base_imb20.log 2>&1 &
```

#### bacon_aircraft_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Pseudo > outputs/logs/bacon_aircraft_Pseudo_imb20.log 2>&1 &
```

#### bacon_aircraft_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Prototype > outputs/logs/bacon_aircraft_Prototype_imb20.log 2>&1 &
```

#### bacon_aircraft_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/bacon_aircraft_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_aircraft_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Full_wo_Tail > outputs/logs/bacon_aircraft_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_aircraft_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Full > outputs/logs/bacon_aircraft_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_aircraft_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Base > outputs/logs/bacon_aircraft_Base_imb50.log 2>&1 &
```

#### bacon_aircraft_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Pseudo > outputs/logs/bacon_aircraft_Pseudo_imb50.log 2>&1 &
```

#### bacon_aircraft_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Prototype > outputs/logs/bacon_aircraft_Prototype_imb50.log 2>&1 &
```

#### bacon_aircraft_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/bacon_aircraft_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_aircraft_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Full_wo_Tail > outputs/logs/bacon_aircraft_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_aircraft_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Full > outputs/logs/bacon_aircraft_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_aircraft_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Base > outputs/logs/bacon_aircraft_Base_imb100.log 2>&1 &
```

#### bacon_aircraft_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Pseudo > outputs/logs/bacon_aircraft_Pseudo_imb100.log 2>&1 &
```

#### bacon_aircraft_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Prototype > outputs/logs/bacon_aircraft_Prototype_imb100.log 2>&1 &
```

#### bacon_aircraft_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/bacon_aircraft_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_aircraft_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Full_wo_Tail > outputs/logs/bacon_aircraft_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_aircraft_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Full > outputs/logs/bacon_aircraft_Full_imb100.log 2>&1 &
```

## Dataset: HERBARIUM 19 (`herb19`)

### Imbalance Ratio: imb20

#### bacon_herb19_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Base > outputs/logs/bacon_herb19_Base_imb20.log 2>&1 &
```

#### bacon_herb19_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Pseudo > outputs/logs/bacon_herb19_Pseudo_imb20.log 2>&1 &
```

#### bacon_herb19_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Prototype > outputs/logs/bacon_herb19_Prototype_imb20.log 2>&1 &
```

#### bacon_herb19_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/bacon_herb19_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_herb19_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Full_wo_Tail > outputs/logs/bacon_herb19_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_herb19_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Full > outputs/logs/bacon_herb19_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_herb19_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Base > outputs/logs/bacon_herb19_Base_imb50.log 2>&1 &
```

#### bacon_herb19_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Pseudo > outputs/logs/bacon_herb19_Pseudo_imb50.log 2>&1 &
```

#### bacon_herb19_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Prototype > outputs/logs/bacon_herb19_Prototype_imb50.log 2>&1 &
```

#### bacon_herb19_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/bacon_herb19_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_herb19_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Full_wo_Tail > outputs/logs/bacon_herb19_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_herb19_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Full > outputs/logs/bacon_herb19_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_herb19_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Base > outputs/logs/bacon_herb19_Base_imb100.log 2>&1 &
```

#### bacon_herb19_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Pseudo > outputs/logs/bacon_herb19_Pseudo_imb100.log 2>&1 &
```

#### bacon_herb19_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Prototype > outputs/logs/bacon_herb19_Prototype_imb100.log 2>&1 &
```

#### bacon_herb19_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/bacon_herb19_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_herb19_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Full_wo_Tail > outputs/logs/bacon_herb19_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_herb19_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Full > outputs/logs/bacon_herb19_Full_imb100.log 2>&1 &
```

## Dataset: INATURALIST 2018 (`inat2018`)

### Imbalance Ratio: imb20

#### bacon_inat2018_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 20 --config Base > outputs/logs/bacon_inat2018_Base_imb20.log 2>&1 &
```

#### bacon_inat2018_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 20 --config Pseudo > outputs/logs/bacon_inat2018_Pseudo_imb20.log 2>&1 &
```

#### bacon_inat2018_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 20 --config Prototype > outputs/logs/bacon_inat2018_Prototype_imb20.log 2>&1 &
```

#### bacon_inat2018_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 20 --config Prototype_Tailaware > outputs/logs/bacon_inat2018_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_inat2018_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 20 --config Full_wo_Tail > outputs/logs/bacon_inat2018_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_inat2018_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 20 --config Full > outputs/logs/bacon_inat2018_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_inat2018_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 50 --config Base > outputs/logs/bacon_inat2018_Base_imb50.log 2>&1 &
```

#### bacon_inat2018_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 50 --config Pseudo > outputs/logs/bacon_inat2018_Pseudo_imb50.log 2>&1 &
```

#### bacon_inat2018_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 50 --config Prototype > outputs/logs/bacon_inat2018_Prototype_imb50.log 2>&1 &
```

#### bacon_inat2018_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 50 --config Prototype_Tailaware > outputs/logs/bacon_inat2018_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_inat2018_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 50 --config Full_wo_Tail > outputs/logs/bacon_inat2018_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_inat2018_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 50 --config Full > outputs/logs/bacon_inat2018_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_inat2018_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 100 --config Base > outputs/logs/bacon_inat2018_Base_imb100.log 2>&1 &
```

#### bacon_inat2018_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 100 --config Pseudo > outputs/logs/bacon_inat2018_Pseudo_imb100.log 2>&1 &
```

#### bacon_inat2018_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 100 --config Prototype > outputs/logs/bacon_inat2018_Prototype_imb100.log 2>&1 &
```

#### bacon_inat2018_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 100 --config Prototype_Tailaware > outputs/logs/bacon_inat2018_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_inat2018_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 100 --config Full_wo_Tail > outputs/logs/bacon_inat2018_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_inat2018_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio 100 --config Full > outputs/logs/bacon_inat2018_Full_imb100.log 2>&1 &
```

### Imbalance Ratio: natural

#### bacon_inat2018_Base_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio natural --config Base > outputs/logs/bacon_inat2018_Base_natural.log 2>&1 &
```

#### bacon_inat2018_Pseudo_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio natural --config Pseudo > outputs/logs/bacon_inat2018_Pseudo_natural.log 2>&1 &
```

#### bacon_inat2018_Prototype_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio natural --config Prototype > outputs/logs/bacon_inat2018_Prototype_natural.log 2>&1 &
```

#### bacon_inat2018_Prototype_Tailaware_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio natural --config Prototype_Tailaware > outputs/logs/bacon_inat2018_Prototype_Tailaware_natural.log 2>&1 &
```

#### bacon_inat2018_Full_wo_Tail_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio natural --config Full_wo_Tail > outputs/logs/bacon_inat2018_Full_wo_Tail_natural.log 2>&1 &
```

#### bacon_inat2018_Full_natural
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --baseline bacon --dataset-name inat2018 --imb-ratio natural --config Full > outputs/logs/bacon_inat2018_Full_natural.log 2>&1 &
```
