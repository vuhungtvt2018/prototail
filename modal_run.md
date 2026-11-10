# MODAL RUN GUIDE — TOÀN BỘ CẤU HÌNH (MODAL CLOUD GPU — TẮT MÁY ĐI NGỦ)

> **CẬP NHẬT CẤU HÌNH FIX V4 CHUẨN KAGGLE (TỰ ĐỘNG)**:
> - **Entrypoint**: `modal run -d modal_train.py::launch ...`
> - **KMeans Backend**: Mặc định `sklearn_cpu` (k-means++ chuẩn).
> - **Tail-Aware Warm-Start**: Tự động bật Epoch 0 cho cấu hình `Prototype_Tailaware` và `Full`.
> - **Fire-and-forget (`-d` / `--detach`)**: Job chạy trên GPU Cloud (A10G/A100) độc lập, bạn có thể **tắt máy đi ngủ** ngay lập tức.
> - **Kiểm tra tiến độ**: `modal app list` hoặc `modal app logs prototail-train`.

---

# ===========================================================================
# MÔ HÌNH: SIMGCD
# ===========================================================================

## Dataset: CUB (`cub`)

### Imbalance Ratio: imb20

#### simgcd_cub_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Base
```

#### simgcd_cub_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Pseudo
```

#### simgcd_cub_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_cub_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 50 --config Base
```

#### simgcd_cub_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 50 --config Pseudo
```

#### simgcd_cub_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_cub_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 100 --config Base
```

#### simgcd_cub_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 100 --config Pseudo
```

#### simgcd_cub_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 100 --config Prototype
```

#### simgcd_cub_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_cub_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_cub_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cub --imb-ratio 100 --config Full
```

## Dataset: CARS (`cars`)

### Imbalance Ratio: imb20

#### simgcd_cars_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 20 --config Base
```

#### simgcd_cars_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 20 --config Pseudo
```

#### simgcd_cars_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 20 --config Prototype
```

#### simgcd_cars_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_cars_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_cars_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_cars_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 50 --config Base
```

#### simgcd_cars_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 50 --config Pseudo
```

#### simgcd_cars_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 50 --config Prototype
```

#### simgcd_cars_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_cars_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_cars_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_cars_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 100 --config Base
```

#### simgcd_cars_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 100 --config Pseudo
```

#### simgcd_cars_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 100 --config Prototype
```

#### simgcd_cars_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_cars_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_cars_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name cars --imb-ratio 100 --config Full
```

## Dataset: AIRCRAFT (`aircraft`)

### Imbalance Ratio: imb20

#### simgcd_aircraft_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Base
```

#### simgcd_aircraft_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Pseudo
```

#### simgcd_aircraft_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Prototype
```

#### simgcd_aircraft_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_aircraft_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_aircraft_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_aircraft_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Base
```

#### simgcd_aircraft_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Pseudo
```

#### simgcd_aircraft_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Prototype
```

#### simgcd_aircraft_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_aircraft_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_aircraft_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_aircraft_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Base
```

#### simgcd_aircraft_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Pseudo
```

#### simgcd_aircraft_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Prototype
```

#### simgcd_aircraft_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_aircraft_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_aircraft_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name aircraft --imb-ratio 100 --config Full
```

## Dataset: HERBARIUM 19 (`herb19`)

### Imbalance Ratio: imb20

#### simgcd_herb19_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Base
```

#### simgcd_herb19_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Pseudo
```

#### simgcd_herb19_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Prototype
```

#### simgcd_herb19_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Prototype_Tailaware
```

#### simgcd_herb19_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Full_wo_Tail
```

#### simgcd_herb19_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### simgcd_herb19_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Base
```

#### simgcd_herb19_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Pseudo
```

#### simgcd_herb19_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Prototype
```

#### simgcd_herb19_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Prototype_Tailaware
```

#### simgcd_herb19_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Full_wo_Tail
```

#### simgcd_herb19_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### simgcd_herb19_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Base
```

#### simgcd_herb19_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Pseudo
```

#### simgcd_herb19_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Prototype
```

#### simgcd_herb19_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Prototype_Tailaware
```

#### simgcd_herb19_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Full_wo_Tail
```

#### simgcd_herb19_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name herb19 --imb-ratio 100 --config Full
```

## Dataset: INATURALIST 2018 (`inat2018`)

#### simgcd_inat2018_Base
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name inat2018 --config Base
```

#### simgcd_inat2018_Pseudo
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name inat2018 --config Pseudo
```

#### simgcd_inat2018_Prototype
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name inat2018 --config Prototype
```

#### simgcd_inat2018_Prototype_Tailaware
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name inat2018 --config Prototype_Tailaware
```

#### simgcd_inat2018_Full_wo_Tail
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name inat2018 --config Full_wo_Tail
```

#### simgcd_inat2018_Full
```bash
modal run -d modal_train.py::launch --baseline simgcd --dataset-name inat2018 --config Full
```

# ===========================================================================
# MÔ HÌNH: BACON
# ===========================================================================



## Dataset: CUB (`cub`)

### Imbalance Ratio: imb20

#### bacon_cub_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 20 --config Base
```

#### bacon_cub_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 20 --config Pseudo
```

#### bacon_cub_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 20 --config Prototype
```

#### bacon_cub_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_cub_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_cub_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_cub_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 50 --config Base
```

#### bacon_cub_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 50 --config Pseudo
```

#### bacon_cub_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 50 --config Prototype
```

#### bacon_cub_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_cub_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_cub_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_cub_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 100 --config Base
```

#### bacon_cub_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 100 --config Pseudo
```

#### bacon_cub_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 100 --config Prototype
```

#### bacon_cub_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_cub_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_cub_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cub --imb-ratio 100 --config Full
```

## Dataset: CARS (`cars`)

### Imbalance Ratio: imb20

#### bacon_cars_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 20 --config Base
```

#### bacon_cars_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 20 --config Pseudo
```

#### bacon_cars_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 20 --config Prototype
```

#### bacon_cars_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_cars_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_cars_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_cars_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 50 --config Base
```

#### bacon_cars_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 50 --config Pseudo
```

#### bacon_cars_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 50 --config Prototype
```

#### bacon_cars_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_cars_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_cars_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_cars_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 100 --config Base
```

#### bacon_cars_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 100 --config Pseudo
```

#### bacon_cars_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 100 --config Prototype
```

#### bacon_cars_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_cars_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_cars_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name cars --imb-ratio 100 --config Full
```

## Dataset: AIRCRAFT (`aircraft`)

### Imbalance Ratio: imb20

#### bacon_aircraft_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Base
```

#### bacon_aircraft_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Pseudo
```

#### bacon_aircraft_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Prototype
```

#### bacon_aircraft_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_aircraft_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_aircraft_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_aircraft_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Base
```

#### bacon_aircraft_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Pseudo
```

#### bacon_aircraft_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Prototype
```

#### bacon_aircraft_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_aircraft_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_aircraft_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_aircraft_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Base
```

#### bacon_aircraft_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Pseudo
```

#### bacon_aircraft_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Prototype
```

#### bacon_aircraft_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_aircraft_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_aircraft_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name aircraft --imb-ratio 100 --config Full
```

## Dataset: HERBARIUM 19 (`herb19`)

### Imbalance Ratio: imb20

#### bacon_herb19_Base_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Base
```

#### bacon_herb19_Pseudo_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Pseudo
```

#### bacon_herb19_Prototype_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Prototype
```

#### bacon_herb19_Prototype_Tailaware_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Prototype_Tailaware
```

#### bacon_herb19_Full_wo_Tail_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Full_wo_Tail
```

#### bacon_herb19_Full_imb20
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 20 --config Full
```

### Imbalance Ratio: imb50

#### bacon_herb19_Base_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Base
```

#### bacon_herb19_Pseudo_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Pseudo
```

#### bacon_herb19_Prototype_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Prototype
```

#### bacon_herb19_Prototype_Tailaware_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Prototype_Tailaware
```

#### bacon_herb19_Full_wo_Tail_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Full_wo_Tail
```

#### bacon_herb19_Full_imb50
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 50 --config Full
```

### Imbalance Ratio: imb100

#### bacon_herb19_Base_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Base
```

#### bacon_herb19_Pseudo_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Pseudo
```

#### bacon_herb19_Prototype_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Prototype
```

#### bacon_herb19_Prototype_Tailaware_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Prototype_Tailaware
```

#### bacon_herb19_Full_wo_Tail_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Full_wo_Tail
```

#### bacon_herb19_Full_imb100
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Full
```

## Dataset: INATURALIST 2018 (`inat2018`)

#### bacon_inat2018_Base
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name inat2018 --config Base
```

#### bacon_inat2018_Pseudo
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name inat2018 --config Pseudo
```

#### bacon_inat2018_Prototype
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name inat2018 --config Prototype
```

#### bacon_inat2018_Prototype_Tailaware
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name inat2018 --config Prototype_Tailaware
```

#### bacon_inat2018_Full_wo_Tail
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name inat2018 --config Full_wo_Tail
```

#### bacon_inat2018_Full
```bash
modal run -d modal_train.py::launch --baseline bacon --dataset-name inat2018 --config Full
```
