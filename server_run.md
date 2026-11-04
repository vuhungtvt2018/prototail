# SERVER RUN GUIDE — 108 CẤU HÌNH (TẮT MÁY ĐI NGỦ VẪN CHẠY)

> **Cách chạy ngầm trên Server (Linux/SSH)**:
> Mọi lệnh đều sử dụng `nohup ... > <log_file> 2>&1 &`.
> Khi lệnh được kích hoạt, tiến trình sẽ tách hoàn toàn khỏi phiên SSH (decoupled from terminal).
> Bạn có thể **exit SSH, đóng terminal, tắt máy tính đi ngủ**, server vẫn chạy liên tục cho đến khi hoàn thành.
> Để kiểm tra tiến trình đang chạy: `ps aux | grep python` hoặc `nvidia-smi`.
> Để xem tiến độ log theo thời gian thực: `tail -f outputs/logs/<exp_name>.log`.
> *(Lưu ý: Trên Server chạy bằng Python thường, KHÔNG dùng `::launch`)*

---

# ===========================================================================
# MÔ HÌNH: SIMGCD
# ===========================================================================

## Dataset: CUB (cub)

### Imbalance Ratio: imb20

#### simgcd_cub_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_cub_Base_imb20 > outputs/logs/simgcd_cub_Base_imb20.log 2>&1 &
```

#### simgcd_cub_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_cub_Pseudo_imb20 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Pseudo_imb20.log 2>&1 &
```

#### simgcd_cub_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_cub_Prototype_imb20 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_cub_Prototype_imb20.log 2>&1 &
```

#### simgcd_cub_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_cub_Prototype_Tailaware_imb20 --use_parts > outputs/logs/simgcd_cub_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_cub_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_cub_Full_wo_Tail_imb20 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_cub_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_cub_Full_imb20 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_cub_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_cub_Base_imb50 > outputs/logs/simgcd_cub_Base_imb50.log 2>&1 &
```

#### simgcd_cub_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_cub_Pseudo_imb50 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Pseudo_imb50.log 2>&1 &
```

#### simgcd_cub_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_cub_Prototype_imb50 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_cub_Prototype_imb50.log 2>&1 &
```

#### simgcd_cub_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_cub_Prototype_Tailaware_imb50 --use_parts > outputs/logs/simgcd_cub_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_cub_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_cub_Full_wo_Tail_imb50 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_cub_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_cub_Full_imb50 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_cub_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_cub_Base_imb100 > outputs/logs/simgcd_cub_Base_imb100.log 2>&1 &
```

#### simgcd_cub_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_cub_Pseudo_imb100 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Pseudo_imb100.log 2>&1 &
```

#### simgcd_cub_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_cub_Prototype_imb100 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_cub_Prototype_imb100.log 2>&1 &
```

#### simgcd_cub_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_cub_Prototype_Tailaware_imb100 --use_parts > outputs/logs/simgcd_cub_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_cub_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_cub_Full_wo_Tail_imb100 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_cub_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_cub_Full_imb100 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_cub_Full_imb100.log 2>&1 &
```

## Dataset: CAR (scars)

### Imbalance Ratio: imb20

#### simgcd_car_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_car_Base_imb20 > outputs/logs/simgcd_car_Base_imb20.log 2>&1 &
```

#### simgcd_car_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_car_Pseudo_imb20 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Pseudo_imb20.log 2>&1 &
```

#### simgcd_car_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_car_Prototype_imb20 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_car_Prototype_imb20.log 2>&1 &
```

#### simgcd_car_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_car_Prototype_Tailaware_imb20 --use_parts > outputs/logs/simgcd_car_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_car_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_car_Full_wo_Tail_imb20 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_car_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_car_Full_imb20 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_car_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_car_Base_imb50 > outputs/logs/simgcd_car_Base_imb50.log 2>&1 &
```

#### simgcd_car_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_car_Pseudo_imb50 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Pseudo_imb50.log 2>&1 &
```

#### simgcd_car_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_car_Prototype_imb50 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_car_Prototype_imb50.log 2>&1 &
```

#### simgcd_car_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_car_Prototype_Tailaware_imb50 --use_parts > outputs/logs/simgcd_car_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_car_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_car_Full_wo_Tail_imb50 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_car_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_car_Full_imb50 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_car_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_car_Base_imb100 > outputs/logs/simgcd_car_Base_imb100.log 2>&1 &
```

#### simgcd_car_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_car_Pseudo_imb100 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Pseudo_imb100.log 2>&1 &
```

#### simgcd_car_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_car_Prototype_imb100 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_car_Prototype_imb100.log 2>&1 &
```

#### simgcd_car_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_car_Prototype_Tailaware_imb100 --use_parts > outputs/logs/simgcd_car_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_car_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_car_Full_wo_Tail_imb100 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_car_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name scars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_car_Full_imb100 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_car_Full_imb100.log 2>&1 &
```

## Dataset: AIR (aircraft)

### Imbalance Ratio: imb20

#### simgcd_air_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_air_Base_imb20 > outputs/logs/simgcd_air_Base_imb20.log 2>&1 &
```

#### simgcd_air_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_air_Pseudo_imb20 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Pseudo_imb20.log 2>&1 &
```

#### simgcd_air_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_air_Prototype_imb20 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_air_Prototype_imb20.log 2>&1 &
```

#### simgcd_air_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_air_Prototype_Tailaware_imb20 --use_parts > outputs/logs/simgcd_air_Prototype_Tailaware_imb20.log 2>&1 &
```

#### simgcd_air_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_air_Full_wo_Tail_imb20 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Full_wo_Tail_imb20.log 2>&1 &
```

#### simgcd_air_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name simgcd_air_Full_imb20 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### simgcd_air_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_air_Base_imb50 > outputs/logs/simgcd_air_Base_imb50.log 2>&1 &
```

#### simgcd_air_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_air_Pseudo_imb50 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Pseudo_imb50.log 2>&1 &
```

#### simgcd_air_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_air_Prototype_imb50 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_air_Prototype_imb50.log 2>&1 &
```

#### simgcd_air_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_air_Prototype_Tailaware_imb50 --use_parts > outputs/logs/simgcd_air_Prototype_Tailaware_imb50.log 2>&1 &
```

#### simgcd_air_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_air_Full_wo_Tail_imb50 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Full_wo_Tail_imb50.log 2>&1 &
```

#### simgcd_air_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name simgcd_air_Full_imb50 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### simgcd_air_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_air_Base_imb100 > outputs/logs/simgcd_air_Base_imb100.log 2>&1 &
```

#### simgcd_air_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_air_Pseudo_imb100 --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Pseudo_imb100.log 2>&1 &
```

#### simgcd_air_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_air_Prototype_imb100 --use_parts --ablate_adaptive_capacity > outputs/logs/simgcd_air_Prototype_imb100.log 2>&1 &
```

#### simgcd_air_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_air_Prototype_Tailaware_imb100 --use_parts > outputs/logs/simgcd_air_Prototype_Tailaware_imb100.log 2>&1 &
```

#### simgcd_air_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_air_Full_wo_Tail_imb100 --use_parts --ablate_adaptive_capacity --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Full_wo_Tail_imb100.log 2>&1 &
```

#### simgcd_air_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name simgcd_air_Full_imb100 --use_parts --enable_pseudo_labeling --enable_novel_pseudo > outputs/logs/simgcd_air_Full_imb100.log 2>&1 &
```

# ===========================================================================
# MÔ HÌNH: BACON
# ===========================================================================

## Dataset: CUB (cub200)

### Imbalance Ratio: imb20

#### bacon_cub_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_cub_Base_imb20 > outputs/logs/bacon_cub_Base_imb20.log 2>&1 &
```

#### bacon_cub_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_cub_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Pseudo_imb20.log 2>&1 &
```

#### bacon_cub_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_cub_Prototype_imb20 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_cub_Prototype_imb20.log 2>&1 &
```

#### bacon_cub_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_cub_Prototype_Tailaware_imb20 --use-parts > outputs/logs/bacon_cub_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_cub_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_cub_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_cub_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_cub_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_cub_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_cub_Base_imb50 > outputs/logs/bacon_cub_Base_imb50.log 2>&1 &
```

#### bacon_cub_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_cub_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Pseudo_imb50.log 2>&1 &
```

#### bacon_cub_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_cub_Prototype_imb50 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_cub_Prototype_imb50.log 2>&1 &
```

#### bacon_cub_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_cub_Prototype_Tailaware_imb50 --use-parts > outputs/logs/bacon_cub_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_cub_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_cub_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_cub_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_cub_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_cub_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_cub_Base_imb100 > outputs/logs/bacon_cub_Base_imb100.log 2>&1 &
```

#### bacon_cub_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_cub_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Pseudo_imb100.log 2>&1 &
```

#### bacon_cub_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_cub_Prototype_imb100 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_cub_Prototype_imb100.log 2>&1 &
```

#### bacon_cub_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_cub_Prototype_Tailaware_imb100 --use-parts > outputs/logs/bacon_cub_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_cub_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_cub_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_cub_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name cub200 --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_cub_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_cub_Full_imb100.log 2>&1 &
```

## Dataset: CAR (stanford_cars)

### Imbalance Ratio: imb20

#### bacon_car_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_car_Base_imb20 > outputs/logs/bacon_car_Base_imb20.log 2>&1 &
```

#### bacon_car_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_car_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Pseudo_imb20.log 2>&1 &
```

#### bacon_car_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_car_Prototype_imb20 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_car_Prototype_imb20.log 2>&1 &
```

#### bacon_car_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_car_Prototype_Tailaware_imb20 --use-parts > outputs/logs/bacon_car_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_car_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_car_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_car_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_car_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_car_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_car_Base_imb50 > outputs/logs/bacon_car_Base_imb50.log 2>&1 &
```

#### bacon_car_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_car_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Pseudo_imb50.log 2>&1 &
```

#### bacon_car_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_car_Prototype_imb50 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_car_Prototype_imb50.log 2>&1 &
```

#### bacon_car_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_car_Prototype_Tailaware_imb50 --use-parts > outputs/logs/bacon_car_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_car_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_car_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_car_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_car_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_car_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_car_Base_imb100 > outputs/logs/bacon_car_Base_imb100.log 2>&1 &
```

#### bacon_car_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_car_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Pseudo_imb100.log 2>&1 &
```

#### bacon_car_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_car_Prototype_imb100 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_car_Prototype_imb100.log 2>&1 &
```

#### bacon_car_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_car_Prototype_Tailaware_imb100 --use-parts > outputs/logs/bacon_car_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_car_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_car_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_car_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name stanford_cars --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_car_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_car_Full_imb100.log 2>&1 &
```

## Dataset: AIR (aircraft)

### Imbalance Ratio: imb20

#### bacon_air_Base_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_air_Base_imb20 > outputs/logs/bacon_air_Base_imb20.log 2>&1 &
```

#### bacon_air_Pseudo_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_air_Pseudo_imb20 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Pseudo_imb20.log 2>&1 &
```

#### bacon_air_Prototype_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_air_Prototype_imb20 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_air_Prototype_imb20.log 2>&1 &
```

#### bacon_air_Prototype_Tailaware_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_air_Prototype_Tailaware_imb20 --use-parts > outputs/logs/bacon_air_Prototype_Tailaware_imb20.log 2>&1 &
```

#### bacon_air_Full_wo_Tail_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_air_Full_wo_Tail_imb20 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Full_wo_Tail_imb20.log 2>&1 &
```

#### bacon_air_Full_imb20
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 20 --exp_name bacon_air_Full_imb20 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Full_imb20.log 2>&1 &
```

### Imbalance Ratio: imb50

#### bacon_air_Base_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_air_Base_imb50 > outputs/logs/bacon_air_Base_imb50.log 2>&1 &
```

#### bacon_air_Pseudo_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_air_Pseudo_imb50 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Pseudo_imb50.log 2>&1 &
```

#### bacon_air_Prototype_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_air_Prototype_imb50 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_air_Prototype_imb50.log 2>&1 &
```

#### bacon_air_Prototype_Tailaware_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_air_Prototype_Tailaware_imb50 --use-parts > outputs/logs/bacon_air_Prototype_Tailaware_imb50.log 2>&1 &
```

#### bacon_air_Full_wo_Tail_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_air_Full_wo_Tail_imb50 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Full_wo_Tail_imb50.log 2>&1 &
```

#### bacon_air_Full_imb50
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 50 --exp_name bacon_air_Full_imb50 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Full_imb50.log 2>&1 &
```

### Imbalance Ratio: imb100

#### bacon_air_Base_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_air_Base_imb100 > outputs/logs/bacon_air_Base_imb100.log 2>&1 &
```

#### bacon_air_Pseudo_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_air_Pseudo_imb100 --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Pseudo_imb100.log 2>&1 &
```

#### bacon_air_Prototype_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_air_Prototype_imb100 --use-parts --ablate-adaptive-capacity > outputs/logs/bacon_air_Prototype_imb100.log 2>&1 &
```

#### bacon_air_Prototype_Tailaware_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_air_Prototype_Tailaware_imb100 --use-parts > outputs/logs/bacon_air_Prototype_Tailaware_imb100.log 2>&1 &
```

#### bacon_air_Full_wo_Tail_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_air_Full_wo_Tail_imb100 --use-parts --ablate-adaptive-capacity --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Full_wo_Tail_imb100.log 2>&1 &
```

#### bacon_air_Full_imb100
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset_name aircraft --backbone dinov2_vitb14 --epochs 200 --batch_size 128 --imb_ratio 100 --exp_name bacon_air_Full_imb100 --use-parts --enable-pseudo-labeling --enable-novel-pseudo > outputs/logs/bacon_air_Full_imb100.log 2>&1 &
```
