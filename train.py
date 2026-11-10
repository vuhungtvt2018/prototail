"""Unified entrypoint for ProtoTail GCD training (Local, Server, or Cloud).

Supports both:
  1. High-level preset mode:
     python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Base
     python train.py --baseline simgcd --dataset-name cub --imb-ratio 20 --config Full
  2. Direct YAML / argument mode:
     python train.py --config configs/experiment_simgcd.yaml --dataset-name cub --imb-ratio 20 ...
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

# Delegate to kaggle_train main runner which handles data discovery,
# config resolution, kmeans backend defaults, and warm-start automation.
from kaggle_train import main

if __name__ == "__main__":
    main()
