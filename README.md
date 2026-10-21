Từ thư mục ProtoTail-GCD

Chạy SimGCD thuần:
python -m prototail_gcd.train --config configs/experiment_simgcd.yaml --data-root CUB_200_2011" --splits-root data_uq_idxs_bacon"

Chạy BaCon thuần:
python -m prototail_gcd.train --config configs/experiment_bacon.yaml --data-root CUB_200_2011" --splits-root data_uq_idxs_bacon"

Chạy Prototail với SimGCD
python -m prototail_gcd.train --config configs/experiment_simgcd.yaml ` --data-root CUB_200_2011" --splits-root data_uq_idxs_bacon" --enable-prototail --run-name simgcd-prototail

Chạy Prototail với BaCon
python -m prototail_gcd.train --config configs/experiment_bacon.yaml ` --data-root CUB_200_2011" --splits-root data_uq_idxs_bacon" --enable-prototail --run-name bacon-prototail