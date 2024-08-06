accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --dataset_name sgp-bench/sit_10k


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --dataset_name sgp-bench/sit_25k


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --dataset_name sgp-bench/sit_40k


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --dataset_name sgp-bench/sit_55k
