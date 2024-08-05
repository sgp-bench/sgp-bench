accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --data_file instruction_tuning_svg_mini_10k.json --block_size 64 --n_butterfly_factor 1


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --data_file instruction_tuning_svg_mini_25k.json --block_size 64 --n_butterfly_factor 1


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --data_file instruction_tuning_svg_mini_40k.json --block_size 64 --n_butterfly_factor 1


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_oft.py --data_file instruction_tuning_svg_mini_55k.json --block_size 64 --n_butterfly_factor 1


python hold.py
