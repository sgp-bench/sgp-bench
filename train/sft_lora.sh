accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_lora.py --data_file instruction_tuning_svg_mini_10k.json


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_lora.py --data_file instruction_tuning_svg_mini_25k.json


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_lora.py --data_file instruction_tuning_svg_mini_40k.json


accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 8 sft_lora.py --data_file instruction_tuning_svg_mini_55k.json


python hold.py
