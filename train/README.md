# SIT (Symbolic Instruction Tuning)

## Overview
Inspired by how visual instruction tuning enables large vision-language models (e.g., LLaVa) to understand images with visual-question-answering (VQA) data, we aim to perform symbolic instruction tuning for LLMs to better bridge the gap between the semantic understanding and the symbolic reasoning within the graphics programs. We provide our generated semantic instruction-following datasets over symbolic programs and training scripts for performing Symbolic Instruction Tuning (SIT). For more information, please refer to our paper.


## Prerequisites

Some additional python packages has to be installed before performing evaluation.
```bash
pip install trl
pip install peft
pip install accelerate
pip install transformers==4.44.0
```

## Data

You can download the symbolic instruction tuning dataset from the huggingface dataset [SIT Data](https://huggingface.co/sgp-bench).


## Symbolic Instruction Tuning

### Running the training
Run the following bash scripts to finetune the base LLM model for SIT.
```bash
bash sft_lora.sh
```
or
```bash
bash sft_oft.sh
```

### Evaluation
After symbolic instruction tuning, the base model with merged adapter weights is saved in the `checkpoint` directory. 

1. **Run the evaluation on sgp-bench svg** to obtain the model responses (`*_query.json` file) by following the instructions for evaluating open-sourced LLM.

2. **Perform LLM-based evaluation** (see `evaluation` directory) to reproduce the results reported in the paper.


## Acknowledgment
The code and methodology for this evaluation are adapted from the following GitHub project: [trl](https://github.com/huggingface/trl)