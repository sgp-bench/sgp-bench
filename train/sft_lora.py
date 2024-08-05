# -----------------------------------------------------------------------------
# This file contains code borrowed and adapted from the following GitHub project:
# 
# Project: trl
# Repository: https://github.com/huggingface/trl
# File: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# 
# The original code is licensed under the Apache license. For more details, 
# please refer to the original project repository.
# -----------------------------------------------------------------------------


import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed

from trl import SFTTrainer, SFTConfig
from trl.trainer import ConstantLengthDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_file", type=str, default="alpaca.json", required=True)
    parser.add_argument("--subset", type=str, default="data/finetune")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000000000, type=int)
    parser.add_argument("--save_strategy", type=str, default="no")

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
            if len(tokenizer(text).tokens()) > 4096:
                print(f"token lenght: {len(tokenizer(text).tokens())}")
        else:
            total_tokens += len(tokenizer.tokenize(text))
            if len(tokenizer.tokenize(text)) > 4096:
                print(f"token lenght: {len(tokenizer.tokenize(text))}")

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {example['instruction']}

    ### Input:
    {example['input']}

    ### Response:
    {example['output']}""" + EOS_TOKEN

    return text


def create_datasets(tokenizer, args):
    ## Dataset preparation
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}
    
    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    train_dataset = load_dataset("json", data_files = args.data_file, split = "train")
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)

    return train_dataset


def run_training(args, train_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        eval_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        # max_steps=10,
        eval_steps=args.eval_freq,
        save_strategy=args.save_strategy,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="llama-7b-finetuned",
        # report_to="wandb",
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        max_seq_length=args.seq_length,
        dataset_text_field = "text",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        # device_map="auto",
        # load_in_8bit=True, 
        # device_map={"": Accelerator().process_index}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        peft_config=lora_config,
        packing=False,
    )

    print_trainable_parameters(trainer.model)

    trainer.train()
    
    print("Saving last checkpoint of the model")

    trainer.model = trainer.model.merge_and_unload()
    trainer.save_model(os.path.join(args.output_dir, str(args.num_train_epochs)))


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    run_name = args.data_file.split(".")[0]
    args.output_dir = os.path.join(args.output_dir, "lora-llama3-8B", run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving checkpoints in {args.output_dir}")

    logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    EOS_TOKEN = tokenizer.eos_token
    train_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset)