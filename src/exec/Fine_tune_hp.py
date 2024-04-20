import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.append("src")
import argparse
import random

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from peft import  get_peft_model, LoraConfig
from dataset.HorryPotter import HP

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--cache_dir", type=str, default=".cache", help="Cache directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of warmup steps"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--save_dir", type=str, default="files/models/hp", help="Save dir"
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = args_parser()
    set_seed(args.seed)
    if "llama" in args.model_name:
        if_llama = True
    dataset = HP("HP",if_llama=if_llama)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[pad]"})
    dataset = dataset.build_pretrain_dataset(tokenizer)
    train_dataset = dataset["train"]

    # test_dataset = dataset["test"]
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_dir="logs",
        logging_steps=50,
        save_steps=50,
        # evaluation_strategy="steps",
        # eval_steps=10,
        save_total_limit=1,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        greater_is_better=False,
        output_dir=args.save_dir,
        optim = "adamw_torch",
        bf16=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    if "llama" in args.model_name:
        model.resize_token_embeddings(len(tokenizer))
        peft_config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=find_all_linear_names(model), 
            lora_dropout=0.05,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    if "llama" in args.model_name:
        model = model.merge_and_unload()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
