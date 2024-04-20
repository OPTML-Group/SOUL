import os
import sys

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


import torch
sys.path.append("src")
from dataset import get_dataset
from metrics import eval_copyright, eval_few_shots, eval_ppl, eval_toxic, eval_tofu
from optim import create_sophia_optimizer
from unlearn import get_unlearn_method
from peft import  get_peft_model, LoraConfig

class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.unlearn_method = kwargs["unlearn_method"]
        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.alpha = kwargs.get("alpha", None)
        self.gamma = kwargs.get("gamma", None)
        self.task_name = kwargs.get("task_name", None)
        self.sophia = kwargs.get("sophia", False)
        self.betas_low = kwargs.get("betas_low", 0.9)
        self.betas_high = kwargs.get("betas_high", 0.95)
        self.betas = (self.betas_low, self.betas_high)
        self.rho = kwargs.get("rho", 0.03)
        self.forget_epoch = kwargs.get("forget_epoch", 1)
        self.if_llama = "llama" in self.model_name
        self.use_lora = kwargs.get("use_lora", False)
        self.resume_path = kwargs.get("resume_path", None)
    def init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        if self.use_lora:
            peft_config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=["q_proj","v_proj"], 
                lora_dropout=0.05,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())
        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except:
            self.device = torch.device("cuda:0")

    def init_dataset(self):
        unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_dataset = test_dataset
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
            self.batch_size * self.gradient_accumulation_steps * self.num_devices
        )
        self.steps_per_epoch = len(unlearn_dataset) // (
            self.batch_size * self.gradient_accumulation_steps * self.num_devices
        )

    def init_unlearner(self, logger):
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        if self.unlearn_method == "origin" or self.unlearn_method == "sys":
            self.unlearner = None
            return
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            output_dir=unlearn_checkpoint,
            optim="adamw_torch",
            save_steps=self.steps_per_epoch,
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            save_total_limit=1,
        )
        if self.optimizer is not None:
            self.unlearner = get_unlearn_method(
                name=self.unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                forget_epoch=self.forget_epoch,
                optimizers=(self.optimizer, None),
            )
        else:
            self.unlearner = get_unlearn_method(
                name=self.unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                forget_epoch=self.forget_epoch,
            )

    def init_optimizer(self):
        if self.sophia:
            self.optimizer = create_sophia_optimizer(
                self.model,
                lr=self.lr,
                betas=self.betas,
                rho=self.rho,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = None

    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        if self.unlearn_method == "sys":
            if_system = True
        else:
            if_system = False
        if self.resume_path is not None:
            model_name = self.resume_path
        else:
            model_name = os.path.join(root, "checkpoints")
        if self.task_name != 'tofu':
            eval_ppl(model_name=model_name, output_dir=root)
            eval_few_shots(model_name=model_name, output_dir=root)
        torch.cuda.empty_cache()
        if self.task_name == "toxic":
            eval_toxic(
                model_name=model_name, output_dir=root, dataset=self.unlearn_dataset
            )
        elif self.task_name == "copyright":
            eval_copyright(model_name=model_name, output_dir=root,batch_size=16,if_llama=self.if_llama)
        elif self.task_name == "tofu":
            forget_subset = self.dataset_names["forget"].split("_")[1]
            retain_subset = self.dataset_names["retain"].split("_")[1]

            eval_tofu(
                model_name=model_name,
                output_dir=root,
                forget_subset=forget_subset,
                retain_subset=retain_subset,
                if_llama=self.if_llama,
                if_system=if_system,
            )

    def save(self, logger):
        logger.save_ckpt("model", self.model, self.use_lora)
        logger.save_ckpt("tokenizer", self.tokenizer, self.use_lora)

    def run(self, logger):
        if self.resume_path is None:
            self.init_model()
            self.init_optimizer()
            self.init_dataset()
            self.init_unlearner(logger)
            if self.unlearner:
                self.unlearner.train()
            self.save(logger)
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
        self.eval(logger)


def get(**kwargs):
    return Unlearn(**kwargs)
