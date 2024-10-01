import csv
import random
from collections import defaultdict

import torch
from datasets import load_dataset


from .Base import BaseDataset


class ToFU(BaseDataset):
    def __init__(self, dataset_name, subset="forget01", if_llama=False):
        self.dataset_name = dataset_name
        self.dataset = defaultdict()
        self.if_llama = if_llama
        self.question_start_token = "[INST] " if self.if_llama else "### Question: "
        self.question_end_token = " [/INST]" if if_llama else "\n"
        self.answer_start_token = " " if if_llama else "### Answer: "
        self.subset = subset
        self.dataset = self.get_dataset()

    def get_dataset(self):
        key = f"{self.subset}"
        train_dataset = load_dataset(
            "locuslab/TOFU", key, cache_dir="./.cache", split="train"
        )
        test_key = f"{self.subset}_perturbed"
        if "retain" in self.subset:
            test_key = f"retain_perturbed"
        elif "real_authors" in self.subset:
            test_key = f"real_authors_perturbed"
        elif "world_facts" in self.subset:
            test_key = f"world_facts_perturbed"
        elif "full" in self.subset:
            test_key = f"full"
        test_dataset = load_dataset(
            "locuslab/TOFU", test_key, cache_dir="./.cache", split="train"
        )
        dataset = defaultdict()
        dataset["train"] = train_dataset
        dataset["test"] = test_dataset
        return dataset

    def __preprocess__(self, tokenizer):
        refusal_answers = []
        with open(
            "files/data/polite_refusal_responses/polite_refusal_responses_tofu.csv", "r"
        ) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                refusal_answers.append(row[0])

        def preprocess_train(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "label": [],
                "refused_label": [],
                "question_length": [],
            }

            for i in range(len(examples["question"])):
                prompt = examples["question"][i]
                question = self.question_start_token + prompt + self.question_end_token
                full_text = question + self.answer_start_token + examples["answer"][i]
                tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,
                )
                num_question_token = len(
                    tokenizer.tokenize(question, add_special_tokens=True)
                )
                pad_length = 512 - len(tokenized.input_ids)
                pad_input_ids = (
                    tokenized.input_ids + [tokenizer.pad_token_id] * pad_length
                )
                pad_attention_mask = tokenized.attention_mask + [0] * pad_length
                if len(tokenized.input_ids) == 512:
                    label = tokenized.input_ids
                else:
                    label = (
                        tokenized.input_ids
                        + [tokenizer.eos_token_id]
                        + [-100] * (pad_length - 1)
                    )

                for i in range(num_question_token):
                    label[i] = -100

                results["input_ids"].append(torch.tensor(pad_input_ids))
                results["attention_mask"].append(torch.tensor(pad_attention_mask))
                results["label"].append(torch.tensor(label))
                results["question_length"].append(torch.tensor(num_question_token))
                refusal_answer = random.choice(refusal_answers)
                refusal_tokenized = tokenizer(
                    refusal_answer,
                    truncation=True,
                    padding=False,  # Don't pad here, we will pad later if necessary
                    add_special_tokens=True,
                )
                refusal_label = (
                    tokenized.input_ids[: num_question_token + 1]
                    + refusal_tokenized.input_ids[1:]
                )
                if len(refusal_label) < 512:
                    refusal_label = refusal_label + [-100] * (512 - len(refusal_label))
                for i in range(num_question_token):
                    refusal_label[i] = -100
                results["refused_label"].append(torch.tensor(refusal_label))
            return results

        train_dataset = self.dataset["train"].map(
            preprocess_train, batched=True, remove_columns=["question", "answer"]
        )
        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label",
                "refused_label",
                "question_length",
            ],
        )
        self.dataset["train"] = train_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)

        return self.dataset

    def build_pretrain_dataset(self, tokenizer, subset="full"):
        train_dataset = load_dataset(
            "locuslab/TOFU", subset, cache_dir="./.cache", split="train"
        )

        def preprocess_train(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "label": [],
            }

            for i in range(len(examples["question"])):
                prompt = examples["question"][i]
                question = self.question_start_token + prompt + self.question_end_token
                full_text = question + self.answer_start_token + examples["answer"][i]
                tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    add_special_tokens=True,
                )
                num_question_token = len(
                    tokenizer.tokenize(question, add_special_tokens=True)
                )
                pad_length = 512 - len(tokenized.input_ids)
                pad_input_ids = (
                    tokenized.input_ids + [tokenizer.pad_token_id] * pad_length
                )
                pad_attention_mask = tokenized.attention_mask + [0] * pad_length
                if len(tokenized.input_ids) == 512:
                    label = tokenized.input_ids
                else:
                    label = tokenized.input_ids + [-100] * pad_length

                for i in range(num_question_token):
                    label[i] = -100

                results["input_ids"].append(torch.tensor(pad_input_ids))
                results["attention_mask"].append(torch.tensor(pad_attention_mask))
                results["label"].append(torch.tensor(label))
            return results

        train_dataset = self.dataset["train"].map(
            preprocess_train, batched=True, remove_columns=["question", "answer"]
        )
        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label",
            ],
        )

        def collector(samples):
            res = {}
            res["input_ids"] = torch.stack([sample["input_ids"] for sample in samples])
            res["attention_mask"] = torch.stack(
                [sample["attention_mask"] for sample in samples]
            )
            res["labels"] = torch.stack([sample["label"] for sample in samples])
            return res

        train_collector = collector

        return train_dataset, train_collector
