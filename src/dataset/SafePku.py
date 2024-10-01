import copy
import csv
import random
from collections import defaultdict

import torch
from datasets import load_dataset

from .Base import BaseDataset


class SafePkuDataset(BaseDataset):
    def __init__(self, dataset_name, with_retain=False, if_llama=False):
        super().__init__(dataset_name, with_retain, if_llama)
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        train_dataset = load_dataset(
            "PKU-Alignment/PKU-SafeRLHF", cache_dir="./.cache",revision="ff7ba91063016c78a225b0f74e1c0860bb18230f"
        )["train"]
        dataset = defaultdict()
        dataset["train"] = train_dataset
        test_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", cache_dir="./.cache",revision="ff7ba91063016c78a225b0f74e1c0860bb18230f")[
            "test"
        ]
        dataset["test"] = test_dataset

        return dataset

    def __preprocess__(self, tokenizer):
        refusal_answers = []
        with open(
            "files/data/polite_refusal_responses/polite_refusal_responses.csv", "r"
        ) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                refusal_answers.append(row[0])

        def preprocess(examples):
            results = {
                "input_ids": [],
                "attention_mask": [],
                "label": [],
                "refused_label": [],
                "question_length": [],
            }
            for i in range(len(examples["prompt"])):
                prompt = examples["prompt"][i]
                unsafe_responses = []
                if not examples["is_response_0_safe"][i]:
                    unsafe_responses.append(examples["response_0"][i])
                if not examples["is_response_1_safe"][i]:
                    unsafe_responses.append(examples["response_1"][i])

                for response in unsafe_responses:
                    text = (
                        self.question_start_token
                        + prompt
                        + self.question_end_token
                        + self.answer_start_token
                        + response
                    )
                    tokenized = tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        add_special_tokens=True,
                    )
                    num_question_token = len(
                        tokenizer.tokenize(
                            self.question_start_token
                            + prompt
                            + self.question_end_token,
                            add_special_tokens=True,
                        )
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
                        self.answer_start_token + refusal_answer,
                        truncation=True,
                        padding=False,  # Don't pad here, we will pad later if necessary
                        add_special_tokens=True,
                    )
                    refusal_label = (
                        copy.deepcopy(pad_input_ids[: num_question_token + 1])
                        + refusal_tokenized.input_ids[1:]
                    )
                    if len(refusal_label) < 512:
                        refusal_label += [-100] * (512 - len(refusal_label))

                    for i in range(num_question_token):
                        refusal_label[i] = -100
                    results["refused_label"].append(torch.tensor(refusal_label))
            return results

        train_dataset = self.dataset["train"].map(
            preprocess,
            batched=True,
            remove_columns=[
                "prompt",
                "response_0",
                "response_1",
                "is_response_0_safe",
                "is_response_1_safe",
                "better_response_id",
                "safer_response_id",
            ],
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
        test_dataset = self.dataset["test"]

        test_dataset = test_dataset.map(
            preprocess,
            batched=True,
            remove_columns=[
                "prompt",
                "response_0",
                "response_1",
                "is_response_0_safe",
                "is_response_1_safe",
                "better_response_id",
                "safer_response_id",
            ],
        )
        self.dataset["test"] = test_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)

        return self.dataset