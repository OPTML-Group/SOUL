import csv
import random
from collections import defaultdict

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
import copy
from .Base import BaseDataset


class HP(BaseDataset):
    def __init__(self, dataset_name, with_retain=False, if_llama=False):
        super().__init__(dataset_name, with_retain, if_llama)
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset = defaultdict()
        qa_dataset_path = "files/data/hp/hp_qa.jsonl"
        qa_dataset = Dataset.from_json(qa_dataset_path)
        dataset["train"] = qa_dataset
        dataset["test"] = None

        return dataset

    def __preprocess__(self, tokenizer):
        refusal_answers = []
        with open(
            "files/data/polite_refusal_responses/polite_refusal_responses_copyright.csv",
            "r",
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
                response = examples["response"][i]
                refusal_label = random.choice(refusal_answers)
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
                        self.question_start_token + prompt + self.question_end_token,
                        add_special_tokens=True,
                    )   
                )

                pad_length = 1024 - len(tokenized.input_ids)
                pad_input_ids = (
                    tokenized.input_ids + [tokenizer.pad_token_id] * pad_length
                )
                pad_attention_mask = tokenized.attention_mask + [0] * pad_length

                if len(tokenized.input_ids) == 1024:
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
                refusal_tokenized = tokenizer(
                    self.answer_start_token + refusal_label,
                    truncation=True,
                    padding=False,  # Don't pad here, we will pad later if necessary
                    add_special_tokens=True,
                )

                refusal_label = (
                    copy.deepcopy(tokenized.input_ids[: num_question_token + 1])
                    + refusal_tokenized.input_ids[1:]
                )

                if len(refusal_label) < 1024:
                    refusal_label += [-100] * (1024 - len(refusal_label))

                for i in range(num_question_token):
                    refusal_label[i] = -100
                
                results["refused_label"].append(torch.tensor(refusal_label))

            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["prompt","response"]
        )

        test_dataset = None

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
        self.dataset["test"] = test_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)

        return self.dataset

    def build_pretrain_dataset(self, tokenizer):
        original_dataset_path = "files/data/hp/hp.jsonl"
        qa_dataset_path = "files/data/hp/hp_qa.jsonl"
        original_dataset = Dataset.from_json(original_dataset_path)
        qa_dataset = Dataset.from_json(qa_dataset_path)
        
        def preprocess_qa(examples):
            results = {"text":[]}
            for i in range(len(examples["prompt"])):
                results["text"].append(self.question_start_token + examples["prompt"][i] + self.question_end_token + self.answer_start_token + examples["response"][i])

            return results
        def preprocess_original(examples):
            results = {"text":[]}
            for i in range(len(examples["text"])):
                results["text"].append(self.question_start_token + examples["text"][i]+ self.question_end_token)
            return results
        qa_dataset = qa_dataset.map(preprocess_qa, batched=True, remove_columns=["prompt", "response"])

        dataset = concatenate_datasets([qa_dataset, original_dataset])
        def tokenize_function(examples):
            # Adjust "max_length" as needed based on your model's maximum input length
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        return DatasetDict(
            {
                "train": dataset.map(tokenize_function, batched=True),
                "test": None,
            }
        )
    def build_test_dataset(self, tokenizer, path):
        dataset = Dataset.from_json(path)

        def preprocess(examples):
            results = {"text": [], "prompt": [], "response": []}
            for i in range(len(examples["prompt"])):
                results["text"].append(
                    self.question_start_token
                    + examples["prompt"][i]
                    + self.question_end_token
                    + self.answer_start_token
                    + examples["response"][i]
                )
                results["prompt"].append(self.question_start_token + examples["prompt"][i] + self.question_end_token)
                results["response"].append(self.answer_start_token + examples["response"][i])
            return results

        dataset = dataset.map(preprocess, batched=True)

        def tokenize_function(examples):
            return tokenizer(examples["prompt"], padding=True, truncation=True, max_length=1024)

        return dataset.map(tokenize_function, batched=True)

