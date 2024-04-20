from collections import defaultdict

from datasets import load_dataset

from .Base import BaseDataset


class BookCorpus(BaseDataset):
    def __init__(self, dataset_name, seed=42, ratio=1.0):
        self.dataset_name = dataset_name
        self.ratio = ratio
        self.seed = seed
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        total_data = load_dataset("bookcorpus", cache_dir="./.cache")["train"]
        length = len(total_data)
        total_data = total_data.shuffle(seed=self.seed)
        total_data = total_data.select(range(int(length * self.ratio)))
        dataset = defaultdict()
        total_data = total_data.train_test_split(test_size=0.1, seed=self.seed)
        dataset["test"] = total_data["test"]
        dataset["train"] = total_data["train"]

        return dataset

    def __preprocess__(self, tokenizer):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "label": []}
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            results["input_ids"] = tokenized.input_ids
            results["attention_mask"] = tokenized.attention_mask
            results["label"] = tokenized.input_ids
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["text"]
        )
        test_dataset = self.dataset["test"].map(
            preprocess, batched=True, remove_columns=["text"]
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        self.dataset["train"] = train_dataset
        self.dataset["test"] = test_dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)

        return self.dataset
