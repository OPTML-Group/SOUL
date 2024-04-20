from collections import defaultdict

from datasets import load_dataset

from .Base import BaseDataset


class C4(BaseDataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):

        dataset = defaultdict()
        train_dataset = load_dataset(        
            "allenai/c4",
            data_files={"train": "en/c4-train.00001-of-01024.json.gz"},
            split="train",
            cache_dir="./.cache"
            )
        

        dataset["train"] = train_dataset
        print(f"Train dataset: {len(dataset['train'])}")
        dataset["test"] = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            cache_dir="./.cache"
            )

        return dataset

    def __preprocess__(self, tokenizer):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "label": []}
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                add_special_tokens=True,
                max_length=1024,
            )
            results["input_ids"] = tokenized.input_ids
            results["attention_mask"] = tokenized.attention_mask
            results["label"] = tokenized.input_ids
            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["text","timestamp","url"]
        )
        test_dataset = self.dataset["test"].map(
            preprocess, batched=True, remove_columns=["text","timestamp","url"]
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        self.dataset["train"] = train_dataset
        self.dataset["test"] = test_dataset
        return self.dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)
        return self.dataset
