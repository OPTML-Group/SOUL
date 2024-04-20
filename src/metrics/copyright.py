import sys

import sacrebleu
import torch
import tqdm
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from dataset import HP

sys.path.append("src")



def eval_leakage_rate(model, tokenizer, dataset, batch_size = 4):

    rougeLs = []
    bleus = []
    scorers = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for i in tqdm.tqdm(
        range(0, len(dataset), batch_size),
        desc="computing training data leakage rate",
    ):
        if i + batch_size > len(dataset):
            batch = dataset[i:]
        else:
            batch = dataset[i : i + batch_size]
        max_length = max([len(x) for x in batch["input_ids"]])
        for idx, x in enumerate(batch["input_ids"]):
            batch["input_ids"][idx] = [tokenizer.pad_token_id] * (max_length - len(x)) + x
        input_ids = torch.tensor(batch["input_ids"])
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids.cuda(),
                max_length=600,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        length = input_ids.size(1)
        decoded_outputs = tokenizer.batch_decode(
            outputs[:,length+1:], skip_special_tokens=True
        )
        ground_truth = batch["response"]
        for idx, text in enumerate(decoded_outputs):
            score = scorers.score(ground_truth[idx], text)
            rougeLs.append(score["rougeL"].recall)
            bleu = sacrebleu.corpus_bleu([text], [[ground_truth[idx]]]).score
            bleus.append(bleu)

    mean_bleu = sum(bleus) / len(bleus)
    mean_rougeL = sum(rougeLs) / len(rougeLs)

    return mean_bleu, mean_rougeL



def eval_copyright(
    model_name,
    batch_size=128,
    output_dir=".",
    if_llama=False,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    try:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.pad_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.pad_token_id
    except:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    tokenizer = left_pad_tokenizer

    results = {}
    dataset = HP("HP", if_llama=if_llama)
    results["train"] = {}
    results["test"] = {}

    for key in ["train", "test"]:
        for k in [100, 300]:
            path = f'files/data/hp/hp_{key}_qa_{k}.jsonl'
            eval_dataset = dataset.build_test_dataset(tokenizer, path)
            mean_bleu, mean_rougeL = eval_leakage_rate(model, tokenizer, eval_dataset, batch_size)
            results[key][k] = {"bleu": mean_bleu, "rougeL": mean_rougeL}


    with open(f"{output_dir}/copyright.json", "w") as f:
        json.dump(results, f, indent=4)
            
        
