import random
import sys

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from scipy.stats import ks_2samp
sys.path.append("src")
import difflib
import json
import sacrebleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

from dataset import ToFU

LABLES = ["A", "B", "C", "D", "E", "F"]
LABLES_ANSWER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
import re

with open("files/data/authors.json") as f:
    authors = json.load(f)
    Name = authors["Name"]

sys_prompt = f"<<SYS>>Please refrain from responding to the following authors' information: " + ", ".join(Name) + "\n<<\SYS>>"

def get_batch_loss(output, label):
    shift_logits = output[..., :-1, :].contiguous()
    shift_labels = label[..., 1:].contiguous()
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(shift_logits.transpose(-1, -2), shift_labels).sum(dim=-1)
    return loss

def compute_prob(model, prompt, answer, tokenizer, if_llama=False,if_system=False):
    question_start_token = "[INST] " if if_llama else "### Question: "
    if if_system:
        question_start_token = "[INST] " + sys_prompt + " " if if_llama else "### Question: " + sys_prompt + " "
    question_end_token = " [\INST]" if if_llama else "\n"
    answer_start_token = " " if if_llama else "### Answer: "
    question = question_start_token + prompt + question_end_token
    full_text = question + answer_start_token + answer
    tokenized = tokenizer(
        full_text,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    num_question_token = len(tokenizer.tokenize(question, add_special_tokens=True))
    label = tokenized.input_ids.clone()
    for i in range(num_question_token):
        label[:, i] = -100
    with torch.no_grad():
        outputs = model(tokenized.input_ids.cuda(), tokenized.attention_mask.cuda())
    loss = get_batch_loss(outputs.logits, label.cuda())
    num_token_answer = (label != -100).sum(-1)
    loss_per_token = loss.item() / num_token_answer
    prob = torch.exp(-loss_per_token)
    return prob.item()


def generate_answer(model, tokenizer, prompt, if_llama=False, if_system=False):
    question_start_token = "[INST] " if if_llama else "### Question: "
    if if_system:
        question_start_token = "[INST] " + sys_prompt + " " if if_llama else "### Question: " + sys_prompt + " "
        max_length = 300
    else:
        max_length = 200
    question_end_token = " [\INST]" if if_llama else "\n"
    question = question_start_token + prompt + question_end_token
    len_question = len(tokenizer.tokenize(question, add_special_tokens=True))
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenizer(question, return_tensors="pt").input_ids.cuda(),
            max_length=max_length,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0, len_question+1:], skip_special_tokens=True)


def eval_tofu_forget(model, tokenizer, subset="forget01", if_llama=False,if_system=False):
    dataset = ToFU("TOFU", subset=subset)
    dataset = dataset.build_dataset(tokenizer)
    test_dataset = dataset["test"]
    mean_truth_ratio = 0
    mean_truth_prob = 0
    mean_rougeL_score = 0
    scorers = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    corr = 0
    total = 0
    truth_ratios = []
    generated_answers = []
    original_answers = []
    sentencemodel = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    for example in tqdm.tqdm(test_dataset, desc=f"evaluating TOFU {subset} dataset"):
        total += 1
        prompt = example["paraphrased_question"]
        paraphrased_answer = example["paraphrased_answer"]
        paraphrased_answer_prob = compute_prob(
            model, prompt, paraphrased_answer, tokenizer, if_llama=if_llama, if_system=if_system
        )
        false_answers_probs = []
        for false_answer in example["perturbed_answer"]:
            false_answer_prob = compute_prob(
                model, prompt, false_answer, tokenizer, if_llama=if_llama, if_system=if_system
            )
            false_answers_probs.append(false_answer_prob)
        ### compyuete truth ratio
        truth_ratio = (
            sum(false_answers_probs)
            / len(false_answers_probs)
            / (paraphrased_answer_prob+1e-12)
        )
        mean_truth_ratio += truth_ratio
        truth_ratios.append(truth_ratio)
        ### classification
        generated_ph_answer = generate_answer(
            model, tokenizer, prompt, if_llama=if_llama, if_system=if_system
        ).replace("[pad]", "")
        generated_ph_answer = generated_ph_answer.replace("<pad>", "")
        generated_answers.append(generated_ph_answer)
        scores = []
        generated_ph_answer_embedding = sentencemodel.encode(
            generated_ph_answer, convert_to_tensor=True
        )
        ph_answer_embedding = sentencemodel.encode(
            paraphrased_answer, convert_to_tensor=True
        )
        scores.append(
            util.pytorch_cos_sim(generated_ph_answer_embedding, ph_answer_embedding)
        )
        for false_answer in example["perturbed_answer"]:
            false_answer_embedding = sentencemodel.encode(
                false_answer, convert_to_tensor=True
            )
            scores.append(
                util.pytorch_cos_sim(
                    generated_ph_answer_embedding, false_answer_embedding
                )
            )
        if max(scores) == scores[0]:
            corr += 1
        prompt = example["question"]
        truth_answer = example["answer"]
        truth_answer_prob = compute_prob(
            model, prompt, truth_answer, tokenizer, if_llama=if_llama, if_system=if_system
        )
        mean_truth_prob += truth_answer_prob
        generated_answer = generate_answer(model, tokenizer, prompt, if_llama=if_llama, if_system=if_system)
        original_answers.append(generated_answer)
        score = scorers.score(truth_answer, generated_answer)
        mean_rougeL_score += score["rougeL"].recall
    mean_truth_prob /= len(test_dataset)
    mean_truth_ratio /= len(test_dataset)
    mean_rougeL_score /= len(test_dataset)
    return (
        truth_ratios,
        mean_truth_ratio,
        mean_truth_prob,
        mean_rougeL_score,
        corr / total,
        generated_answers,
        original_answers,
    )


def eval_tofu_retain(model, tokenizer, subset="retain", if_llama=False, if_system=False):
    dataset = ToFU("TOFU", subset=subset)
    dataset = dataset.build_dataset(tokenizer)
    test_dataset = dataset["test"]
    mean_truth_ratio = 0
    mean_truth_prob = 0
    mean_rougeL_score = 0
    scorers = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    corr = 0
    total = 0
    generated_answers = []
    truth_ratios = []
    sentencemodel = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    for example in tqdm.tqdm(test_dataset, desc=f"evaluating TOFU {subset} dataset"):
        total += 1
        prompt = example["paraphrased_question"]
        paraphrased_answer = example["paraphrased_answer"]
        paraphrased_answer_prob = compute_prob(
            model, prompt, paraphrased_answer, tokenizer, if_llama=if_llama, if_system=if_system
        )
        false_answers_probs = []
        for false_answer in example["perturbed_answer"]:
            false_answer_prob = compute_prob(
                model, prompt, false_answer, tokenizer, if_llama=if_llama, if_system=if_system
            )
            false_answers_probs.append(false_answer_prob)
        generated_ph_answer = generate_answer(
            model, tokenizer, prompt, if_llama=if_llama, if_system=if_system
        ).replace("[pad]", "")
        generated_ph_answer = generated_ph_answer.replace("<pad>", "")
        generated_answers.append(generated_ph_answer)
        scores = []
        generated_ph_answer_embedding = sentencemodel.encode(
            generated_ph_answer, convert_to_tensor=True
        )
        ph_answer_embedding = sentencemodel.encode(
            paraphrased_answer, convert_to_tensor=True
        )
        scores.append(
            util.pytorch_cos_sim(generated_ph_answer_embedding, ph_answer_embedding)
        )
        for false_answer in example["perturbed_answer"]:
            false_answer_embedding = sentencemodel.encode(
                false_answer, convert_to_tensor=True
            )
            scores.append(
                util.pytorch_cos_sim(
                    generated_ph_answer_embedding, false_answer_embedding
                )
            )
        if max(scores) == scores[0]:
            corr += 1
        truth_ratio = (
            sum(false_answers_probs)
            / len(false_answers_probs)
            / (paraphrased_answer_prob+1e-12)
        )
        mean_truth_ratio += truth_ratio
        truth_ratios.append(truth_ratio)
        prompt = example["question"]
        truth_answer = example["answer"]
        truth_answer_prob = compute_prob(
            model, prompt, truth_answer, tokenizer, if_llama=if_llama, if_system=if_system
        )
        mean_truth_prob += truth_answer_prob
        generated_answer = generate_answer(model, tokenizer, prompt, if_llama=if_llama, if_system=if_system)
        score = scorers.score(truth_answer, generated_answer)
        mean_rougeL_score += score["rougeL"].recall
    mean_truth_prob /= len(test_dataset)
    mean_truth_ratio /= len(test_dataset)
    mean_rougeL_score /= len(test_dataset)
    return (
        truth_ratios,
        mean_truth_ratio,
        mean_truth_prob,
        mean_rougeL_score,
        corr / total,
        generated_answers,
    )


def eval_tofu_other(model, tokenizer, subset="retain", if_llama=False,if_system=False):
    dataset = ToFU("TOFU", subset=subset)
    dataset = dataset.build_dataset(tokenizer)
    test_dataset = dataset["test"]
    mean_truth_ratio = 0
    mean_truth_prob = 0
    mean_rougeL_score = 0
    corr = 0
    total = 0
    generated_answers = []
    truth_ratios = []
    scorers = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for example in tqdm.tqdm(test_dataset, desc=f"evaluating TOFU {subset} dataset"):
        total += 1
        prompt = example["question"]
        false_answers_prob = []
        truth_answer = example["answer"]
        truth_answer_prob = compute_prob(
            model, prompt, truth_answer, tokenizer, if_llama=if_llama, if_system=if_system
        )
        mean_truth_prob += truth_answer_prob
        generated_answer = generate_answer(
            model, tokenizer, prompt, if_llama=if_llama, if_system=if_system
        ).replace("[pad]", "")

        generated_answers.append(generated_answer)
        for false_answer in example["perturbed_answer"]:
            false_answer_prob = compute_prob(
                model, prompt, false_answer, tokenizer, if_llama=if_llama, if_system=if_system
            )
            false_answers_prob.append(false_answer_prob)
        pattern = re.compile(re.escape(truth_answer), re.IGNORECASE)
        if pattern.search(generated_answer) is not None:
            corr += 1
        truth_ratio = (
            sum(false_answers_prob)
            / len(false_answers_prob)
            / (truth_answer_prob+1e-12)
        )
        mean_truth_ratio += truth_ratio
        truth_ratios.append(truth_ratio)
        score = scorers.score(truth_answer, generated_answer)
        mean_rougeL_score += score["rougeL"].recall
    mean_truth_prob /= len(test_dataset)
    mean_truth_ratio /= len(test_dataset)
    mean_rougeL_score /= len(test_dataset)
    return (
        truth_ratios,
        mean_truth_ratio,
        mean_truth_prob,
        mean_rougeL_score,
        corr / total,
        generated_answers,
    )


def eval_tofu(
    model_name,
    forget_subset="forget01",
    retain_subset="retain99",
    output_dir=".",
    if_llama=False,
    if_system=False,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    try:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    except:
        left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
        left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    tokenizer = left_pad_tokenizer
    (
        forget_truth_ratios,
        mean_forget_truth_ratio,
        mean_forget_truth_prob,
        mean_forget_rougeL_score,
        mean_forget_acc,
        forget_generated_answers,
        forget_original_answers,
    ) = eval_tofu_forget(model, tokenizer, forget_subset, if_llama=if_llama,if_system=if_system)
    print(mean_forget_acc)
    (
        retain_truth_ratios,
        mean_retain_truth_ratio,
        mean_retain_truth_prob,
        mean_retain_rougeL_score,
        mean_retain_acc,
        retain_generated_answers,
    ) = eval_tofu_retain(model, tokenizer, retain_subset, if_llama=if_llama,if_system=if_system)
    (
        real_author_truth_ratios,
        mean_real_author_truth_ratio,
        mean_real_author_truth_prob,
        mean_real_author_rougeL_score,
        mean_real_author_acc,
        real_author_generated_answers,
    ) = eval_tofu_other(model, tokenizer, "real_authors", if_llama=if_llama,if_system=if_system)
    print(mean_real_author_acc)
    (
        world_fact_truth_ratios,
        mean_world_fact_truth_ratio,
        mean_world_fact_truth_prob,
        mean_world_fact_rougeL_score,
        mean_world_fact_acc,
        world_fact_generated_answers,
    ) = eval_tofu_other(model, tokenizer, "world_facts", if_llama=if_llama,if_system=if_system)
    print(mean_world_fact_acc)

    test_res = ks_2samp(forget_truth_ratios, retain_truth_ratios)
    result = {
        "forget": {
            "truth_ratio": mean_forget_truth_ratio,
            "truth_prob": mean_forget_truth_prob,
            "rougeL_score": mean_forget_rougeL_score,
            "acc": mean_forget_acc,
            "generated_answers": forget_generated_answers,
            "original_answers": forget_original_answers,
        },
        "retain": {
            "truth_ratio": mean_retain_truth_ratio,
            "truth_prob": mean_retain_truth_prob,
            "rougeL_score": mean_retain_rougeL_score,
            "acc": mean_retain_acc,
            "generated_answers": retain_generated_answers,
        },
        "real_author": {
            "truth_ratio": mean_real_author_truth_ratio,
            "truth_prob": mean_real_author_truth_prob,
            "rougeL_score": mean_real_author_rougeL_score,
            "acc": mean_real_author_acc,
            "generated_answers": real_author_generated_answers,
        },
        "world_fact": {
            "truth_ratio": mean_world_fact_truth_ratio,
            "truth_prob": mean_world_fact_truth_prob,
            "rougeL_score": mean_world_fact_rougeL_score,
            "acc": mean_world_fact_acc,
            "generated_answers": world_fact_generated_answers,
        },
        "Forget Quality": test_res.pvalue
    }
    with open(f"{output_dir}/tofu.json", "w") as f:
        json.dump(result, f, indent=4)
