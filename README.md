<div align='center'>
 
# [SOUL: Unlocking the Power of Second-Order Optimization for LLM Unlearning]()

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2404.18239&color=B31B1B)](https://arxiv.org/pdf/2404.18239)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<table align="center">
  <tr>
    <td align="center"> 
      <img src="assets/teaser.png" alt="Image 1" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Performance highlight using second-order optimization.</em>
    </td>
  </tr>
</table>
</div>

Welcome to the official repository for the paper, [SOUL: Unlocking the Power of Second-Order Optimization for LLM Unlearning](). This repository contains the code for the experiments used in the paper.

## Abstract
Large Language Models (LLMs) have highlighted the necessity of effective unlearning mechanisms to comply with data regulations and ethical AI practices. LLM unlearning aims at removing undesired data influences and associated model capabilities without compromising utility out of the scope of unlearning. While interest in studying LLM unlearning is growing,the impact of the optimizer choice for LLM unlearning remains under-explored. In this work, we shed light on the significance of optimizer selection in LLM unlearning for the first time, establishing a clear connection between {second-order optimization} and influence unlearning (a classical approach using influence functions to update the model for data influence removal).
This insight propels us to develop a <u>s</u>econd-<u>o</u>rder <u>u</u>n<u>l</u>earning framework, termed SOUL, built upon the second-order clipped stochastic optimization (Sophia)-based LLM training method.  SOUL  extends the static, one-shot model update using influence unlearning to a dynamic, iterative unlearning process. Our extensive experiments show that  SOUL consistently outperforms conventional first-order methods across various unlearning tasks, models, and metrics, suggesting the promise of second-order optimization in providing a scalable and easily implementable solution for LLM unlearning.


## 1) Installation
You can install the required dependencies using the following command:
```
bash create_env.sh
```

## 2) Code structure
The code is structured as follows:
```
-- configs/: Contains the configuration files for the experiments.
    -- Different folders for different experiments (Tofu, etc.)
-- files/: 
    -- data/: Contains the data files nessary for the experiments.
    -- results/: the log and results of experiments will stored in this directory.
-- lm-evaluation-harness: official repository for the evaluation of LLMs from      
  https://github.com/EleutherAI/lm-evaluation-harness.
-- src/: Contains the source code for the experiments.
    -- dataset/: Contains the data processing and dataloader creation codes.
    -- model/: Contains the main unlearning class which will conduct load model, 
      unlearn,evaluation.
    -- optim/: Contains the optimizer code.
    -- metrics/: Contains the evaluation code.
    -- loggers/: Contains the logger code.
    -- unlearn/: Contains different unlearning methods' code.
    -- exec/:
        -- Fine_tune_hp.py: Code for finetuning on harry potter books.
        -- unlearn_model.py: The main file to run the unlearning experiments.
```

## 3) Running the experiments
To run the experiments, you can use the following command:
```
python src/exec/unlearn_model.py --config_file ${config-file} ${args you want to change from the config file}
```

## 4) Cite This Work
```
@misc{jia2024soul,
      title={SOUL: Unlocking the Power of Second-Order Optimization for LLM Unlearning}, 
      author={Jinghan Jia and Yihua Zhang and Yimeng Zhang and Jiancheng Liu and Bharat Runwal and James Diffenderfer and Bhavya Kailkhura and Sijia Liu},
      year={2024},
      eprint={2404.18239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
