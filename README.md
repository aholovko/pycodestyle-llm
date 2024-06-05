# Fine-Tuning LLM for Code Style Analysis

This repository contains the source code and resources for the research paper titled "Fine-Tuning LLM for Code Style
Analysis: An Approach Augmented with DFA". Study explores the integration of Deterministic Finite Automata (DFA) into
the fine-tuning process of Large Language Models (LLMs), specifically focusing on Llama-2 7B and Llama-3 8B models.

The primary objective is to improve the models' accuracy in distinguishing between PEP-8 compliant and non-compliant
code indentation, particularly under conditions of limited training data.

## Installation

```bash
$ git clone https://github.com/aholovko/pycodestyle-llm.git
$ cd pycodestyle-llm
$ python3.11 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Usage

### Run zero-shot evaluation

```bash
$ python zero-shot-eval.py --model llama2 --iterations 10
```

### Run W&B sweeps to search the hyperparameter space

```bash
$ wandb sweep sweep.yaml
$ wandb agent <sweep_id> -p pycodestyle-llm -e one-cleancode
```

### Fine-tuning model with specific parameters

```bash
$ python llama-tune.py --batch_size=16 --epochs=4 --learning_rate=0.001 --lora_alpha=64 --lora_dropout=0.05 --lora_r=8 --model=llama3 --output_dir=llama-tuned-local
```
