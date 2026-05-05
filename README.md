# TinyLoRA on GSM8K with Qwen2.5-7B-Instruct

This repository provides a clean Hugging Face / TRL training setup for reproducing the **TinyLoRA + GRPO** GSM8K setting from **Learning to Reason in 13 Parameters** using **PEFT's TinyLoRA implementation**.

## What this repo matches from the paper

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Training method: **GRPO**
- Task: **GSM8K**
- Reward: **exact match** on the final numeric answer
- KL penalty: **0.0**
- Epochs: **3**
- Samples / generations per problem: **4**
- Global batch size target: **64 prompts**
- Max completion length: **4096**
- TinyLoRA defaults here:
  - `r = 2` (recommended by PEFT TinyLoRA docs)
  - `u = 13`
  - `weight_tying = 1.0` for the 13-parameter regime
  - target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

<!-- ## Important note about PEFT version

TinyLoRA is currently documented on the **PEFT `main` docs** and may require installing PEFT from source instead of the latest PyPI release.

This repo therefore installs:

```bash
pip install git+https://github.com/huggingface/peft.git
``` -->

## Repository layout

```text
.
├── configs/
│   ├── qwen25_7b_tinylora_gsm8k.yaml ##use this to change config 
│   └── qwen25_7b_tinylora_gsm8k_debug.yaml
├── scripts/
│   ├── train.sh
│   └── eval.sh
├── src/tinylora_gsm8k/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eval_gsm8k.py
│   ├── prompts.py
│   ├── rewards.py
│   ├── train_grpo.py
│   └── utils.py
└── requirements.txt
```

## Installation

```bash
conda create -n tinylora-gsm8k python=3.11 -y
conda activate tinylora-gsm8k
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## Training

Paper-like config:

```bash
bash scripts/train.sh configs/qwen25_7b_tinylora_gsm8k.yaml
```

Small debug run:

```bash
bash scripts/train.sh configs/qwen25_7b_tinylora_gsm8k_debug.yaml
```

You can also call the trainer directly:

```bash
python -m tinylora_gsm8k.train_grpo --config configs/qwen25_7b_tinylora_gsm8k.yaml
```

## Evaluation

```bash
bash scripts/eval.sh outputs/qwen25-7b-gsm8k-tinylora
```

or:

```bash
python -m tinylora_gsm8k.eval_gsm8k \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/qwen25-7b-gsm8k-tinylora/checkpoint-final \
  --output_path outputs/qwen25-7b-gsm8k-tinylora/eval.json
```

## Repro guidance

This repo aims to match the paper's **algorithmic setting**, but exact leaderboard numbers can still vary because of:

- PEFT / TRL / Transformers version drift
- generation backend differences (the paper used VERL + vLLM in their RL stack)
- optimizer / distributed setup details
- hardware and random seed effects

## Practical notes

1. The paper's GSM8K setup uses **4 samples per problem**. In TRL this maps naturally to `num_generations=4`.
2. TRL requires the effective batch size to be divisible by `num_generations`. This repo checks that.
3. The exact-match reward is implemented on the **normalized final numeric answer**, extracted from GSM8K's reference answer and from the model's generated completion.
4. The prompt template asks the model to reason normally and end with a line of the form:

   ```text
   Final answer: <answer>
   ```

   This makes exact-match scoring more stable.

## Suggested hardware

For paper-like training with `Qwen/Qwen2.5-7B-Instruct`, bf16, long completions, and GRPO, use multi-GPU hardware if possible. A single-GPU debug path is provided, but the full paper-like setting is compute-heavy.
