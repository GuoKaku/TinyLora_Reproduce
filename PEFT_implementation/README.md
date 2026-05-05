# TinyLoRA Reproduction

This repository is a lightweight, shareable version of my reproduction project for the paper **"Learning to Reason in 13 Parameters"**.

The project focuses on reproducing TinyLoRA + GRPO results on **GSM8K**, along with a small Figure 8-style sweep on Qwen2.5-3B-Instruct.

This repository contains **only the part of our project that uses the PEFT implementation of TinyLoRA**. It is not the full codebase for our overall reproduction effort.

## What This Repo Contains

This repo is meant to help other people:

- understand the reproduction setup
- inspect the training and evaluation code
- reuse the scripts for their own experiments
- view small result summaries and reproduced figures

This repo does **not** include heavy outputs such as merged models, adapter weights, checkpoints, or raw large log files.

It also does **not** include code for **our own custom TinyLoRA implementation**.

## Scope

This repo should be understood as:

- a PEFT-based reproduction subproject
- a record of the scripts and small artifacts from this part of the work
- a shareable GitHub version of one implementation path

This repo should **not** be understood as:

- the complete code for the whole reproduction project
- the repository for our custom TinyLoRA implementation
- a combined release of all collaborators' code

## What Is Included

- `scripts/gsm8k/`
  Main scripts for the core GSM8K reproduction
- `scripts/fig8/`
  Scripts for the Figure 8 sweep and plotting
- `experiments/`
  Experiment-specific script variants for `u=1`, `u=13`, `u=120`, and `u=120` with `lr=1e-6`
- `assets/figures/`
  Reproduced plots
- `assets/prompts/`
  Prompt examples used during evaluation
- `results/`
  Small summary files kept for reference

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── scripts/
│   ├── gsm8k/
│   │   ├── train_tinylora_grpo.py
│   │   ├── eval_gsm8k.py
│   │   └── eval_baseline.py
│   └── fig8/
│       ├── train_fig8.py
│       ├── eval_fig8.py
│       ├── run_sweep.sh
│       ├── run_eval_sweep.sh
│       └── plot_fig8.py
├── experiments/
│   ├── u1/
│   ├── u13_lr5e5/
│   ├── u120/
│   └── u120_lr1e6/
├── assets/
│   ├── figures/
│   └── prompts/
└── results/
```

## Requirements

You will need:

- Python 3.10+
- PyTorch-compatible GPU setup for training and vLLM-based evaluation
- access to Hugging Face model downloads

The scripts use:

- `transformers`
- `datasets`
- `peft`
- `trl`
- `vllm`
- `matplotlib`

## Installation

Create a virtual environment and install the lightweight dependency set:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the original full package snapshot from the experiment environment:

```bash
pip install -r requirements-lock.txt
```

## How To Use This Repo

### 1. Main GSM8K reproduction

The core training script is:

```bash
python scripts/gsm8k/train_tinylora_grpo.py
```

The corresponding evaluation script is:

```bash
python scripts/gsm8k/eval_gsm8k.py
```

What these scripts do:

- load `Qwen/Qwen2.5-7B-Instruct`
- train TinyLoRA with GRPO on GSM8K
- evaluate on the GSM8K test split
- compare reproduced accuracy against the paper headline result

### 2. Experiment variants

The `experiments/` folder contains script variants for specific settings:

- `experiments/u1/`
- `experiments/u13_lr5e5/`
- `experiments/u120/`
- `experiments/u120_lr1e6/`

These are useful if you want to inspect or rerun a particular configuration without editing the main script.

### 3. Figure 8 sweep

For the sweep experiments, use:

```bash
python scripts/fig8/train_fig8.py --u 13 --n_tie 256 --output_dir /path/to/run
python scripts/fig8/eval_fig8.py --run_dir /path/to/run
```

To plot the sweep results:

```bash
python scripts/fig8/plot_fig8.py --sweep_dir /path/to/fig8_sweep --out fig8_reproduction.png
```

## Important Notes Before Running

Some scripts still contain the original experiment paths, such as:

- `/workspace/runs/...`
- `/workspace/fig8_sweep`
- `/workspace/venvs/...`

So if you want to rerun the code yourself, you will probably need to edit:

- output directories
- shell script paths
- virtual environment paths

In other words, this repo is fully useful for reading and adapting the code, but it is not yet a one-command reproducibility package.

## Included Result Files

This lightweight repo keeps only small result artifacts:

- `results/fig8_sweep_summary.txt`
- `results/3b_baseline_summary.json`
- figures in `assets/figures/`

These are included so others can quickly see the output style and summary numbers without downloading large artifacts.


## Relation To The Other Project Code

Our full reproduction work has two parts:

- this repository: PEFT-based TinyLoRA experiments
- a separate codebase from my collaborators: our own TinyLoRA implementation


