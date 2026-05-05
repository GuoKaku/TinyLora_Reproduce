# TinyLoRA on GSM8K with Qwen2.5-7B-Instruct

This repository provides a clean Hugging Face / TRL training setup for reproducing the **TinyLoRA + GRPO** GSM8K setting from **Learning to Reason in 13 Parameters** using **PEFT's TinyLoRA implementation**.

## High level structure of this repo

We implemented Tinylora both from scratch and based on PEFT library. For PEFT library based implementation, please refer to `PEFT_implementation` folder, which contains the complete scripts/files to run training and evaluation experiments. For other folders/files, they are used to construct TinyLora from scratch.


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
plot_scripts/
scripts/
├── download.sh. ##for downloading dataset and model
├── eval.sh. ##evaluate ckpts, no need to manually run
├── train_nopeft.sh.    ##run this to start training
├── train_nopeft_slurm.sh. ## run this to submit slurm job for training
├── src/tinylora_gsm8k/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py 
│   ├── eval_gsm8k.py
│   ├── prompts.py
│   ├── rewards.py  
│   ├── train_grpo.py. ## core file to construct TinyLora
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

### 1. Download model and dataset

By default, models and datasets are downloaded to `~/.cache/huggingface`.

```bash
bash scripts/download.sh
```

You can optionally specify a custom cache directory:

```bash
HF_DIR=/path/to/cache bash scripts/download.sh
```

---

### 2. Run training (single GPU)

```bash
HF_DIR=/path/to/cache \
DATASET=gsm8k \
CONFIG_PATH=configs/qwen25_7b_tinylora.yaml \
bash scripts/train_nopeft.sh
```

* `HF_DIR`: HuggingFace cache directory (must match the download step)
* `DATASET`: dataset name (e.g., `gsm8k`, `math`)
* `CONFIG_PATH`: training config

---

### 3. Run training with Slurm (cluster)

```bash
HF_DIR=/path/to/cache \
DATASET=gsm8k \
CONFIG_PATH=configs/qwen25_7b_tinylora.yaml \
sbatch scripts/train_nopeft_slurm.sh
```

> ⚠️ You may need to modify the `#SBATCH` fields (e.g., partition, account) in the script based on your cluster.

---

### Notes

* If `local_dataset_path` in the config is `null`, the script will automatically use:

  ```
  $HF_DIR/gsm8k_local
  ```

* Offline mode is enabled by default. To allow downloading during training:

```bash
OFFLINE=0 bash scripts/train_nopeft.sh
```

* Outputs are saved to:

```
outputs/<job_name>_job<id>_<timestamp>/
```





## Evaluation

Evaluation will be automatically done with the progress of training. More specificlly, evaluation will be done on the beginning and end of training, as well as the every steps specified in configs. If you want to run it manually, do

```bash
bash scripts/eval.sh /path/to/the/checkpoint
```


## Repro guidance

This repo aims to match the paper's **algorithmic setting**, but exact leaderboard numbers can still vary because of:

- TRL / Transformers version drift
- generation backend differences (the paper used VERL + vLLM in their RL stack)
- optimizer / distributed setup details
- hardware and random seed effects

## Practical notes

1. The paper's GSM8K setup uses **4 samples per problem**. In TRL this maps naturally to `num_generations=4`.
2. TRL requires the effective batch size to be divisible by `num_generations`. This repo checks that.
3. The exact-match reward is implemented on the **normalized final numeric answer**, extracted from GSM8K's reference answer and from the model's generated completion.

## Suggested hardware

We did our expeiments on a single H200 GPU with 144GB CUDA memory. During training, we found our used CUDA memory to be around 120GB. On such setting, an expeiment with the default setting will typically run 4~5 hours.
