#!/usr/bin/env bash
set -euo pipefail

HF_DIR=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface

mkdir -p "${HF_DIR}"

export HF_HOME="${HF_DIR}"
export TRANSFORMERS_CACHE="${HF_DIR}"
export HF_DATASETS_CACHE="${HF_DIR}"

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

python - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

model_name = "${MODEL_NAME}"
cache_dir = "${HF_DIR}"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print("Downloading model...")
AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype="auto",
)

# print("Downloading GSM8K...")
# load_dataset("gsm8k", "main", cache_dir=cache_dir)

print("Done.")
EOF