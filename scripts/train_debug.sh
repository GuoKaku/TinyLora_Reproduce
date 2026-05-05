#!/usr/bin/env bash

export HF_HOME=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

set -euo pipefail

CONFIG_PATH="${1:-configs/qwen25_7b_tinylora_gsm8k.yaml}"
JOB_NAME="${JOB_NAME:-tinylora_gsm8k_local_debug}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# ==== 关键：生成唯一 run id ====
RUN_TAG="${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT_DIR="outputs/${RUN_TAG}"
RUNTIME_CONFIG="configs/_runtime_${RUN_TAG}.yaml"

mkdir -p configs

python - <<PY
from pathlib import Path
import yaml

src = Path("${CONFIG_PATH}")
dst = Path("${RUNTIME_CONFIG}")

cfg = yaml.safe_load(src.read_text())
cfg["output_dir"] = "${RUN_OUTPUT_DIR}"

dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(f"Wrote runtime config to: {dst}")
print(f"output_dir = ${RUN_OUTPUT_DIR}")
PY

echo
echo "Running with:"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "RUNTIME_CONFIG=$RUNTIME_CONFIG"
echo "OUTPUT_DIR=$RUN_OUTPUT_DIR"
echo

python -m tinylora_gsm8k.train_grpo_debug --config "$RUNTIME_CONFIG"