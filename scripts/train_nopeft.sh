#!/usr/bin/env bash

export HF_HOME=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

set -euo pipefail

DATASET="${DATASET:-gsm8k}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen25_7b_tinylora.yaml}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

RUNTIME_CONFIG=$(python - <<PY
import sys, tempfile, yaml
from pathlib import Path

cfg = yaml.safe_load(Path("${CONFIG_PATH}").read_text())
dataset_cfg = cfg.pop("datasets")["${DATASET}"]
cfg.update(dataset_cfg)

tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
tmp.close()
print(tmp.name)
PY
)

python -m tinylora_gsm8k.train_grpo_nopeft --config "$RUNTIME_CONFIG"

rm -f "$RUNTIME_CONFIG"