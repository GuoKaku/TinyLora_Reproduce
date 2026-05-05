#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1}"
OUTPUT_PATH="${2:-${CHECKPOINT_PATH}/eval.json}"

DATASET="${DATASET:-gsm8k}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen25_7b_tinylora.yaml}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface

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

if [[ "$CHECKPOINT_PATH" == *"nopeft"* ]]; then
    IMPL_FLAG="--use_nopeft"
else
    IMPL_FLAG=""
fi

python -m tinylora_gsm8k.eval_gsm8k \
    --config "$RUNTIME_CONFIG" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_path "$OUTPUT_PATH" \
    $IMPL_FLAG \
    --use_vllm

rm -f "$RUNTIME_CONFIG"