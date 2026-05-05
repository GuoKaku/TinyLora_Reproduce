#!/usr/bin/env bash
set -euo pipefail

HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"
DATASET="${DATASET:-gsm8k}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen25_7b_tinylora.yaml}"
OFFLINE="${OFFLINE:-1}"

export HF_DIR
export HF_HOME="$HF_DIR"
export TRANSFORMERS_CACHE="$HF_DIR"
export HF_DATASETS_CACHE="$HF_DIR"

if [[ "$OFFLINE" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

RUNTIME_CONFIG=$(python - <<PY
import tempfile, yaml, os
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
dataset = "${DATASET}"
hf_dir = os.environ.get("HF_DIR", str(Path.home() / ".cache" / "huggingface"))

cfg = yaml.safe_load(config_path.read_text())
dataset_cfg = cfg.pop("datasets")[dataset]

# If local_dataset_path is null, optionally fill conventional local paths.
if dataset_cfg.get("local_dataset_path") is None:
    if dataset == "gsm8k":
        dataset_cfg["local_dataset_path"] = os.path.join(hf_dir, "gsm8k_local")
    elif dataset == "math":
        dataset_cfg["local_dataset_path"] = os.path.join(hf_dir, "math_local")

cfg.update(dataset_cfg)

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
tmp.close()
print(tmp.name)
PY
)
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

python -m tinylora_gsm8k.train_grpo_nopeft --config "$RUNTIME_CONFIG"
