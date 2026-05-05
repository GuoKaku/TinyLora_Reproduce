#!/bin/bash
#SBATCH --job-name=tinylora_nopeft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=your_partition_name
#SBATCH --account=your_account_name
#SBATCH --time=18:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

DATASET="${DATASET:-gsm8k}"
JOB_NAME="${JOB_NAME:-tinylora_${DATASET}_nopeft}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen25_7b_tinylora.yaml}"
HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"
CONDA_ENV="${CONDA_ENV:-tinylora}"
OFFLINE="${OFFLINE:-1}"

export HF_DIR

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

STDOUT_LOG="$LOG_DIR/${SLURM_JOB_ID}_${JOB_NAME}.out"
STDERR_LOG="$LOG_DIR/${SLURM_JOB_ID}_${JOB_NAME}.err"

exec >"$STDOUT_LOG" 2>"$STDERR_LOG"

echo "=============================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job Name:      ${JOB_NAME}"
echo "Dataset:       ${DATASET}"
echo "Node:          $(hostname)"
echo "Start Time:    $(date)"
echo "Config Path:   ${CONFIG_PATH}"
echo "HF_DIR:        ${HF_DIR}"
echo "=============================="

source ~/.bashrc
conda activate "$CONDA_ENV"

export HF_HOME="$HF_DIR"
export TRANSFORMERS_CACHE="$HF_DIR"
export HF_DATASETS_CACHE="$HF_DIR"

if [[ "$OFFLINE" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

RUN_TAG="${JOB_NAME}_job${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT_DIR="${OUTPUT_ROOT:-outputs}/${RUN_TAG}"
RUNTIME_CONFIG="configs/_runtime_${RUN_TAG}.yaml"

mkdir -p configs "$RUN_OUTPUT_DIR"

python - <<PY
from pathlib import Path
import os
import yaml

src = Path("${CONFIG_PATH}")
dst = Path("${RUNTIME_CONFIG}")
cfg = yaml.safe_load(src.read_text())

dataset = "${DATASET}"
hf_dir = os.environ["HF_DIR"]

dataset_cfg = cfg.pop("datasets")[dataset]

if dataset_cfg.get("local_dataset_path") is None:
    if dataset == "gsm8k":
        dataset_cfg["local_dataset_path"] = os.path.join(hf_dir, "gsm8k_local")
    elif dataset == "math":
        dataset_cfg["local_dataset_path"] = os.path.join(hf_dir, "math_local")

cfg.update(dataset_cfg)
cfg["output_dir"] = "${RUN_OUTPUT_DIR}"

dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(f"Wrote runtime config to: {dst}")
print(f"dataset = {dataset}")
print(f"output_dir = ${RUN_OUTPUT_DIR}")
print(f"local_dataset_path = {cfg.get('local_dataset_path')}")
PY

echo
echo "Environment checks:"
which python
python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME=$HF_HOME"
echo "Dataset: $DATASET"
echo "RUN_OUTPUT_DIR=$RUN_OUTPUT_DIR"
echo "RUNTIME_CONFIG=$RUNTIME_CONFIG"
echo

nvidia-smi || true
echo

echo "Launching training..."
python -m tinylora_gsm8k.train_grpo_nopeft --config "$RUNTIME_CONFIG"

echo
echo "Finished at $(date)"
