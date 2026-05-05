#!/bin/bash
#SBATCH --job-name=tinylora_nopeft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=mengdiw
#SBATCH --time=18:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

DATASET="${DATASET:-gsm8k}"
JOB_NAME="${JOB_NAME:-tinylora_${DATASET}_nopeft}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen25_7b_tinylora.yaml}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/${SLURM_JOB_ID}_${JOB_NAME}.out"
STDERR_LOG="$LOG_DIR/${SLURM_JOB_ID}_${JOB_NAME}.err"

echo "Logging to:"
echo "  $STDOUT_LOG"
echo "  $STDERR_LOG"
echo

exec >"$STDOUT_LOG" 2>"$STDERR_LOG"

echo "=============================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $JOB_NAME"
echo "Dataset:       $DATASET"
echo "Node:          $(hostname)"
echo "Start Time:    $(date)"
echo "Config Path:   $CONFIG_PATH"
echo "=============================="

source ~/.bashrc
conda activate tinylora
set -euo pipefail

export HF_HOME=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

RUN_TAG="${JOB_NAME}_job${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT_DIR="outputs/${RUN_TAG}"
RUNTIME_CONFIG="configs/_runtime_${RUN_TAG}.yaml"

mkdir -p configs

python - <<PY
from pathlib import Path
import yaml

src = Path("${CONFIG_PATH}")
dst = Path("${RUNTIME_CONFIG}")
cfg = yaml.safe_load(src.read_text())

dataset = "${DATASET}"
dataset_cfg = cfg.pop("datasets")[dataset]
cfg.update(dataset_cfg)
cfg["output_dir"] = "${RUN_OUTPUT_DIR}"

dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print(f"Wrote runtime config to: {dst}")
print(f"dataset = {dataset}")
print(f"output_dir = ${RUN_OUTPUT_DIR}")
PY

echo
echo "Environment checks:"
which python
python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME=$HF_HOME"
echo "Dataset:       $DATASET"
echo "RUN_OUTPUT_DIR=$RUN_OUTPUT_DIR"
echo "RUNTIME_CONFIG=$RUNTIME_CONFIG"
echo

nvidia-smi || true
echo

echo "Launching training..."
python -m tinylora_gsm8k.train_grpo_nopeft --config "$RUNTIME_CONFIG"

echo
echo "Finished at $(date)"