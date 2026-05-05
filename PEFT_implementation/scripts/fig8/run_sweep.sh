#!/bin/bash
set -u

SWEEP_DIR="/workspace/fig8_sweep"
SCRIPT="$SWEEP_DIR/scripts/train_fig8.py"
LOG_DIR="$SWEEP_DIR/logs"
RUN_DIR="$SWEEP_DIR/runs"
VENV="/workspace/venvs/tinylora/bin/activate"

mkdir -p "$LOG_DIR" "$RUN_DIR"
source "$VENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
  "1 256"   "4 256"   "16 256"   "64 256"
  "1 64"    "4 64"    "16 64"    "64 64"
  "1 8"     "4 8"     "16 8"     "64 8"
  "1 1"     "4 1"     "16 1"     "64 1"
)

echo "=== Sweep started $(date) ==="
echo "Total configs: ${#CONFIGS[@]}"

for cfg in "${CONFIGS[@]}"; do
  read -r u ntie <<< "$cfg"
  name="u${u}_ntie${ntie}"
  log="$LOG_DIR/${name}.log"
  run_path="$RUN_DIR/${name}"
  done_marker="$run_path/.done"

  if [[ -f "$done_marker" ]]; then
    echo "[skip] $name"
    continue
  fi

  echo "[start] $name at $(date)"
  python "$SCRIPT" --u "$u" --n_tie "$ntie" --output_dir "$run_path" > "$log" 2>&1

  if [[ $? -eq 0 ]]; then
    touch "$done_marker"
    echo "[done]  $name at $(date)"
  else
    echo "[FAIL]  $name (last 15 lines):"
    tail -15 "$log" | sed 's/^/        /'
  fi

  sleep 20
  nvidia-smi | head -10
done

echo "=== Sweep finished $(date) ==="
