#!/bin/bash
set -u

SWEEP_DIR="/workspace/fig8_sweep"
SCRIPT="$SWEEP_DIR/scripts/eval_fig8.py"
LOG_DIR="$SWEEP_DIR/logs/eval"
RUN_DIR="$SWEEP_DIR/runs"
VENV="/workspace/venvs/tinylora/bin/activate"

mkdir -p "$LOG_DIR"
source "$VENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Eval sweep started $(date) ==="

for run_path in "$RUN_DIR"/u*_ntie*; do
    name=$(basename "$run_path")
    log="$LOG_DIR/${name}.log"
    eval_marker="$run_path/.eval_done"

    if [[ -f "$eval_marker" ]]; then
        echo "[skip] $name (eval done)"
        continue
    fi

    if [[ ! -d "$run_path/final" ]]; then
        echo "[skip] $name (no adapter)"
        continue
    fi

    echo "[eval] $name at $(date)"
    python "$SCRIPT" --run_dir "$run_path" > "$log" 2>&1

    if [[ $? -eq 0 ]]; then
        touch "$eval_marker"
        # Extract and display the three scores
        grep -E "hash_only:|strict:|flexible:" "$log"
        echo ""
    else
        echo "[FAIL] $name:"
        tail -10 "$log" | sed 's/^/      /'
    fi

    sleep 10
done

echo "=== Eval sweep finished $(date) ==="

# Build summary table
echo ""
echo "=== Summary ==="
printf "%-15s %10s %10s %10s\n" "config" "hash_only" "strict" "flexible"
for log in "$LOG_DIR"/u*_ntie*.log; do
    name=$(basename "$log" .log)
    h=$(grep "hash_only:" "$log" | sed -E 's/.*= ([0-9.]+)%.*/\1/')
    s=$(grep "strict:"    "$log" | sed -E 's/.*= ([0-9.]+)%.*/\1/')
    x=$(grep "flexible:"  "$log" | sed -E 's/.*= ([0-9.]+)%.*/\1/')
    printf "%-15s %10s %10s %10s\n" "$name" "$h" "$s" "$x"
done | sort | tee "$SWEEP_DIR/logs/eval/summary.txt"
