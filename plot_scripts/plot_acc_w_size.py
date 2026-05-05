"""
Plot final averaged accuracy vs. number of trainable parameters.

- x-axis: trainable parameters (log scale)
- y-axis: final average accuracy (%)
- includes an untrained base model point
- annotates each point with percentage
- uses a tight y-range to highlight trends

Usage:
    python plot_final_acc_vs_trainable.py
    python plot_final_acc_vs_trainable.py --output final_acc_vs_trainable.png
"""

import re
import glob
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Configure your log dir here ──────────────────────────────────────────────
LOG_DIR = "/scratch/gpfs/MENGDIW/jg8305/TinyLoRA/logs"

# Each tuple: ([job ids], label, trainable_params)
GROUPS = [
    ([
        "7189607",
        "7189621",
        "7189645",
    ], "u=5,n=f,r=2", 5),

    ([
        "7189517",
        "7189546",
        "7189577",
    ], "u=13,n=f,r=2", 13),

    ([
        "7198290",
        "7198859",
        "7199334",
    ], "u=80,n=f,r=2", 80),

    ([
        "7199490",
        "7199513",
        "7218402",
    ], "u=196,n=f,r=2", 196),

    ([
        "7207646",
        "7207712",
        "7208020",
    ], "u=1,n=1,r=2", 196),

    ([
        "7207525",
        "7207547",
        "7207562",
    ], "u=13,n=1,r=2", 2458),

    ([
        "7221734",
        "7224354",
        "7224382",
    ], "u=32,n=1,r=2", 6272),
]
# ──────────────────────────────────────────────────────────────────────────────


PAT_EVAL = re.compile(r'\[Eval\].*?mode=(\S+)\s+accuracy=([\d.]+)\s+\((\d+)/(\d+)\)')
PAT_STEP = re.compile(r"['\"]step['\"]\s*:\s*(\d+)")


def id_to_path(job_id: str, log_dir: str = LOG_DIR) -> str:
    """Find the .out log file whose name starts with <job_id>_."""
    pat = re.compile(rf'^{re.escape(job_id)}_')
    matches = [
        f for f in glob.glob(f"{log_dir}/*")
        if pat.search(os.path.basename(f)) and f.endswith(".out")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No .out log file found for job ID '{job_id}' in '{log_dir}'"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple .out files match job ID '{job_id}': {matches}. "
            f"Job IDs must be unique."
        )
    return matches[0]


def parse_log(path: str):
    """
    Returns a list of dicts (one per eval checkpoint):
        {'label': str, 'step': int or None, 'acc': float, 'correct': int, 'total': int}
    """
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    eval_hits = []
    for i, line in enumerate(lines):
        m = PAT_EVAL.search(line)
        if m:
            eval_hits.append(
                (i, m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4)))
            )

    results = []
    seen_base = False

    for hit_idx, (line_idx, mode, acc, correct, total) in enumerate(eval_hits):
        step = None
        for j in range(max(0, line_idx - 5), min(len(lines), line_idx + 60)):
            ms = PAT_STEP.search(lines[j])
            if ms:
                step = int(ms.group(1))
                break

        if mode == "base_v0" and not seen_base:
            label = "Base model"
            step = 0
            seen_base = True
        elif hit_idx == 1:
            label = "LoRA init"
            step = 0
        else:
            label = f"Step {step}" if step is not None else f"Eval {hit_idx}"

        results.append({
            "label": label,
            "step": step,
            "acc": acc,
            "correct": correct,
            "total": total,
        })

    return results


def average_seeds(all_results):
    """
    Average eval checkpoints across seeds.
    """
    n_seeds = len(all_results)
    lengths = [len(r) for r in all_results]
    if len(set(lengths)) > 1:
        print(f"Warning: seeds have different eval counts {lengths}. Using min={min(lengths)}.")
    n_evals = min(lengths)

    averaged = []
    for i in range(n_evals):
        accs = [all_results[s][i]["acc"] for s in range(n_seeds)]
        averaged.append({
            "label": all_results[0][i]["label"],
            "step": all_results[0][i]["step"],
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if n_seeds > 1 else 0.0,
            "acc_per_seed": accs,
        })

    return averaged


def find_base_and_final(averaged):
    """
    Extract:
      - base model acc (first base_v0)
      - final trained acc (last eval point)
    """
    if not averaged:
        raise ValueError("No eval entries found.")

    base_entry = None
    for r in averaged:
        if r["label"] == "Base model":
            base_entry = r
            break
    if base_entry is None:
        raise ValueError("Base model eval not found in averaged results.")

    final_entry = averaged[-1]
    return base_entry, final_entry


def collect_group_results(groups):
    """
    For each configuration group:
      - parse 3 seeds
      - average them
      - extract final accuracy
    Also extract one shared base-model point from the first group.
    """
    plot_rows = []
    shared_base = None

    for seed_ids, label, trainable in groups:
        log_paths = [id_to_path(job_id) for job_id in seed_ids]

        all_results = []
        for job_id, path in zip(seed_ids, log_paths):
            print(f"Parsing {label} | job {job_id}: {path}")
            results = parse_log(path)
            print(f"  -> {len(results)} eval checkpoints found")
            all_results.append(results)

        averaged = average_seeds(all_results)
        base_entry, final_entry = find_base_and_final(averaged)

        if shared_base is None:
            shared_base = {
                "label": "Base model",
                "trainable": 1.0,   # pseudo-x for log scale
                "mean": base_entry["acc_mean"] * 100,
                "std": base_entry["acc_std"] * 100,
            }

        plot_rows.append({
            "label": label,
            "trainable": float(trainable),
            "mean": final_entry["acc_mean"] * 100,
            "std": final_entry["acc_std"] * 100,
        })

    return shared_base, sorted(plot_rows, key=lambda x: x["trainable"])


def make_tight_ylim(y_values, y_errs=None):
    """
    Make y-axis range tight enough to highlight small differences,
    while still leaving room for text annotations.
    """
    if y_errs is None:
        y_errs = [0.0] * len(y_values)

    low = min(y - e for y, e in zip(y_values, y_errs))
    high = max(y + e for y, e in zip(y_values, y_errs))

    spread = high - low
    spread = max(spread, 0.15)   # avoid completely flat axis

    pad_bottom = max(0.08, 0.18 * spread)
    pad_top = max(0.25, 0.35 * spread)   # more room for labels

    ymin = low - pad_bottom
    ymax = high + pad_top

    ymin = max(0.0, ymin)
    ymax = min(100.0, ymax)

    # avoid overly tiny range if values are extremely close
    if ymax - ymin < 0.5:
        mid = (ymax + ymin) / 2
        ymin = max(0.0, mid - 0.25)
        ymax = min(100.0, mid + 0.25)

    return ymin, ymax


def annotate_points(ax, xs, ys, labels):
    """
    Annotate each point with its percentage.
    Slight alternating vertical offsets to reduce overlap.
    """
    for i, (x, y, txt) in enumerate(zip(xs, ys, labels)):
        dy = 10 if i % 2 == 0 else -14
        va = "bottom" if dy > 0 else "top"
        ax.annotate(
            txt,
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
        )


def plot_results(base_row, trained_rows, output_path):
    rows = [base_row] + trained_rows

    # ===== 1. 如果有重复 x，保留最高 y =====
    best_per_x = {}
    for r in rows:
        x = r["trainable"]
        y = r["mean"]
        if x not in best_per_x or y > best_per_x[x]["mean"]:
            best_per_x[x] = r

    # 排序（按 x）
    rows = sorted(best_per_x.values(), key=lambda x: x["trainable"])

    xs = [r["trainable"] for r in rows]
    ys = [r["mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    # ===== 2. 连线（加粗）=====
    ax.plot(
        xs, ys,
        linewidth=3,         # 线更粗
        marker='o',
        markersize=10,       # 点更大
    )

    # ===== 3. log x =====
    ax.set_xscale("log")

    # ===== 4. annotate（字体更大）=====
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.2f}%",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=15,     # 字体变大
            fontweight="bold"
        )

    # ===== 5. x ticks =====
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["Base"] + [str(int(r["trainable"])) for r in rows[1:]],
        fontsize=18
    )

    # ===== 6. label/title =====
    ax.set_xlabel("Trainable parameters (log scale)", fontsize=18)
    ax.set_ylabel("Final average accuracy (%)", fontsize=18)
    ax.set_title("Final Accuracy vs Trainable Params", fontsize=20)

    # ===== 7. y范围（更紧一点）=====
    ymin = min(ys)
    ymax = max(ys)
    spread = max(ymax - ymin, 0.2)

    ax.set_ylim(
        ymin - 0.2 * spread,
        ymax + 0.3 * spread
    )

    # ===== 8. grid & style =====
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to: {output_path}")
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Plot final average accuracy vs. trainable parameters."
    )
    parser.add_argument(
        "--output",
        default="final_acc_vs_trainable.png",
        help="Output figure path"
    )
    args = parser.parse_args()

    base_row, trained_rows = collect_group_results(GROUPS)
    plot_results(base_row, trained_rows, args.output)


if __name__ == "__main__":
    main()