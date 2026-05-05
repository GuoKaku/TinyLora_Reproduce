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


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


PAPER_REPORT = [
    {"trainable": 1.0,  "display_x": "0",    "mean": 79.2},   # 0 不能上 log 轴，用 1.0 占位
    {"trainable": 4.0, "display_x": "13",   "mean": 84.3},
    {"trainable": 7.0, "display_x": "49",   "mean": 85.2},
    {"trainable": 9.0,"display_x": "196",  "mean": 84.5},
    {"trainable": 13.0,"display_x": "392",  "mean": 90.2},
    {"trainable": 30.0,"display_x": "6272","mean": 92.5},
    {"trainable": 70.0, "display_x": "13",   "mean": 93.1},
    {"trainable": 80.0, "display_x": "49",   "mean": 92.4},
    {"trainable": 96.0,"display_x": "196",  "mean": 94.2},
    {"trainable": 196.0,"display_x": "392",  "mean": 94.8},
    # {"trainable": 6272.0,"display_x": "6272","mean": 91.9},
]


def plot_results(base_row, trained_rows, output_path):
    # ===== our reproduction =====
    our_rows = [base_row] + trained_rows

    # 同一个 x 只保留最高 y
    best_our = {}
    for r in our_rows:
        x = float(r["trainable"])
        y = float(r["mean"])
        if x not in best_our or y > best_our[x]["mean"]:
            best_our[x] = r
    our_rows = sorted(best_our.values(), key=lambda r: r["trainable"])

    # ===== paper report =====
    best_paper = {}
    for r in PAPER_REPORT:
        x = float(r["trainable"])
        y = float(r["mean"])
        if x not in best_paper or y > best_paper[x]["mean"]:
            best_paper[x] = r
    paper_rows = sorted(best_paper.values(), key=lambda r: r["trainable"])

    our_x = [r["trainable"] for r in our_rows]
    our_y = [r["mean"] for r in our_rows]

    paper_x = [r["trainable"] for r in paper_rows]
    paper_y = [r["mean"] for r in paper_rows]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # ===== our line + points =====
    ax.plot(
        our_x, our_y,
        marker="o",
        markersize=11,
        linewidth=3.2,
        label="Our reproduction",
        zorder=3,
    )

    # ===== paper line + points =====
    ax.plot(
        paper_x, paper_y,
        marker="^",
        markersize=12,
        linewidth=3.2,
        label="Paper report",
        zorder=4,
    )

    ax.set_xscale("log")

    # ===== annotate our points =====
    for x, y in zip(our_x, our_y):
        ax.annotate(
            f"{y:.2f}%",
            xy=(x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # ===== annotate paper points =====
    for x, y in zip(paper_x, paper_y):
        ax.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    # ===== xticks =====
    tick_map = {
        1.0: "0",       # log 轴上不能放 0，所以还是用 1.0 占位，但显示成 0
        4.0: "13",
        7.0: "49",
        9.0: "196",
        13.0: "392",
        30.0: "6272",
        70.0: "13",
        80.0: "49",
        96.0: "196",
        196.0: "392",
        6272.0: "6272",
    }

    # 把我们自己的真实 trainable 也加入 tick
    for r in our_rows:
        x = float(r["trainable"])
        if x not in tick_map:
            tick_map[x] = str(int(x))

    xticks = sorted(tick_map.keys())
    ax.set_xticks(xticks)
    ax.set_xticklabels([tick_map[x] for x in xticks], fontsize=10)

    # ===== labels / title =====
    ax.set_xlabel("Trainable parameters (log scale)", fontsize=18)
    ax.set_ylabel("Final average accuracy (%)", fontsize=18)
    ax.set_title("Final Accuracy vs Trainable Parameters", fontsize=20)

    ax.tick_params(axis="y", labelsize=15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ===== tighter y range =====
    all_y = our_y + paper_y
    ymin = min(all_y)
    ymax = max(all_y)
    spread = max(ymax - ymin, 0.25)

    ax.set_ylim(
        ymin - 0.18 * spread,
        ymax + 0.28 * spread
    )

    # ===== style =====
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
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