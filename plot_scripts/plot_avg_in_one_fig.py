"""
Parse training logs from multiple groups (each with multiple seeds),
and plot all groups on a single line chart with error bars.

Each SEED_IDS_* entry is a tuple: ([id1, id2, id3], "label").
Add all groups you want to compare to the GROUPS list at the top.

File names are expected to match:  <ID>_*.out
e.g.  7189621_tinylora_gsm8k_nopeft.out

Usage:
    python parse_eval_plot.py
    python parse_eval_plot.py --output my_plot.png
"""

import re
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Configure your groups here ───────────────────────────────────────────────
LOG_DIR = "/scratch/gpfs/MENGDIW/jg8305/TinyLoRA/logs"

SEED_IDS_U1_N1_R2_T196 = ([
    "7207646",
    "7207712",
    "7208020",
], "u=1,n=1,r=2,t=196")

SEED_IDS_U13_N1_R2_T2458 = ([
    "7207525",
    "7207547",
    "7207562",
], "u=13,n=1,r=2,t=2458")

# SEED_IDS_U196_NF_R2_T196 = ([
#     "7199490",
#     "7199513",
#     "xxxxxxx",
# ], "u=196,n=f,r=2,t=196")

SEED_IDS_U80_NF_R2_T80 = ([
    "7198290",
    "7198859",
    "7199334",
], "u=80,n=f,r=2,t=80")

SEED_IDS_U13_NF_R2_T13 = ([
    "7189517",
    "7189546",
    "7189577",
], "u=13,n=f,r=2,t=13")

SEED_IDS_U5_NF_R2_T5 = ([
    "7189607",
    "7189621",
    "7189645",
], "u=5,n=f,r=2,t=5")

# ── Add/remove groups to plot here ────────────────────────────────────────────
GROUPS = [
    SEED_IDS_U5_NF_R2_T5,
    SEED_IDS_U13_NF_R2_T13,
    SEED_IDS_U80_NF_R2_T80,
    SEED_IDS_U13_N1_R2_T2458,
    SEED_IDS_U1_N1_R2_T196,
]
# ──────────────────────────────────────────────────────────────────────────────


def id_to_path(job_id: str, log_dir: str = LOG_DIR) -> str:
    """Find the .out log file whose name starts with <job_id>_."""
    pat = re.compile(rf'^{re.escape(job_id)}_')
    matches = [
        f for f in glob.glob(f"{log_dir}/*")
        if pat.search(os.path.basename(f)) and f.endswith('.out')
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


PAT_EVAL = re.compile(r'\[Eval\].*?mode=(\S+)\s+accuracy=([\d.]+)\s+\((\d+)/(\d+)\)')
PAT_STEP = re.compile(r"['\"]step['\"]\s*:\s*(\d+)")


def parse_log(path: str):
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()

    eval_hits = []
    for i, line in enumerate(lines):
        m = PAT_EVAL.search(line)
        if m:
            eval_hits.append((i, m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))))

    results = []
    seen_base = False

    for hit_idx, (line_idx, mode, acc, correct, total) in enumerate(eval_hits):
        step = None
        for j in range(max(0, line_idx - 5), min(len(lines), line_idx + 60)):
            ms = PAT_STEP.search(lines[j])
            if ms:
                step = int(ms.group(1))
                break

        if mode == 'base_v0' and not seen_base:
            label = 'Base model'
            step  = 0
            seen_base = True
        elif hit_idx == 1:
            label = 'LoRA init\n(step 0)'
            step  = 0
        else:
            label = f'Step {step}' if step is not None else f'Eval {hit_idx}'

        results.append({'label': label, 'step': step, 'acc': acc, 'correct': correct, 'total': total})

    return results


def average_seeds(all_results):
    n_seeds = len(all_results)
    lengths = [len(r) for r in all_results]
    if len(set(lengths)) > 1:
        print(f"Warning: seeds have different eval counts {lengths}. Using min={min(lengths)}.")
    n_evals = min(lengths)

    averaged = []
    for i in range(n_evals):
        accs = [all_results[s][i]['acc'] for s in range(n_seeds)]
        averaged.append({
            'label':        all_results[0][i]['label'],
            'step':         all_results[0][i]['step'],
            'acc_mean':     np.mean(accs),
            'acc_std':      np.std(accs, ddof=1) if n_seeds > 1 else 0.0,
            'acc_per_seed': accs,
        })

    return averaged


def plot_all(group_results, output_path: str):
    """
    group_results: list of (title, averaged) tuples
    """
    # Use x-ticks from the group with the most evals (usually all the same)
    ref_averaged = max(group_results, key=lambda g: len(g[1]))[1]
    labels = [r['label'] for r in ref_averaged]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(9, len(x) * 1.5), 5.5))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (title, averaged) in enumerate(group_results):
        n = len(averaged)
        xi = list(range(n))
        means = [r['acc_mean'] * 100 for r in averaged]
        stds  = [r['acc_std']  * 100 for r in averaged]
        color = colors[idx % len(colors)]

        ax.plot(
            xi, means,
            color=color, linewidth=2,
            marker='o', markersize=5,
            label=title, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Eval Accuracy Comparison (3-seed average)', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Tight ylim: based on mean values only (no error bars to accommodate)
    all_means = [
        r['acc_mean'] * 100
        for _, averaged in group_results
        for r in averaged
    ]
    data_min = min(all_means)
    data_max = max(all_means)
    spread = max(data_max - data_min, 0.2)
    padding = spread * 0.2
    ax.set_ylim(data_min - padding, min(100.0, data_max + padding))

    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Console table
    for title, averaged in group_results:
        n_seeds = len(averaged[0]['acc_per_seed'])
        print(f"\n=== {title} ({n_seeds} seeds) ===")
        print(f"{'#':<4} {'Label':<25} {'Mean Acc':>10} {'Std':>8}")
        print("-" * 52)
        for i, r in enumerate(averaged):
            label_flat = r['label'].replace('\n', ' ')
            print(f"{i:<4} {label_flat:<25} {r['acc_mean']*100:>9.2f}% {r['acc_std']*100:>7.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Compare eval accuracy across multiple groups with error bars.'
    )
    parser.add_argument(
        '--output', default='eval_accuracy_comparison.png',
        help='Output plot file (default: eval_accuracy_comparison.png)'
    )
    args = parser.parse_args()

    group_results = []
    for seed_id_list, title in GROUPS:
        print(f"\n── Group: {title} ──")
        log_paths = []
        for job_id in seed_id_list:
            path = id_to_path(job_id)
            log_paths.append(path)

        all_results = []
        for job_id, path in zip(seed_id_list, log_paths):
            print(f"  Parsing job {job_id}: {path}")
            results = parse_log(path)
            print(f"    -> {len(results)} eval checkpoints found")
            all_results.append(results)

        averaged = average_seeds(all_results)
        group_results.append((title, averaged))

    plot_all(group_results, args.output)


if __name__ == '__main__':
    main()