"""
Parse training logs from multiple seeds, average eval accuracies, and plot a line chart with error bars.

Edit SEED_IDS and LOG_DIR at the top of this file.
File names are expected to match the pattern:  <ID>_*.out
e.g.  7189621_tinylora_gsm8k_nopeft.out

Expected eval entries per log (7 total):
  0: base model  — [Eval] mode=base_v0  accuracy=...
  1: lora init   — [Eval] mode=lora_init accuracy=...  (step 0, before training)
  2-6: mid/end   — [Eval] mode=lora_init accuracy=...  (step 300, 600, 900, 1200, end)

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

# ─── Configure your seed job IDs here ─────────────────────────────────────────
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

SEED_IDS_U196_NF_R2_T196 = ([
    "7199490",
    "7199513",
    "7218402",
], "u=196,n=f,r=2,t=196")

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

SEED_IDS_U32_N1_R2_T6272 = ([
    "7221734",
    "7224354",
    "7224382",
], "u=32,n=1,r=2,t=6272")



SEED_IDS = SEED_IDS_U196_NF_R2_T196
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
    """
    Returns a list of dicts (one per eval checkpoint):
        {'label': str, 'step': int or None, 'acc': float, 'correct': int, 'total': int}
    """
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
        # Find nearest step number in surrounding lines
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
    """
    Given results from N seeds (each a list of dicts), return averaged results.
    Labels and steps are taken from the first seed.
    """
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


def plot(averaged, log_paths, title: str, output_path: str):
    if not averaged:
        print("No eval entries found.")
        return

    n_seeds = len(log_paths)
    labels  = [r['label'] for r in averaged]
    means   = [r['acc_mean'] * 100 for r in averaged]
    stds    = [r['acc_std']  * 100 for r in averaged]
    x       = list(range(len(averaged)))

    fig, ax = plt.subplots(figsize=(max(9, len(x) * 1.5), 5.5))

    # Individual seed lines (light dashed, for reference)
    seed_colors = ['#a8c8e8', '#f4a8a8', '#a8e8b8']
    for s, path in enumerate(log_paths):
        seed_accs = [r['acc_per_seed'][s] * 100 for r in averaged]
        ax.plot(x, seed_accs,
                color=seed_colors[s % len(seed_colors)],
                linewidth=1, linestyle='--', alpha=0.7,
                label=f'Seed {s+1}', zorder=2)

    # Mean line with error bars
    ax.errorbar(
        x, means, yerr=stds,
        color='steelblue', linewidth=2.2,
        marker='o', markersize=6,
        capsize=5, capthick=1.5, elinewidth=1.5,
        label=f'Mean ± std (n={n_seeds})',
        zorder=4,
    )

    # Annotate mean points
    for xi, (mean, std) in enumerate(zip(means, stds)):
        ax.annotate(
            f'{mean:.2f}%\n±{std:.2f}%',
            xy=(xi, mean),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=8, color='#222',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Eval Accuracy — {title} (3-seed average)', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Tight ylim: based on actual data range across all seeds
    all_seed_accs = [r['acc_per_seed'][s] * 100 for r in averaged for s in range(n_seeds)]
    data_min = min(all_seed_accs)
    data_max = max(all_seed_accs)
    spread = max(data_max - data_min, 0.2)  # at least 0.2% spread
    padding = spread * 0.4
    ymin = data_min - padding
    ymax = min(100.0, data_max + padding + spread * 0.3)  # extra for annotations
    ax.set_ylim(ymin, ymax)

    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Console table
    print(f"\n=== Averaged Eval Results ({n_seeds} seeds) ===")
    print(f"{'#':<4} {'Label':<25} {'Mean Acc':>10} {'Std':>8}  Per-seed accs")
    print("-" * 70)
    for i, r in enumerate(averaged):
        label_flat = r['label'].replace('\n', ' ')
        per_seed_str = '  '.join(f"{v*100:.2f}%" for v in r['acc_per_seed'])
        print(f"{i:<4} {label_flat:<25} {r['acc_mean']*100:>9.2f}% {r['acc_std']*100:>7.2f}%  {per_seed_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Average eval accuracies across seeds and plot a line chart with error bars.'
    )
    parser.add_argument(
        '--output', default='eval_accuracy.png',
        help='Output plot file (default: eval_accuracy.png)'
    )
    args = parser.parse_args()

    seed_id_list, title = SEED_IDS

    log_paths = []
    for job_id in seed_id_list:
        path = id_to_path(job_id)
        log_paths.append(path)

    all_results = []
    for job_id, path in zip(seed_id_list, log_paths):
        print(f"Parsing job {job_id}: {path}")
        results = parse_log(path)
        print(f"  -> {len(results)} eval checkpoints found")
        all_results.append(results)

    averaged = average_seeds(all_results)
    plot(averaged, log_paths, title, f"eval_accuracy_{title.replace(' ', '_')}.png")


if __name__ == '__main__':
    main()