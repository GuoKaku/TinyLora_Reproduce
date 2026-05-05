"""
Parse training log to extract eval accuracies and plot a line chart.

Expected eval entries (7 total):
  0: base model  — [Eval] mode=base_v0  accuracy=...
  1: lora init   — [Eval] mode=lora_init accuracy=...   (step 0, before training)
  2-6: mid/end   — [Eval] mode=lora_init accuracy=...   (step 300, 600, 900, 1200, end)

All lines share the same format:
    [Eval] mode=<mode> accuracy=<float> (<correct>/<total>)

Step numbers are inferred from nearby log lines containing 'step': <int>.

Usage:
    python parse_eval_plot.py <log_file>
    python parse_eval_plot.py <log_file> --output my_plot.png
"""

import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Matches any [Eval] accuracy line
PAT_EVAL = re.compile(
    r'\[Eval\].*?mode=(\S+)\s+accuracy=([\d.]+)\s+\((\d+)/(\d+)\)'
)
# Matches step number from a metrics dict line
PAT_STEP = re.compile(r"['\"]step['\"]\s*:\s*(\d+)")


def parse_log(path: str):
    """
    Returns a list of dicts:
        {'label': str, 'step': int or None, 'acc': float, 'correct': int, 'total': int}
    in the order they appear in the log.
    """
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()

    eval_hits = []   # (line_idx, mode, acc, correct, total)
    for i, line in enumerate(lines):
        m = PAT_EVAL.search(line)
        if m:
            mode    = m.group(1)
            acc     = float(m.group(2))
            correct = int(m.group(3))
            total   = int(m.group(4))
            eval_hits.append((i, mode, acc, correct, total))

    results = []
    seen_base = False

    for hit_idx, (line_idx, mode, acc, correct, total) in enumerate(eval_hits):
        # --- Determine step number by searching surrounding lines ---
        step = None
        search_start = max(0, line_idx - 5)
        search_end   = min(len(lines), line_idx + 60)
        for j in range(search_start, search_end):
            ms = PAT_STEP.search(lines[j])
            if ms:
                step = int(ms.group(1))
                break

        # --- Assign human-readable label ---
        if mode == 'base_v0' and not seen_base:
            label = 'Base model'
            step  = 0          # conceptually step 0
            seen_base = True
        elif hit_idx == 1:
            # second entry = lora_init before any gradient step
            label = 'LoRA init\n(step 0)'
            step  = 0
        else:
            label = f'Step {step}' if step is not None else f'Eval {hit_idx}'

        results.append({
            'label':   label,
            'step':    step,
            'acc':     acc,
            'correct': correct,
            'total':   total,
        })

    return results


def plot(results, output_path: str):
    if not results:
        print("No eval entries found — check your log file.")
        return

    labels  = [r['label'] for r in results]
    accs    = [r['acc'] * 100 for r in results]
    details = [f"{r['correct']}/{r['total']}" for r in results]
    x       = list(range(len(results)))

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 1.8), 6.2))

    # ===== Line + markers: bigger & thicker =====
    ax.plot(
        x, accs,
        linewidth=3.2,
        marker='o',
        markersize=10.5,
        zorder=3,
    )

    # ===== Annotate each point: larger text =====
    for xi, (acc, detail) in enumerate(zip(accs, details)):
        ax.annotate(
            f'{acc:.2f}%\n({detail})',
            xy=(xi, acc),
            xytext=(0, 14),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=13,
            fontweight='bold',
        )

    # ===== Axes / labels =====
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=17)
    ax.set_title('Eval Accuracy across Training', fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # ===== Tighter y-range =====
    ymin = min(accs)
    ymax = max(accs)
    spread = max(ymax - ymin, 0.3)

    ax.set_ylim(
        ymin - 0.18 * spread,
        min(100.5, ymax + 0.28 * spread)
    )

    # ===== Grid / style =====
    ax.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    print(f"Plot saved to: {output_path}")

    # Console table
    print("\n=== Extracted Eval Results ===")
    print(f"{'#':<4} {'Label':<28} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 60)
    for i, r in enumerate(results):
        label_flat = r['label'].replace('\n', ' ')
        frac = f"{r['correct']}/{r['total']}"
        print(f"{i:<4} {label_flat:<28} {r['acc']*100:>9.2f}% {frac:>15}")

def main():
    parser = argparse.ArgumentParser(
        description='Parse training log and plot eval accuracies as a line chart.'
    )
    parser.add_argument('log_file', help='Path to the training log file')
    parser.add_argument(
        '--output', default='eval_accuracy.png',
        help='Output plot file (default: eval_accuracy.png)'
    )
    args = parser.parse_args()

    results = parse_log(args.log_file)
    plot(results, args.output)


if __name__ == '__main__':
    main()