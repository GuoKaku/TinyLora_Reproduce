"""
Reproduce Figure 8 (Qwen2.5-3B-Instruct panel) from sweep eval results.

Reads per-config eval_results.jsonl files, computes the three pass@1 scores,
and plots them against trainable parameter count on a log x-axis, grouped
by n_tie (one line per n_tie value).

Run from your Mac:
    python plot_fig8.py \\
        --sweep_dir ~/Desktop/tinylora_reproduction/fig8_sweep \\
        --out fig8_reproduction.png
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


# Qwen2.5-3B-Instruct total linear module count for target_modules list
N_MODULES = 252  # 36 layers × 7 modules


def parse_config_name(name: str):
    """Parse 'u4_ntie8' -> (u=4, n_tie=8)."""
    m = re.match(r"u(\d+)_ntie(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def trainable_params(u: int, n_tie: int) -> int:
    """params = n × m × u / n_tie, clamped when n_tie > n×m."""
    if n_tie >= N_MODULES:
        return u
    # Use round() to handle non-integer divisions (matches PEFT's behavior)
    return round(N_MODULES * u / n_tie)


def load_eval_results(jsonl_path: Path):
    """Count correct predictions under each of the three extractors."""
    n, h, s, x = 0, 0, 0, 0
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            n += 1
            h += int(d["hash_ok"])
            s += int(d["strict_ok"])
            x += int(d["flex_ok"])
    return n, h, s, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="fig8_reproduction.png")
    ap.add_argument("--metric", type=str, default="flexible",
                    choices=["hash_only", "strict", "flexible"],
                    help="Which extractor to plot as the main metric")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs_dir = sweep_dir / "runs"

    # Collect {n_tie: [(params, hash_only, strict, flexible), ...]}
    data = {}
    missing = []
    for run_dir in sorted(runs_dir.glob("u*_ntie*")):
        u, n_tie = parse_config_name(run_dir.name)
        if u is None:
            continue
        jsonl = run_dir / "eval_results.jsonl"
        if not jsonl.exists():
            missing.append(run_dir.name)
            continue
        n, h, s, x = load_eval_results(jsonl)
        params = trainable_params(u, n_tie)
        data.setdefault(n_tie, []).append({
            "u": u, "params": params,
            "hash_only": h / n, "strict": s / n, "flexible": x / n,
            "n": n,
        })
        print(f"  {run_dir.name:15s}  params={params:6d}  "
              f"hash={h/n:.3f}  strict={s/n:.3f}  flex={x/n:.3f}")

    if missing:
        print(f"\n[warn] missing eval_results.jsonl for: {missing}")

    if not data:
        print("[error] no eval data found")
        return

    # Sort points within each line by parameter count
    for n_tie in data:
        data[n_tie].sort(key=lambda d: d["params"])

    # ---- Plot: one panel per extractor ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    extractors = ["hash_only", "strict", "flexible"]
    titles = [
        "hash_only (#### only)",
        "strict (####, last_num fallback)",
        "flexible (####, boxed, last_num)",
    ]

    # Color scheme matching paper: viridis-like, dark-to-yellow
    n_tie_values = sorted(data.keys())
    colors = plt.cm.viridis([0.0, 0.3, 0.6, 0.9][:len(n_tie_values)])
    color_map = dict(zip(n_tie_values, colors))

    for ax, extractor, title in zip(axes, extractors, titles):
        for n_tie in n_tie_values:
            xs = [d["params"] for d in data[n_tie]]
            ys = [d[extractor] for d in data[n_tie]]
            ax.plot(xs, ys, marker="o", color=color_map[n_tie],
                    label=f"n_tie={n_tie}", lw=2, markersize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.3, 1.0)

    axes[0].set_ylabel("GSM8K test pass@1")
    axes[0].legend(loc="lower right", title="n_tie", fontsize=9)
    fig.suptitle("Figure 8 reproduction — Qwen2.5-3B-Instruct, 1 epoch, lr=5e-5",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")

    # ---- Paper-style single-panel version ----
    fig2, ax = plt.subplots(figsize=(7, 5))
    for n_tie in n_tie_values:
        xs = [d["params"] for d in data[n_tie]]
        ys = [d[args.metric] for d in data[n_tie]]
        ax.plot(xs, ys, marker="o", color=color_map[n_tie],
                label=f"{n_tie}", lw=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel(f"GSM8K test pass@1 ({args.metric})")
    ax.set_title("Qwen2.5-3B-Instruct")
    ax.legend(title="n_tie", loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 0.95)
    plt.tight_layout()
    out2 = args.out.replace(".png", "_paper_style.png")
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
