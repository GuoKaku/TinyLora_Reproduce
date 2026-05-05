#!/usr/bin/env python3
"""
Show wrong predictions from eval.json
Usage: python scripts/show_wrong.py <eval_json_path> [--n N] [--tail N]
Output: wrong_examples.txt in the same directory as eval.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_path", type=str, help="Path to eval.json")
    parser.add_argument("--n", type=int, default=None, help="Number of wrong examples to show (default: all)")
    parser.add_argument("--tail", type=int, default=400, help="Show last N chars of completion")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_path = eval_path.parent / "wrong_examples.txt"

    with open(eval_path) as f:
        data = json.load(f)

    predictions = data["predictions"]
    wrong = [p for p in predictions if not p["correct"]]
    show_n = args.n if args.n is not None else len(wrong)

    lines = []
    lines.append(f"Total    : {data['num_samples']}")
    lines.append(f"Correct  : {data['correct']}")
    lines.append(f"Wrong    : {len(wrong)}")
    lines.append(f"Accuracy : {data['accuracy']:.4f}")
    lines.append("")

    for i, p in enumerate(wrong[:show_n]):
        lines.append("=" * 60)
        lines.append(f"Wrong #{i+1} / {len(wrong)}")
        lines.append("=" * 60)
        lines.append(f"Question : {p['question']}")
        lines.append(f"Gold     : {p['gold']}")
        lines.append(f"Pred     : {p['pred']}")
        lines.append(f"Completion (last {args.tail} chars):")
        lines.append(f"...{p['completion'][-args.tail:]}")
        lines.append("")

    output = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    print(f"\n✅ Saved to: {output_path}")


if __name__ == "__main__":
    main()