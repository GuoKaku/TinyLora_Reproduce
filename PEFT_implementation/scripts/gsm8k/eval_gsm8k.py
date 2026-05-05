"""
Evaluate a trained TinyLoRA adapter on GSM8K test split.
Reports pass@1 (greedy decoding) to compare against paper's 91% headline.

Strategy: load base model + TinyLoRA adapter via PEFT, merge into base weights,
then run inference with vLLM on the merged model. This avoids vLLM's LoRA kernel
limitations (which don't support TinyLoRA natively).
"""
import re
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- Config ----------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "/workspace/runs/tinylora_u13_gsm8k/final"
MERGED_DIR = "/workspace/runs/tinylora_u13_gsm8k/merged"
RESULTS_PATH = "/workspace/runs/tinylora_u13_gsm8k/eval_results.jsonl"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "End your response with the final numeric answer on its own line in the "
    "form: #### <number>"
)

def extract_gold(answer_field: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return m.group(1).replace(",", "") if m else ""

def extract_pred(text: str) -> str:
    matches = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""

def numeric_equal(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False

def merge_adapter():
    """Load base + TinyLoRA adapter, merge weights, save to disk."""
    if os.path.exists(os.path.join(MERGED_DIR, "config.json")):
        print(f"Merged model already exists at {MERGED_DIR}, skipping merge.")
        return

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print("Loading TinyLoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    # Count adapter params before merge
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Adapter trainable params: {n_trainable}")

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}...")
    merged.save_pretrained(MERGED_DIR)

    # Also copy tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.save_pretrained(MERGED_DIR)
    import gc
    del merged, base, model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print("Merge complete, GPU memory released.")

def evaluate():
    """Run greedy eval on GSM8K test with the merged model via vLLM."""
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)

    # Load test set
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Test set size: {len(ds)}")

    # Build prompts
    prompts, golds, questions = [], [], []
    for ex in ds:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))
        golds.append(extract_gold(ex["answer"]))
        questions.append(ex["question"])

    # vLLM on merged model — no LoRA needed
    llm = LLM(
        model=MERGED_DIR,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )

    sampling = SamplingParams(
        temperature=0.0,   # greedy for pass@1
        max_tokens=MAX_NEW_TOKENS,
    )

    print(f"Running inference on {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling)

    # Score
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    n_correct = 0
    with open(RESULTS_PATH, "w") as f:
        for q, gold, out in zip(questions, golds, outputs):
            completion = out.outputs[0].text
            pred = extract_pred(completion)
            correct = numeric_equal(pred, gold)
            n_correct += int(correct)
            f.write(json.dumps({
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "completion": completion,
            }) + "\n")

    acc = n_correct / len(prompts)
    print(f"\n{'='*40}")
    print(f"  RESULTS")
    print(f"{'='*40}")
    print(f"  Correct: {n_correct} / {len(prompts)}")
    print(f"  pass@1:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Paper:   0.9100    (91.00%)")
    print(f"  Delta:   {(acc - 0.91)*100:+.2f} pp")
    print(f"{'='*40}")
    print(f"\nPer-example results: {RESULTS_PATH}")

def main():
    merge_adapter()
    evaluate()

if __name__ == "__main__":
    main()
