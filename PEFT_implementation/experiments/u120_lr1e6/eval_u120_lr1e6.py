"""
Evaluate a trained TinyLoRA adapter on GSM8K test split.
Reports three pass@1 numbers from ONE generation pass:

  - hash_only: accept only `#### <number>`. No fallback. Truly strict.
  - strict:    exact replica of the original eval_gsm8k.py extract_pred:
               try `####` first, fall back to the last number in the text.
               This is what produced the 90.98% number previously.
  - flexible:  try `####`, then `\boxed{<n>}`, then last number.

All three scores come from the SAME generated outputs, so any difference
between them is purely about answer-extraction, not sampling or prompting.

Strategy: load base + TinyLoRA adapter via PEFT, merge into base weights,
then run inference with vLLM on the merged model. Identical to original.
"""
import re
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- Config (unchanged from original) ----------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "/workspace/runs/tinylora_u120_lr1e6_gsm8k/final"
MERGED_DIR = "/workspace/runs/tinylora_u120_lr1e6_gsm8k/merged"
RESULTS_PATH = "/workspace/runs/tinylora_u120_lr1e6_gsm8k/eval_results.jsonl"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "End your response with the final numeric answer on its own line in the "
    "form: #### <number>"
)

# ---------- Gold extraction (unchanged) ----------
def extract_gold(answer_field: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return m.group(1).replace(",", "") if m else ""

# ---------- Three extractors, all operating on the SAME generation ----------

HASH_RE   = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
BOXED_RE  = re.compile(r"\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}")
ANYNUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract_hash_only(text: str):
    """Truly strict: only accept `#### <n>`. Returns (pred, method)."""
    m = HASH_RE.findall(text)
    if m:
        return m[-1].replace(",", ""), "hash"
    return "", "none"

def extract_strict_original(text: str):
    """
    Bit-for-bit match to the original extract_pred:
      1. try `####` (last match)
      2. fall back to the last plain number in the text (commas stripped)
    """
    matches = HASH_RE.findall(text)
    if matches:
        return matches[-1].replace(",", ""), "hash"
    nums = ANYNUM_RE.findall(text.replace(",", ""))
    if nums:
        return nums[-1], "last_num"
    return "", "none"

def extract_flexible(text: str):
    """
    Adds `\\boxed{}` as an intermediate fallback between #### and last_num.
      1. ####   -> 'hash'
      2. \\boxed{n} -> 'boxed'
      3. last number -> 'last_num'
    """
    m = HASH_RE.findall(text)
    if m:
        return m[-1].replace(",", ""), "hash"
    b = BOXED_RE.findall(text)
    if b:
        return b[-1].replace(",", ""), "boxed"
    nums = ANYNUM_RE.findall(text.replace(",", ""))
    if nums:
        return nums[-1], "last_num"
    return "", "none"

def numeric_equal(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False

# ---------- Merge step (unchanged from original) ----------
def merge_adapter():
    if os.path.exists(os.path.join(MERGED_DIR, "config.json")):
        print(f"Merged model already exists at {MERGED_DIR}, skipping merge.")
        return

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print("Loading TinyLoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Adapter trainable params: {n_trainable}")

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}...")
    merged.save_pretrained(MERGED_DIR)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.save_pretrained(MERGED_DIR)
    import gc
    del merged, base, model, tok
    gc.collect()
    torch.cuda.empty_cache()
    print("Merge complete, GPU memory released.")

# ---------- Eval ----------
def evaluate():
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Test set size: {len(ds)}")

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

    llm = LLM(
        model=MERGED_DIR,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    print(f"Running inference on {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling)

    # Score all three modes on the same outputs
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    n = len(prompts)
    hits = {"hash_only": 0, "strict": 0, "flexible": 0}
    flex_methods = {"hash": 0, "boxed": 0, "last_num": 0, "none": 0}
    strict_methods = {"hash": 0, "last_num": 0, "none": 0}

    with open(RESULTS_PATH, "w") as f:
        for q, gold, out in zip(questions, golds, outputs):
            completion = out.outputs[0].text

            h_pred, _        = extract_hash_only(completion)
            s_pred, s_method = extract_strict_original(completion)
            x_pred, x_method = extract_flexible(completion)

            h_ok = numeric_equal(h_pred, gold)
            s_ok = numeric_equal(s_pred, gold)
            x_ok = numeric_equal(x_pred, gold)

            hits["hash_only"] += int(h_ok)
            hits["strict"]    += int(s_ok)
            hits["flexible"]  += int(x_ok)
            strict_methods[s_method] += 1
            flex_methods[x_method]   += 1

            f.write(json.dumps({
                "question": q,
                "gold": gold,
                "hash_only_pred": h_pred,
                "strict_pred": s_pred,   "strict_method": s_method,
                "flexible_pred": x_pred, "flexible_method": x_method,
                "hash_only_correct": h_ok,
                "strict_correct":    s_ok,
                "flexible_correct":  x_ok,
                "completion": completion,
            }) + "\n")

    # Report
    print(f"\n{'='*60}")
    print(f"  RESULTS  (adapter={ADAPTER_DIR})")
    print(f"{'='*60}")
    print(f"  hash_only (#### only):           "
          f"{hits['hash_only']}/{n} = {100*hits['hash_only']/n:.2f}%")
    print(f"  strict    (####, then last_num): "
          f"{hits['strict']}/{n} = {100*hits['strict']/n:.2f}%   "
          f"<-- matches original eval_gsm8k.py")
    print(f"  flexible  (####, boxed, last):   "
          f"{hits['flexible']}/{n} = {100*hits['flexible']/n:.2f}%")
    print(f"\n  Strict method breakdown:   {strict_methods}")
    print(f"  Flexible method breakdown: {flex_methods}")
    print(f"\n  Paper u=13 claim: 91.00%")
    print(f"{'='*60}")
    print(f"\nPer-example results: {RESULTS_PATH}")

def main():
    merge_adapter()
    evaluate()

if __name__ == "__main__":
    main()
