"""
Parameterized unified eval for the Fig 8 sweep.
Merges adapter, runs vLLM inference, reports hash_only / strict / flexible scores.
"""
import argparse
import os
import re
import json
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "End your response with the final numeric answer on its own line in the "
    "form: #### <number>"
)
MAX_NEW_TOKENS = 1024

HASH_RE   = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
BOXED_RE  = re.compile(r"\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}")
ANYNUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract_gold(a):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", a)
    return m.group(1).replace(",", "") if m else ""

def extract_hash_only(t):
    m = HASH_RE.findall(t)
    return (m[-1].replace(",", ""), "hash") if m else ("", "none")

def extract_strict(t):
    m = HASH_RE.findall(t)
    if m: return m[-1].replace(",", ""), "hash"
    n = ANYNUM_RE.findall(t.replace(",", ""))
    return (n[-1], "last_num") if n else ("", "none")

def extract_flexible(t):
    m = HASH_RE.findall(t)
    if m: return m[-1].replace(",", ""), "hash"
    b = BOXED_RE.findall(t)
    if b: return b[-1].replace(",", ""), "boxed"
    n = ANYNUM_RE.findall(t.replace(",", ""))
    return (n[-1], "last_num") if n else ("", "none")

def numeq(a, b):
    try: return float(a) == float(b)
    except: return False

def merge(base_id, adapter_dir, merged_dir):
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"[merge] exists, skipping")
        return
    print(f"[merge] loading base {base_id}")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    print(f"[merge] attaching adapter {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    print(f"[merge] merging")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(merged_dir)
    del merged, base, model
    gc.collect()
    torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to run dir containing final/")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    adapter_dir = os.path.join(args.run_dir, "final")
    merged_dir  = os.path.join(args.run_dir, "merged")
    results_path = os.path.join(args.run_dir, "eval_results.jsonl")

    if not os.path.exists(adapter_dir):
        print(f"[error] no adapter at {adapter_dir}")
        return

    merge(args.base, adapter_dir, merged_dir)

    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    ds = load_dataset("openai/gsm8k", "main", split="test")

    prompts, golds, questions = [], [], []
    for ex in ds:
        msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ex["question"]}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        golds.append(extract_gold(ex["answer"]))
        questions.append(ex["question"])

    llm = LLM(model=merged_dir, max_model_len=2048, gpu_memory_utilization=0.9, dtype="bfloat16")
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))

    n = len(outputs)
    hits = {"hash_only": 0, "strict": 0, "flexible": 0}
    with open(results_path, "w") as f:
        for q, g, out in zip(questions, golds, outputs):
            comp = out.outputs[0].text
            h, _  = extract_hash_only(comp)
            s, sm = extract_strict(comp)
            x, xm = extract_flexible(comp)
            ho, so, xo = numeq(h,g), numeq(s,g), numeq(x,g)
            hits["hash_only"] += int(ho)
            hits["strict"]    += int(so)
            hits["flexible"]  += int(xo)
            f.write(json.dumps({"q":q,"gold":g,"hash":h,"strict":s,"flex":x,
                                "hash_ok":ho,"strict_ok":so,"flex_ok":xo,
                                "strict_m":sm,"flex_m":xm,"comp":comp})+"\n")

    print(f"\n=== {os.path.basename(args.run_dir)} ===")
    print(f"  hash_only:  {hits['hash_only']}/{n} = {100*hits['hash_only']/n:.2f}%")
    print(f"  strict:     {hits['strict']}/{n} = {100*hits['strict']/n:.2f}%")
    print(f"  flexible:   {hits['flexible']}/{n} = {100*hits['flexible']/n:.2f}%")

if __name__ == "__main__":
    main()
