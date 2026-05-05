from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You are a helpful math assistant. Solve the problem step by step. End your response with the final numeric answer on its own line in the form: #### <number>"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ds = load_dataset("openai/gsm8k", "main", split="test")

prompts, golds = [], []
for ex in ds:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}]
    prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    m = re.search(r"####\s*(-?[\d,]+)", ex["answer"])
    golds.append(m.group(1).replace(",", "") if m else "")

llm = LLM(model=MODEL_ID, max_model_len=2048, gpu_memory_utilization=0.9, dtype="bfloat16")
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))

correct = 0
for g, o in zip(golds, outputs):
    matches = re.findall(r"####\s*(-?[\d,]+)", o.outputs[0].text)
    pred = matches[-1].replace(",", "") if matches else ""
    try:
        correct += float(pred) == float(g)
    except:
        pass

print(f"Baseline: {correct}/{len(prompts)} = {correct/len(prompts)*100:.2f}%")
