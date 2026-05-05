"""
Reproduce TinyLoRA Figure 1 headline: Qwen2.5-7B-Instruct + GRPO on GSM8K
with u=1, weight_tying=1.0 -> 13 trainable parameters.

Paper: Learning to Reason in 13 Parameters (Morris et al., 2026)
"""
import re
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import TinyLoraConfig
from trl import GRPOConfig, GRPOTrainer

# ---------- Config ----------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/workspace/runs/tinylora_u1_gsm8k"
SEED = 42

# TinyLoRA: r=2 (paper's main setting), u=1, full weight tying -> 13 params total
tinylora_config = TinyLoraConfig(
    r=2,
    u=1,
    weight_tying=1.0,  # 1.0 = all target modules share one v vector
    projection_seed=SEED,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# ---------- Dataset ----------
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "End your response with the final numeric answer on its own line in the "
    "form: #### <number>"
)

def extract_gold(answer_field: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return m.group(1).replace(",", "") if m else ""

def extract_pred(text: str) -> str:
    """Pull the final '#### <number>' from a model completion."""
    matches = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(",", "")
    # Fallback: last number in the completion
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""

def build_dataset(tokenizer):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    def to_chat(ex):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "gold": extract_gold(ex["answer"])}
    ds = ds.map(to_chat, remove_columns=ds.column_names)
    return ds

# ---------- Reward ----------
def exact_match_reward(completions, gold, **kwargs):
    """1.0 if the extracted numeric answer matches gold, else 0.0."""
    rewards = []
    for comp, g in zip(completions, gold):
        pred = extract_pred(comp)
        try:
            rewards.append(1.0 if float(pred) == float(g) else 0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards

# ---------- Main ----------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(tokenizer)
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Example prompt:\n{train_ds[0]['prompt'][:500]}")
    print(f"Example gold: {train_ds[0]['gold']}")

    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        # --- Training schedule ---
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,  # effective batch = 64
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        bf16=True,
        gradient_checkpointing=True,
        # --- GRPO ---
        num_generations=4,
        max_completion_length=1024,  # paper uses 4096; cut for speed, extend if you have time
        temperature=1.0,
        beta=0.0,          # paper: no KL penalty for GSM8K
        num_iterations=1,
        scale_rewards=True,
        # --- vLLM for rollouts (colocate on same GPU) ---
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.45,
        vllm_max_model_length=2048,
        # --- Logging / saving ---
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=exact_match_reward,
        args=args,
        train_dataset=train_ds,
        peft_config=tinylora_config,
        processing_class=tokenizer,
    )

    # Print trainable parameter count — should be exactly 13
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"\n=== Parameter count ===")
    print(f"Trainable: {n_trainable}")
    print(f"Total:     {n_total}")
    print(f"Ratio:     {n_trainable / n_total:.2e}")
    print("=======================\n")

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))

if __name__ == "__main__":
    main()
