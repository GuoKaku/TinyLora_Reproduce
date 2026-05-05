"""
Parameterized TinyLoRA GRPO training for the Figure 8 reproduction sweep.
Maps paper's `n_tie` integer to PEFT's `weight_tying` float.

Usage: python train_fig8.py --u 13 --n_tie 256 --output_dir /path/to/run
"""
import argparse
import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import TinyLoraConfig
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "End your response with the final numeric answer on its own line in the "
    "form: #### <number>"
)


def extract_gold(ans):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans)
    return m.group(1).replace(",", "") if m else ""


def extract_pred(text):
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def exact_match_reward(completions, gold, **kwargs):
    out = []
    for c, g in zip(completions, gold):
        p = extract_pred(c)
        try:
            out.append(1.0 if float(p) == float(g) else 0.0)
        except (ValueError, TypeError):
            out.append(0.0)
    return out


def ntie_to_weight_tying(n_tie: int) -> float:
    """Paper's n_tie -> PEFT's weight_tying. Verified: weight_tying = 1 - 1/n_tie."""
    if n_tie <= 1:
        return 0.0
    return 1.0 - 1.0 / n_tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=int, required=True)
    ap.add_argument("--n_tie", type=int, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    weight_tying = ntie_to_weight_tying(args.n_tie)

    tinylora_config = TinyLoraConfig(
        r=2,
        u=args.u,
        weight_tying=weight_tying,
        projection_seed=args.seed,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("openai/gsm8k", "main", split="train")
    def to_chat(ex):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
            "gold": extract_gold(ex["answer"]),
        }
    ds = ds.map(to_chat, remove_columns=ds.column_names)
    print(f"Dataset size: {len(ds)}")

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,  # effective batch = 64
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        bf16=True,
        gradient_checkpointing=True,
        num_generations=4,
        max_completion_length=1024,
        temperature=1.0,
        beta=0.0,
        num_iterations=1,
        scale_rewards=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.45,
        vllm_max_model_length=2048,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=exact_match_reward,
        args=grpo_args,
        train_dataset=ds,
        peft_config=tinylora_config,
        processing_class=tokenizer,
    )

    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n=== u={args.u}  n_tie={args.n_tie} (wt={weight_tying:.4f})  "
          f"trainable={n_trainable}  lr={args.learning_rate} ===\n")

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\n[ok] saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
