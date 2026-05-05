from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import TinyLoraConfig, get_peft_model

from .config import ExperimentConfig
from .data import load_gsm8k_prompt_only_dataset
from .rewards import gsm8k_exact_match_reward
from .utils import (
    compute_effective_batch_size,
    count_trainable_parameters,
    get_world_size,
    maybe_torch_dtype,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def enable_offline_mode(cfg: ExperimentConfig) -> str | None:
    # 强制走本地缓存，避免 compute node 访问外网
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # 统一缓存目录：优先读配置，其次读环境变量
    cache_dir = getattr(cfg, "cache_dir", None) or os.environ.get("HF_HOME")

    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

    return cache_dir


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    cache_dir = enable_offline_mode(cfg)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    world_size = get_world_size()
    effective_batch = compute_effective_batch_size(
        cfg.per_device_train_batch_size,
        cfg.gradient_accumulation_steps,
        world_size,
    )
    if effective_batch % cfg.num_generations != 0:
        raise ValueError(
            f"Effective batch size ({effective_batch}) must be divisible by num_generations ({cfg.num_generations})."
        )

    print(f"ckpt 0: offline mode enabled, cache_dir={cache_dir}, effective_batch_size={effective_batch}, world_size={world_size}")

    # 只加载 tokenizer，模型让 TRL 自己处理
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_cfg = TinyLoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.tinylora_r,
        u=cfg.tinylora_u,
        weight_tying=cfg.tinylora_weight_tying,
        projection_seed=cfg.projection_seed,
        save_projection=cfg.save_projection,
        init_v_bound=cfg.init_v_bound,
        target_modules=cfg.target_modules,
        tinylora_dropout=cfg.tinylora_dropout,
        bias=cfg.bias,
        init_weights=cfg.init_weights,
    )

    local_dataset_path = getattr(cfg, "local_dataset_path", None)

    train_ds = load_gsm8k_prompt_only_dataset(
        cfg.dataset_name,
        cfg.dataset_config_name,
        cfg.train_split,
        cfg.system_prompt,
        cfg.max_train_samples,
        cache_dir=cache_dir,
        local_dataset_path=local_dataset_path,
        tokenizer=tokenizer,
    )
    eval_ds = load_gsm8k_prompt_only_dataset(
        cfg.dataset_name,
        cfg.dataset_config_name,
        cfg.eval_split,
        cfg.system_prompt,
        cfg.max_eval_samples,
        cache_dir=cache_dir,
        local_dataset_path=local_dataset_path,
        tokenizer=tokenizer,
    )

    print("ckpt 1: tokenizer and datasets prepared, starting training loop...")

    grpo_args = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        beta=cfg.beta,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        tf32=cfg.tf32,
        remove_unused_columns=False,
        report_to=cfg.report_to,
        seed=cfg.seed,
        use_vllm=cfg.use_vllm,
        log_completions=cfg.log_completions,
        save_only_model=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.45,
        vllm_max_model_length=2048,
        scale_rewards=True,
        num_iterations=1,
    )

    trainer = GRPOTrainer(
        model=cfg.model_name_or_path,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=gsm8k_exact_match_reward,
    )

    print("ckpt 2: trainer initialized, starting training...")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"Trainable parameters: {trainable} / {total} ({trainable / total:.2%})")

    metadata = {
        "model_name_or_path": cfg.model_name_or_path,
        "dataset_name": cfg.dataset_name,
        "dataset_config_name": cfg.dataset_config_name,
        "effective_batch_size": effective_batch,
        "world_size": world_size,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
        "cache_dir": cache_dir,
        "offline_env": {
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        },
        "run_config_summary": {
            "global_batch_size_target": 64,
            "num_generations": cfg.num_generations,
            "num_train_epochs": cfg.num_train_epochs,
            "beta": cfg.beta,
            "max_completion_length": cfg.max_completion_length,
        }
    }
    with open(os.path.join(cfg.output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("ckpt 3: metadata saved, starting training loop...")
    print(json.dumps(metadata, indent=2))

    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to: {final_dir}")


if __name__ == "__main__":
    main()