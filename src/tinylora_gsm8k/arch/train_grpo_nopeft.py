from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
# from peft import TinyLoraConfig, get_peft_model
from .tinylora import apply_tinylora, untie_tinylora_shared_v

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


class PatchedGRPOTrainer(GRPOTrainer):

    def _patch_sync_weights(self):
        from .tinylora import TinyLoraLinear

        def patched_sync():
            # 不调用 original_sync，完全自己处理
            print("[patched_sync] called, syncing weights to vLLM...")
            llm_model = self.vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model
            
            weights_to_sync = []
            
            for name, module in self.model.named_modules():
                if isinstance(module, TinyLoraLinear):
                    # 合并后的 weight
                    merged = (module.weight + module.delta_weight()).detach()
                    # vllm 的参数名格式：去掉 "model." 前缀
                    vllm_name = self.vllm_generation._fix_param_name_to_vllm(name + ".weight")
                    weights_to_sync.append((vllm_name, merged))
                    if module.bias is not None:
                        vllm_name_bias = self.vllm_generation._fix_param_name_to_vllm(name + ".bias")
                        weights_to_sync.append((vllm_name_bias, module.bias.detach()))
                
            # 其他普通参数（非 TinyLoRA 层）
            tinylora_module_names = {
                name for name, module in self.model.named_modules()
                if isinstance(module, TinyLoraLinear)
            }
            for name, param in self.model.named_parameters():
                # 跳过所有 TinyLoRA 相关参数
                is_tinylora_param = any(
                    name.startswith(mod_name + ".") 
                    for mod_name in tinylora_module_names
                )
                if not is_tinylora_param:
                    vllm_name = self.vllm_generation._fix_param_name_to_vllm(name)
                    weights_to_sync.append((vllm_name, param.detach()))

            # 一次性 load 给 vLLM
            llm_model.load_weights(weights_to_sync)
            
            # reset prefix cache
            self.vllm_generation.llm.reset_prefix_cache()

        self.vllm_generation.sync_weights = patched_sync
           
    def training_step(self, model, inputs, num_items_in_batch=None):
        # 第一次调用时 patch sync_weights
        if not getattr(self, "_sync_patched", False):
            if hasattr(self, "vllm_generation") and self.vllm_generation is not None:
                self._patch_sync_weights()
                self._sync_patched = True
                print("[PatchedGRPOTrainer] sync_weights patched successfully.")
        return super().training_step(model, inputs, num_items_in_batch)

    def _save_checkpoint(self, model, trial):
        from .tinylora import TinyLoraLinear
        shared_v_backup = {}
        for name, module in model.named_modules():
            if isinstance(module, TinyLoraLinear):
                shared_v_backup[name] = module.v

        untie_tinylora_shared_v(model)
        result = super()._save_checkpoint(model, trial)

        for name, module in model.named_modules():
            if isinstance(module, TinyLoraLinear) and name in shared_v_backup:
                module.v = shared_v_backup[name]

        return result  
    
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


def build_model_and_tokenizer(cfg: ExperimentConfig):
    print(f"Building model and tokenizer for {cfg.model_name_or_path} with torch_dtype={cfg.torch_dtype}...")
    torch_dtype = maybe_torch_dtype(cfg.torch_dtype)
    cache_dir = getattr(cfg, "cache_dir", None) or os.environ.get("HF_HOME")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "cache_dir": cache_dir,
        "local_files_only": True,
    }
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        **model_kwargs,
    )
    
    print(f"[ATTN] attn_implementation={model.config._attn_implementation}")

    model.config.use_cache = False

    model = apply_tinylora(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable} / {total} ({trainable/total:.2%})")

    return model, tokenizer


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

    model, tokenizer = build_model_and_tokenizer(cfg)

    print("Model and tokenizer loaded, counting trainable parameters...")
    trainable, total = count_trainable_parameters(model)
    print(f"Model loaded. Trainable parameters: {trainable} / {total} ({trainable / total:.2%})")

    local_dataset_path = getattr(cfg, "local_dataset_path", None)

    train_ds = load_gsm8k_prompt_only_dataset(
        cfg.dataset_name,
        cfg.dataset_config_name,
        cfg.train_split,
        cfg.system_prompt,
        cfg.max_train_samples,
        cache_dir=cache_dir,
        local_dataset_path=local_dataset_path,
    )

    eval_ds = load_gsm8k_prompt_only_dataset(
        cfg.dataset_name,
        cfg.dataset_config_name,
        cfg.eval_split,
        cfg.system_prompt,
        cfg.max_eval_samples,
        cache_dir=cache_dir,
        local_dataset_path=local_dataset_path,
    )

    print("ckpt 1: model and tokenizer loaded, datasets prepared, starting training loop...")

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

    trainer = PatchedGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[gsm8k_exact_match_reward],
    )

    # 尽早 patch sync_weights，确保第一次 rollout 就用合并后的权重
    if hasattr(trainer, "vllm_generation") and trainer.vllm_generation is not None:
        trainer._patch_sync_weights()
        trainer._sync_patched = True
        print("[main] sync_weights patched after trainer init.")
    else:
        print("[WARN] vllm_generation not available at init time, will patch at first training_step.")

    print("ckpt 2: trainer initialized, starting training...")

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

    untie_tinylora_shared_v(trainer.model)

    final_dir = os.path.join(cfg.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to: {final_dir}")

if __name__ == "__main__":
    main()