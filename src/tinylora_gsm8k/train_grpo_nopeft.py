from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

# Transformers treats scikit-learn as an optional backend, but if a broken
# sklearn/SciPy binary is present it can fail during import before training
# starts. This GRPO path does not use sklearn-backed generation helpers.
if os.environ.get("TINYLORA_DISABLE_SKLEARN", "1") == "1":
    sys.modules.setdefault("sklearn", None)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from .tinylora import apply_tinylora, untie_tinylora_shared_v
from tqdm import tqdm
from .config import ExperimentConfig
from .data import load_gsm8k_prompt_only_dataset, load_math_prompt_only_dataset
from .rewards import gsm8k_exact_match_reward, math_exact_match_reward
from .utils import (
    compute_effective_batch_size,
    count_trainable_parameters,
    get_world_size,
    maybe_torch_dtype,
    set_seed,
)

sys.stdout.reconfigure(line_buffering=True)

REWARD_FNS = {
    "gsm8k": gsm8k_exact_match_reward,
    "math": math_exact_match_reward,
}

DATASET_LOADERS = {
    "gsm8k": load_gsm8k_prompt_only_dataset,
    "math": load_math_prompt_only_dataset,
}


class PatchedGRPOTrainer(GRPOTrainer):

    def _patch_sync_weights(self):
        from .tinylora import TinyLoraLinear

        def patched_sync():
            print("[patched_sync] called, syncing weights to vLLM...")
            llm_model = self.vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model

            weights_to_sync = []

            for name, module in self.model.named_modules():
                if isinstance(module, TinyLoraLinear):
                    merged = (module.weight + module.delta_weight()).detach()
                    vllm_name = self.vllm_generation._fix_param_name_to_vllm(name + ".weight")
                    weights_to_sync.append((vllm_name, merged))
                    if module.bias is not None:
                        vllm_name_bias = self.vllm_generation._fix_param_name_to_vllm(name + ".bias")
                        weights_to_sync.append((vllm_name_bias, module.bias.detach()))

            tinylora_module_names = {
                name for name, module in self.model.named_modules()
                if isinstance(module, TinyLoraLinear)
            }
            for name, param in self.model.named_parameters():
                is_tinylora_param = any(
                    name.startswith(mod_name + ".")
                    for mod_name in tinylora_module_names
                )
                if not is_tinylora_param:
                    vllm_name = self.vllm_generation._fix_param_name_to_vllm(name)
                    weights_to_sync.append((vllm_name, param.detach()))

            llm_model.load_weights(weights_to_sync)
            self.vllm_generation.llm.reset_prefix_cache()

        self.vllm_generation.sync_weights = patched_sync

    def training_step(self, model, inputs, num_items_in_batch=None):
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

    def set_eval_config(self, cfg, tokenizer, eval_raw_dataset, dataset_type):
        """在 main() 里 trainer 初始化后调用一次，传入 eval 所需的上下文"""
        self._eval_cfg = cfg
        self._eval_tokenizer = tokenizer
        self._eval_raw_dataset = eval_raw_dataset  # 原始 HF dataset（非 prompt-only）
        self._eval_dataset_type = dataset_type

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", zero_v=False):
        from .tinylora import TinyLoraLinear

        # 备份并清零 v（测 pure base）
        v_backup = {}
        if zero_v:
            seen_ptrs = set()
            for name, module in self.model.named_modules():
                if isinstance(module, TinyLoraLinear):
                    ptr = module.v.data_ptr()
                    if ptr not in seen_ptrs:
                        seen_ptrs.add(ptr)
                        v_backup[ptr] = (module.v, module.v.data.clone())
                        module.v.data.zero_()
            print(f"[Eval] zero_v=True: zeroed {len(v_backup)} unique v tensors → pure base model mode")

        try:
            metrics = {}

            cfg = self._eval_cfg
            tokenizer = self._eval_tokenizer
            ds = self._eval_raw_dataset
            dataset_type = self._eval_dataset_type

            if dataset_type == "gsm8k":
                question_field, answer_field = "question", "answer"
            else:
                question_field, answer_field = "problem", "solution"

            if dataset_type == "gsm8k":
                from .utils import extract_pred_answer, extract_reference_answer
                def judge(completion_text, raw_answer):
                    pred = extract_pred_answer(completion_text)
                    gold = extract_reference_answer(raw_answer)
                    try:
                        is_correct = float(pred) == float(gold)
                    except ValueError:
                        is_correct = False
                    return is_correct
            else:
                from math_verify import verify, parse as math_parse
                def _extract_boxed(text):
                    idx = text.rfind(r"\boxed{")
                    if idx == -1:
                        return ""
                    start = idx + len(r"\boxed{")
                    depth = 1
                    for i in range(start, len(text)):
                        if text[i] == "{": depth += 1
                        elif text[i] == "}":
                            depth -= 1
                            if depth == 0:
                                return text[start:i].strip()
                    return ""
                def judge(completion_text, raw_answer):
                    pred = _extract_boxed(completion_text)
                    gold = _extract_boxed(raw_answer)
                    try:
                        return bool(verify(math_parse(pred), math_parse(gold)))
                    except Exception:
                        return False

            all_prompts = []
            for q in ds[question_field]:
                msgs = [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": q},
                ]
                all_prompts.append(
                    tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                )

            self.model.cpu()
            torch.cuda.empty_cache()
            self.model.cuda()
            self.model.eval()
            model = self.model

            BATCH_SIZE = 64
            max_new_tokens = getattr(cfg, "eval_max_new_tokens", 1024)
            temperature = getattr(cfg, "eval_temperature", 0.0)

            correct = 0
            total = len(ds)

            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"

            with torch.no_grad():
                for i in tqdm(range(0, total, BATCH_SIZE), desc="[Eval]"):
                    answers = ds[answer_field][i: i + BATCH_SIZE]
                    prompts = all_prompts[i: i + BATCH_SIZE]

                    inputs = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    )
                    prompt_len = inputs["input_ids"].shape[1]
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    for j, a in enumerate(answers):
                        completion_ids = generated[j][prompt_len:]
                        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                        correct += int(judge(completion_text, a))

            tokenizer.padding_side = original_padding_side

            self.model.train()
            torch.cuda.empty_cache()

            accuracy = correct / total if total else 0.0
            mode_tag = "base_v0" if zero_v else "lora_init"
            metrics = {
                f"{metric_key_prefix}/reward_acc": accuracy,
                f"{metric_key_prefix}/correct": correct,
                f"{metric_key_prefix}/total": total,
            }
            print(f"[Eval] mode={mode_tag} accuracy={accuracy:.4f} ({correct}/{total})")
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )

            return metrics

        finally:
            # 无论是否异常，都还原 v
            if zero_v:
                for ptr, (v_param, backup) in v_backup.items():
                    v_param.data.copy_(backup)
                print(f"[Eval] v restored ({len(v_backup)} tensors).")
            
            
            
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def enable_offline_mode(cfg: ExperimentConfig) -> str | None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    attn_implementation = cfg.attn_implementation
    if attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        attn_implementation = "sdpa"
        print("[ATTN] flash_attention_2 requested but flash_attn is not installed; falling back to sdpa.")

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
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

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

    # 根据 dataset_type 选择 loader 和 reward 函数
    dataset_type = getattr(cfg, "dataset_type", "gsm8k")
    if dataset_type not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose from {list(DATASET_LOADERS.keys())}")

    load_fn = DATASET_LOADERS[dataset_type]
    reward_fn = REWARD_FNS[dataset_type]
    print(f"Using dataset_type={dataset_type}, reward_fn={reward_fn.__name__}")

    local_dataset_path = getattr(cfg, "local_dataset_path", None)

    train_ds = load_fn(
        cfg.dataset_name,
        cfg.dataset_config_name,
        cfg.train_split,
        cfg.system_prompt,
        cfg.max_train_samples,
        cache_dir=cache_dir,
        local_dataset_path=local_dataset_path,
        tokenizer=tokenizer,
    )
    eval_ds = load_fn(
        cfg.dataset_name,
        cfg.dataset_config_name,
        cfg.eval_split,
        cfg.system_prompt,
        cfg.max_eval_samples,
        cache_dir=cache_dir,
        local_dataset_path=local_dataset_path,
        tokenizer=tokenizer,
    )
    
    # 在 train_ds / eval_ds 加载之后，额外拿一份原始的 eval set 给 evaluate() 用
    if local_dataset_path and Path(local_dataset_path).exists():
        from datasets import load_from_disk
        eval_raw_ds = load_from_disk(local_dataset_path)[cfg.eval_split]
    else:
        from datasets import load_dataset
        eval_raw_ds = load_dataset(
            cfg.dataset_name, cfg.dataset_config_name, split=cfg.eval_split
        )
    if cfg.max_eval_samples:
        eval_raw_ds = eval_raw_ds.select(range(min(cfg.max_eval_samples, len(eval_raw_ds))))

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
        eval_strategy="steps" if cfg.eval_steps is not None else "no",
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
        vllm_max_model_length=getattr(cfg, "max_prompt_length", 1024) + getattr(cfg, "max_completion_length", 1024),
        scale_rewards=True,
        num_iterations=1,
    )

    trainer = PatchedGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[reward_fn],
    )
    
    trainer.set_eval_config(cfg, tokenizer, eval_raw_ds, dataset_type)

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
        "dataset_type": dataset_type,
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

    # print("Running initial evaluation before training...")
    # trainer.evaluate()
    
    print("Running initial evaluation before training...")
    trainer.evaluate(metric_key_prefix="eval_pre_base",      zero_v=True)   # pure base
    trainer.evaluate(metric_key_prefix="eval_pre_lora_init", zero_v=False)  # init lora 未训练

    trainer.train()

    print("Running final evaluation after training...")
    trainer.evaluate(metric_key_prefix="eval_final")

    # trainer.train()

    # # 训练后评估
    # print("Running final evaluation after training...")
    # trainer.evaluate()

    untie_tinylora_shared_v(trainer.model)

    final_dir = os.path.join(cfg.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to: {final_dir}")


if __name__ == "__main__":
    main()
