from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .config import ExperimentConfig
from .prompts import build_gsm8k_prompt
from .utils import extract_pred_answer, extract_reference_answer, maybe_torch_dtype, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--use_nopeft", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--check_weights", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")  
    return parser.parse_args()


def _load_state_from_checkpoint_dir(checkpoint_path: Path):
    index_path   = checkpoint_path / "model.safetensors.index.json"
    full_path    = checkpoint_path / "model.safetensors"
    adapter_path = checkpoint_path / "adapter_model.safetensors"
    bin_path     = checkpoint_path / "pytorch_model.bin"

    if index_path.exists():
        import json
        from safetensors.torch import load_file
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state = {}
        for shard in shard_files:
            shard_path = checkpoint_path / shard
            print(f"[EVAL] Loading shard: {shard_path}")
            state.update(load_file(shard_path))
        print(f"[EVAL] Loaded sharded safetensors from {checkpoint_path}")
        return state, str(index_path)

    elif full_path.exists():
        from safetensors.torch import load_file
        state = load_file(full_path)
        print(f"[EVAL] Loading state from {full_path}")
        return state, str(full_path)

    elif adapter_path.exists():
        from safetensors.torch import load_file
        state = load_file(adapter_path)
        print(f"[EVAL] Loading state from {adapter_path}")
        return state, str(adapter_path)

    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        print(f"[EVAL] Loading state from {bin_path}")
        return state, str(bin_path)

    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
    

def _tensor_max_abs_diff(t1: torch.Tensor, t2: torch.Tensor) -> float:
    return (t1.detach().float().cpu() - t2.detach().float().cpu()).abs().max().item()


def _compare_model_to_base(base_model, loaded_model, max_print=30):
    base_sd = base_model.state_dict()
    loaded_sd = loaded_model.state_dict()

    base_keys = set(base_sd.keys())
    loaded_keys = set(loaded_sd.keys())

    common_keys = sorted(base_keys & loaded_keys)
    only_base = sorted(base_keys - loaded_keys)
    only_loaded = sorted(loaded_keys - base_keys)

    changed = []
    unchanged = []

    max_diff = -1.0
    worst_key = None

    for k in common_keys:
        t1 = base_sd[k]
        t2 = loaded_sd[k]

        if t1.shape != t2.shape:
            changed.append((k, float("inf"), "shape_mismatch"))
            worst_key = k
            max_diff = float("inf")
            continue

        diff = _tensor_max_abs_diff(t1, t2)
        if diff > 0:
            changed.append((k, diff, "value_changed"))
            if diff > max_diff:
                max_diff = diff
                worst_key = k
        else:
            unchanged.append(k)

    print("\n" + "=" * 80)
    print("[CHECK] Compare loaded model vs fresh base model")
    print("=" * 80)
    print(f"[CHECK] base keys      : {len(base_keys)}")
    print(f"[CHECK] loaded keys    : {len(loaded_keys)}")
    print(f"[CHECK] common keys    : {len(common_keys)}")
    print(f"[CHECK] only in base   : {len(only_base)}")
    print(f"[CHECK] only in loaded : {len(only_loaded)}")
    print(f"[CHECK] changed tensors: {len(changed)}")
    print(f"[CHECK] unchanged      : {len(unchanged)}")

    if only_base:
        print("\n[CHECK] First keys only in base:")
        for k in only_base[:max_print]:
            print("   ", k)

    if only_loaded:
        print("\n[CHECK] First keys only in loaded:")
        for k in only_loaded[:max_print]:
            print("   ", k)

    if changed:
        changed_sorted = sorted(changed, key=lambda x: (x[1] if x[1] != float("inf") else 1e30), reverse=True)
        print("\n[CHECK] Top changed tensors:")
        for k, diff, reason in changed_sorted[:max_print]:
            print(f"   {k} | diff={diff} | {reason}")
        print(f"\n[CHECK] Largest diff: {max_diff} at {worst_key}")
    else:
        print("\n[CHECK] Loaded model is EXACTLY identical to fresh base model.")

    return {
        "base_keys": len(base_keys),
        "loaded_keys": len(loaded_keys),
        "common_keys": len(common_keys),
        "only_base": only_base,
        "only_loaded": only_loaded,
        "changed": changed,
        "unchanged": unchanged,
        "max_diff": max_diff,
        "worst_key": worst_key,
    }


def _inspect_tinylora_modules(model, max_print=20):
    print("\n" + "=" * 80)
    print("[CHECK] Inspect TinyLoRA modules")
    print("=" * 80)

    found = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "tinylora" in cls_name:
            found += 1
            msg = [f"  {name} ({module.__class__.__name__})"]

            if hasattr(module, "v") and isinstance(module.v, torch.nn.Parameter):
                msg.append(f"v_norm={module.v.detach().float().norm().item():.8g}")
            if hasattr(module, "u") and isinstance(module.u, torch.nn.Parameter):
                msg.append(f"u_norm={module.u.detach().float().norm().item():.8g}")
            if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                msg.append(f"weight_norm={module.weight.detach().float().norm().item():.8g}")

            print(" | ".join(msg))
            if found >= max_print:
                break

    if found == 0:
        print("[CHECK] No TinyLoRA-like modules found.")


def merge_tinylora_to_state_dict(model) -> dict:

    from .tinylora import TinyLoraLinear
    merged_sd = {}
    for name, module in model.named_modules():
        if isinstance(module, TinyLoraLinear):
            merged_sd[name + ".weight"] = (module.weight + module.delta_weight()).detach().cpu()
            if module.bias is not None:
                merged_sd[name + ".bias"] = module.bias.detach().cpu()

    tinylora_prefixes = tuple(
        name + "." for name, module in model.named_modules()
        if module.__class__.__name__ == "TinyLoraLinear"
    )
    for name, param in model.state_dict().items():
        if not name.startswith(tinylora_prefixes):
            merged_sd[name] = param.detach().cpu()

    return merged_sd


def load_model(args, cfg, torch_dtype, cache_dir):
    checkpoint_path = Path(args.checkpoint_path)
    print(f"[EVAL] checkpoint_path = {checkpoint_path}")
    print(f"[EVAL] use_nopeft      = {args.use_nopeft}")

    fresh_base_model = None
    if args.check_weights:
        fresh_base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        fresh_base_model.config.use_cache = True
        fresh_base_model.eval()

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model.config.use_cache = True

    if checkpoint_path.name == "checkpoint-000":
        print("[EVAL] checkpoint-000 detected, evaluating pure base model.")
        model.eval()
        if args.check_weights:
            _compare_model_to_base(fresh_base_model, model)
        return model

    if args.use_nopeft:
        print("[EVAL] Loading custom TinyLoRA (nopeft) implementation...")
        from .tinylora import apply_tinylora

        if args.check_weights:
            fresh_base_model = apply_tinylora(fresh_base_model, cfg)
            fresh_base_model.eval()

        model = apply_tinylora(model, cfg)

        state, state_src = _load_state_from_checkpoint_dir(checkpoint_path)

        model_state = model.state_dict()
        ckpt_keys = set(state.keys())
        model_keys = set(model_state.keys())

        print(f"[EVAL] state source                = {state_src}")
        print(f"[EVAL] #checkpoint state keys     = {len(ckpt_keys)}")
        print(f"[EVAL] #current model state keys  = {len(model_keys)}")
        print(f"[EVAL] #common keys               = {len(ckpt_keys & model_keys)}")
        print(f"[EVAL] #missing in checkpoint     = {len(model_keys - ckpt_keys)}")
        print(f"[EVAL] #unexpected in checkpoint  = {len(ckpt_keys - model_keys)}")

        missing, unexpected = model.load_state_dict(state, strict=False)

        print(f"[EVAL] load_state_dict finished.")
        print(f"[EVAL] missing count    = {len(missing)}")
        print(f"[EVAL] unexpected count = {len(unexpected)}")

        if missing:
            print("[WARN] First 30 missing keys:")
            for k in missing[:30]:
                print("   ", k)

        if unexpected:
            print("[WARN] First 30 unexpected keys:")
            for k in unexpected[:30]:
                print("   ", k)

        model.eval()
        _inspect_tinylora_modules(model)
        if args.check_weights:
            _compare_model_to_base(fresh_base_model, model)

    else:
        print("[EVAL] Loading PEFT TinyLoRA implementation...")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint_path))
        model.eval()

        if args.check_weights:
            try:
                from peft import PeftConfig, get_peft_model
                peft_cfg = PeftConfig.from_pretrained(str(checkpoint_path))
                fresh_base_model = get_peft_model(fresh_base_model, peft_cfg)
                fresh_base_model.eval()
                _compare_model_to_base(fresh_base_model, model)
            except Exception as e:
                print(f"[WARN] Failed to build PEFT-wrapped fresh base model for comparison: {e}")
                print("[WARN] Skip exact compare for PEFT branch.")

        print(f"[EVAL] Loaded PEFT checkpoint from {checkpoint_path}")

    return model


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    cache_dir = getattr(cfg, "cache_dir", None) or os.environ.get("HF_HOME")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)

    set_seed(cfg.seed)

    dataset_type = getattr(cfg, "dataset_type", "gsm8k")
    print(f"[EVAL] dataset_type={dataset_type}")

    torch_dtype_str = args.torch_dtype or cfg.torch_dtype
    torch_dtype = maybe_torch_dtype(torch_dtype_str)
    temperature = args.temperature if args.temperature is not None else cfg.eval_temperature
    max_new_tokens = args.max_new_tokens or cfg.eval_max_new_tokens

    print(f"[EVAL] model={cfg.model_name_or_path}")
    print(f"[EVAL] checkpoint={args.checkpoint_path}")
    print(f"[EVAL] torch_dtype={torch_dtype}, temperature={temperature}, max_new_tokens={max_new_tokens}")
    print(f"[EVAL] cache_dir={cache_dir}, use_vllm={args.use_vllm}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    eval_local_path = getattr(cfg, "eval_local_dataset_path", None)
    local_dataset_path = getattr(cfg, "local_dataset_path", None)
    load_path = eval_local_path or local_dataset_path

    if load_path and Path(load_path).exists():
        print(f"[EVAL] Loading dataset from: {load_path}")
        ds = load_from_disk(load_path)[args.split]
    else:
        ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config_name,
            split=args.split,
        )

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"[EVAL] Evaluating on {len(ds)} samples...")


    if dataset_type == "gsm8k":
        question_field = "question"
        answer_field = "answer"
    else:  # math / math_hard
        question_field = "problem"
        answer_field = "solution"


    all_prompts = []
    for q in ds[question_field]:
        msgs = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_prompts.append(prompt_text)


    if dataset_type == "gsm8k":
        def judge(completion_text: str, raw_answer: str):
            pred = extract_pred_answer(completion_text)
            gold = extract_reference_answer(raw_answer)
            try:
                is_correct = float(pred) == float(gold)
            except ValueError:
                is_correct = False
            return pred, gold, is_correct
    else:
        from math_verify import verify, parse as math_parse

        def _extract_boxed(text: str) -> str:
            idx = text.rfind(r"\boxed{")
            if idx == -1:
                return ""
            start = idx + len(r"\boxed{")
            depth = 1
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i].strip()
            return ""

        def judge(completion_text: str, raw_answer: str):
            pred = _extract_boxed(completion_text)
            gold = _extract_boxed(raw_answer)
            try:
                is_correct = bool(verify(math_parse(pred), math_parse(gold)))
            except Exception:
                is_correct = False
            return pred, gold, is_correct

    correct = 0
    outputs = []

    if args.use_vllm:
        from vllm import LLM, SamplingParams
        from transformers import AutoModelForCausalLM as _AMCL
        import tempfile
        
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        print("[EVAL] vLLM mode: merging TinyLoRA weights into a temp checkpoint...")

        model = load_model(args, cfg, torch_dtype, cache_dir)
        model.eval()

        if args.use_nopeft:
            merged_sd = merge_tinylora_to_state_dict(model)

            base_model = _AMCL.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=cfg.trust_remote_code,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            missing, unexpected = base_model.load_state_dict(merged_sd, strict=False)
            print(f"[EVAL] merge load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

            tmp_dir = tempfile.mkdtemp(prefix="tinylora_merged_")
            base_model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
            print(f"[EVAL] Merged model saved to temp dir: {tmp_dir}")
            vllm_model_path = tmp_dir
        else:
            print("[EVAL] PEFT + vLLM: merging adapter into base model...")

            from peft import PeftModel
            from transformers import AutoModelForCausalLM as _AMCL
            import tempfile


            base_model = _AMCL.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=cfg.trust_remote_code,
                cache_dir=cache_dir,
                local_files_only=True,
            )


            model = PeftModel.from_pretrained(base_model, str(args.checkpoint_path))


            model = model.merge_and_unload()


            tmp_dir = tempfile.mkdtemp(prefix="peft_merged_")
            model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)

            print(f"[EVAL] Merged PEFT model saved to: {tmp_dir}")

            vllm_model_path = tmp_dir

        del model
        torch.cuda.empty_cache()

        llm = LLM(
            model=vllm_model_path,
            dtype=torch_dtype_str,
            gpu_memory_utilization=0.9,
            max_model_len=getattr(cfg, "max_prompt_length", 1024) + getattr(cfg, "max_completion_length", 1024),
            trust_remote_code=cfg.trust_remote_code,
            enforce_eager=True, 
        )

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
            top_p=1.0,
        )

        print("[EVAL] Running vLLM generation...")
        vllm_outputs = llm.generate(all_prompts, sampling_params)

        for i, (q, a) in enumerate(zip(ds[question_field], ds[answer_field])):
            completion_text = vllm_outputs[i].outputs[0].text
            pred, gold, is_correct = judge(completion_text, a)
            correct += int(is_correct)
            outputs.append({
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "completion": completion_text,
            })


    else:
        BATCH_SIZE = 128
        print(f"[EVAL] HF generate mode, batch_size={BATCH_SIZE}")

        model = load_model(args, cfg, torch_dtype, cache_dir)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        print(f"[EVAL] Model on device: {next(model.parameters()).device}")

        for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating"):
            batch = ds[i: i + BATCH_SIZE]
            questions = batch[question_field]
            answers = batch[answer_field]
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

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for j, (q, a) in enumerate(zip(questions, answers)):
                completion_ids = generated[j][prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                pred, gold, is_correct = judge(completion_text, a)
                correct += int(is_correct)
                outputs.append({
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                    "completion": completion_text,
                })


    accuracy = correct / len(ds) if len(ds) else 0.0
    result = {
        "accuracy": accuracy,
        "num_samples": len(ds),
        "correct": correct,
        "dataset_type": dataset_type,
        "checkpoint_path": args.checkpoint_path,
        "model_name_or_path": cfg.model_name_or_path,
        "predictions": outputs,
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary = {k: v for k, v in result.items() if k != "predictions"}
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()