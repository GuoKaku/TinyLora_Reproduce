from __future__ import annotations

import argparse
import json
import os
import sys
import torch
from safetensors.torch import load_file

DEFAULT_CONFIG = "configs/qwen25_7b_tinylora_gsm8k.yaml"


def _resolve_base_model_path(config_path: str = DEFAULT_CONFIG) -> str:
    """从 yaml config 里读 model_name_or_path，再找到本地 snapshot 路径。"""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name_or_path = cfg.get("model_name_or_path", "")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # 如果已经是本地绝对路径，直接用
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # 否则从 HF_HOME 里找 snapshot
    # model_name_or_path 形如 "Qwen/Qwen2.5-7B-Instruct"
    model_id = model_name_or_path.replace("/", "--")
    snapshot_dir = os.path.join(hf_home, f"models--{model_id}", "snapshots")

    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(
            f"Cannot find snapshot dir: {snapshot_dir}\n"
            f"model_name_or_path={model_name_or_path}, HF_HOME={hf_home}"
        )

    candidates = sorted([
        os.path.join(snapshot_dir, d)
        for d in os.listdir(snapshot_dir)
        if os.path.isdir(os.path.join(snapshot_dir, d))
    ])
    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")

    resolved = candidates[-1]  # 取最新的
    print(f"  [base model] auto-resolved from config: {resolved}")
    return resolved


def load_state_dict_from_dir(ckpt_dir: str, base_model_path: str = None) -> dict:
    """
    自动识别单文件或分片 safetensors，返回完整 state_dict。
    若 ckpt_dir 以 checkpoint-000 结尾，则加载 base_model_path 的权重。
    """
    if os.path.basename(ckpt_dir.rstrip("/")) == "checkpoint-000":
        resolved = base_model_path or _resolve_base_model_path()
        print(f"  [base model] checkpoint-000 detected, loading from: {resolved}")
        return load_state_dict_from_dir(resolved)

    index_path   = os.path.join(ckpt_dir, "model.safetensors.index.json")
    full_path    = os.path.join(ckpt_dir, "model.safetensors")
    adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    bin_path     = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.exists(index_path):
        print(f"  [sharded] index: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state = {}
        for shard in shard_files:
            shard_path = os.path.join(ckpt_dir, shard)
            print(f"  [sharded] loading shard: {shard_path}")
            state.update(load_file(shard_path))
        return state

    elif os.path.exists(full_path):
        print(f"  [single] loading: {full_path}")
        return load_file(full_path)

    elif os.path.exists(adapter_path):
        print(f"  [adapter] loading: {adapter_path}")
        return load_file(adapter_path)

    elif os.path.exists(bin_path):
        print(f"  [bin] loading: {bin_path}")
        return torch.load(bin_path, map_location="cpu")

    else:
        raise FileNotFoundError(
            f"No model weights found in {ckpt_dir}\n"
            f"Looked for: model.safetensors.index.json, model.safetensors, "
            f"adapter_model.safetensors, pytorch_model.bin"
        )


def compare_ckpts(dir1: str, dir2: str, base_model_path: str = None,
                  atol: float = 0.0, rtol: float = 0.0):
    print("Loading checkpoints:")
    print(f"  ckpt1: {dir1}")
    w1 = load_state_dict_from_dir(dir1, base_model_path)
    print(f"  ckpt2: {dir2}")
    w2 = load_state_dict_from_dir(dir2, base_model_path)
    print()

    keys1 = set(w1.keys())
    keys2 = set(w2.keys())

    if keys1 != keys2:
        print("❌ Tensor keys differ.")
        only1 = sorted(keys1 - keys2)
        only2 = sorted(keys2 - keys1)
        if only1:
            print("\nOnly in ckpt1:")
            for k in only1:
                print(f"  {k}")
        if only2:
            print("\nOnly in ckpt2:")
            for k in only2:
                print(f"  {k}")
        return False

    print(f"✅ Same tensor keys: {len(keys1)} tensors")
    print()

    all_equal = True
    max_diff = -1.0
    worst_key = None
    diff_count = 0

    for k in sorted(keys1):
        t1 = w1[k]
        t2 = w2[k]

        if t1.shape != t2.shape:
            print(f"❌ Shape mismatch: {k}")
            print(f"   ckpt1: {tuple(t1.shape)}")
            print(f"   ckpt2: {tuple(t2.shape)}")
            all_equal = False
            diff_count += 1
            continue

        if t1.dtype != t2.dtype:
            print(f"⚠️  Dtype mismatch: {k} ({t1.dtype} vs {t2.dtype})")

        diff = (t1.float() - t2.float()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            worst_key = k

        equal = torch.allclose(t1, t2, atol=atol, rtol=rtol)
        if not equal:
            all_equal = False
            diff_count += 1
            mean_diff = (t1.float() - t2.float()).abs().mean().item()
            print(f"⚠️  Not equal: {k} | max diff = {diff:.8g}, mean diff = {mean_diff:.8g}")

    print("\n====== RESULT ======")
    if all_equal:
        print("🎉 All tensors are exactly equal.")
    else:
        print(f"❌ Checkpoints differ in {diff_count} tensor(s).")
        print(f"Largest difference: {max_diff:.8g} at tensor: {worst_key}")

    return all_equal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", type=str, help="第一个 checkpoint 目录")
    parser.add_argument("dir2", type=str, help="第二个 checkpoint 目录")
    parser.add_argument("--base_model", type=str, default=None,
                        help="base model 路径（可选，默认从 config yaml 自动解析）")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"yaml config 路径，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    compare_ckpts(args.dir1, args.dir2,
                  base_model_path=args.base_model,
                  atol=args.atol, rtol=args.rtol)