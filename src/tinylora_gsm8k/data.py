from __future__ import annotations

import os

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from .prompts import build_gsm8k_prompt
from .utils import extract_reference_answer



def load_gsm8k_prompt_only_dataset(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    system_prompt: str,
    max_samples: int | None = None,
    cache_dir: str | None = None,
    local_dataset_path: str | None = None,
    tokenizer=None,  # ← 新增
) -> Dataset:
    ds = None

    if local_dataset_path is not None and os.path.exists(local_dataset_path):
        loaded = load_from_disk(local_dataset_path)
        if isinstance(loaded, DatasetDict):
            ds = loaded[split]
        else:
            ds = loaded

    if ds is None:
        ds = load_dataset(
            dataset_name,
            dataset_config_name,
            split=split,
            cache_dir=cache_dir,
        )

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def _map(example: dict) -> dict:
        question = example["question"]
        answer = example["answer"]

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            # print(prompt)
        else:
            # fallback: 直接返回 message list，让 TRL 自己处理
            prompt = msgs

        return {
            "prompt": prompt,
            "ground_truth": extract_reference_answer(answer),
        }

    ds = ds.map(_map, remove_columns=ds.column_names)
    return ds


def load_math_prompt_only_dataset(
    dataset_name: str,
    dataset_config_name: str | None,
    split: str,
    system_prompt: str,
    max_samples: int | None = None,
    cache_dir: str | None = None,
    local_dataset_path: str | None = None,
    tokenizer=None,
) -> Dataset:
    ds = None
    if local_dataset_path is not None and os.path.exists(local_dataset_path):
        loaded = load_from_disk(local_dataset_path)
        if isinstance(loaded, DatasetDict):
            ds = loaded[split]
        else:
            ds = loaded

    if ds is None:
        ds = load_dataset(
            dataset_name,
            dataset_config_name,
            split=split,
            cache_dir=cache_dir,
        )

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def _map(example: dict) -> dict:
        question = example["problem"]
        solution = example["solution"]
        gold = _extract_boxed_from_solution(solution)

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = msgs

        return {
            "prompt": prompt,
            "ground_truth": gold,
        }

    ds = ds.map(_map, remove_columns=ds.column_names)
    # 过滤掉提取不到答案的样本
    ds = ds.filter(lambda x: x["ground_truth"] != "")
    return ds


def _extract_boxed_from_solution(solution: str) -> str:
    """从 solution 字段提取 \\boxed{} 内容"""
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
    return ""