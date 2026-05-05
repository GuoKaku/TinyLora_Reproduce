from __future__ import annotations

import math
import random
import re
from typing import Any

import numpy as np
import torch


# FINAL_ANSWER_RE = re.compile(r"Final answer\s*:\s*(.+)", re.IGNORECASE)
# NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_commas(text: str) -> str:
    return text.replace(",", "")


def canonicalize_numeric_string(text: str) -> str:
    text = text.strip()
    text = strip_commas(text)
    if text.endswith("."):
        text = text[:-1].strip()
    return text


# def extract_reference_answer(answer_text: str) -> str:
#     # GSM8K reference answers typically end with "#### 42"
#     if "####" in answer_text:
#         answer_text = answer_text.split("####")[-1]
#     answer_text = answer_text.strip()
#     nums = NUMBER_RE.findall(answer_text)
#     if nums:
#         return canonicalize_numeric_string(nums[-1])
#     return canonicalize_numeric_string(answer_text)


# def extract_pred_answer(completion: str) -> str:
#     match = FINAL_ANSWER_RE.search(completion)
#     if match:
#         tail = match.group(1).strip()
#     else:
#         tail = completion.strip().splitlines()[-1] if completion.strip() else ""
#     nums = NUMBER_RE.findall(tail)
#     if nums:
#         return canonicalize_numeric_string(nums[-1])
#     return canonicalize_numeric_string(tail)

FINAL_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")

def extract_reference_answer(answer_text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_text)
    return m.group(1).replace(",", "") if m else ""

def extract_pred_answer(completion: str) -> str:
    matches = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", completion)
    if matches:
        return matches[-1].replace(",", "")
    # fallback: 最后一个数字
    nums = re.findall(r"-?\d+(?:\.\d+)?", completion.replace(",", ""))
    return nums[-1] if nums else ""


def compute_effective_batch_size(
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> int:
    return per_device_train_batch_size * gradient_accumulation_steps * world_size


def maybe_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype string: {dtype_name}")
    return mapping[key]


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


