from __future__ import annotations

from typing import Any

from .utils import extract_pred_answer





def gsm8k_exact_match_reward(completions, ground_truth, **kwargs):
    scores: list[float] = []
    for completion, gold in zip(completions, ground_truth):
        # print(f"Debug: completion type: {type(completion)}, value (last 300 chars): ...{str(completion)[-300:]}")
        # print(f"Debug: ground_truth type: {type(gold)}, value: {gold}")
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        else:
            text = completion
        pred = extract_pred_answer(text)
        try:
            scores.append(1.0 if float(pred) == float(gold) else 0.0)
        except ValueError:
            scores.append(0.0)
    return scores


from math_verify import verify, parse as math_parse

def _extract_boxed(text: str) -> str:
    """从文本中提取最后一个 \\boxed{} 的内容，支持嵌套花括号"""
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


def math_exact_match_reward(completions, ground_truth, **kwargs):
    scores: list[float] = []
    for completion, gold in zip(completions, ground_truth):
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        else:
            text = completion

        pred = _extract_boxed(text)

        try:
            correct = verify(math_parse(pred), math_parse(gold))
            scores.append(1.0 if correct else 0.0)
        except Exception:
            scores.append(0.0)

    return scores