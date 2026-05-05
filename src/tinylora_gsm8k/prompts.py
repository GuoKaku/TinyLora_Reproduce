from __future__ import annotations


def build_gsm8k_prompt(question: str, system_prompt: str) -> list[dict[str, str]]:
    user_prompt = (
        "Solve the following GSM8K math word problem.\n\n"
        f"Question: {question}\n\n"
        "Reason step by step. Your final line must be exactly:\n"
        "Final answer: <answer>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
