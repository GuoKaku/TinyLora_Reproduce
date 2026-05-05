from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import yaml


@dataclass
class ExperimentConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "openai/gsm8k"
    dataset_config_name: str = "main"
    dataset_type: str = "gsm8k"  # or "math"
    local_dataset_path: str | None = None
    eval_local_dataset_path: str | None = None
    train_split: str = "train"
    eval_split: str = "test"
    output_dir: str = "outputs/qwen25-7b-gsm8k-tinylora"
    seed: int = 42

    # Prompting / generation
    system_prompt: str = (
        "You are a careful math solver. Show your reasoning, then end with a single line exactly in the format: "
        "Final answer: <answer>."
    )
    max_prompt_length: int = 1024
    max_completion_length: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0

    # Paper-like GSM8K GRPO setup
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant"
    beta: float = 0.0
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    per_device_eval_batch_size: int = 2
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int | None = None
    save_total_limit: int = 2
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    report_to: str | list[str] = "none"
    use_vllm: bool = False
    log_completions: bool = True

    # TinyLoRA
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    tinylora_r: int = 2
    tinylora_u: int = 13
    tinylora_weight_tying: float = 1.0
    tinylora_dropout: float = 0.0
    projection_seed: int = 42
    save_projection: bool = True
    init_weights: str | bool = "uniform"
    init_v_bound: float = 0.02
    bias: str = "none"

    # Data limits
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Evaluation generation
    eval_temperature: float = 0.0
    eval_max_new_tokens: int = 1024

    # Optional extras
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)
