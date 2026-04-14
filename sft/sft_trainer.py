"""SFT training using HuggingFace TRL.
Covers: SFT (Supervised Fine-Tuning): curating high-quality instruction-response pairs

Trains a causal LM with LoRA adapters using TRL's SFTTrainer.
Config is loaded from configs/sft_config.yaml.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    """SFT training configuration."""

    model_name: str = "microsoft/phi-2"
    dataset_path: str = "data/sft_train.jsonl"
    output_dir: str = "checkpoints/sft"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    seed: int = 42
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    system_prompt: str = "You are a helpful, harmless, and honest assistant."
    fp16: bool = False
    bf16: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Support nested "sft" key or flat structure
        if "sft" in data:
            data = data["sft"]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_sft_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    system_prompt: str = "You are a helpful, harmless, and honest assistant.",
) -> Dataset:
    """Load JSONL with {instruction, response}, format as chat template.

    Args:
        data_path: Path to JSONL file. Each line: {instruction, response}.
        tokenizer: HuggingFace tokenizer (must have apply_chat_template or EOS token).
        max_length: Maximum token length; longer examples are filtered out.
        system_prompt: System message prepended to every conversation.

    Returns:
        HuggingFace Dataset with "text" column ready for SFTTrainer.
    """
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No data found in {data_path}")

    texts = []
    for rec in records:
        instruction = rec.get("instruction", rec.get("prompt", ""))
        response = rec.get("response", rec.get("output", ""))
        if not instruction or not response:
            continue

        # Format using chat template if available, else manual format
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback: Alpaca-style format
            text = (
                f"### System:\n{system_prompt}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{response}{tokenizer.eos_token or ''}"
            )
        texts.append(text)

    # Filter by tokenized length
    filtered = []
    for text in texts:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count <= max_length:
            filtered.append({"text": text})

    print(
        f"Loaded {len(records)} records → {len(texts)} valid → "
        f"{len(filtered)} within max_length={max_length}"
    )

    return Dataset.from_list(filtered)


# ---------------------------------------------------------------------------
# Loss logging callback
# ---------------------------------------------------------------------------


class LossLoggerCallback(TrainerCallback):
    """Records training loss at each logging step for later plotting."""

    def __init__(self):
        self.loss_history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_history.append(
                {"step": state.global_step, "loss": logs["loss"]}
            )

    def save_loss_curve(self, output_dir: str):
        """Save loss history as JSON and optionally plot."""
        path = Path(output_dir) / "loss_curve.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.loss_history, f, indent=2)
        print(f"Loss curve saved to {path}")

        try:
            import matplotlib.pyplot as plt

            steps = [entry["step"] for entry in self.loss_history]
            losses = [entry["loss"] for entry in self.loss_history]
            plt.figure(figsize=(10, 5))
            plt.plot(steps, losses, linewidth=1.5, color="steelblue")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("SFT Training Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = Path(output_dir) / "loss_curve.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Loss curve plot saved to {plot_path}")
        except ImportError:
            print("matplotlib not installed; skipping loss curve plot")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_sft(config_path: str) -> None:
    """Train a causal LM with LoRA using TRL's SFTTrainer.

    Args:
        config_path: Path to YAML config file (sft_config.yaml).
    """
    # Load config
    cfg = SFTConfig.from_yaml(config_path)
    print(f"SFT Config:\n{cfg}")

    # Set seed
    torch.manual_seed(cfg.seed)

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32),
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset: {cfg.dataset_path}")
    train_dataset = load_sft_dataset(
        data_path=cfg.dataset_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        system_prompt=cfg.system_prompt,
    )

    # Optional eval split (last 5%)
    split = train_dataset.train_test_split(test_size=0.05, seed=cfg.seed)
    train_data = split["train"]
    eval_data = split["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    loss_logger = LossLoggerCallback()

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg.max_length,
        callbacks=[loss_logger],
    )

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save final checkpoint
    output_dir = Path(cfg.output_dir)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")

    # Save loss curve
    loss_logger.save_loss_curve(cfg.output_dir)

    # Save config snapshot
    config_snapshot = {
        "model_name": cfg.model_name,
        "dataset_path": cfg.dataset_path,
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "max_length": cfg.max_length,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "seed": cfg.seed,
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
    }
    config_path_out = output_dir / "training_config.json"
    with open(config_path_out, "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("SFT training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT Trainer")
    parser.add_argument(
        "--config",
        default="configs/sft_config.yaml",
        help="Path to SFT config YAML",
    )
    args = parser.parse_args()
    train_sft(args.config)
