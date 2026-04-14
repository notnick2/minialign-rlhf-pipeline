"""ORPO: Odds Ratio Preference Optimization.
Covers: ORPO — combines SFT + preference in one loss, no reference model needed

ORPO unifies SFT and preference alignment in a single training objective.
It penalizes rejected responses via a log odds-ratio term while simultaneously
training on chosen responses with standard SFT cross-entropy.
No reference model or reward model is required.

Reference: Hong et al. (2024) "ORPO: Monolithic Preference Optimization without
           Reference Model"
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# ORPO loss
# ---------------------------------------------------------------------------


def odds_ratio_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    lambda_: float = 0.1,
) -> dict[str, torch.Tensor]:
    """ORPO: combined SFT + odds-ratio preference loss.

    The odds of a response is: odds = P(response) / (1 - P(response))
    Working in log-space: log_odds = logps - log(1 - exp(logps) + eps)

    The log odds-ratio between chosen and rejected serves as an implicit reward
    signal without requiring a reference model.

    Args:
        chosen_logps:   Summed log-probs for chosen responses,   shape (batch,).
        rejected_logps: Summed log-probs for rejected responses, shape (batch,).
        lambda_:        Weight for the odds-ratio loss relative to SFT loss.

    Returns:
        Dict with keys: total_loss, sft_loss, or_loss (all scalar tensors).
    """
    # SFT loss: standard cross-entropy on chosen responses
    sft_loss = -chosen_logps.mean()

    # Convert log-probs to log-odds
    # log_odds = log(p) - log(1 - p) = logp - log(1 - exp(logp))
    # Numerically stable: clamp to avoid log(0)
    chosen_logps_clamped = chosen_logps.clamp(max=-1e-6)
    rejected_logps_clamped = rejected_logps.clamp(max=-1e-6)

    log_odds_chosen = chosen_logps_clamped - torch.log1p(-torch.exp(chosen_logps_clamped) + 1e-8)
    log_odds_rejected = rejected_logps_clamped - torch.log1p(-torch.exp(rejected_logps_clamped) + 1e-8)

    # Log odds-ratio: how much more likely is chosen than rejected?
    log_odds_ratio = log_odds_chosen - log_odds_rejected

    # Preference loss: maximize probability that chosen > rejected
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # Combined ORPO objective
    total_loss = sft_loss + lambda_ * or_loss

    return {
        "total_loss": total_loss,
        "sft_loss": sft_loss,
        "or_loss": or_loss,
    }


# ---------------------------------------------------------------------------
# Dataset (same pair format as DPO)
# ---------------------------------------------------------------------------


def _tokenize_response(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int,
) -> dict:
    """Tokenize a (prompt, response) and create labels for response tokens."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_text = prompt + " " + response
    full_ids = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[:max_length]

    pad_len = max_length - len(full_ids)
    input_ids = full_ids + [tokenizer.pad_token_id or 0] * pad_len
    attention_mask = [1] * len(full_ids) + [0] * pad_len
    labels_padded = labels + [-100] * pad_len

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels_padded[:max_length],
    }


class ORPODataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        prompt = pair.get("prompt", pair.get("instruction", ""))
        chosen = pair.get("chosen", "")
        rejected = pair.get("rejected", "")

        chosen_enc = _tokenize_response(prompt, chosen, self.tokenizer, self.max_length)
        rejected_enc = _tokenize_response(prompt, rejected, self.tokenizer, self.max_length)

        return {
            "chosen_input_ids": torch.tensor(chosen_enc["input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_enc["attention_mask"], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_enc["labels"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_enc["input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_enc["attention_mask"], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_enc["labels"], dtype=torch.long),
        }


def _compute_sequence_logprobs(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """Compute mean log-prob over response label tokens."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(
        dim=2, index=shift_labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)

    label_mask = (shift_labels != -100).float()
    n_tokens = label_mask.sum(dim=-1).clamp(min=1)
    # Use mean log-prob (per-token) to avoid length bias
    return (token_logprobs * label_mask).sum(dim=-1) / n_tokens


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_orpo(config_path: str) -> None:
    """Train with ORPO: unified SFT + preference optimization (no reference model).

    Args:
        config_path: Path to orpo_config.yaml.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("orpo", raw)

    sft_checkpoint = cfg.get("sft_checkpoint", cfg.get("model_name", "microsoft/phi-2"))
    dataset_path = cfg.get("dataset_path", "data/dpo_dataset.jsonl")
    output_dir = Path(cfg.get("output_dir", "checkpoints/orpo"))
    num_epochs = cfg.get("num_epochs", 2)
    batch_size = cfg.get("batch_size", 4)
    learning_rate = cfg.get("learning_rate", 8e-6)
    max_length = cfg.get("max_length", 512)
    lambda_ = cfg.get("lambda_", 0.1)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    seed = cfg.get("seed", 42)
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    logging_steps = cfg.get("logging_steps", 10)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ORPO Training | device={device} | model={sft_checkpoint}")
    print("Note: No reference model needed with ORPO!")

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    # Load pairs
    pairs: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    dataset = ORPODataset(pairs, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log: list[dict] = []
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_sft_loss = 0.0
        epoch_or_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            chosen_logps = _compute_sequence_logprobs(model, chosen_ids, chosen_mask, chosen_labels)
            rejected_logps = _compute_sequence_logprobs(model, rejected_ids, rejected_mask, rejected_labels)

            losses = odds_ratio_loss(chosen_logps, rejected_logps, lambda_=lambda_)
            loss = losses["total_loss"]

            (loss / grad_accum).backward()
            epoch_total_loss += loss.item()
            epoch_sft_loss += losses["sft_loss"].item()
            epoch_or_loss += losses["or_loss"].item()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "total_loss": losses["total_loss"].item(),
                    "sft_loss": losses["sft_loss"].item(),
                    "or_loss": losses["or_loss"].item(),
                }
                step_log.append(log_entry)

                if global_step % logging_steps == 0:
                    print(
                        f"  Step {global_step:5d} | "
                        f"total={losses['total_loss'].item():.4f} | "
                        f"sft={losses['sft_loss'].item():.4f} | "
                        f"or={losses['or_loss'].item():.4f}"
                    )

        n = len(dataloader)
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"avg_total={epoch_total_loss/n:.4f} | "
            f"avg_sft={epoch_sft_loss/n:.4f} | "
            f"avg_or={epoch_or_loss/n:.4f}"
        )

    # Save
    checkpoint_dir = output_dir / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    with open(output_dir / "orpo_step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)

    print(f"ORPO training complete. Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orpo_config.yaml")
    args = parser.parse_args()
    train_orpo(args.config)
