"""SimPO: Simple Preference Optimization.
Covers: SimPO — length-normalized reward, no reference model needed

SimPO uses length-normalized log-probabilities as implicit rewards and adds
a target reward margin gamma. This eliminates both the reference model and
the reward model, while the length normalization prevents the model from
exploiting shorter responses to inflate reward scores.

Reference: Meng et al. (2024) "SimPO: Simple Preference Optimization with a
           Reference-Free Reward"
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
# SimPO loss
# ---------------------------------------------------------------------------


def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """SimPO loss: length-normalized preference optimization.

    Computes length-normalized implicit rewards and optimizes the margin
    between chosen and rejected with a minimum target margin gamma.

    Args:
        policy_chosen_logps:   Summed log-probs for chosen,   shape (batch,).
        policy_rejected_logps: Summed log-probs for rejected, shape (batch,).
        chosen_lengths:        Number of response tokens per chosen,   shape (batch,).
        rejected_lengths:      Number of response tokens per rejected, shape (batch,).
        beta:   Scaling temperature for the preference margin.
        gamma:  Target reward margin (minimum gap between chosen and rejected scores).

    Returns:
        Scalar SimPO loss.
    """
    # Length-normalized reward: (1/|y|) * log π(y|x) - gamma
    chosen_score = (policy_chosen_logps / chosen_lengths.float().clamp(min=1)) - gamma
    rejected_score = (policy_rejected_logps / rejected_lengths.float().clamp(min=1)) - gamma

    # Preference loss: chosen score should exceed rejected by beta-scaled margin
    loss = -F.logsigmoid(beta * (chosen_score - rejected_score)).mean()
    return loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _count_response_tokens(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int,
) -> dict:
    """Tokenize (prompt + response) and count response tokens."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_text = prompt + " " + response
    full_ids = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    # Number of response tokens (after prompt, capped by max_length)
    n_response_tokens = max(1, len(full_ids) - len(prompt_ids))

    # Build labels: -100 for prompt tokens
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
        "n_response_tokens": n_response_tokens,
    }


class SimPODataset(Dataset):
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

        chosen_enc = _count_response_tokens(prompt, chosen, self.tokenizer, self.max_length)
        rejected_enc = _count_response_tokens(prompt, rejected, self.tokenizer, self.max_length)

        return {
            "chosen_input_ids": torch.tensor(chosen_enc["input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_enc["attention_mask"], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_enc["labels"], dtype=torch.long),
            "chosen_length": torch.tensor(chosen_enc["n_response_tokens"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_enc["input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_enc["attention_mask"], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_enc["labels"], dtype=torch.long),
            "rejected_length": torch.tensor(rejected_enc["n_response_tokens"], dtype=torch.long),
        }


def _compute_summed_logprobs(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """Compute summed log-prob over response label tokens."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(
        dim=2, index=shift_labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)

    label_mask = (shift_labels != -100).float()
    return (token_logprobs * label_mask).sum(dim=-1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_simpo(config_path: str) -> None:
    """Train with SimPO: length-normalized preference optimization without reference model.

    Args:
        config_path: Path to simpo_config.yaml.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("simpo", raw)

    sft_checkpoint = cfg.get("sft_checkpoint", cfg.get("model_name", "microsoft/phi-2"))
    dataset_path = cfg.get("dataset_path", "data/dpo_dataset.jsonl")
    output_dir = Path(cfg.get("output_dir", "checkpoints/simpo"))
    num_epochs = cfg.get("num_epochs", 2)
    batch_size = cfg.get("batch_size", 4)
    learning_rate = cfg.get("learning_rate", 8e-6)
    max_length = cfg.get("max_length", 512)
    beta = cfg.get("beta", 2.0)
    gamma = cfg.get("gamma", 0.5)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    seed = cfg.get("seed", 42)
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    logging_steps = cfg.get("logging_steps", 10)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SimPO Training | device={device} | model={sft_checkpoint}")
    print("Note: No reference model needed with SimPO!")

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

    dataset = SimPODataset(pairs, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log: list[dict] = []
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_norm_gap = 0.0

        for batch_idx, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            chosen_lengths = batch["chosen_length"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            rejected_lengths = batch["rejected_length"].to(device)

            chosen_logps = _compute_summed_logprobs(model, chosen_ids, chosen_mask, chosen_labels)
            rejected_logps = _compute_summed_logprobs(model, rejected_ids, rejected_mask, rejected_labels)

            loss = simpo_loss(
                chosen_logps, rejected_logps,
                chosen_lengths, rejected_lengths,
                beta=beta, gamma=gamma,
            )
            (loss / grad_accum).backward()
            epoch_loss += loss.item()

            # Length-normalized reward gap for logging
            norm_gap = (
                chosen_logps / chosen_lengths.float().clamp(min=1)
                - rejected_logps / rejected_lengths.float().clamp(min=1)
            ).mean().item()
            epoch_norm_gap += norm_gap

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "length_normalized_reward_gap": norm_gap,
                }
                step_log.append(log_entry)

                if global_step % logging_steps == 0:
                    print(
                        f"  Step {global_step:5d} | loss={loss.item():.4f} | "
                        f"norm_reward_gap={norm_gap:.4f}"
                    )

        n = len(dataloader)
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"avg_loss={epoch_loss/n:.4f} | "
            f"avg_norm_gap={epoch_norm_gap/n:.4f}"
        )

    # Save
    checkpoint_dir = output_dir / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    with open(output_dir / "simpo_step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)

    print(f"SimPO training complete. Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simpo_config.yaml")
    args = parser.parse_args()
    train_simpo(args.config)
