"""DPO: Direct Preference Optimization.
Covers: DPO (Direct Preference Optimization): preference learning without separate RM

DPO eliminates the need for a separate reward model by showing that the
optimal policy under RLHF can be derived analytically from the reference model.
The loss directly optimizes the preference probability using paired data.

Reference: Rafailov et al. (2023) "Direct Preference Optimization:
           Your Language Model is Secretly a Reward Model"
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
# Log-probability computation
# ---------------------------------------------------------------------------


def compute_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log-probabilities for label tokens.

    Args:
        model: Causal LM (HuggingFace).
        input_ids: Token IDs, shape (batch, seq_len).
        attention_mask: Mask, shape (batch, seq_len).
        labels: Label IDs, shape (batch, seq_len). -100 tokens are ignored.

    Returns:
        Per-example summed log-prob of label tokens, shape (batch,).
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits  # (batch, seq_len, vocab_size)

    # Shift: predict token[i+1] from token[i]
    shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    shift_labels = labels[:, 1:]       # (batch, seq_len-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch, seq_len-1, vocab)

    # Gather log-prob of the actual label tokens
    token_logprobs = log_probs.gather(
        dim=2, index=shift_labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)  # (batch, seq_len-1)

    # Mask out padding / ignored positions (-100 labels)
    label_mask = (shift_labels != -100).float()
    sequence_logprobs = (token_logprobs * label_mask).sum(dim=-1)  # (batch,)

    return sequence_logprobs


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    Computes the implicit reward ratio and optimizes the preference probability.

    Args:
        policy_chosen_logps:   Policy log-probs for chosen responses,   shape (batch,).
        policy_rejected_logps: Policy log-probs for rejected responses, shape (batch,).
        ref_chosen_logps:      Reference log-probs for chosen,          shape (batch,).
        ref_rejected_logps:    Reference log-probs for rejected,        shape (batch,).
        beta: KL penalty coefficient. Higher beta = closer to reference.

    Returns:
        Scalar DPO loss.
    """
    # Implicit reward = log(π_policy / π_ref) scaled by beta
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    # Preference probability under Bradley-Terry model
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()
    return loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _tokenize_pair(instruction: str, response: str, tokenizer, max_length: int) -> dict:
    """Tokenize a (prompt, response) pair and create labels for response tokens only."""
    prompt_enc = tokenizer.encode(instruction, add_special_tokens=True)
    full_text = instruction + " " + response
    full_enc = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    # Labels: -100 for prompt tokens, actual token IDs for response tokens
    labels = [-100] * len(prompt_enc) + full_enc[len(prompt_enc):]
    labels = labels[:max_length]

    # Pad to max_length
    pad_len = max_length - len(full_enc)
    input_ids = full_enc + [tokenizer.pad_token_id or 0] * pad_len
    attention_mask = [1] * len(full_enc) + [0] * pad_len
    labels_padded = labels + [-100] * pad_len

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels_padded[:max_length],
    }


class DPODataset(Dataset):
    """Dataset for DPO training — yields tokenized (chosen, rejected) pairs."""

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

        chosen_enc = _tokenize_pair(prompt, chosen, self.tokenizer, self.max_length)
        rejected_enc = _tokenize_pair(prompt, rejected, self.tokenizer, self.max_length)

        return {
            "chosen_input_ids": torch.tensor(chosen_enc["input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_enc["attention_mask"], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_enc["labels"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_enc["input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_enc["attention_mask"], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_enc["labels"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_dpo(config_path: str) -> None:
    """Train a causal LM using DPO on preference pairs.

    Args:
        config_path: Path to dpo_config.yaml.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("dpo", raw)

    model_name = cfg.get("model_name", "microsoft/phi-2")
    sft_checkpoint = cfg.get("sft_checkpoint", cfg.get("model_name", model_name))
    dataset_path = cfg.get("dataset_path", "data/dpo_dataset.jsonl")
    output_dir = Path(cfg.get("output_dir", "checkpoints/dpo"))
    num_epochs = cfg.get("num_epochs", 1)
    batch_size = cfg.get("batch_size", 2)
    learning_rate = cfg.get("learning_rate", 5e-5)
    max_length = cfg.get("max_length", 512)
    beta = cfg.get("beta", 0.1)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    seed = cfg.get("seed", 42)
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    logging_steps = cfg.get("logging_steps", 10)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DPO Training | device={device} | model={sft_checkpoint}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (trainable) with LoRA
    policy_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    policy_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model = policy_model.to(device)
    policy_model.print_trainable_parameters()

    # Reference model (frozen SFT checkpoint)
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    ref_model = ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load dataset
    pairs: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    dataset = DPODataset(pairs, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    step_log: list[dict] = []
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        policy_model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # Policy log-probs (with gradient)
            policy_chosen_logps = compute_logprobs(
                policy_model, chosen_ids, chosen_mask, chosen_labels
            )
            policy_rejected_logps = compute_logprobs(
                policy_model, rejected_ids, rejected_mask, rejected_labels
            )

            # Reference log-probs (no gradient)
            with torch.no_grad():
                ref_chosen_logps = compute_logprobs(
                    ref_model, chosen_ids, chosen_mask, chosen_labels
                )
                ref_rejected_logps = compute_logprobs(
                    ref_model, rejected_ids, rejected_mask, rejected_labels
                )

            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )
            (loss / grad_accum).backward()

            epoch_loss += loss.item()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                reward_margin = (
                    (policy_chosen_logps - policy_rejected_logps).mean().item()
                )
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "reward_margin": reward_margin,
                }
                step_log.append(log_entry)

                if global_step % logging_steps == 0:
                    print(
                        f"  Step {global_step:5d} | loss={loss.item():.4f} | "
                        f"reward_margin={reward_margin:.4f}"
                    )

        print(f"Epoch {epoch+1}/{num_epochs} complete | avg_loss={epoch_loss/len(dataloader):.4f}")

    # Save
    checkpoint_dir = output_dir / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    with open(output_dir / "dpo_step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)

    print(f"DPO training complete. Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    args = parser.parse_args()
    train_dpo(args.config)
