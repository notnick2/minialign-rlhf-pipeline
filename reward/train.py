"""Train reward model on preference data.
Covers: Reward Modeling

Trains the RewardModel using the Bradley-Terry pairwise preference loss.
Evaluates accuracy (% chosen > rejected) and reward gap.
Produces a calibration analysis plot (reward distribution for chosen vs rejected).
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .model import RewardModel


# ---------------------------------------------------------------------------
# Bradley-Terry loss
# ---------------------------------------------------------------------------


def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise preference loss.

    Maximizes the probability that the chosen response has a higher reward
    than the rejected response under a logistic model.

    Args:
        reward_chosen:   Scalar rewards for chosen responses, shape (batch,).
        reward_rejected: Scalar rewards for rejected responses, shape (batch,).

    Returns:
        Scalar loss tensor.
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PreferencePairDataset(Dataset):
    """Dataset of (prompt+chosen, prompt+rejected) pairs for reward training."""

    def __init__(
        self,
        pairs: list[dict],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        prompt = pair.get("prompt", "")
        chosen = pair.get("chosen", "")
        rejected = pair.get("rejected", "")

        chosen_text = prompt + " " + chosen
        rejected_text = prompt + " " + rejected

        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_reward_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "reward_model" in cfg:
        cfg = cfg["reward_model"]
    defaults = {
        "backbone": "distilbert-base-uncased",
        "dataset_path": "data/dpo_dataset.jsonl",
        "output_dir": "checkpoints/reward_model",
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "max_length": 512,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "seed": 42,
        "val_ratio": 0.1,
    }
    defaults.update(cfg)
    return defaults


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_reward_model(config_path: str) -> None:
    """Train the reward model on preference pairs.

    Loads config, trains with Bradley-Terry loss, saves the best checkpoint
    (lowest validation loss), and produces a reward distribution plot.

    Args:
        config_path: Path to reward_model_config.yaml.
    """
    cfg = _load_reward_config(config_path)
    print(f"Reward model config: {json.dumps(cfg, indent=2)}")

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data_path = Path(cfg["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    all_pairs: list[dict] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(json.loads(line))

    # Split
    import random
    random.seed(cfg["seed"])
    random.shuffle(all_pairs)
    val_n = max(1, int(len(all_pairs) * cfg["val_ratio"]))
    train_pairs = all_pairs[val_n:]
    val_pairs = all_pairs[:val_n]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    train_ds = PreferencePairDataset(train_pairs, tokenizer, cfg["max_length"])
    val_ds = PreferencePairDataset(val_pairs, tokenizer, cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # Model
    model = RewardModel(backbone=cfg["backbone"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Linear warmup scheduler
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = cfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(cfg["num_epochs"]):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            loss = bradley_terry_loss(reward_chosen, reward_rejected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            reward_gap = (reward_chosen - reward_rejected).mean().item()
            epoch_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                print(
                    f"  Step {global_step:5d} | loss={loss.item():.4f} | "
                    f"reward_gap={reward_gap:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_gaps: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                chosen_ids = batch["chosen_input_ids"].to(device)
                chosen_mask = batch["chosen_attention_mask"].to(device)
                rejected_ids = batch["rejected_input_ids"].to(device)
                rejected_mask = batch["rejected_attention_mask"].to(device)

                r_chosen = model(chosen_ids, chosen_mask)
                r_rejected = model(rejected_ids, rejected_mask)

                loss = bradley_terry_loss(r_chosen, r_rejected)
                val_loss += loss.item()

                correct = (r_chosen > r_rejected).sum().item()
                val_correct += correct
                val_total += len(r_chosen)
                val_gaps.extend((r_chosen - r_rejected).cpu().tolist())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        val_accuracy = val_correct / max(1, val_total)
        mean_gap = sum(val_gaps) / max(1, len(val_gaps))

        print(
            f"Epoch {epoch+1}/{cfg['num_epochs']} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_accuracy:.3f} | mean_gap={mean_gap:.4f}"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(str(output_dir / "best"))
            print(f"  → New best checkpoint saved (val_loss={best_val_loss:.4f})")

    # Save final checkpoint
    model.save_pretrained(str(output_dir / "final"))

    # Save loss history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Reward distribution plot
    _plot_reward_distributions(model, val_ds, val_loader, device, output_dir)

    print(f"\nReward model training complete. Outputs saved to {output_dir}")


def _plot_reward_distributions(
    model: RewardModel,
    val_ds: PreferencePairDataset,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Plot chosen vs rejected reward distributions and save as PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping reward distribution plot")
        return

    model.eval()
    chosen_rewards: list[float] = []
    rejected_rewards: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            r_chosen = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            r_rejected = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )
            chosen_rewards.extend(r_chosen.cpu().tolist())
            rejected_rewards.extend(r_rejected.cpu().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(chosen_rewards, bins=40, alpha=0.6, label="Chosen", color="steelblue")
    axes[0].hist(rejected_rewards, bins=40, alpha=0.6, label="Rejected", color="salmon")
    axes[0].set_xlabel("Reward Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reward Distribution: Chosen vs Rejected")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter chosen vs rejected
    axes[1].scatter(rejected_rewards, chosen_rewards, alpha=0.4, s=15, color="purple")
    min_r = min(rejected_rewards + chosen_rewards)
    max_r = max(rejected_rewards + chosen_rewards)
    axes[1].plot([min_r, max_r], [min_r, max_r], "k--", linewidth=1, label="y=x")
    axes[1].set_xlabel("Reward (Rejected)")
    axes[1].set_ylabel("Reward (Chosen)")
    axes[1].set_title("Chosen vs Rejected Reward Scatter")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "reward_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Reward distribution plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_reward_model(
    model_path: str,
    eval_data: list[dict],
    max_length: int = 512,
    batch_size: int = 32,
) -> dict:
    """Evaluate a trained reward model on preference pairs.

    Args:
        model_path: Path to saved RewardModel directory.
        eval_data: List of {prompt, chosen, rejected} dicts.
        max_length: Maximum token length.
        batch_size: Inference batch size.

    Returns:
        Dict with keys: accuracy, mean_reward_gap, mean_chosen_reward,
        mean_rejected_reward, n_pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RewardModel.load_pretrained(model_path)
    model = model.to(device)
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model.backbone_name)

    ds = PreferencePairDataset(eval_data, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    chosen_rewards: list[float] = []
    rejected_rewards: list[float] = []

    with torch.no_grad():
        for batch in loader:
            r_chosen = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            r_rejected = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )
            chosen_rewards.extend(r_chosen.cpu().tolist())
            rejected_rewards.extend(r_rejected.cpu().tolist())

    n = len(chosen_rewards)
    accuracy = sum(c > r for c, r in zip(chosen_rewards, rejected_rewards)) / max(1, n)
    gaps = [c - r for c, r in zip(chosen_rewards, rejected_rewards)]
    mean_gap = sum(gaps) / max(1, n)

    metrics = {
        "accuracy": accuracy,
        "mean_reward_gap": mean_gap,
        "mean_chosen_reward": sum(chosen_rewards) / max(1, n),
        "mean_rejected_reward": sum(rejected_rewards) / max(1, n),
        "n_pairs": n,
    }

    print(f"\nReward Model Evaluation ({n} pairs):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--config",
        default="configs/reward_model_config.yaml",
        help="Path to reward model config YAML",
    )
    args = parser.parse_args()
    train_reward_model(args.config)
