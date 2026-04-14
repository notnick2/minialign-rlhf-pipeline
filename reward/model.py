"""Reward model architecture.
Covers: Reward Modeling: training models on human preference comparisons

A lightweight reward model built on DistilBERT that maps a
(prompt + response) text to a single scalar reward score.

Design choices:
- DistilBERT: fast, small (66M params), easily trainable on consumer hardware.
- Single scalar head: industry standard for Bradley-Terry preference learning.
- Tied tokenizer: saved alongside the model for reproducible inference.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RewardModel(nn.Module):
    """Scalar reward model: text → reward score.

    Architecture:
        DistilBERT encoder → [CLS] representation → Linear(hidden_size, 1)

    The [CLS] token embedding is used as the sequence representation,
    consistent with classification fine-tuning of BERT-family models.

    Args:
        backbone: HuggingFace model name or path for the encoder.
        dropout_prob: Dropout applied before the linear head.
    """

    DEFAULT_BACKBONE = "distilbert-base-uncased"

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.reward_head = nn.Linear(hidden_size, 1)

        # Initialize head near zero for stable early training
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar reward scores.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len).
            attention_mask: Attention mask, shape (batch_size, seq_len).

        Returns:
            Reward scores, shape (batch_size,) — one scalar per example.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # DistilBERT returns last_hidden_state; index [CLS] at position 0
        if hasattr(outputs, "last_hidden_state"):
            cls_hidden = outputs.last_hidden_state[:, 0, :]
        else:
            # Fallback for pooler-equipped models
            cls_hidden = outputs.pooler_output

        cls_hidden = self.dropout(cls_hidden)
        reward = self.reward_head(cls_hidden).squeeze(-1)  # (batch_size,)
        return reward

    def save_pretrained(self, path: str) -> None:
        """Save model weights and config to directory.

        Args:
            path: Directory path. Created if it does not exist.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch weights
        torch.save(self.state_dict(), save_dir / "reward_model.pt")

        # Save backbone encoder (HuggingFace format for easy re-loading)
        self.encoder.save_pretrained(str(save_dir / "backbone"))

        # Save tokenizer alongside the model
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
            tokenizer.save_pretrained(str(save_dir))
        except Exception as exc:
            # Tokenizer saving is a best-effort convenience; avoid failing save_pretrained().
            print(f"Warning: could not save tokenizer: {exc}")

        # Save meta config
        meta = {
            "backbone": self.backbone_name,
            "hidden_size": self.encoder.config.hidden_size,
            "dropout_prob": self.dropout.p,
        }
        with open(save_dir / "reward_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"RewardModel saved to {save_dir}")

    @classmethod
    def load_pretrained(cls, path: str) -> "RewardModel":
        """Load a saved reward model from directory.

        Args:
            path: Directory containing reward_model.pt and reward_config.json.

        Returns:
            Loaded RewardModel in eval mode.
        """
        load_dir = Path(path)
        config_path = load_dir / "reward_config.json"

        if config_path.exists():
            with open(config_path) as f:
                meta = json.load(f)
            backbone = meta.get("backbone", cls.DEFAULT_BACKBONE)
            dropout_prob = meta.get("dropout_prob", 0.1)
        else:
            backbone = cls.DEFAULT_BACKBONE
            dropout_prob = 0.1

        model = cls(backbone=backbone, dropout_prob=dropout_prob)

        weights_path = load_dir / "reward_model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        print(f"RewardModel loaded from {load_dir}")
        return model
