"""Reward modeling module for MiniAlign.

Provides:
- RewardModel: DistilBERT-based scalar reward model
- bradley_terry_loss: Bradley-Terry preference loss
- train_reward_model: Full training loop with preference data
- evaluate_reward_model: Accuracy and reward gap metrics
"""

from .model import RewardModel
from .train import bradley_terry_loss, evaluate_reward_model, train_reward_model

__all__ = [
    "RewardModel",
    "bradley_terry_loss",
    "train_reward_model",
    "evaluate_reward_model",
]
