"""Training module for MiniAlign.

Provides implementations of five preference learning algorithms:
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization via TRL)
- GRPO (Group Relative Policy Optimization)
- ORPO (Odds Ratio Preference Optimization)
- SimPO (Simple Preference Optimization)
"""

from .dpo_trainer import compute_logprobs, dpo_loss, train_dpo
from .grpo_trainer import compute_group_advantages, grpo_loss, train_grpo
from .orpo_trainer import odds_ratio_loss, train_orpo
from .ppo_trainer import train_ppo
from .simpo_trainer import simpo_loss, train_simpo

__all__ = [
    # DPO
    "compute_logprobs",
    "dpo_loss",
    "train_dpo",
    # PPO
    "train_ppo",
    # GRPO
    "compute_group_advantages",
    "grpo_loss",
    "train_grpo",
    # ORPO
    "odds_ratio_loss",
    "train_orpo",
    # SimPO
    "simpo_loss",
    "train_simpo",
]
