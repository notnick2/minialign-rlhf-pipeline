"""GRPO: Group Relative Policy Optimization.
Covers: GRPO

GRPO (introduced with DeepSeek-R1) replaces the value function critic with
a group-relative advantage estimate: for each prompt, G responses are sampled
and advantages are computed within the group by mean/std normalization.
This avoids the need for a separate value model while preserving PPO-style clipping.

Reference: DeepSeek-AI (2025) "DeepSeek-R1: Incentivizing Reasoning Capability
           in LLMs via Reinforcement Learning"
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# GRPO core functions
# ---------------------------------------------------------------------------


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Normalize rewards within each group to get group-relative advantages.

    Args:
        rewards: Tensor of shape (batch_size, G) where G is the group size.
                 Each row contains reward scores for G sampled responses
                 to the same prompt.

    Returns:
        Advantage tensor of shape (batch_size, G), zero-mean and unit-variance
        within each group row.
    """
    mean = rewards.mean(dim=1, keepdim=True)          # (batch, 1)
    std = rewards.std(dim=1, keepdim=True) + 1e-8     # (batch, 1) — avoid div-by-zero
    advantages = (rewards - mean) / std                # (batch, G)
    return advantages


def grpo_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ref_logprobs: torch.Tensor,
    beta: float = 0.1,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """GRPO objective: clipped policy gradient + KL penalty.

    Args:
        logprobs:     Current policy log-probs per response, shape (N,).
                      N = batch_size * G (flattened group dimension).
        advantages:   Normalized advantages, shape (N,).
        ref_logprobs: Reference policy log-probs, shape (N,).
        beta:         KL penalty coefficient.
        epsilon:      PPO clip range (ratio in [1-epsilon, 1+epsilon]).

    Returns:
        Scalar GRPO loss.
    """
    # Probability ratio π_policy / π_ref
    ratio = torch.exp(logprobs - ref_logprobs)  # (N,)

    # Clipped surrogate objective (PPO-style)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # KL divergence penalty: E[logprobs - ref_logprobs] ≈ KL(π || π_ref)
    kl_penalty = beta * (logprobs - ref_logprobs).mean()

    return policy_loss + kl_penalty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: int,
) -> torch.Tensor:
    """Compute summed log-prob of the response portion of a sequence."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (1, seq, vocab)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # shift

    # Response tokens only
    response_ids = input_ids[:, 1:]  # shifted labels
    token_logprobs = log_probs.gather(
        dim=2, index=response_ids.clamp(min=0).unsqueeze(2)
    ).squeeze(2)  # (1, seq-1)

    # Only sum over response positions
    resp_len = token_logprobs.shape[1] - response_start_idx
    if resp_len <= 0:
        return torch.tensor(0.0, device=input_ids.device)

    response_logprobs = token_logprobs[:, response_start_idx:]
    return response_logprobs.sum(dim=-1).squeeze(0)  # scalar


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_grpo(config_path: str) -> None:
    """Train with GRPO: group-relative policy optimization.

    For each prompt, generates G responses, scores them with a reward model,
    computes normalized group advantages, and applies GRPO loss.

    Args:
        config_path: Path to grpo_config.yaml.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("grpo", raw)

    sft_checkpoint = cfg.get("sft_checkpoint", cfg.get("model_name", "microsoft/phi-2"))
    reward_model_path = cfg.get("reward_model_path", "checkpoints/reward_model/best")
    dataset_path = cfg.get("dataset_path", "data/sft_train.jsonl")
    output_dir = Path(cfg.get("output_dir", "checkpoints/grpo"))
    num_steps = cfg.get("num_steps", 200)
    batch_size = cfg.get("batch_size", 4)
    group_size = cfg.get("group_size", 4)  # G
    learning_rate = cfg.get("learning_rate", 5e-6)
    max_new_tokens = cfg.get("max_new_tokens", 128)
    max_input_length = cfg.get("max_input_length", 256)
    beta = cfg.get("beta", 0.1)
    epsilon = cfg.get("epsilon", 0.2)
    seed = cfg.get("seed", 42)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)
    logging_steps = cfg.get("logging_steps", 10)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GRPO Training | device={device} | model={sft_checkpoint} | G={group_size}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model with LoRA
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

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Reward model
    from reward.model import RewardModel
    reward_model = RewardModel.load_pretrained(reward_model_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    reward_model = reward_model.to(device)
    reward_model.eval()

    # Load prompts
    prompts: list[str] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                p = obj.get("instruction", obj.get("prompt", ""))
                if p:
                    prompts.append(p)
    print(f"Loaded {len(prompts)} prompts")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    step_log: list[dict] = []

    import random
    random.seed(seed)

    for step in range(num_steps):
        # Sample batch of prompts
        batch_prompts = random.choices(prompts, k=batch_size)

        batch_logprobs: list[torch.Tensor] = []
        batch_ref_logprobs: list[torch.Tensor] = []
        batch_rewards: list[list[float]] = []

        policy_model.eval()  # eval mode for generation

        for prompt in batch_prompts:
            prompt_enc = tokenizer(
                prompt,
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids = prompt_enc["input_ids"].to(device)
            prompt_len = prompt_ids.shape[1]

            group_logprobs: list[torch.Tensor] = []
            group_ref_logprobs: list[torch.Tensor] = []
            group_rewards: list[float] = []

            for _ in range(group_size):
                # Generate one response
                with torch.no_grad():
                    generated = policy_model.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                full_ids = generated  # (1, prompt_len + response_len)
                full_mask = torch.ones_like(full_ids)

                # Compute policy log-probs for response tokens
                policy_model.train()
                lp = _compute_sequence_logprobs(
                    policy_model, full_ids, full_mask, response_start_idx=prompt_len - 1
                )
                group_logprobs.append(lp)
                policy_model.eval()

                # Compute reference log-probs
                with torch.no_grad():
                    ref_lp = _compute_sequence_logprobs(
                        ref_model, full_ids, full_mask, response_start_idx=prompt_len - 1
                    )
                group_ref_logprobs.append(ref_lp)

                # Score with reward model
                response_text = tokenizer.decode(
                    generated[0][prompt_len:], skip_special_tokens=True
                )
                full_text = prompt + " " + response_text
                rew_enc = reward_tokenizer(
                    full_text,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                with torch.no_grad():
                    score = reward_model(
                        rew_enc["input_ids"].to(device),
                        rew_enc["attention_mask"].to(device),
                    ).item()
                group_rewards.append(score)

            batch_logprobs.append(torch.stack(group_logprobs))      # (G,)
            batch_ref_logprobs.append(torch.stack(group_ref_logprobs))  # (G,)
            batch_rewards.append(group_rewards)

        # Compute advantages
        rewards_tensor = torch.tensor(batch_rewards, device=device)  # (B, G)
        advantages = compute_group_advantages(rewards_tensor)         # (B, G)

        # Flatten for loss computation
        logprobs_flat = torch.stack(batch_logprobs).view(-1)        # (B*G,)
        ref_logprobs_flat = torch.stack(batch_ref_logprobs).view(-1)
        advantages_flat = advantages.view(-1)

        # GRPO loss and backward
        policy_model.train()
        loss = grpo_loss(logprobs_flat, advantages_flat, ref_logprobs_flat, beta, epsilon)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        group_reward_mean = rewards_tensor.mean().item()
        advantage_std = advantages_flat.std().item()

        log_entry = {
            "step": step + 1,
            "loss": loss.item(),
            "group_reward_mean": group_reward_mean,
            "advantage_std": advantage_std,
        }
        step_log.append(log_entry)

        if (step + 1) % logging_steps == 0:
            print(
                f"  Step {step+1:5d} | loss={loss.item():.4f} | "
                f"group_reward_mean={group_reward_mean:.4f} | "
                f"advantage_std={advantage_std:.4f}"
            )

    # Save
    checkpoint_dir = output_dir / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    with open(output_dir / "grpo_step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)

    print(f"GRPO training complete. Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    args = parser.parse_args()
    train_grpo(args.config)
