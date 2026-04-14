"""PPO: Proximal Policy Optimization using TRL.
Covers: PPO (Proximal Policy Optimization): RL fine-tuning using reward signal

PPO is the classic RL algorithm applied to language models.
A frozen reference model provides KL regularization; a separate reward
model (or function) scores generated responses.

TRL's PPOTrainer handles the full actor-critic loop including:
  - Value function estimation
  - Advantage computation via GAE
  - Clipped surrogate objective
  - Value function loss
  - Entropy bonus

Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
           Ziegler et al. (2019) "Fine-Tuning Language Models from Human Preferences"
"""

import json
from pathlib import Path
from typing import Optional

import torch
import yaml
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


def _load_ppo_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("ppo", cfg)


def train_ppo(config_path: str) -> None:
    """Train a causal LM using PPO with a reward model scorer.

    Args:
        config_path: Path to ppo_config.yaml.
    """
    cfg = _load_ppo_config(config_path)

    sft_checkpoint = cfg.get("sft_checkpoint", cfg.get("model_name", "microsoft/phi-2"))
    reward_model_path = cfg.get("reward_model_path", "checkpoints/reward_model/best")
    dataset_path = cfg.get("dataset_path", "data/sft_train.jsonl")
    output_dir = Path(cfg.get("output_dir", "checkpoints/ppo"))
    batch_size = cfg.get("batch_size", 8)
    mini_batch_size = cfg.get("mini_batch_size", 4)
    ppo_epochs = cfg.get("ppo_epochs", 4)
    num_steps = cfg.get("num_steps", 200)
    learning_rate = cfg.get("learning_rate", 1.4e-5)
    kl_coef = cfg.get("kl_coef", 0.2)
    max_new_tokens = cfg.get("max_new_tokens", 128)
    max_input_length = cfg.get("max_input_length", 256)
    seed = cfg.get("seed", 42)
    lora_r = cfg.get("lora_r", 16)
    lora_alpha = cfg.get("lora_alpha", 32)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PPO Training | device={device} | sft={sft_checkpoint}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation

    # Policy model with value head
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_checkpoint,
        peft_config=lora_config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    policy_model = policy_model.to(device)

    # Reference model (frozen)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
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
    reward_model = reward_model.to(device)
    reward_model.eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

    # Load prompts dataset
    prompts: list[str] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompt = obj.get("instruction", obj.get("prompt", ""))
                if prompt:
                    prompts.append(prompt)

    if not prompts:
        raise ValueError(f"No prompts found in {dataset_path}")

    print(f"Loaded {len(prompts)} prompts")

    # PPO config
    ppo_config = PPOConfig(
        model_name=sft_checkpoint,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        ppo_epochs=ppo_epochs,
        kl_penalty="kl",
        init_kl_coef=kl_coef,
        adap_kl_ctrl=True,
        target_kl=6.0,
        seed=seed,
        log_with=None,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    step_log: list[dict] = []

    # Generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
    }

    print(f"Starting PPO training for {num_steps} steps...")

    import random
    random.seed(seed)

    for step in range(num_steps):
        # Sample a batch of prompts
        batch_prompts = random.choices(prompts, k=batch_size)

        # Tokenize prompts
        queries: list[torch.Tensor] = []
        for prompt in batch_prompts:
            enc = tokenizer(
                prompt,
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            queries.append(enc["input_ids"].squeeze(0).to(device))

        # Generate responses from policy
        responses: list[torch.Tensor] = []
        for query in queries:
            response_ids = ppo_trainer.generate(
                query.unsqueeze(0), **gen_kwargs
            )
            # response_ids includes the prompt; slice it off
            response_only = response_ids[0][len(query):]
            responses.append(response_only)

        # Decode for reward scoring
        decoded_responses = [
            tokenizer.decode(r, skip_special_tokens=True) for r in responses
        ]

        # Score with reward model
        rewards: list[torch.Tensor] = []
        reward_model.eval()
        with torch.no_grad():
            for prompt, response in zip(batch_prompts, decoded_responses):
                full_text = prompt + " " + response
                enc = reward_tokenizer(
                    full_text,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                score = reward_model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                )
                rewards.append(score.squeeze())

        # PPO step
        stats = ppo_trainer.step(queries, responses, rewards)

        mean_reward = torch.stack(rewards).mean().item()
        kl_div = stats.get("objective/kl", 0.0)
        policy_loss = stats.get("ppo/loss/policy", 0.0)

        log_entry = {
            "step": step + 1,
            "mean_reward": mean_reward,
            "kl_divergence": kl_div,
            "policy_loss": policy_loss,
        }
        step_log.append(log_entry)

        if (step + 1) % 10 == 0:
            print(
                f"  Step {step+1:5d} | mean_reward={mean_reward:.4f} | "
                f"kl={kl_div:.4f} | policy_loss={policy_loss:.4f}"
            )

    # Save checkpoint
    checkpoint_dir = output_dir / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ppo_trainer.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    with open(output_dir / "ppo_step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)

    print(f"PPO training complete. Checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_config.yaml")
    args = parser.parse_args()
    train_ppo(args.config)
