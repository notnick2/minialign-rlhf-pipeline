# MiniAlign

MiniAlign is a complete, end-to-end RLHF alignment training pipeline that implements the full spectrum of modern preference learning algorithms — from supervised fine-tuning and human annotation collection through reward modeling and policy optimization — in a single, well-structured Python codebase designed as a portfolio-quality reference implementation for alignment research and engineering.

---

## Architecture

```
RedSynth ──► [SFT Data] ──────────────────────────────────────────┐
             [Contrastive Pairs]                                    │
                                                                    ▼
Annotation Interface (Gradio)                              MiniAlign Training
  Tab 1: Classification/NER/Sentiment/Intent                │
  Tab 2: Pairwise Preference A/B      ──► Reward Model ────►│──► PPO
  Tab 3: Instruction Quality 1-5      ──► DPO Dataset  ────►│──► DPO
  Tab 4: Factuality Annotation                              │──► GRPO
  Tab 5: Toxicity/Bias Labeling                            │──► ORPO
  IAA: Cohen's Kappa + Krippendorff's α                    └──► SimPO
                                                                    │
Constitutional AI / RLAIF ──────────────────────────────────────────┘
  (Critique → Revise → Preference pairs)                            │
                                                                    ▼
                                                          Aligned Model Checkpoints
                                                                    │
                                                                    ▼
                                                           EvalForge (evaluation)
```

Data flows left-to-right and top-to-bottom:
1. **RedSynth** (external project) supplies raw instruction-response pairs and synthetic data.
2. **Annotation Interface** collects human preference labels across 5 annotation types.
3. **Constitutional AI / RLAIF** generates preference pairs via autonomous critique-revision cycles.
4. **MiniAlign Training** uses all three data sources to train alignment algorithms.
5. **EvalForge** (external project) evaluates resulting checkpoints.

---

## Taxonomy Coverage

Every bullet from all 6 alignment engineering categories is implemented:

### Category 1 — Data Collection & Annotation
| Bullet | Module |
|--------|--------|
| Classification, NER, Sentiment, Intent annotation | `annotation/` Tab 1 |
| Pairwise A/B preference annotation | `annotation/` Tab 2 |
| Instruction quality scoring (1–5) | `annotation/` Tab 3 |
| Factuality annotation | `annotation/` Tab 4 |
| Toxicity/bias labeling | `annotation/` Tab 5 |
| Inter-annotator agreement (Cohen's κ + Krippendorff's α) | `annotation/` IAA module |

### Category 2 — Data Curation & Augmentation
| Bullet | Module |
|--------|--------|
| High-quality SFT pair curation | `sft/dataset_curator.py` |
| Persona-conditioned generation (6 personas) | `sft/persona_generator.py` |
| Contrastive pair generation (3 methods) | `data/contrastive_pairs.py` |
| Dataset deduplication and length filtering | `data/dataset_utils.py` |
| Constitutional AI revisions as preference pairs | `data/contrastive_pairs.py` + `constitutional/` |

### Category 3 — Supervised Fine-Tuning
| Bullet | Module |
|--------|--------|
| SFT training with HuggingFace TRL | `sft/sft_trainer.py` |
| LoRA parameter-efficient fine-tuning | `sft/sft_trainer.py` (via PEFT) |
| Chat template formatting | `data/dataset_utils.py` |
| Training loss logging and plotting | `sft/sft_trainer.py` (LossLoggerCallback) |

### Category 4 — Reward Modeling
| Bullet | Module |
|--------|--------|
| Reward model architecture (DistilBERT → scalar) | `reward/model.py` |
| Bradley-Terry pairwise preference loss | `reward/train.py` |
| Training loop with val accuracy and reward gap | `reward/train.py` |
| Reward distribution calibration plot | `reward/train.py` |
| Evaluation metrics (accuracy, mean gap) | `reward/train.py` |

### Category 5 — Preference Learning Algorithms
| Bullet | Module |
|--------|--------|
| DPO — Direct Preference Optimization | `training/dpo_trainer.py` |
| PPO — Proximal Policy Optimization | `training/ppo_trainer.py` |
| GRPO — Group Relative Policy Optimization | `training/grpo_trainer.py` |
| ORPO — Odds Ratio Preference Optimization | `training/orpo_trainer.py` |
| SimPO — Simple Preference Optimization | `training/simpo_trainer.py` |
| Constitutional AI / RLAIF pipeline | `constitutional/rlaif.py` |

### Category 6 — Evaluation & Reproducibility
| Bullet | Module |
|--------|--------|
| Prompt versioning (SHA-256 hashing) | `tracking/experiment_tracker.py` |
| Experiment tracking with SQLite | `tracking/experiment_tracker.py` |
| Run comparison (config diff + metric diff) | `tracking/experiment_tracker.py` |
| Reproducible config snapshots (YAML) | `configs/` |
| Run export as JSON | `tracking/experiment_tracker.py` |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/notnick2/minialign-rlhf-pipeline.git
cd minialign-rlhf-pipeline
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### 2. Set API key

MiniAlign uses the Anthropic API for persona generation, constitutional AI, and synthetic data degradation.

```bash
# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Or create a .env file
echo ANTHROPIC_API_KEY=sk-ant-... > .env
```

### 3. Verify installation

```python
from sft.persona_generator import PERSONAS
from tracking.experiment_tracker import ExperimentTracker
from reward.model import RewardModel
print("MiniAlign ready.", list(PERSONAS.keys()))
```

---

## Module Descriptions

### `annotation/` — Human Annotation Interface
A Gradio web app with 5 annotation tabs and inter-annotator agreement (IAA) computation. Annotations are persisted in SQLite and can be exported as contrastive pairs for DPO training.

**Key design decision:** SQLite storage (not CSV) enables multi-annotator workflows and atomic writes without file locking conflicts.

### `sft/` — Supervised Fine-Tuning
- `dataset_curator.py` — Filters, deduplicates, and quality-scores raw data before SFT.
- `persona_generator.py` — Generates 6 stylistically distinct responses per instruction (domain expert, curious student, layperson, skeptic, non-native speaker, child) using `claude-haiku-4-5-20251001` for cost-effective large-scale generation.
- `sft_trainer.py` — TRL SFTTrainer with LoRA adapters. Uses cosine LR schedule with linear warmup. Saves loss curve as both JSON and PNG.

**Key design decision:** LoRA over full fine-tuning — 3–10× fewer trainable parameters, enabling training on a single consumer GPU with models up to 7B parameters.

### `data/` — Dataset Utilities and Contrastive Pair Generation
- `dataset_utils.py` — Core I/O primitives (load/save JSONL), train/val splitting, chat formatting, deduplication, and length filtering.
- `contrastive_pairs.py` — Three complementary methods for building `{prompt, chosen, rejected}` datasets:
  1. **Degradation**: Claude synthetically worsens good responses (fast, scalable, no human labels needed).
  2. **Annotations**: Reads pairwise preferences directly from the annotation SQLite DB.
  3. **RLAIF**: Constitutional AI revised responses become chosen; originals become rejected.

**Key design decision:** Three complementary sources ensure diverse preference signals and avoid over-reliance on any single annotation modality.

### `constitutional/` — Constitutional AI / RLAIF
Implements the Anthropic Constitutional AI pipeline: given a response, Claude critiques it against a set of constitutional principles, then revises it. The original/revised pair forms a preference training example.

### `reward/` — Reward Modeling
- `model.py` — `RewardModel`: DistilBERT encoder + single linear head → scalar reward. Includes `save_pretrained()` / `load_pretrained()` for reproducible checkpointing.
- `train.py` — Full training loop with Bradley-Terry loss, validation accuracy, reward gap logging, best-checkpoint saving, and a reward distribution calibration plot.

**Key design decision:** DistilBERT (66M params) over larger encoders for fast iteration during development. The reward model is swappable — any HuggingFace encoder works by changing `backbone` in the config.

**Why Bradley-Terry?** It's the theoretically grounded model for pairwise comparison data, maximizing the probability that the chosen response beats the rejected under a logistic model. It is the same loss used in InstructGPT and most RLHF pipelines.

### `training/` — Preference Learning Algorithms

#### DPO (`dpo_trainer.py`)
**Why DPO over PPO?** DPO eliminates the reward model and RL training loop entirely. It can be shown that RLHF with a Bradley-Terry reward model has a closed-form optimal policy expressible as a log-ratio of the policy to the reference. DPO directly optimizes this, resulting in a simple supervised objective on preference pairs. It is significantly more stable and compute-efficient than PPO for most alignment tasks.

#### PPO (`ppo_trainer.py`)
**When to use PPO?** PPO is preferred when the reward signal is non-differentiable (e.g., a code execution outcome, a factuality verifier, or an external tool call). It remains the gold standard for complex RL-from-feedback scenarios. Uses TRL's `PPOTrainer` with an adaptive KL controller to maintain proximity to the reference model.

#### GRPO (`grpo_trainer.py`)
**Key innovation:** Replaces PPO's value function with group-relative advantage normalization. For each prompt, G=4 responses are sampled; advantages are (reward - group_mean) / group_std. This eliminates the separate critic model while retaining PPO-style clipping. Introduced in DeepSeek-R1 for reasoning tasks.

#### ORPO (`orpo_trainer.py`)
**Key innovation:** Unifies SFT and preference alignment in one objective: `L = L_SFT + λ * L_OR`. No reference model or reward model needed. The odds-ratio term penalizes the model for assigning high probability to rejected responses while simultaneously maximizing chosen response probability. Reduces memory and compute requirements by ~50% compared to DPO (no reference model inference).

#### SimPO (`simpo_trainer.py`)
**Key innovation:** Uses length-normalized log-probability as an implicit reward: `score = (1/|y|) * log π(y|x) - γ`. The length normalization prevents the model from gaming the reward with shorter responses. The target margin γ provides a minimum quality gap. Like ORPO, no reference model is needed, making it especially lightweight.

### `tracking/` — Experiment Tracking
SQLite-backed tracker with four tables: `runs`, `metrics`, `checkpoints`, `prompts`.

**Key features:**
- Prompt templates are SHA-256 hashed for version tracking — you can see which runs used which prompt versions.
- Config diff between any two runs for ablation analysis.
- WAL journal mode for concurrent access safety.
- Full run export to JSON for sharing and reproducibility.

---

## Usage Examples

### Running the Annotation Interface

```bash
# Start the Gradio annotation app
python -m annotation.app

# App will open at http://localhost:7860
# Navigate tabs for different annotation tasks
```

### Generating Persona-Conditioned SFT Data

```bash
# Generate responses for all 6 personas
python -m sft.persona_generator \
  --instructions data/raw_instructions.jsonl \
  --output data/persona_sft.jsonl \
  --api-key $ANTHROPIC_API_KEY

# Generate only for specific personas
python -m sft.persona_generator \
  --instructions data/raw_instructions.jsonl \
  --output data/expert_layperson.jsonl \
  --personas domain_expert layperson
```

### Training with SFT

```bash
python -m sft.sft_trainer --config configs/sft_config.yaml
# Checkpoint saved to: checkpoints/sft/final/
# Loss curve: checkpoints/sft/loss_curve.png
```

### Building a Contrastive Pair Dataset

```python
from data.contrastive_pairs import (
    pairs_from_degradation, pairs_from_annotations,
    pairs_from_rlaif, combine_all_sources
)

# Combine all three sources
pairs = combine_all_sources(
    degradation_path="data/degradation_pairs.jsonl",
    annotations_db="annotations.db",
    rlaif_path="constitutional/rlaif_results.jsonl",
    output_path="data/dpo_dataset.jsonl",
)
print(f"Combined {len(pairs)} preference pairs")
```

### Training with Each Algorithm

```bash
# DPO (recommended first choice — no RM needed, stable training)
python -m training.dpo_trainer --config configs/dpo_config.yaml

# PPO (RL with reward model — use when reward is non-differentiable)
python -m training.ppo_trainer --config configs/ppo_config.yaml

# GRPO (group-relative advantages — good for reasoning tasks)
python -m training.grpo_trainer --config configs/grpo_config.yaml

# ORPO (no reference model — unified SFT + preference)
python -m training.orpo_trainer --config configs/orpo_config.yaml

# SimPO (length-normalized, no reference model)
python -m training.simpo_trainer --config configs/simpo_config.yaml
```

### Training the Reward Model

```bash
python -m reward.train --config configs/reward_model_config.yaml
# Best checkpoint: checkpoints/reward_model/best/
# Distribution plot: checkpoints/reward_model/reward_distribution.png
```

### Evaluating the Reward Model

```python
from reward.train import evaluate_reward_model
from data.dataset_utils import load_jsonl

eval_data = load_jsonl("data/eval_pairs.jsonl")
metrics = evaluate_reward_model(
    model_path="checkpoints/reward_model/best",
    eval_data=eval_data,
)
# {"accuracy": 0.84, "mean_reward_gap": 0.42, ...}
```

### Experiment Tracking

```python
from tracking.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("experiments.db")

# Start a run
run_id = tracker.start_run(
    config={
        "algorithm": "dpo",
        "model_name": "microsoft/phi-2",
        "learning_rate": 5e-5,
        "beta": 0.1,
        "num_epochs": 2,
    },
    prompt_template="### Instruction:\n{instruction}\n\n### Response:",
    notes="Baseline DPO run with phi-2",
)

# Log metrics during training
for step in range(100):
    tracker.log_metrics(run_id, step, {
        "loss": 0.8 - step * 0.005,
        "reward_margin": step * 0.003,
    })

# Log checkpoint
tracker.log_checkpoint(run_id, "checkpoints/dpo/final", eval_score=0.72)

# Compare two runs
comparison = tracker.compare_runs(run_id_a, run_id_b)
print("Config differences:", comparison["config_diff"])
print("Metric comparison:", comparison["metrics_comparison"])

# List all runs
runs = tracker.list_runs()
for run in runs:
    print(f"{run['algorithm']:10s} | {run['timestamp'][:19]} | steps={run['n_steps_logged']}")

# Export run as JSON
tracker.export_run(run_id, f"exports/{run_id}.json")
```

---

## Configuration

All training runs are configured via YAML files in `configs/`. Key fields:

| Config file | Algorithm | Key params |
|-------------|-----------|------------|
| `sft_config.yaml` | SFT | model_name, lora_r, num_epochs, max_length |
| `dpo_config.yaml` | DPO | beta (KL coef), sft_checkpoint |
| `ppo_config.yaml` | PPO | kl_coef, num_steps, reward_model_path |
| `grpo_config.yaml` | GRPO | group_size (G), beta, epsilon |
| `orpo_config.yaml` | ORPO | lambda_ (OR loss weight) |
| `simpo_config.yaml` | SimPO | beta, gamma (target margin) |
| `reward_model_config.yaml` | Reward Model | backbone, val_ratio |

---

## Project Structure

```
MiniAlign/
├── annotation/             # Gradio annotation interface
│   ├── app.py              #   5-tab annotation UI
│   └── storage/            #   SQLite annotation store + IAA
│       └── annotation_store.py
├── configs/                # YAML training configs
│   ├── sft_config.yaml
│   ├── dpo_config.yaml
│   ├── ppo_config.yaml
│   ├── grpo_config.yaml
│   ├── orpo_config.yaml
│   ├── simpo_config.yaml
│   └── reward_model_config.yaml
├── constitutional/         # Constitutional AI / RLAIF
│   ├── principles.py       #   Constitutional principles definitions
│   └── rlaif.py            #   Critique-revise pipeline
├── data/                   # Dataset utilities
│   ├── __init__.py
│   ├── dataset_utils.py    #   I/O, split, dedup, filter
│   └── contrastive_pairs.py#   3-source preference pair generation
├── reward/                 # Reward model
│   ├── __init__.py
│   ├── model.py            #   DistilBERT → scalar reward
│   └── train.py            #   Bradley-Terry training loop
├── sft/                    # Supervised fine-tuning
│   ├── __init__.py
│   ├── dataset_curator.py  #   Data quality curation
│   ├── persona_generator.py#   6-persona response generation
│   └── sft_trainer.py      #   TRL SFTTrainer + LoRA
├── tracking/               # Experiment tracking
│   ├── __init__.py
│   └── experiment_tracker.py # SQLite tracker + run comparison
├── training/               # Preference learning trainers
│   ├── __init__.py
│   ├── dpo_trainer.py      #   DPO (Rafailov et al.)
│   ├── ppo_trainer.py      #   PPO (Schulman et al.) via TRL
│   ├── grpo_trainer.py     #   GRPO (DeepSeek-R1)
│   ├── orpo_trainer.py     #   ORPO (Hong et al.)
│   └── simpo_trainer.py    #   SimPO (Meng et al.)
├── requirements.txt
├── setup.py
└── README.md
```

---

## References

- **DPO**: Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). NeurIPS 2023.

- **PPO**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

- **RLHF with PPO**: Ziegler, D. M., et al. (2019). [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593).

- **InstructGPT**: Ouyang, L., et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.

- **GRPO / DeepSeek-R1**: DeepSeek-AI (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948).

- **ORPO**: Hong, J., Lee, N., & Thorne, J. (2024). [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691).

- **SimPO**: Meng, Y., Xia, M., & Chen, D. (2024). [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734).

- **Constitutional AI**: Bai, Y., et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). Anthropic.

- **WizardLM / Evol-Instruct**: Xu, C., et al. (2023). [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244).

- **LoRA**: Hu, E., et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.

---

## License

MIT License. See LICENSE for details.
