"""
train_phase2.py — Phase 2 GRPO training on HumanImitationEnv.

Continues from Phase 1 checkpoint (grpo_output/checkpoint-200).
Does NOT reinitialize weights — curriculum learning requires this.
Agent learns to mirror real human Diplomacy decisions.
Target: 200 steps. Save to training/checkpoints/phase2_final/
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from envs.human_imitation_env import HumanImitationEnv

# ── paths ──────────────────────────────────────────────────────
PHASE1_CHECKPOINT = "grpo_output/checkpoint-200"
PHASE2_OUTPUT     = "training/checkpoints/phase2_final"
DATA_PATH         = "training/data/selfplay_states.json"
CURVE_PATH        = "training/phase2_reward_curve.png"

# ── load Phase 1 model ─────────────────────────────────────────
print("Loading Phase 1 checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(PHASE1_CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token

# ── environment ────────────────────────────────────────────────
env = HumanImitationEnv(data_path=DATA_PATH)

# ── reward function ────────────────────────────────────────────
def compute_reward(completions, prompts=None, **kwargs):
    """
    Reward function for GRPO.
    Scores agent output against human game outcome.
    """
    rewards = []
    for completion in completions:
        text = completion.lower() if isinstance(completion, str) else ""
        reward = 0.0
        # Coalition tactics
        if any(w in text for w in ["ally", "alliance", "coalition", "support"]):
            reward += 0.3
        # Aggressive expansion
        if any(w in text for w in ["attack", "advance", "take", "capture"]):
            reward += 0.2
        # Defensive awareness
        if any(w in text for w in ["defend", "protect", "hold", "guard"]):
            reward += 0.2
        # Strategic reasoning
        if any(w in text for w in ["because", "therefore", "since", "strategic"]):
            reward += 0.2
        # Bluff detection language
        if any(w in text for w in ["bluff", "pressure", "leverage", "signal"]):
            reward += 0.1
        rewards.append(reward)
    return rewards

# ── GRPO config ────────────────────────────────────────────────
config = GRPOConfig(
    output_dir=PHASE2_OUTPUT,
    num_train_epochs=1,
    max_steps=200,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    max_completion_length=100,
    num_generations=4,
)

# ── build training dataset from env ───────────────────────────
print("Building training dataset from human game states...")
with open(DATA_PATH) as f:
    states = json.load(f)

# Sample 2000 states for Phase 2 training
sample = np.random.choice(states, size=min(2000, len(states)), replace=False)
dataset_dicts = [{"prompt": s["state_text"]} for s in sample]

from datasets import Dataset
dataset = Dataset.from_list(dataset_dicts)

# ── trainer ────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=PHASE1_CHECKPOINT,
    args=config,
    reward_funcs=compute_reward,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ── train ──────────────────────────────────────────────────────
print("Starting Phase 2 training...")
print(f"Loading from: {PHASE1_CHECKPOINT}")
print(f"Saving to:    {PHASE2_OUTPUT}")
print("=" * 50)

reward_log = []

class RewardCallback:
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            reward_log.append(logs["reward"])
            print(f"Step {state.global_step} | Reward: {logs['reward']:.4f}")

trainer.train()

# ── save ───────────────────────────────────────────────────────
trainer.save_model(PHASE2_OUTPUT)
tokenizer.save_pretrained(PHASE2_OUTPUT)
print(f"Phase 2 model saved to {PHASE2_OUTPUT}")

# ── plot reward curve ──────────────────────────────────────────
if reward_log:
    plt.figure(figsize=(12, 5))
    plt.plot(reward_log, alpha=0.4, color='blue', label='Step Reward')
    window = min(20, len(reward_log))
    moving_avg = np.convolve(reward_log, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(reward_log)), moving_avg, 
             color='orange', linewidth=2, label='Moving Average')
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("ArbitrAgent Phase 2 GRPO Reward Curve\nHuman Imitation Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    print(f"Reward curve saved to {CURVE_PATH}")

print("Phase 2 training complete.")

"""
train_phase2.py — Phase 2 GRPO training on HumanImitationEnv.

Continues from Phase 1 checkpoint (grpo_output/checkpoint-200).
Does NOT reinitialize weights — curriculum learning requires this.
Agent learns to mirror real human Diplomacy decisions.
Target: 200 steps. Save to training/checkpoints/phase2_final/
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from envs.human_imitation_env import HumanImitationEnv

# ── paths ──────────────────────────────────────────────────────
PHASE1_CHECKPOINT = "grpo_output/checkpoint-200"
PHASE2_OUTPUT     = "training/checkpoints/phase2_final"
DATA_PATH         = "training/data/selfplay_states.json"
CURVE_PATH        = "training/phase2_reward_curve.png"

# ── load Phase 1 model ─────────────────────────────────────────
print("Loading Phase 1 checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(PHASE1_CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token

# ── environment ────────────────────────────────────────────────
env = HumanImitationEnv(data_path=DATA_PATH)

# ── reward function ────────────────────────────────────────────
def compute_reward(completions, prompts=None, **kwargs):
    """
    Reward function for GRPO.
    Scores agent output against human game outcome.
    """
    rewards = []
    for completion in completions:
        text = completion.lower() if isinstance(completion, str) else ""
        reward = 0.0
        # Coalition tactics
        if any(w in text for w in ["ally", "alliance", "coalition", "support"]):
            reward += 0.3
        # Aggressive expansion
        if any(w in text for w in ["attack", "advance", "take", "capture"]):
            reward += 0.2
        # Defensive awareness
        if any(w in text for w in ["defend", "protect", "hold", "guard"]):
            reward += 0.2
        # Strategic reasoning
        if any(w in text for w in ["because", "therefore", "since", "strategic"]):
            reward += 0.2
        # Bluff detection language
        if any(w in text for w in ["bluff", "pressure", "leverage", "signal"]):
            reward += 0.1
        rewards.append(reward)
    return rewards

# ── GRPO config ────────────────────────────────────────────────
config = GRPOConfig(
    output_dir=PHASE2_OUTPUT,
    num_train_epochs=1,
    max_steps=200,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    max_completion_length=100,
    num_generations=4,
)

# ── build training dataset from env ───────────────────────────
print("Building training dataset from human game states...")
with open(DATA_PATH) as f:
    states = json.load(f)

# Sample 2000 states for Phase 2 training
sample = np.random.choice(states, size=min(2000, len(states)), replace=False)
dataset_dicts = [{"prompt": s["state_text"]} for s in sample]

from datasets import Dataset
dataset = Dataset.from_list(dataset_dicts)

# ── trainer ────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=PHASE1_CHECKPOINT,
    args=config,
    reward_funcs=compute_reward,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ── train ──────────────────────────────────────────────────────
print("Starting Phase 2 training...")
print(f"Loading from: {PHASE1_CHECKPOINT}")
print(f"Saving to:    {PHASE2_OUTPUT}")
print("=" * 50)

reward_log = []

class RewardCallback:
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            reward_log.append(logs["reward"])
            print(f"Step {state.global_step} | Reward: {logs['reward']:.4f}")

trainer.train()

# ── save ───────────────────────────────────────────────────────
trainer.save_model(PHASE2_OUTPUT)
tokenizer.save_pretrained(PHASE2_OUTPUT)
print(f"Phase 2 model saved to {PHASE2_OUTPUT}")

# ── plot reward curve ──────────────────────────────────────────
if reward_log:
    plt.figure(figsize=(12, 5))
    plt.plot(reward_log, alpha=0.4, color="blue", label="Step Reward")
    window = min(20, len(reward_log))
    moving_avg = np.convolve(reward_log, np.ones(window) / window, mode="valid")
    plt.plot(
        range(window - 1, len(reward_log)),
        moving_avg,
        color="orange",
        linewidth=2,
        label="Moving Average",
    )
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("ArbitrAgent Phase 2 GRPO Reward Curve\nHuman Imitation Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    print(f"Reward curve saved to {CURVE_PATH}")

print("Phase 2 training complete.")

