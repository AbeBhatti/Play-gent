"""
train_unified.py — GRPO on ArbitrAgentEnv (unified 3-signal reward).

Loads from training/checkpoints/phase2_final. Does NOT reinitialize weights.
Environment: ArbitrAgentEnv. 200 steps, lr 5e-6, batch size 2.
Logs accuracy / outcome / bluff separately; saves to unified_final and plot.
"""

import os
import sys
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from envs.arbitragent_env import ArbitrAgentEnv, _extract_human_orders

# ── paths ──────────────────────────────────────────────────────
PHASE2_CHECKPOINT = "training/checkpoints/phase2_final"
UNIFIED_OUTPUT = "training/checkpoints/unified_final"
DATA_PATH = "training/data/selfplay_states.json"
CURVE_PATH = "training/unified_reward_curve.png"

# ── load Phase 2 model ─────────────────────────────────────────
print("Loading Phase 2 checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(PHASE2_CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token

# ── encoder for reward (shared with env logic) ──────────────────
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── reward logs for plotting and trainer_state ───────────────────
accuracy_log = []
outcome_log = []
bluff_log = []
total_log = []


def _outcome_score(text: str) -> float:
    """Same keyword logic as ArbitrAgentEnv._outcome_reward."""
    reward = 0.0
    if any(w in text for w in ["ally", "alliance", "coalition", "support", "another buyer", "trade offer from another"]):
        reward += 0.4
    if any(w in text for w in ["pressure", "leverage", "can you do", "less urgent", "make the numbers work"]):
        reward += 0.3
    if any(w in text for w in ["deal", "agree", "accept", "close"]):
        reward += 0.2
    if any(w in text for w in ["ok $30", "accept 30", "take it at 30", "deal at 30"]):
        reward -= 0.6
    if any(w in text for w in ["final offer", "lowest you can go", "that's your final"]):
        reward -= 0.3
    return float(np.clip(reward, -1.0, 1.0))


def _bluff_score_from_action(action_lower: str) -> float:
    """Bluff reward: synthetic message is a bluff; reward pressure language."""
    from envs.arbitragent_env import SYNTHETIC_BLUFF_MESSAGE, SYNTHETIC_BLUFF_PROFILE, SYNTHETIC_THREAD
    from agent.bluff_detector import analyze_bluff
    signals = analyze_bluff(SYNTHETIC_BLUFF_PROFILE, SYNTHETIC_THREAD, SYNTHETIC_BLUFF_MESSAGE, turn=2)
    formulaic_present = signals.formulaic_tell > 0
    pressure_in_action = any(
        w in action_lower for w in ["another seller", "trade offer from another", "less urgent", "can you do"]
    )
    if signals.is_bluff and pressure_in_action:
        return float(signals.bluff_score)
    if formulaic_present and not pressure_in_action:
        return -0.5
    return 0.0


def compute_reward(completions, prompts=None, **kwargs):
    """
    Unified reward: 0.35*accuracy + 0.35*outcome + 0.30*bluff.
    Logs each component to global lists for plotting and trainer_state.
    """
    if prompts is None:
        prompts = [""] * len(completions)
    rewards = []
    for completion, prompt in zip(completions, prompts):
        if isinstance(completion, list) and completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            action = completion[-1]["content"].strip()
        elif isinstance(completion, str):
            action = completion.strip()
        else:
            action = ""
        action_lower = action.lower()

        # Accuracy: cosine sim with human action from state_text (prompt)
        human_action_text = _extract_human_orders(prompt if isinstance(prompt, str) else "")
        action_emb = encoder.encode(action or " ", convert_to_numpy=True)
        human_emb = encoder.encode(human_action_text, convert_to_numpy=True)
        dot = float(np.dot(action_emb, human_emb))
        norm_a = float(np.linalg.norm(action_emb)) or 1e-8
        norm_h = float(np.linalg.norm(human_emb)) or 1e-8
        accuracy = float(np.clip(dot / (norm_a * norm_h), -1.0, 1.0))

        outcome = _outcome_score(action_lower)
        bluff = _bluff_score_from_action(action_lower)
        total = 0.35 * accuracy + 0.35 * outcome + 0.30 * bluff

        accuracy_log.append(accuracy)
        outcome_log.append(outcome)
        bluff_log.append(bluff)
        total_log.append(total)
        rewards.append(total)
    return rewards


# ── GRPO config ─────────────────────────────────────────────────
config = GRPOConfig(
    output_dir=UNIFIED_OUTPUT,
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
    bf16=False,
)

# ── build dataset from ArbitrAgentEnv data ──────────────────────
print("Building training dataset from selfplay states...")
with open(DATA_PATH) as f:
    states = json.load(f)
sample = list(np.random.choice(states, size=min(2000, len(states)), replace=False))
dataset_dicts = [{"prompt": s["state_text"]} for s in sample]
dataset = Dataset.from_list(dataset_dicts)

# ── trainer ─────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=PHASE2_CHECKPOINT,
    args=config,
    reward_funcs=compute_reward,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ── train ──────────────────────────────────────────────────────
print("Starting unified training...")
print(f"Loading from: {PHASE2_CHECKPOINT}")
print(f"Saving to:    {UNIFIED_OUTPUT}")
print("=" * 50)

trainer.train()

# ── save ────────────────────────────────────────────────────────
trainer.save_model(UNIFIED_OUTPUT)
tokenizer.save_pretrained(UNIFIED_OUTPUT)
print(f"Unified model saved to {UNIFIED_OUTPUT}")

# ── write reward components for logging ─────────────────────────
step_size = max(1, len(total_log) // 200)
acc_agg = [np.mean(accuracy_log[i:i + step_size]) for i in range(0, len(accuracy_log), step_size)][:200]
out_agg = [np.mean(outcome_log[i:i + step_size]) for i in range(0, len(outcome_log), step_size)][:200]
bluff_agg = [np.mean(bluff_log[i:i + step_size]) for i in range(0, len(bluff_log), step_size)][:200]
total_agg = [np.mean(total_log[i:i + step_size]) for i in range(0, len(total_log), step_size)][:200]
reward_log_path = os.path.join(UNIFIED_OUTPUT, "unified_reward_log.json")
with open(reward_log_path, "w") as f:
    json.dump({"accuracy": acc_agg, "outcome": out_agg, "bluff": bluff_agg, "total": total_agg}, f, indent=2)

# ── plot reward curve (three lines) ─────────────────────────────
x = range(1, len(total_agg) + 1)
plt.figure(figsize=(12, 5))
plt.plot(x, acc_agg, alpha=0.8, label="accuracy", color="C0")
plt.plot(x, out_agg, alpha=0.8, label="outcome", color="C1")
plt.plot(x, bluff_agg, alpha=0.8, label="bluff", color="C2")
plt.plot(x, total_agg, alpha=0.9, label="total", color="black", linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.title("ArbitrAgent Unified GRPO — Accuracy / Outcome / Bluff")
plt.legend()
plt.tight_layout()
plt.savefig(CURVE_PATH)
print(f"Reward curve saved to {CURVE_PATH}")

# ── print final reward values ───────────────────────────────────
n = min(20, len(accuracy_log))
print("\nFinal reward values (last {} steps):".format(n))
print("  accuracy: {:.4f}".format(np.mean(accuracy_log[-n:])))
print("  outcome:  {:.4f}".format(np.mean(outcome_log[-n:])))
print("  bluff:    {:.4f}".format(np.mean(bluff_log[-n:])))
print("  total:    {:.4f}".format(np.mean(total_log[-n:])))

print("\nUnified training complete.")
