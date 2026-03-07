import argparse
import json
import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_model import DiplomacyRewardModel, score_state


def build_prompt(state_text: str, power_name: str) -> str:
    """Construct the text prompt given the current state description."""
    return (
        "You are a Diplomacy negotiation agent. Your goal is to maximize long-term strategic advantage "
        f"for {power_name} by capturing and holding supply centers while avoiding elimination.\n\n"
        "Current game state:\n\n"
        f"{state_text}\n\n"
        f"What is your next strategic move as {power_name}? "
        "Respond with a concise description of your intent and planned orders."
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training loop for Diplomacy negotiation using TRL.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for GRPO (default: 1e-5).",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="reward_model.pt",
        help="Path to trained reward model weights (default: reward_model.pt).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Approximate number of GRPO optimization steps (default: 100).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="selfplay_states.json",
        help="Path to self-play states JSON file (default: selfplay_states.json).",
    )
    args = parser.parse_args()

    # Device / GPU info
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Policy model identifier (1B chat model, ungated).
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading policy model: {model_name}")

    # Load trained reward model (DistilBERT-based scalar scorer).
    reward_model_path = args.reward_model_path.strip()
    print("Loading reward model from:", reward_model_path)
    reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    reward_model = DiplomacyRewardModel().to(device)
    reward_model.load_state_dict(
        torch.load(reward_model_path, map_location=device)
    )
    reward_model.eval()

    # Load self-play dataset and build GRPO prompts.
    print(f"Loading self-play dataset from: {args.dataset_path}")
    with open(args.dataset_path, "r") as f:
        raw_states: List[Dict[str, Any]] = json.load(f)

    prompts: List[str] = []
    base_rewards: List[float] = []
    for ex in raw_states:
        state_text = ex.get("state_text", "")
        power = ex.get("power", "ENGLAND")
        r = float(ex.get("reward", 0.0))
        prompts.append(build_prompt(state_text, power))
        base_rewards.append(r)

    dataset = Dataset.from_dict({"prompt": prompts, "reward": base_rewards})
    print(f"Dataset size: {len(dataset)} examples")

    # Custom reward function for GRPO: reward model + repetition & length penalties.
    def reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """
        Score each completion with reward model, then apply repetition and length penalties.
        """
        scores: List[float] = []
        for completion in completions:
            text = completion.strip()
            words = text.split()

            # Reward model score
            inputs = reward_tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                hidden = reward_model.encoder(**inputs).last_hidden_state[:, 0, :]
                rm_score = reward_model.head(hidden).squeeze().item()

            # Repetition penalty — unique words ratio
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
            else:
                unique_ratio = 0.0

            if unique_ratio < 0.3:
                repetition_penalty = -2.0
            elif unique_ratio < 0.5:
                repetition_penalty = -0.5
            else:
                repetition_penalty = 0.0

            # Penalty for very short completions
            length_penalty = -0.5 if len(words) < 5 else 0.0

            combined = float(rm_score) + repetition_penalty + length_penalty
            scores.append(torch.tensor(combined, dtype=torch.float32))
        return scores

    # GRPO configuration: small batch, multiple generations per prompt.
    grpo_config = GRPOConfig(
        output_dir="grpo_output",
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        num_generations=4,
        max_completion_length=64,
        max_steps=args.episodes,
        logging_steps=1,
        save_steps=100,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Extract reward metrics: prefer internal _metrics, fallback to log_history.
    try:
        rewards = trainer._metrics["train"]["reward"]
        print(f"Total reward datapoints from _metrics: {len(rewards)}")
    except (KeyError, AttributeError):
        rewards = []

    if not rewards:
        rewards = []
        for entry in trainer.state.log_history:
            for key in ["reward", "rewards/reward_fn/mean"]:
                if key in entry:
                    rewards.append(entry[key])
                    break

    print(f"Total reward datapoints: {len(rewards)}")

    if rewards:
        plt.figure(figsize=(12, 5))
        plt.plot(rewards, alpha=0.6, label="Step Reward")
        if len(rewards) >= 10:
            window = max(len(rewards) // 10, 1)
            moving_avg = [
                sum(rewards[max(0, i - window) : i + 1]) / min(i + 1, window)
                for i in range(len(rewards))
            ]
            plt.plot(moving_avg, linewidth=2, label="Moving Average")
        plt.xlabel("Training Step")
        plt.ylabel("Reward")
        plt.title("Play-gent GRPO Reward Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("ppo_reward_curve.png")
        print(
            f"Curve saved. Final: {rewards[-1]:.3f}, "
            f"Best: {max(rewards):.3f}"
        )


if __name__ == "__main__":
    main()

