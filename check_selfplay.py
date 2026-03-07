import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np


def check_selfplay_dataset(path: str = "selfplay_states.json") -> None:
    """
    Verify the quality of the self-play dataset.

    Prints:
      - Total training examples
      - Unique games
      - Reward distribution: how many +2, +1, 0, -1, -2 (and any others)
      - Average phases per game
      - Average SC count across all states
      - 2 random example state_text samples
      - Confirms no empty state_text fields
      - Flags any data quality issues clearly
    """
    print(f"Loading self-play states from {path}...")
    with open(path, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total_examples = len(data)
    print(f"Total training examples: {total_examples}")

    if not data:
        print("Dataset is empty. No further checks possible.")
        return

    # Unique games.
    game_ids = {ex.get("game_id") for ex in data if ex.get("game_id") is not None}
    print(f"Unique games: {len(game_ids)}")

    # Reward distribution for key buckets.
    reward_counts: Counter = Counter()
    other_rewards: Counter = Counter()
    for ex in data:
        r = ex.get("reward")
        try:
            rf = float(r)
        except (TypeError, ValueError):
            other_rewards[str(r)] += 1
            continue
        if rf == 2.0:
            reward_counts["+2"] += 1
        elif rf == 1.0:
            reward_counts["+1"] += 1
        elif rf == 0.0:
            reward_counts["0"] += 1
        elif rf == -1.0:
            reward_counts["-1"] += 1
        elif rf == -2.0:
            reward_counts["-2"] += 1
        else:
            other_rewards[f"{rf}"] += 1

    print("Reward distribution (key buckets):")
    for bucket in ["+2", "+1", "0", "-1", "-2"]:
        print(f"  {bucket}: {reward_counts.get(bucket, 0)}")
    if other_rewards:
        print("Other reward values observed:")
        for val, cnt in other_rewards.items():
            print(f"  {val}: {cnt}")

    # Average phases per game: use distinct phase labels per game.
    phases_per_game: Dict[str, set] = defaultdict(set)
    for ex in data:
        gid = ex.get("game_id")
        phase = ex.get("phase")
        if gid is None or phase is None:
            continue
        phases_per_game[gid].add(phase)

    if phases_per_game:
        avg_phases = np.mean([len(phases) for phases in phases_per_game.values()])
    else:
        avg_phases = 0.0
    print(f"Average phases per game (approx): {avg_phases:.2f}")

    # Average SC count across all states.
    sc_counts = [ex.get("sc_count", 0) for ex in data]
    try:
        avg_sc = float(np.mean(sc_counts)) if sc_counts else 0.0
    except TypeError:
        avg_sc = 0.0
    print(f"Average SC count across all examples: {avg_sc:.2f}")

    # Check for empty state_text.
    empty_text_examples = [i for i, ex in enumerate(data) if not str(ex.get("state_text", "")).strip()]
    if empty_text_examples:
        print(f"[WARNING] Found {len(empty_text_examples)} examples with empty state_text.")
    else:
        print("No empty state_text fields found.")

    # Show 2 random example state_text samples.
    print("\nRandom state_text samples:")
    indices = list(range(total_examples))
    random.shuffle(indices)
    sample_indices = indices[:2]
    for idx in sample_indices:
        ex = data[idx]
        print(f"\n--- Example index {idx} ---")
        print(f"Game: {ex.get('game_id')} | Phase: {ex.get('phase')} | Power: {ex.get('power')}")
        print(ex.get("state_text", ""))

    # Additional quality checks.
    issues: List[str] = []

    # Check that every example has a power and phase.
    missing_power = sum(1 for ex in data if not ex.get("power"))
    missing_phase = sum(1 for ex in data if not ex.get("phase"))
    if missing_power:
        issues.append(f"{missing_power} examples missing 'power'.")
    if missing_phase:
        issues.append(f"{missing_phase} examples missing 'phase'.")

    # Check that sc_delta is present and numeric.
    non_numeric_sc_delta = sum(
        1 for ex in data if not isinstance(ex.get("sc_delta", 0), (int, float))
    )
    if non_numeric_sc_delta:
        issues.append(f"{non_numeric_sc_delta} examples have non-numeric sc_delta.")

    if issues:
        print("\n[DATA QUALITY ISSUES]")
        for msg in issues:
            print(f"- {msg}")
    else:
        print("\nNo major data quality issues detected.")


def main() -> None:
    check_selfplay_dataset()


if __name__ == "__main__":
    main()

