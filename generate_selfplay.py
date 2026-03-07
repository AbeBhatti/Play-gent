import argparse
import json
import random
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format  # noqa: F401 (imported for completeness, not used yet)
from tqdm import tqdm


POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def _format_units(units_list: List[str]) -> List[str]:
    """Convert engine-style unit strings into human-readable strings."""
    formatted: List[str] = []
    for unit in units_list:
        # Units may be like 'A PAR', 'F IRI', or prefixed with '*' for dislodged.
        dislodged = unit.startswith("*")
        if dislodged:
            unit = unit[1:]
        if not unit:
            continue
        unit_type_char = unit[0]
        loc = unit[2:] if len(unit) > 2 else ""
        if unit_type_char == "A":
            unit_type = "Army"
        elif unit_type_char == "F":
            unit_type = "Fleet"
        else:
            unit_type = unit_type_char
        label = f"{unit_type} {loc}".strip()
        if dislodged:
            label += " (dislodged)"
        formatted.append(label)
    return formatted


def run_single_game(seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run a single self-play Diplomacy game to completion using random orders.

    Returns a list of state dicts, one per phase, each containing:
      - phase_name (e.g. "S1901M")
      - board_state: full game.get_state() dict for that phase
      - supply_centers: dict of power -> list of owned SCs
      - units: dict of power -> list of human-readable units
      - orders_submitted: dict of power -> list of orders that turn
      - sc_counts: dict of power -> int count of supply centers
      - winner: which power won (or None if draw / multi-winner)
      - game_id: uuid string
    """
    if seed is not None:
        random.seed(seed)

    game = Game()
    game_id = str(uuid.uuid4())

    states: List[Dict[str, Any]] = []

    # Play until the engine marks the game as completed.
    while not game.is_game_done:
        # Snapshot board state at the start of the phase (before processing orders).
        board_state = game.get_state()
        phase_name = board_state.get("name") or game._phase_abbr()

        # Extract supply centers and units from board_state.
        centers = board_state.get("centers", {})
        units_raw = board_state.get("units", {})

        supply_centers: Dict[str, List[str]] = {
            power_name: list(centers.get(power_name, [])) for power_name in centers
        }
        units: Dict[str, List[str]] = {
            power_name: _format_units(units_raw.get(power_name, []))
            for power_name in units_raw
        }
        sc_counts: Dict[str, int] = {
            power_name: len(supply_centers.get(power_name, [])) for power_name in centers
        }

        # Generate random legal orders for this phase.
        orders_submitted: Dict[str, List[str]] = {}
        all_possible_orders = game.get_all_possible_orders()
        orderable_locations_by_power = game.get_orderable_locations()

        for power_name, locs in orderable_locations_by_power.items():
            power_orders: List[str] = []
            for loc in locs:
                loc = loc.upper()
                loc_orders = all_possible_orders.get(loc, [])
                if not loc_orders:
                    continue
                order = random.choice(list(loc_orders))
                power_orders.append(order)

            if power_orders:
                game.set_orders(power_name, power_orders)
                orders_submitted[power_name] = power_orders
            else:
                # Power has no orderable units or no valid orders; skip silently.
                orders_submitted[power_name] = []

        # Record state for this phase.
        states.append(
            {
                "game_id": game_id,
                "phase_name": phase_name,
                "board_state": board_state,
                "supply_centers": supply_centers,
                "units": units,
                "orders_submitted": orders_submitted,
                "sc_counts": sc_counts,
                # winner will be filled after game completion
                "winner": None,
            }
        )

        # Advance the game to the next phase.
        game.process()

    # Determine winner(s) once the game is complete.
    outcome = getattr(game, "outcome", None)
    winners: List[str] = []
    if isinstance(outcome, list) and len(outcome) >= 2:
        # outcome = [phase_abbr, WINNER1, WINNER2, ...]
        winners = [w.upper() for w in outcome[1:]]

    # If exactly one winner, record it; otherwise treat as draw (None).
    if len(winners) == 1:
        winner_name: Optional[str] = winners[0]
    else:
        winner_name = None

    for state in states:
        state["winner"] = winner_name

    return states


def compute_rewards(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute per-power rewards for a sequence of states from a single game.

    For each state and each power:
      - +1.0 if power gained supply centers this phase
      - -1.0 if power lost supply centers this phase
      - +2.0 if power won the game on this phase
      - -2.0 if power was eliminated on this phase
      - 0.0 otherwise

    Adds to each state:
      - rewards: dict power -> reward value
      - sc_delta: dict power -> int supply center delta vs previous phase
      - is_eliminated: dict power -> bool (became eliminated on this phase)
    """
    if not states:
        return states

    # Determine winners from final state (constant across the game).
    final_state = states[-1]
    winner_name = final_state.get("winner")
    winners: List[str] = [winner_name] if isinstance(winner_name, str) else []

    prev_sc_counts: Dict[str, int] = {}
    prev_unit_counts: Dict[str, int] = {}

    for idx, state in enumerate(states):
        sc_counts: Dict[str, int] = state.get("sc_counts", {}) or {}
        units: Dict[str, List[str]] = state.get("units", {}) or {}

        # Compute top-2 SC thresholds for this phase (for intermediate rewards).
        if sc_counts:
            unique_counts = sorted(set(sc_counts.values()), reverse=True)
            top_sc_thresholds = set(unique_counts[:2])
        else:
            top_sc_thresholds = set()

        # Collect all powers seen so far.
        power_names = set(sc_counts.keys()) | set(prev_sc_counts.keys()) | set(units.keys())
        rewards: Dict[str, float] = {}
        sc_delta: Dict[str, int] = {}
        is_eliminated: Dict[str, bool] = {}

        for power in power_names:
            curr_sc = sc_counts.get(power, 0)
            prev_sc = prev_sc_counts.get(power, curr_sc)
            delta = curr_sc - prev_sc
            sc_delta[power] = delta

            curr_units = len(units.get(power, []))
            prev_units = prev_unit_counts.get(power, curr_units)

            # Elimination: transitioned from having presence (SCs or units) to none.
            eliminated_now = (prev_sc > 0 or prev_units > 0) and curr_sc == 0 and curr_units == 0
            is_eliminated[power] = eliminated_now

            reward = 0.0

            # Supply center gain/loss.
            if delta > 0:
                reward += 1.0
            elif delta < 0:
                reward -= 1.0

            # Units gain/loss.
            if curr_units > prev_units:
                reward += 0.5
            elif curr_units < prev_units:
                reward -= 0.5

            # Top-2 SC bonus for this phase.
            if top_sc_thresholds and curr_sc in top_sc_thresholds:
                reward += 0.3

            # Elimination penalty.
            if eliminated_now:
                reward -= 2.0

            # Winner bonus only on the final phase.
            if idx == len(states) - 1 and power in winners:
                reward += 2.0

            rewards[power] = float(reward)

        state["rewards"] = rewards
        state["sc_delta"] = sc_delta
        state["is_eliminated"] = is_eliminated

        prev_sc_counts = {p: sc_counts.get(p, 0) for p in power_names}
        prev_unit_counts = {p: len(units.get(p, [])) for p in power_names}

    return states


def state_to_text(state_dict: Dict[str, Any], power: str) -> str:
    """
    Convert a state dict into a readable text string from a specific power's perspective.

    Example format:

    DIPLOMACY GAME STATE
    Phase: S1901M
    Playing as: ENGLAND
    ...
    """
    power = power.upper()
    phase_name = state_dict.get("phase_name", "UNKNOWN")

    supply_centers: Dict[str, List[str]] = state_dict.get("supply_centers", {}) or {}
    units: Dict[str, List[str]] = state_dict.get("units", {}) or {}
    orders_submitted: Dict[str, List[str]] = state_dict.get("orders_submitted", {}) or {}
    sc_counts: Dict[str, int] = state_dict.get("sc_counts", {}) or {}
    sc_delta_map: Dict[str, int] = state_dict.get("sc_delta", {}) or {}
    rewards_map: Dict[str, float] = state_dict.get("rewards", {}) or {}

    my_units = units.get(power, [])
    my_scs = supply_centers.get(power, [])
    my_sc_count = sc_counts.get(power, len(my_scs))
    my_orders = orders_submitted.get(power, [])
    my_sc_delta = sc_delta_map.get(power, 0)
    my_reward = rewards_map.get(power, 0.0)

    lines: List[str] = []
    lines.append("DIPLOMACY GAME STATE")
    lines.append(f"Phase: {phase_name}")
    lines.append(f"Playing as: {power}")
    lines.append("")
    lines.append("My units: " + (", ".join(my_units) if my_units else "None"))
    lines.append(
        "My supply centers: "
        + (", ".join(my_scs) if my_scs else "None")
        + f" ({my_sc_count} centers)"
    )
    lines.append(
        "My orders this turn: "
        + (", ".join(my_orders) if my_orders else "None")
    )
    lines.append("")
    lines.append("Other powers:")

    for other_power in sorted(set(sc_counts.keys()) | set(units.keys())):
        if other_power == power:
            continue
        other_sc_count = sc_counts.get(other_power, 0)
        other_units = units.get(other_power, [])
        unit_str = ", ".join(other_units) if other_units else "None"
        lines.append(
            f"  {other_power}: {other_sc_count} SCs | Units: {unit_str}"
        )

    lines.append("")
    lines.append(f"Supply center delta this phase: {my_sc_delta}")
    lines.append(f"Reward: {my_reward:.1f}")

    return "\n".join(lines)


def generate_dataset(
    n_games: int = 200,
    output_path: str = "selfplay_states.json",
) -> None:
    """
    Generate a self-play dataset by running n_games random Diplomacy games.

    For each game:
      - run_single_game()
      - compute_rewards()
      - for each state and each power, generate a text example via state_to_text()

    Saves a flat list of training examples to output_path.
    """
    all_examples: List[Dict[str, Any]] = []

    for game_idx in tqdm(range(n_games), desc="Generating self-play games"):
        # Seed each game deterministically for reproducibility.
        seed = game_idx
        states = run_single_game(seed=seed)
        states = compute_rewards(states)

        for state in states:
            game_id = state.get("game_id")
            phase_name = state.get("phase_name")
            sc_counts: Dict[str, int] = state.get("sc_counts", {}) or {}
            sc_delta_map: Dict[str, int] = state.get("sc_delta", {}) or {}
            rewards_map: Dict[str, float] = state.get("rewards", {}) or {}
            eliminated_map: Dict[str, bool] = state.get("is_eliminated", {}) or {}

            # Use all known powers in this state; fall back to standard powers list.
            powers_in_state = set(sc_counts.keys()) | set(rewards_map.keys())
            if not powers_in_state:
                powers_in_state = set(POWERS)

            for power in sorted(powers_in_state):
                power_upper = power.upper()
                sc_count = sc_counts.get(power_upper, 0)
                sc_delta = sc_delta_map.get(power_upper, 0)
                reward = float(rewards_map.get(power_upper, 0.0))
                is_eliminated = bool(eliminated_map.get(power_upper, False))

                # winner field is global for the game (or None).
                winner_name = state.get("winner")
                is_winner = isinstance(winner_name, str) and (winner_name.upper() == power_upper)

                state_text = state_to_text(state, power_upper)

                example = {
                    "game_id": game_id,
                    "phase": phase_name,
                    "power": power_upper,
                    "state_text": state_text,
                    "reward": reward,
                    "sc_count": sc_count,
                    "sc_delta": sc_delta,
                    "is_winner": is_winner,
                    "is_eliminated": is_eliminated,
                }
                all_examples.append(example)

        if (game_idx + 1) % 25 == 0:
            print(
                f"Completed {game_idx + 1} games; "
                f"total training examples so far: {len(all_examples)}"
            )

    # Undersample zero-reward examples: keep only ~20% of them.
    non_zero_examples = [ex for ex in all_examples if ex["reward"] != 0.0]
    zero_examples = [ex for ex in all_examples if ex["reward"] == 0.0]

    keep_zero_count = int(len(zero_examples) * 0.2)
    if keep_zero_count > 0:
        random.shuffle(zero_examples)
        kept_zeros = zero_examples[:keep_zero_count]
    else:
        kept_zeros = []

    filtered_examples = non_zero_examples + kept_zeros
    random.shuffle(filtered_examples)

    # Recompute reward distribution on filtered dataset.
    reward_counts: Counter = Counter()
    for ex in filtered_examples:
        reward_counts[ex["reward"]] += 1

    # Save dataset.
    with open(output_path, "w") as f:
        json.dump(filtered_examples, f)

    # Final summary.
    print("\n=== Self-play generation summary ===")
    print(f"Total games: {n_games}")
    print(f"Total training examples (after undersampling zeros): {len(filtered_examples)}")
    print("Reward distribution:")
    for reward_value, count in sorted(reward_counts.items(), key=lambda x: x[0]):
        print(f"  {reward_value:+.1f}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play Diplomacy states via random self-play.")
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of self-play games to generate (default: 200).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="selfplay_states.json",
        help="Output JSON file path (default: selfplay_states.json).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only 3 games to verify the pipeline.",
    )

    args = parser.parse_args()
    n_games = 3 if args.test else args.games

    print(f"Generating {n_games} self-play games...")
    generate_dataset(n_games=n_games, output_path=args.output)
    print("Self-play dataset generation complete.")


if __name__ == "__main__":
    main()

