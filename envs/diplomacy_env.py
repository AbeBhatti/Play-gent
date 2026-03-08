import random
from typing import Any, Dict, Tuple

import numpy as np
from diplomacy import Game
from openenv.env import Env
from sentence_transformers import SentenceTransformer


class DiplomacyNegotiationEnv(Env):
    """
    OpenEnv-compatible wrapper around the diplomacy.Game engine.

    Observation: 384-dim MiniLM embedding of a textual game-state description
    from the perspective of a single power (e.g. ENGLAND).
    Action: free-form text describing strategic intent (logged but not yet parsed).
    """

    def __init__(self, power_name: str = "ENGLAND", seed: int | None = None):
        self._reset_random_power = power_name.upper() == "ENGLAND"  # default: vary power on reset for non-hardcoded obs
        self.power_name = power_name.upper()
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.game: Game | None = None
        self.current_phase: int = 0
        self.prev_sc_count: int = 0
        self.max_phases: int = 50
        if seed is not None:
            random.seed(seed)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the underlying Diplomacy game and return initial observation + info."""
        self.game = Game()
        self.current_phase = 0
        state = self.game.get_state()
        centers = state.get("centers", {})
        # Vary observation across resets: pick random power and optionally advance game
        powers = list(centers.keys())
        if powers and getattr(self, "_reset_random_power", True):
            self.power_name = random.choice(powers)
        # Advance by 0..3 random phases so consecutive resets give different states
        n_advance = random.randint(0, 3)
        for _ in range(n_advance):
            all_possible = self.game.get_all_possible_orders()
            for power, locs in self.game.get_orderable_locations().items():
                orders = []
                for loc in locs:
                    loc_orders = all_possible.get(loc.upper(), [])
                    if loc_orders:
                        orders.append(random.choice(list(loc_orders)))
                if orders:
                    self.game.set_orders(power, orders)
            self.game.process()
            self.current_phase += 1
        state = self.game.get_state()
        centers = state.get("centers", {})
        self.prev_sc_count = len(centers.get(self.power_name, []))
        obs = self._get_observation()
        info = {"phase": state.get("name"), "sc_count": self.prev_sc_count}
        return obs, info

    def step(self, action: str):
        """
        Advance one phase.

        - Currently ignores the semantic content of `action` and instead
          submits random legal orders for all powers.
        - Logs the provided action in the returned info for later analysis.
        """
        if self.game is None:
            raise RuntimeError("Environment must be reset() before step().")

        # Submit random legal orders for all powers.
        all_possible = self.game.get_all_possible_orders()
        for power, locs in self.game.get_orderable_locations().items():
            orders = []
            for loc in locs:
                loc_orders = all_possible.get(loc.upper(), [])
                if loc_orders:
                    orders.append(random.choice(list(loc_orders)))
            if orders:
                self.game.set_orders(power, orders)

        self.game.process()
        self.current_phase += 1

        reward = self._compute_reward()
        obs = self._get_observation()
        done = self.game.is_game_done or self.current_phase >= self.max_phases

        state = self.game.get_state()
        curr_sc = len(state.get("centers", {}).get(self.power_name, []))

        if self.game.is_game_done:
            done_reason = "game_complete"
        elif self.current_phase >= self.max_phases:
            done_reason = "max_phases"
        else:
            done_reason = None

        info = {
            "phase": state.get("name"),
            "sc_count": curr_sc,
            "sc_delta": curr_sc - self.prev_sc_count,
            "action_logged": action,
            "done_reason": done_reason,
        }
        return obs, reward, done, info

    def _compute_reward(self) -> float:
        """Shaped reward based on SC changes, relative rank, and game outcome."""
        if self.game is None:
            return 0.0

        state = self.game.get_state()
        centers = state.get("centers", {})
        curr_sc = len(centers.get(self.power_name, []))
        all_counts = {p: len(c) for p, c in centers.items()}

        delta = curr_sc - self.prev_sc_count
        self.prev_sc_count = curr_sc

        reward = 0.0
        if delta > 0:
            reward += 1.0
        if delta < 0:
            reward -= 1.0
        if curr_sc == 0:
            reward -= 2.0

        # Relative position bonuses/penalties.
        if all_counts:
            sorted_counts = sorted(all_counts.values(), reverse=True)
            top_two = sorted_counts[:2]
            bottom_two = sorted_counts[-2:]
            if curr_sc in top_two:
                reward += 0.3
            if curr_sc in bottom_two and curr_sc > 0:
                reward -= 0.2

        # Game outcome bonus when completed.
        if self.game.is_game_done:
            outcome = getattr(self.game, "outcome", [])
            if isinstance(outcome, list) and len(outcome) > 1:
                if self.power_name in [w.upper() for w in outcome[1:]]:
                    reward += 2.0

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Return a 384-dim MiniLM embedding of the current game state text."""
        text = self._get_state_text()
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        # Ensure consistent dtype for downstream RL code.
        return embedding.astype(np.float32)

    def _get_state_text(self) -> str:
        """Human-readable textual description of the current game state."""
        if self.game is None:
            return "Environment not initialized."

        state = self.game.get_state()
        centers = state.get("centers", {})
        units = state.get("units", {})
        phase = state.get("name", "UNKNOWN")

        my_scs = centers.get(self.power_name, [])
        my_units = units.get(self.power_name, [])
        curr_sc = len(my_scs)
        delta = curr_sc - self.prev_sc_count

        # Coarse strategic position label.
        if curr_sc > 10:
            position = "dominant"
        elif curr_sc >= 7:
            position = "strong"
        elif curr_sc >= 4:
            position = "stable"
        elif curr_sc >= 2:
            position = "weak"
        else:
            position = "critical"

        lines: list[str] = [
            "DIPLOMACY GAME STATE",
            f"Phase: {phase}",
            f"Playing as: {self.power_name}",
            "",
            f"My units: {', '.join(my_units) or 'None'}",
            f"My supply centers: {', '.join(my_scs) or 'None'} ({curr_sc} centers)",
            "",
            "Other powers:",
        ]

        for power in sorted(centers.keys()):
            if power == self.power_name:
                continue
            sc_count = len(centers.get(power, []))
            unit_list = units.get(power, [])
            lines.append(
                f"  {power}: {sc_count} SCs | Units: {', '.join(unit_list) or 'None'}"
            )

        lines += [
            "",
            f"Strategic position: {position}",
            f"Supply center delta: {delta:+d}",
        ]
        return "\n".join(lines)

    def render(self):
        """Print and return the current state text."""
        text = self._get_state_text()
        print(text)
        return text

    def close(self):
        """Clean up the underlying game."""
        self.game = None
        print("Environment closed.")

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {"type": "continuous", "shape": (384,), "dtype": "float32"}

    @property
    def action_space(self) -> Dict[str, Any]:
        return {"type": "text", "description": "Natural language strategic intent"}

