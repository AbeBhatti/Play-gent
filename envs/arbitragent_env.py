"""
ArbitrAgentEnv — OpenEnv 0.2.1 unified environment with three reward signals.

total_reward = 0.35 * accuracy_reward + 0.35 * outcome_reward + 0.30 * bluff_reward

- accuracy_reward: cosine similarity between model action embedding and human action embedding.
- outcome_reward: keyword scoring (coalition, pressure, clean close vs premature concession).
- bluff_reward: BluffDetector on synthetic seller message; reward correct flag, penalize missed bluff.
"""

import json
import random
import re
import numpy as np
from openenv.env import Env
from sentence_transformers import SentenceTransformer

def _extract_human_orders(state_text: str) -> str:
    """Extract 'My orders this turn: ...' from Diplomacy state_text as human action."""
    match = re.search(r"My orders this turn:\s*(.+?)(?:\n|$)", state_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return state_text[:200]


# Synthetic seller message used for bluff_reward when state has no thread (Diplomacy states).
SYNTHETIC_BLUFF_MESSAGE = "look i really cant go lower than $30, thats my final offer. been getting a lot of interest so"
SYNTHETIC_BLUFF_PROFILE = {
    "response_speed": "fast",
    "archetype": "bluffer",
}
SYNTHETIC_THREAD = [
    {"turn": 1, "agent": "Can you do $35?"},
    {"turn": 1, "seller": "How about $32?"},
    {"turn": 2, "agent": "I need a better price."},
    {"turn": 2, "seller": SYNTHETIC_BLUFF_MESSAGE},
]


class ArbitrAgentEnv(Env):
    """
    Single OpenEnv 0.2.1 environment combining accuracy (human move alignment),
    outcome (negotiation language), and bluff (detection) rewards.
    """

    def __init__(self, data_path: str = "training/data/selfplay_states.json", seed=None):
        self.data_path = data_path
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        with open(data_path, "r") as f:
            self.all_states = json.load(f)
        self.current_state = None
        self.round = 0
        self.max_rounds = 10
        self.done = False
        self._last_reward_breakdown = None

    def reset(self):
        self.current_state = random.choice(self.all_states)
        self.round = 0
        self.done = False
        self._last_reward_breakdown = None
        obs = self._get_observation()
        info = {
            "round": self.round,
            "phase": self.current_state.get("phase", ""),
            "power": self.current_state.get("power", ""),
        }
        return obs, info

    def step(self, action: str):
        self.round += 1
        action = action or "(no action)"
        action_lower = action.lower()

        accuracy = self._accuracy_reward(action)
        outcome = self._outcome_reward(action_lower)
        bluff, bluff_signals, seller_bluff_detected = self._bluff_reward(action_lower)

        total = 0.35 * accuracy + 0.35 * outcome + 0.30 * bluff
        self._last_reward_breakdown = {"accuracy": accuracy, "outcome": outcome, "bluff": bluff, "total": total}

        self.current_state = self._get_next_state()
        self.done = (
            self.round >= self.max_rounds
            or self.current_state.get("is_winner", False)
            or self.current_state.get("is_eliminated", False)
        )
        obs = self._get_observation()
        info = {
            "round": self.round,
            "accuracy": accuracy,
            "outcome": outcome,
            "bluff": bluff,
            "total": total,
            "phase": self.current_state.get("phase", ""),
            "power": self.current_state.get("power", ""),
            "bluff_detected": seller_bluff_detected,
            "bluff_signals": bluff_signals,
        }
        return obs, total, self.done, info

    def _accuracy_reward(self, action: str) -> float:
        """Cosine similarity between action embedding and human action embedding."""
        state_text = self.current_state.get("state_text", "")
        human_action_text = _extract_human_orders(state_text)
        action_emb = self.encoder.encode(action, convert_to_numpy=True)
        human_emb = self.encoder.encode(human_action_text, convert_to_numpy=True)
        dot = float(np.dot(action_emb, human_emb))
        norm_a = float(np.linalg.norm(action_emb)) or 1e-8
        norm_h = float(np.linalg.norm(human_emb)) or 1e-8
        cos = dot / (norm_a * norm_h)
        return float(np.clip(cos, -1.0, 1.0))

    def _outcome_reward(self, action_lower: str) -> float:
        """Keyword scoring: reward coalition/pressure/clean close; penalize premature concession."""
        reward = 0.0
        # Positive: coalition language
        if any(w in action_lower for w in ["ally", "alliance", "coalition", "support", "another buyer", "trade offer from another"]):
            reward += 0.4
        # Positive: pressure moves
        if any(w in action_lower for w in ["pressure", "leverage", "can you do", "less urgent", "make the numbers work"]):
            reward += 0.3
        # Positive: clean close
        if any(w in action_lower for w in ["deal", "agree", "accept", "close"]):
            reward += 0.2
        # Negative: premature concession (accepting stated floor)
        if any(w in action_lower for w in ["ok $30", "accept 30", "take it at 30", "deal at 30"]):
            reward -= 0.6
        # Negative: accepting stated floor language
        if any(w in action_lower for w in ["final offer", "lowest you can go", "that's your final"]):
            reward -= 0.3
        return float(np.clip(reward, -1.0, 1.0))

    def _bluff_reward(self, action_lower: str):
        """
        Analyze the synthetic SELLER message for bluff_detected and bluff_signals (for info).
        Bluff reward = score agent for coalition pressure / bluff-calling when seller message is a bluff.
        """
        try:
            from agent.bluff_detector import analyze_bluff, learned_bluff_score
            # Analyze the seller's (synthetic) message for UI signals
            signals = analyze_bluff(
                SYNTHETIC_BLUFF_PROFILE,
                SYNTHETIC_THREAD,
                SYNTHETIC_BLUFF_MESSAGE,
                turn=2,
            )
            learned = learned_bluff_score(SYNTHETIC_BLUFF_MESSAGE, SYNTHETIC_THREAD)
            signals_dict = {
                "timing_tell": round(signals.timing_tell, 3),
                "size_tell": round(signals.size_tell, 3),
                "formulaic_tell": round(signals.formulaic_tell, 3),
                "pattern_tell": round(signals.pattern_tell, 3),
                "learned_score": round(learned, 3),
            }
            # Synthetic state always includes the canonical bluff message; reward agent for coalition pressure
            seller_is_bluff = signals.is_bluff or (signals.bluff_score > 0.25)  # treat synthetic as bluff context
            reward = 0.0
            if seller_is_bluff:
                if any(w in action_lower for w in ["bluff", "other seller", "other buyers", "other deal", "lined up", "two other", "better deal", "isn't urgent", "or i walk", "can you do $", "trade offer from another", "sellers lined up"]):
                    reward += 0.6
                if any(w in action_lower for w in ["lying", "final", "non-negotiable", "counter", "$20", "$22", "$24", "$26", "non negotiable"]):
                    reward += 0.3
            reward = float(np.clip(reward, 0.0, 1.0))
            return reward, signals_dict, True  # synthetic seller message is always bluff for UI
        except Exception:
            return 0.0, {}, False

    def _get_next_state(self):
        current_game_id = self.current_state.get("game_id")
        same_game = [
            s for s in self.all_states
            if s.get("game_id") == current_game_id and s.get("phase") != self.current_state.get("phase")
        ]
        if same_game:
            return random.choice(same_game)
        return random.choice(self.all_states)

    def _get_state_text(self):
        s = self.current_state
        return f"""ARBITRAGENT UNIFIED ENV — Round {self.round}/{self.max_rounds}
Phase: {s.get('phase', '')} | Power: {s.get('power', '')}

{s.get('state_text', '')}

Synthetic seller message (for bluff awareness): "{SYNTHETIC_BLUFF_MESSAGE}"

Your task: Propose a move. If you detect a bluff, use coalition pressure; otherwise negotiate toward a good outcome."""

    def _get_observation(self):
        text = self._get_state_text()
        emb = self.encoder.encode(text, convert_to_numpy=True)
        return emb.astype(np.float32)

    def render(self):
        text = self._get_state_text()
        if self._last_reward_breakdown:
            text += f"\n\nLast reward breakdown: accuracy={self._last_reward_breakdown['accuracy']:.3f}, outcome={self._last_reward_breakdown['outcome']:.3f}, bluff={self._last_reward_breakdown['bluff']:.3f}, total={self._last_reward_breakdown['total']:.3f}"
        return text

    def close(self):
        pass

    @property
    def observation_space(self):
        return {"type": "continuous", "shape": (384,), "dtype": "float32"}

    @property
    def action_space(self):
        return {"type": "text", "description": "Natural language move + reasoning"}
