"""
ContractorNegotiationEnv — OpenEnv 0.2.1 compatible environment.

Real-world negotiation environment where the agent negotiates with multiple
contractors to achieve the lowest price. Demonstrates transfer of Diplomacy
skills (coalition pressure, multi-party bluffing) to real-world negotiation.

Swap guide:
    To change the use case, subclass or copy this file and modify:
    - _get_state_text()  → change the scenario description  
    - _compute_reward()  → change what counts as a good outcome
    - reset()            → change the "parties" and their attributes
    Everything else (obs space, action space, training loop) stays identical.
"""

import random
import numpy as np

try:
    from openenv_base import OpenEnvBase
    BASE = OpenEnvBase
except ImportError:
    class BASE:
        pass

from sentence_transformers import SentenceTransformer


class ContractorNegotiationEnv(BASE):
    def __init__(self, n_contractors=5, budget=10000, seed=None):
        self.n_contractors = n_contractors
        self.budget = budget
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        self.contractors = {}
        self.round = 0
        self.max_rounds = 10
        self.accepted = False

    def reset(self):
        self.contractors = {}
        for i in range(self.n_contractors):
            floor = random.randint(int(self.budget * 0.5), int(self.budget * 0.75))
            ask = random.randint(int(self.budget * 0.8), int(self.budget * 1.1))
            urgency = random.random()
            self.contractors[f"Contractor_{i}"] = {
                "ask": ask,
                "floor": floor,
                "urgency": urgency,
                "rounds": 0,
                "active": True,
                "at_floor": False
            }
        self.round = 0
        self.accepted = False
        obs = self._get_observation()
        return obs, {"round": 0, "best_offer": self._best_offer()}

    def step(self, action: str):
        self.round += 1
        action_lower = action.lower()

        for name, c in self.contractors.items():
            if not c["active"]:
                continue
            c["rounds"] += 1
            drop_rate = 0.05 + (0.1 * c["urgency"])

            if any(w in action_lower for w in ["competing", "other", "beat", "match", "offer"]):
                drop_rate *= 1.5
            if any(w in action_lower for w in ["pressure", "budget", "lower", "reduce"]):
                drop_rate *= 1.2

            new_ask = max(c["floor"], int(c["ask"] * (1 - drop_rate)))
            c["ask"] = new_ask

            if c["ask"] <= c["floor"] * 1.02:
                c["at_floor"] = True

        if "accept" in action_lower:
            self.accepted = True

        reward = self._compute_reward(action_lower)
        obs = self._get_observation()
        done = self.accepted or self.round >= self.max_rounds

        info = {
            "round": self.round,
            "best_offer": self._best_offer(),
            "savings": self.budget - self._best_offer(),
            "savings_pct": (self.budget - self._best_offer()) / self.budget * 100,
            "action_logged": action
        }
        return obs, reward, done, info

    def _compute_reward(self, action_lower):
        best = self._best_offer()
        savings_pct = (self.budget - best) / self.budget
        reward = 0.0

        if savings_pct > 0.20:
            reward += 2.0
        elif savings_pct > 0.15:
            reward += 1.5
        elif savings_pct > 0.10:
            reward += 1.0
        elif savings_pct > 0.05:
            reward += 0.5

        if any(w in action_lower for w in ["competing", "other offer", "beat", "match"]):
            reward += 0.5

        if self.accepted and savings_pct > 0.10:
            reward += 1.0

        if self.round >= self.max_rounds and not self.accepted:
            reward -= 1.0

        return float(reward)

    def _best_offer(self):
        active = [c["ask"] for c in self.contractors.values() if c["active"]]
        return min(active) if active else self.budget

    def _get_state_text(self):
        best = self._best_offer()
        lines = [
            "CONTRACTOR NEGOTIATION STATE",
            f"Round: {self.round}/{self.max_rounds}",
            f"Budget: ${self.budget:,}",
            f"Best current offer: ${best:,}",
            f"Potential savings: ${self.budget - best:,}",
            "",
            "Active contractors:"
        ]
        for name, c in self.contractors.items():
            if not c["active"]:
                continue
            urgency = "HIGH" if c["urgency"] > 0.6 else "LOW"
            lines.append(
                f"  {name}: asking ${c['ask']:,} | urgency: {urgency} | rounds negotiated: {c['rounds']}"
            )
        lines += [
            "",
            f"Rounds remaining: {self.max_rounds - self.round}",
            "What is your next negotiation move?"
        ]
        return "\n".join(lines)

    def _get_observation(self):
        text = self._get_state_text()
        emb = self.encoder.encode(text, convert_to_numpy=True)
        return emb.astype(np.float32)

    def render(self):
        text = self._get_state_text()
        print(text)
        return text

    def close(self):
        print("ContractorNegotiationEnv closed.")

    @property
    def observation_space(self):
        return {"type": "continuous", "shape": (384,), "dtype": "float32"}

    @property
    def action_space(self):
        return {"type": "text", "description": "Natural language negotiation move"}