"""
HumanImitationEnv — OpenEnv 0.2.1 compatible environment.
Phase 2 training environment.
Loads real webDiplomacy game states from selfplay_states.json.
Agent is shown a real human game state and must predict the optimal move.
Reward is based on alignment with actual human outcome recorded in that state.
Human judgment from 211,278 real games is the reward signal.
"""

import json
import random
import numpy as np
from openenv.env import Env
from sentence_transformers import SentenceTransformer


class HumanImitationEnv(Env):
    def __init__(self, data_path="training/data/selfplay_states.json", seed=None):
        self.data_path = data_path
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        with open(data_path, "r") as f:
            self.all_states = json.load(f)
        print(f"Loaded {len(self.all_states)} real human game states.")
        self.current_state = None
        self.round = 0
        self.max_rounds = 10
        self.done = False

    def reset(self):
        self.current_state = random.choice(self.all_states)
        self.round = 0
        self.done = False
        obs = self._get_observation()
        info = {
            "round": self.round,
            "phase": self.current_state["phase"],
            "power": self.current_state["power"],
            "sc_count": self.current_state["sc_count"],
            "human_reward": self.current_state["reward"]
        }
        return obs, info

    def step(self, action: str):
        self.round += 1
        action_lower = action.lower()
        reward = self._compute_reward(action_lower)
        self.current_state = self._get_next_state()
        self.done = (
            self.round >= self.max_rounds or
            self.current_state.get("is_winner", False) or
            self.current_state.get("is_eliminated", False)
        )
        obs = self._get_observation()
        info = {
            "round": self.round,
            "phase": self.current_state["phase"],
            "power": self.current_state["power"],
            "sc_count": self.current_state["sc_count"],
            "sc_delta": self.current_state["sc_delta"],
            "human_reward": self.current_state["reward"],
            "agent_reward": reward,
            "is_winner": self.current_state.get("is_winner", False),
            "is_eliminated": self.current_state.get("is_eliminated", False)
        }
        return obs, reward, self.done, info

    def _compute_reward(self, action_lower):
        human_reward = self.current_state["reward"]
        sc_delta = self.current_state["sc_delta"]
        is_winner = self.current_state.get("is_winner", False)
        is_eliminated = self.current_state.get("is_eliminated", False)
        sc_count = self.current_state["sc_count"]
        reward = 0.0
        reward += human_reward * 0.5
        if sc_delta > 0:
            if any(w in action_lower for w in ["attack", "advance", "move", "push", "take"]):
                reward += 0.3
        if sc_delta < 0:
            if any(w in action_lower for w in ["defend", "hold", "support", "protect", "retreat"]):
                reward += 0.3
        if any(w in action_lower for w in ["ally", "alliance", "support", "cooperate", "together"]):
            reward += 0.2
        if sc_count >= 9 and any(w in action_lower for w in ["dominant", "strong", "win", "control"]):
            reward += 0.2
        if is_eliminated and any(w in action_lower for w in ["survive", "desperate", "last", "hold"]):
            reward += 0.3
        if is_winner and any(w in action_lower for w in ["victory", "win", "dominate", "finish"]):
            reward += 0.5
        return float(np.clip(reward, -1.0, 2.0))

    def _get_next_state(self):
        current_game_id = self.current_state["game_id"]
        same_game = [
            s for s in self.all_states
            if s["game_id"] == current_game_id
            and s["phase"] != self.current_state["phase"]
        ]
        if same_game:
            return random.choice(same_game)
        return random.choice(self.all_states)

    def _get_state_text(self):
        s = self.current_state
        return f"""HUMAN IMITATION ENVIRONMENT — Phase 2 Training
Round: {self.round}/{self.max_rounds}
Source: Real human game — pulled from webDiplomacy.net

{s['state_text']}

Your task: What move would a skilled human player make here?
Explain your reasoning and state your intended orders."""

    def _get_observation(self):
        text = self._get_state_text()
        emb = self.encoder.encode(text, convert_to_numpy=True)
        return emb.astype(np.float32)

    def render(self):
        text = self._get_state_text()
        print(text)
        return text

    def close(self):
        print("HumanImitationEnv closed.")

    @property
    def observation_space(self):
        return {"type": "continuous", "shape": (384,), "dtype": "float32"}

    @property
    def action_space(self):
        return {
            "type": "text",
            "description": "Natural language move + reasoning"
        }

"""
HumanImitationEnv — OpenEnv 0.2.1 compatible environment.
Phase 2 training environment.
Loads real webDiplomacy game states from selfplay_states.json.
Agent is shown a real human game state and must predict the optimal move.
Reward is based on alignment with actual human outcome recorded in that state.
Human judgment from 211,278 real games is the reward signal.
"""

import json
import random
import numpy as np
from openenv.env import Env
from sentence_transformers import SentenceTransformer


class HumanImitationEnv(Env):
    def __init__(self, data_path="training/data/selfplay_states.json", seed=None):
        self.data_path = data_path
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        with open(data_path, "r") as f:
            self.all_states = json.load(f)
        print(f"Loaded {len(self.all_states)} real human game states.")
        self.current_state = None
        self.round = 0
        self.max_rounds = 10
        self.done = False

    def reset(self):
        self.current_state = random.choice(self.all_states)
        self.round = 0
        self.done = False
        obs = self._get_observation()
        info = {
            "round": self.round,
            "phase": self.current_state["phase"],
            "power": self.current_state["power"],
            "sc_count": self.current_state["sc_count"],
            "human_reward": self.current_state["reward"]
        }
        return obs, info

    def step(self, action: str):
        self.round += 1
        action_lower = action.lower()
        reward = self._compute_reward(action_lower)
        self.current_state = self._get_next_state()
        self.done = (
            self.round >= self.max_rounds or
            self.current_state.get("is_winner", False) or
            self.current_state.get("is_eliminated", False)
        )
        obs = self._get_observation()
        info = {
            "round": self.round,
            "phase": self.current_state["phase"],
            "power": self.current_state["power"],
            "sc_count": self.current_state["sc_count"],
            "sc_delta": self.current_state["sc_delta"],
            "human_reward": self.current_state["reward"],
            "agent_reward": reward,
            "is_winner": self.current_state.get("is_winner", False),
            "is_eliminated": self.current_state.get("is_eliminated", False)
        }
        return obs, reward, self.done, info

    def _compute_reward(self, action_lower):
        human_reward = self.current_state["reward"]
        sc_delta = self.current_state["sc_delta"]
        is_winner = self.current_state.get("is_winner", False)
        is_eliminated = self.current_state.get("is_eliminated", False)
        sc_count = self.current_state["sc_count"]
        reward = 0.0
        reward += human_reward * 0.5
        if sc_delta > 0:
            if any(w in action_lower for w in ["attack", "advance", "move", "push", "take"]):
                reward += 0.3
        if sc_delta < 0:
            if any(w in action_lower for w in ["defend", "hold", "support", "protect", "retreat"]):
                reward += 0.3
        if any(w in action_lower for w in ["ally", "alliance", "support", "cooperate", "together"]):
            reward += 0.2
        if sc_count >= 9 and any(w in action_lower for w in ["dominant", "strong", "win", "control"]):
            reward += 0.2
        if is_eliminated and any(w in action_lower for w in ["survive", "desperate", "last", "hold"]):
            reward += 0.3
        if is_winner and any(w in action_lower for w in ["victory", "win", "dominate", "finish"]):
            reward += 0.5
        return float(np.clip(reward, -1.0, 2.0))

    def _get_next_state(self):
        current_game_id = self.current_state["game_id"]
        same_game = [
            s for s in self.all_states
            if s["game_id"] == current_game_id
            and s["phase"] != self.current_state["phase"]
        ]
        if same_game:
            return random.choice(same_game)
        return random.choice(self.all_states)

    def _get_state_text(self):
        s = self.current_state
        return f"""HUMAN IMITATION ENVIRONMENT — Phase 2 Training
Round: {self.round}/{self.max_rounds}
Source: Real human game — pulled from webDiplomacy.net

{s['state_text']}

Your task: What move would a skilled human player make here?
Explain your reasoning and state your intended orders."""

    def _get_observation(self):
        text = self._get_state_text()
        emb = self.encoder.encode(text, convert_to_numpy=True)
        return emb.astype(np.float32)

    def render(self):
        text = self._get_state_text()
        print(text)
        return text

    def close(self):
        print("HumanImitationEnv closed.")

    @property
    def observation_space(self):
        return {"type": "continuous", "shape": (384,), "dtype": "float32"}

    @property
    def action_space(self):
        return {
            "type": "text",
            "description": "Natural language move + reasoning"
        }

