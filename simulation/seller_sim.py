"""
seller_sim.py — CraigslistSellerSim.
LLM-backed seller counterparts for the ArbitrAgent demo.
Each seller has an archetype, personality, hidden floor, and response behavior.
"""

import random
from simulation.seller_profiles import RESPONSE_PROFILES


class CraigslistSellerSim:
    def __init__(self, profile: dict, client=None):
        self.profile = profile
        self.client = client
        self.turn = 0
        self.turns_without_response = 0
        self.current_offer = profile["listing_price"]
        self.concessions_made = 0
        self.thread_history = []
        self._status = "active"
        self.response_profile = RESPONSE_PROFILES[profile["response_speed"]]

    def get_system_prompt(self) -> str:
        p = self.profile
        base = f"""You are a Craigslist seller. You are selling: {p['item']}.
Your listing price is ${p['listing_price']}.
Your absolute floor (never reveal this): ${p['floor']}.
Your personality: {p['personality']}
Respond like a real person texting. Keep it short. Casual tone. Typos ok.
Never reveal your floor price directly."""

        if p["archetype"] == "motivated":
            base += "\nYou want to sell quickly. You will negotiate honestly toward your floor."

        elif p["archetype"] == "bluffer":
            if self.turn >= p.get("bluff_trigger_turn", 3):
                base += f"""
IMPORTANT: On this turn you must say your bluff message EXACTLY:
"{p['bluff_message']}"
This is a bluff — you actually have room left but you want them to think you don't."""
            else:
                base += "\nYou are open to negotiating but will pretend to be firm when pushed."

        elif p["archetype"] == "ghoster":
            base += "\nYou are unreliable. You may not respond."

        elif p["archetype"] == "trade_curious":
            base += "\nYou resist cash offers but get excited about trades."

        return base

    def step(self, agent_message: str) -> str | None:
        self.turn += 1
        self.thread_history.append({"turn": self.turn, "agent": agent_message})

        # Bluffer always sends bluff message at trigger turn (deterministic demo inject)
        p = self.profile
        if p.get("archetype") == "bluffer" and self.turn >= p.get("bluff_trigger_turn", 3):
            response = p["bluff_message"]
            self.thread_history.append({"turn": self.turn, "seller": response})
            return response

        # Check ghost probability
        ghost_prob = self.response_profile["ghost_prob"]
        if self.profile["archetype"] == "ghoster" and self.turn >= 2:
            ghost_prob = 0.70

        if random.random() < ghost_prob:
            self.turns_without_response += 1
            if self.turns_without_response >= 3:
                self._status = "dead"
            return None

        self.turns_without_response = 0

        # Generate response
        response = self._generate_response(agent_message)
        self.thread_history.append({"turn": self.turn, "seller": response})
        return response

    def _generate_response(self, agent_message: str) -> str:
        p = self.profile
        agent_lower = agent_message.lower()

        # Bluffer sends exact bluff message at trigger turn
        if p["archetype"] == "bluffer" and self.turn >= p.get("bluff_trigger_turn", 3):
            return p["bluff_message"]

        # Motivated seller negotiates toward floor
        if p["archetype"] == "motivated":
            if any(w in agent_lower for w in ["lower", "less", "offer", "take"]):
                drop = random.randint(5, 15)
                self.current_offer = max(p["floor"], self.current_offer - drop)
                self.concessions_made += 1
                if self.current_offer <= p["floor"]:
                    return f"ok {self.current_offer} is the lowest i can do, deal?"
                return f"i could do ${self.current_offer}, how does that sound"

        # Trade-curious resists cash, opens for trades
        if p["archetype"] == "trade_curious":
            if any(w in agent_lower for w in ["trade", "swap", "exchange"]):
                self._status = "active"
                return "oh interesting, what did you have in mind for a trade?"
            else:
                return f"hmm not really looking to go lower on cash tbh, listed at ${self.current_offer}"

        # Default response
        drop = random.randint(2, 8)
        self.current_offer = max(p["floor"], self.current_offer - drop)
        return f"best i can do is ${self.current_offer}"

    def is_dead(self) -> bool:
        return self._status == "dead" or self.turns_without_response >= 3

    @property
    def status(self) -> str:
        if self.is_dead():
            return "dead"
        return self._status

