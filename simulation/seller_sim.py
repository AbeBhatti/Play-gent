"""
seller_sim.py — CraigslistSellerSim.
LLM-backed seller counterparts for the ArbitrAgent demo.
Each seller has an archetype, personality, hidden floor, and response behavior.
"""

import os
import random
from simulation.seller_profiles import RESPONSE_PROFILES

try:
    from groq import Groq

    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))
except Exception:
    groq_client = None
    GROQ_AVAILABLE = False


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

        p = self.profile
        # Profile can force always ghost (e.g. seller_ghoster_001)
        if p.get("always_ghost") is True:
            self.turns_without_response += 1
            if self.turns_without_response >= 3:
                self._status = "dead"
            return None

        # Bluffer always sends bluff message at trigger turn (deterministic demo inject)
        if p.get("archetype") == "bluffer" and self.turn >= p.get("bluff_trigger_turn", 3):
            response = p["bluff_message"]
            self.thread_history.append({"turn": self.turn, "seller": response})
            return response

        # Check ghost probability (motivated sellers never ghost)
        ghost_prob = self.response_profile["ghost_prob"]
        if self.profile["archetype"] == "motivated":
            ghost_prob = 0.0
        elif self.profile["archetype"] == "ghoster" and self.turn >= 2:
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

        base_response = None

        # Motivated seller negotiates toward floor
        if p["archetype"] == "motivated":
            if any(w in agent_lower for w in ["lower", "less", "offer", "take"]):
                drop = random.randint(5, 15)
                self.current_offer = max(p["floor"], self.current_offer - drop)
                self.concessions_made += 1
                if self.current_offer <= p["floor"]:
                    base_response = f"ok {self.current_offer} is the lowest i can do, deal?"
                else:
                    templates = [
                        "i could do ${price}",
                        "how about ${price}?",
                        "meet me at ${price}",
                    ]
                    tmpl = random.choice(templates)
                    base_response = tmpl.replace("${price}", str(self.current_offer))

        # Trade-curious resists cash, opens for trades
        if base_response is None and p["archetype"] == "trade_curious":
            if any(w in agent_lower for w in ["trade", "swap", "exchange"]):
                self._status = "active"
                base_response = random.choice(
                    [
                        "oh interesting, what did you have in mind for a trade?",
                        "id consider a trade for the right thing.",
                    ]
                )
            else:
                base_response = random.choice(
                    [
                        f"hmm not really looking to go lower on cash tbh, listed at ${self.current_offer}",
                        "not really looking for cash, got anything to trade?",
                    ]
                )

        # Default response with archetype-specific phrasing
        if base_response is None:
            drop = random.randint(2, 8)
            self.current_offer = max(p["floor"], self.current_offer - drop)
            price = self.current_offer
            if p["archetype"] == "motivated":
                templates = [
                    "i could do ${price}",
                    "how about ${price}?",
                    "meet me at ${price}",
                ]
                tmpl = random.choice(templates)
                base_response = tmpl.replace("${price}", str(price))
            elif p["archetype"] == "bluffer":
                templates = [
                    "firm on ${price}",
                    "cant do it for less than ${price}",
                    "thats my bottom line at ${price}",
                    "been getting interest at ${price}, cant go lower",
                ]
                tmpl = random.choice(templates)
                base_response = tmpl.replace("${price}", str(price))
            elif p["archetype"] == "trade_curious":
                base_response = random.choice(
                    [
                        f"hmm not really looking to go lower on cash tbh, listed at ${price}",
                        "not really looking for cash, got anything to trade?",
                    ]
                )
            else:
                base_response = f"best i can do is ${price}"

        # Optional Groq-backed surface form; keeps semantics from base_response.
        if GROQ_AVAILABLE and groq_client is not None and os.environ.get("GROQ_API_KEY"):
            try:
                system_prompt = self.get_system_prompt()
                user_prompt = (
                    f"The buyer just said: {agent_message}\n"
                    f"You were going to reply: '{base_response}'.\n"
                    "Reply in one short, casual Craigslist-style message with the same intent."
                )
                resp = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=64,
                )
                text = resp.choices[0].message.content.strip()
                if text:
                    return text
            except Exception:
                # Fall back to the rule-based response on any Groq error.
                pass

        return base_response

    def is_dead(self) -> bool:
        return self._status == "dead" or self.turns_without_response >= 3

    @property
    def status(self) -> str:
        if self.is_dead():
            return "dead"
        return self._status

