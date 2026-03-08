"""
agent_llm.py — Lightweight inference wrapper for the trained TinyLlama model.

Lazy-loads unified_final (or phase2_final) and generates negotiation messages
for ArbitrAgent: scout, pressure, and coalition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Lazy-loaded
_MODEL = None
_TOKENIZER = None
_CHECKPOINT_PATH: Optional[Path] = None


def _resolve_checkpoint() -> Path:
    """Unified_final if exists, else phase2_final."""
    root = Path(__file__).resolve().parent.parent
    unified = root / "training" / "checkpoints" / "unified_final"
    phase2 = root / "training" / "checkpoints" / "phase2_final"
    if unified.exists() and (unified / "config.json").exists():
        return unified
    if phase2.exists() and (phase2 / "config.json").exists():
        return phase2
    return unified  # caller will handle missing


def _load():
    global _MODEL, _TOKENIZER, _CHECKPOINT_PATH
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _CHECKPOINT_PATH = _resolve_checkpoint()
    if not _CHECKPOINT_PATH.exists() or not (_CHECKPOINT_PATH / "config.json").exists():
        return
    _TOKENIZER = AutoTokenizer.from_pretrained(str(_CHECKPOINT_PATH))
    _MODEL = AutoModelForCausalLM.from_pretrained(
        str(_CHECKPOINT_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _MODEL.eval()


class AgentLLM:
    """
    Lazy-loads the trained TinyLlama checkpoint (unified_final or phase2_final)
    and provides scout_message, pressure_message, coalition_message.
    """

    def _clean(self, text: str, fallback: str) -> str:
        BAD_PHRASES = [
            "my goal", "more specifically", "focused on helping",
            "value proposition", "helping sellers", "helping buyers",
            "specifically focused", "as an auctioneer", "as a buyer",
            "increasing conversions", "active listener"
        ]
        # Take only first sentence/line
        text = text.strip().split('.')[0].split('\n')[0].strip()
        # If too long or contains bad phrases, use fallback
        if len(text) > 120:
            return fallback
        if any(p in text.lower() for p in BAD_PHRASES):
            return fallback
        # If too short to be meaningful, use fallback
        if len(text) < 10:
            return fallback
        return text

    def generate(self, prompt: str, max_tokens: int = 80) -> str:
        """Generate text from prompt; returns only the generated part (prompt stripped)."""
        _load()
        if _MODEL is None or _TOKENIZER is None:
            return ""
        import torch

        inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL.device)
        prompt_decoded = _TOKENIZER.decode(inputs["input_ids"][0], skip_special_tokens=True)
        with torch.no_grad():
            out = _MODEL.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=_TOKENIZER.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        full = _TOKENIZER.decode(out[0], skip_special_tokens=True)
        if full.startswith(prompt_decoded):
            generated = full[len(prompt_decoded) :].strip()
        else:
            generated = full.strip()
        # First sentence or line
        for sep in ["\n", ".", "!"]:
            if sep in generated:
                generated = generated.split(sep)[0].strip()
                break
        # Fall back to hardcoded if 3+ consecutive repeated words
        words = generated.split()
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return ""
        return generated

    def scout_message(self, item: str, listing_price: float) -> str:
        """Opening inquiry to seller."""
        prompt = (
            f"You are a buyer on Craigslist. Send a short, casual opening message "
            f"asking if the {item} (listed around ${listing_price:.0f}) is still available "
            f"and if there's any room on price. Keep it under 20 words. Message:"
        )
        result = self.generate(prompt, max_tokens=40)
        return self._clean(result, f"hey, is the {item} still available? any room on price?")

    def pressure_message(self, item: str, current_offer: float, turn: int = 0) -> str:
        """Follow-up pressure message when seller hasn't moved much. Rotates through 5 messages by turn."""
        PRESSURE_MESSAGES = [
            "just checking back — any flexibility on your price at all?",
            "I have a trade offer from another seller that makes this less urgent — can you do better?",
            "still interested but my other option is looking more attractive — any movement?",
            "last check — is there any room at all or should I go with my other offer?",
            "I need to make a decision today — can you sharpen your price?",
        ]
        index = (turn - 2) % len(PRESSURE_MESSAGES) if turn >= 2 else 0
        fallback = PRESSURE_MESSAGES[index]
        prompt = (
            f"You are a buyer negotiating for a {item}. Current seller offer is ${current_offer:.0f}. "
            f"Send a short follow-up (turn {turn}) asking for flexibility. Vary the phrasing. Keep it under 25 words. Message:"
        )
        result = self.generate(prompt, max_tokens=40)
        return self._clean(result, fallback)

    def coalition_message(self, item: str, floor_minus_4: int) -> str:
        """Coalition pressure after detecting a bluff; counter at floor_minus_4."""
        prompt = (
            f"You are a buyer for a {item}. You detected the seller is bluffing about a final offer. "
            f"You have another deal lined up. Mention it casually and counter at ${floor_minus_4}. "
            f"Keep it under 25 words. Message:"
        )
        result = self.generate(prompt, max_tokens=50)
        return self._clean(
            result,
            f"I have a trade offer from another seller that makes this less urgent for me — can you do ${floor_minus_4}?",
        )
