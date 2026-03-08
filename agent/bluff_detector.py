"""
bluff_detector.py — bluff signal extraction for ArbitrAgent.

This module exposes a small, deterministic API that inspects a seller's
response in the context of a thread and extracts four bluff signals:

1. timing_tell
2. size_tell
3. formulaic_tell
4. pattern_tell

The overall bluff_score is a weighted sum of these four signals. A response
is flagged as a bluff when bluff_score > 0.6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


FORMULAIC_PHRASES: List[str] = [
    "lowest i can go",
    "lowest i can do",
    "final offer",
    "cant go lower",
    "can't go lower",
    "cant do lower",
    "can't do lower",
    "thats my final offer",
    "that's my final offer",
    "best i can do",
]


@dataclass
class BluffSignals:
    timing_tell: float
    size_tell: float
    formulaic_tell: float
    pattern_tell: float
    bluff_score: float
    is_bluff: bool


DEFAULT_WEIGHTS: Dict[str, float] = {
    "timing_tell": 0.25,
    "size_tell": 0.25,
    "formulaic_tell": 0.25,
    "pattern_tell": 0.25,
}


def analyze_bluff(
    seller_profile: Mapping[str, Any],
    thread_history: Sequence[Mapping[str, Any]],
    seller_message: str,
    turn: Optional[int] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> BluffSignals:
    """
    Analyze a seller response and return bluff signals plus an overall score.

    Parameters
    ----------
    seller_profile:
        The seller profile dict (see simulation.seller_profiles.SELLER_PROFILES).
    thread_history:
        Sequence of {"turn": int, "agent": str} / {"turn": int, "seller": str}
        entries, such as CraigslistSellerSim.thread_history.
    seller_message:
        The seller's latest message to analyze.
    turn:
        Turn index at which this message was sent. If omitted, the maximum
        `turn` value from thread_history is used.
    weights:
        Optional per-signal weights. Keys should be
        {"timing_tell", "size_tell", "formulaic_tell", "pattern_tell"}.
        If omitted, DEFAULT_WEIGHTS are used.
    """
    if turn is None:
        turn = max((entry.get("turn", 0) for entry in thread_history), default=0)

    timing = _timing_tell(seller_profile, turn)
    size = _size_tell(seller_message)
    formulaic = _formulaic_tell(seller_message)
    pattern = _pattern_tell(thread_history, seller_message)

    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_w = float(sum(weights.get(k, 0.0) for k in DEFAULT_WEIGHTS.keys()))
    if total_w <= 0.0:
        norm_weights = DEFAULT_WEIGHTS
        total_w = 1.0
    else:
        norm_weights = {
            key: float(weights.get(key, 0.0)) / total_w
            for key in DEFAULT_WEIGHTS.keys()
        }

    bluff_score = (
        timing * norm_weights["timing_tell"]
        + size * norm_weights["size_tell"]
        + formulaic * norm_weights["formulaic_tell"]
        + pattern * norm_weights["pattern_tell"]
    )
    is_bluff = bluff_score > 0.6

    return BluffSignals(
        timing_tell=timing,
        size_tell=size,
        formulaic_tell=formulaic,
        pattern_tell=pattern,
        bluff_score=bluff_score,
        is_bluff=is_bluff,
    )


def analyze_from_sim(
    sim: Any,
    seller_message: str,
    weights: Optional[Mapping[str, float]] = None,
) -> BluffSignals:
    """
    Convenience wrapper to analyze a CraigslistSellerSim instance directly.

    This avoids circular imports by using a structural interface only:
    the `sim` object must expose `profile`, `thread_history`, and `turn`.
    """
    return analyze_bluff(
        seller_profile=getattr(sim, "profile"),
        thread_history=getattr(sim, "thread_history"),
        seller_message=seller_message,
        turn=getattr(sim, "turn"),
        weights=weights,
    )


def _timing_tell(seller_profile: Mapping[str, Any], turn: int) -> float:
    """
    Timing tell — did a strong floor assertion arrive suspiciously fast?

    Heuristic: sellers with response_speed == "fast" who make this claim in
    the first few turns score as a timing tell.
    """
    speed = str(seller_profile.get("response_speed", "")).lower()
    if speed == "fast" and turn <= 3:
        return 1.0
    if speed == "fast":
        return 0.7
    return 0.0


def _size_tell(seller_message: str) -> float:
    """
    Size tell — concession lands cleanly on a round number (anchoring).
    """
    price = _extract_price(seller_message)
    if price is None:
        return 0.0

    if price % 10 == 0:
        return 1.0
    if price % 5 == 0:
        return 0.7
    return 0.0


def _formulaic_tell(seller_message: str) -> float:
    """
    Formulaic tell — canned bluff phrases like "final offer".
    """
    text = seller_message.lower()
    for phrase in FORMULAIC_PHRASES:
        if phrase in text:
            return 1.0
    return 0.0


def _pattern_tell(
    thread_history: Sequence[Mapping[str, Any]],
    seller_message: str,
) -> float:
    """
    Pattern tell — behavior shift vs. earlier thread history.

    Heuristics:
    - There is at least one prior seller message with a numeric price.
    - Those prior messages show at least one concession (price decreasing).
    - The current message is formulaic (strong finality language).
    """
    seller_msgs: List[str] = [
        str(entry["seller"])
        for entry in thread_history
        if "seller" in entry
    ]
    if not seller_msgs:
        return 0.0

    # Exclude the current message if it's already in thread_history.
    prior_msgs = seller_msgs[:-1] if seller_msgs[-1].strip() == seller_message.strip() else seller_msgs
    if not prior_msgs:
        return 0.0

    prev_prices: List[int] = []
    for msg in prior_msgs:
        price = _extract_price(msg)
        if price is not None:
            prev_prices.append(price)

    if len(prev_prices) < 1:
        return 0.0

    concessions = 0
    for i in range(1, len(prev_prices)):
        if prev_prices[i] < prev_prices[i - 1]:
            concessions += 1

    if concessions == 0:
        return 0.0

    if _formulaic_tell(seller_message) <= 0.0:
        return 0.0

    return 1.0


def _extract_price(text: str) -> Optional[int]:
    """
    Extract the last integer that looks like a price from the text.
    """
    matches = re.findall(r"\$?\s*(\d+)", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None

