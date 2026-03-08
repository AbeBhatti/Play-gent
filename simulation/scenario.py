"""
scenario.py — Deterministic demo scenario.
Same run every time. Seeds all RNG.
Produces the 90-second ArbitrAgent demo.
"""

import random

random.seed(42)

from simulation.seller_profiles import get_profile, TRADE_TARGETS
from simulation.seller_sim import CraigslistSellerSim


def get_scenario():
    """
    Returns the standard demo scenario.
    Buy layer: 3 sellers (Motivated, Bluffer=camera, Ghoster)
    Trade layer: 4 trade targets
    Everything seeded for deterministic replay.
    """
    sellers = [
        CraigslistSellerSim(get_profile("seller_motivated_001")),
        CraigslistSellerSim(get_profile("seller_bluffer_camera")),
        CraigslistSellerSim(get_profile("seller_ghoster_001")),
    ]
    trade_targets = TRADE_TARGETS
    return sellers, trade_targets


def get_extended_scenario():
    """
    Extended demo scenario:
    - 5 sellers including the two additional profiles:
      * seller_aggressive_001 (bluffer, vintage watch)
      * seller_trader_001 (trade-curious, mountain bike)
    - Uses the same trade targets; the demo loop controls turn count.
    """
    sellers = [
        CraigslistSellerSim(get_profile("seller_motivated_001")),
        CraigslistSellerSim(get_profile("seller_bluffer_camera")),
        CraigslistSellerSim(get_profile("seller_ghoster_001")),
        CraigslistSellerSim(get_profile("seller_aggressive_001")),
        CraigslistSellerSim(get_profile("seller_trader_001")),
    ]
    trade_targets = TRADE_TARGETS
    return sellers, trade_targets

