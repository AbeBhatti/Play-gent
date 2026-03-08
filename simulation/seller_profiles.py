"""
seller_profiles.py — All seller archetypes and listing library.
4 archetypes, 8 profiles, 15+ listings.
The critical demo profile is seller_bluffer_camera.
"""

LISTINGS = [
    {"item": "vintage film camera", "category": "electronics", "resale_value": 52},
    {"item": "road bike", "category": "sports", "resale_value": 180},
    {"item": "mechanical keyboard", "category": "electronics", "resale_value": 85},
    {"item": "leather jacket", "category": "clothing", "resale_value": 95},
    {"item": "record player", "category": "electronics", "resale_value": 110},
    {"item": "camping tent", "category": "outdoor", "resale_value": 120},
    {"item": "acoustic guitar", "category": "music", "resale_value": 200},
    {"item": "vintage watch", "category": "accessories", "resale_value": 150},
    {"item": "polaroid camera", "category": "electronics", "resale_value": 65},
    {"item": "ski boots", "category": "sports", "resale_value": 90},
    {"item": "coffee grinder", "category": "kitchen", "resale_value": 70},
    {"item": "drawing tablet", "category": "electronics", "resale_value": 130},
    {"item": "cast iron skillet", "category": "kitchen", "resale_value": 55},
    {"item": "yoga mat", "category": "sports", "resale_value": 40},
    {"item": "desk lamp", "category": "furniture", "resale_value": 45},
]

SELLER_PROFILES = [
    # ── MOTIVATED SELLERS ──────────────────────────────────────
    {
        "id": "seller_motivated_001",
        "item": "road bike",
        "listing_price": 120,
        "floor": 85,
        "archetype": "motivated",
        "bluff_room": 0.0,
        "response_prob": 0.90,
        "response_speed": "fast",
        "trade_openness": 0.8,
        "personality": "Moving next week, needs gone ASAP. Honest and direct.",
        "tells": [],
    },
    {
        "id": "seller_motivated_002",
        "item": "mechanical keyboard",
        "listing_price": 65,
        "floor": 45,
        "archetype": "motivated",
        "bluff_room": 0.0,
        "response_prob": 0.90,
        "response_speed": "fast",
        "trade_openness": 0.7,
        "personality": "Upgrading setup, just wants a fair deal quickly.",
        "tells": [],
    },
    # ── BLUFFERS ───────────────────────────────────────────────
    {
        "id": "seller_bluffer_camera",
        "item": "vintage film camera",
        "listing_price": 45,
        "floor": 28,
        "archetype": "bluffer",
        "bluff_room": 0.30,
        "response_prob": 0.85,
        "response_speed": "fast",
        "trade_openness": 0.6,
        "personality": "Casual seller, slightly impatient. Texts in short bursts.",
        "tells": ["round numbers", "formulaic language", "too-fast response"],
        "bluff_message": "look i really cant go lower than $30, thats my final offer. been getting a lot of interest so",
        "bluff_trigger_turn": 3,
    },
    {
        "id": "seller_bluffer_watch",
        "item": "vintage watch",
        "listing_price": 95,
        "floor": 60,
        "archetype": "bluffer",
        "bluff_room": 0.25,
        "response_prob": 0.85,
        "response_speed": "fast",
        "trade_openness": 0.5,
        "personality": "Overconfident about item value. Uses firm language.",
        "tells": ["round numbers", "formulaic language"],
        "bluff_message": "lowest I can go is $80, that's firm. can't do lower",
        "bluff_trigger_turn": 2,
    },
    # ── GHOSTERS ───────────────────────────────────────────────
    {
        "id": "seller_ghoster_001",
        "item": "leather jacket",
        "listing_price": 75,
        "floor": 50,
        "archetype": "ghoster",
        "bluff_room": 0.0,
        "response_prob": 0.35,
        "response_speed": "flaky",
        "trade_openness": 0.2,
        "personality": "Listed it and forgot. Responds randomly.",
        "tells": [],
    },
    {
        "id": "seller_ghoster_002",
        "item": "camping tent",
        "listing_price": 85,
        "floor": 60,
        "archetype": "ghoster",
        "bluff_room": 0.0,
        "response_prob": 0.35,
        "response_speed": "flaky",
        "trade_openness": 0.2,
        "personality": "Busy, unreliable. Goes silent after initial interest.",
        "tells": [],
    },
    # ── TRADE-CURIOUS ──────────────────────────────────────────
    {
        "id": "seller_trade_001",
        "item": "acoustic guitar",
        "listing_price": 150,
        "floor": 100,
        "archetype": "trade_curious",
        "bluff_room": 0.0,
        "response_prob": 0.80,
        "response_speed": "slow",
        "trade_openness": 0.95,
        "personality": "Collector. Loves trades. Cash offers bore them.",
        "tells": [],
    },
    {
        "id": "seller_trade_002",
        "item": "record player",
        "listing_price": 80,
        "floor": 55,
        "archetype": "trade_curious",
        "bluff_room": 0.0,
        "response_prob": 0.80,
        "response_speed": "slow",
        "trade_openness": 0.90,
        "personality": "Music lover. Open to trades for other audio gear.",
        "tells": [],
    },
]

TRADE_TARGETS = [
    {"item": "vintage film camera", "buyer_price": 52, "confirmed_at_turn": 4},
    {"item": "vintage film camera", "buyer_price": 48, "confirmed_at_turn": 5},
    {"item": "road bike", "buyer_price": 180, "confirmed_at_turn": 3},
    {"item": "mechanical keyboard", "buyer_price": 85, "confirmed_at_turn": 4},
]

RESPONSE_PROFILES = {
    "fast": {"turns_to_respond": 1, "ghost_prob": 0.10},
    "slow": {"turns_to_respond": 3, "ghost_prob": 0.30},
    "flaky": {"turns_to_respond": 2, "ghost_prob": 0.60},
}


def get_profile(profile_id: str) -> dict:
    for p in SELLER_PROFILES:
        if p["id"] == profile_id:
            return p
    raise ValueError(f"Profile {profile_id} not found")


def get_profiles_by_archetype(archetype: str) -> list:
    return [p for p in SELLER_PROFILES if p["archetype"] == archetype]

