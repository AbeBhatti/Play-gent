from simulation.seller_profiles import get_profile
from simulation.seller_sim import CraigslistSellerSim

print("=" * 60)
print("TESTING ALL 4 SELLER ARCHETYPES")
print("=" * 60)

test_cases = [
    (
        "seller_motivated_001",
        [
            "Hi, interested in the bike. Would you take $90?",
            "Could you do $85?",
            "Deal, I'll take it at $85",
        ],
    ),
    (
        "seller_bluffer_camera",
        [
            "Hi, interested in the camera. Would you take $38?",
            "How about $32?",
            "Come on, can you do $30?",
            "I have another offer at $28",
        ],
    ),
    (
        "seller_ghoster_001",
        [
            "Hi, interested in the jacket",
            "Still available?",
            "Hello?",
            "Last message",
        ],
    ),
    (
        "seller_trade_001",
        [
            "Would you take $120 cash?",
            "How about $110?",
            "I could trade you a vintage keyboard for it",
        ],
    ),
]

for profile_id, messages in test_cases:
    print(f"\n{'=' * 40}")
    print(f"Testing: {profile_id}")
    print(f"{'=' * 40}")
    seller = CraigslistSellerSim(get_profile(profile_id))
    for msg in messages:
        print(f"\nAgent: {msg}")
        response = seller.step(msg)
        if response:
            print(f"Seller: {response}")
        else:
            print("Seller: [NO RESPONSE — ghosted]")
        print(f"Status: {seller.status} | Offer: ${seller.current_offer}")
        if seller.is_dead():
            print(">>> ROUTE DEAD — agent should pivot")
            break

print("\n" + "=" * 60)
print("Seller sim test complete.")

