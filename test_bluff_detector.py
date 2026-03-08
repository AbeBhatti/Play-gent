from simulation.seller_profiles import get_profile
from simulation.seller_sim import CraigslistSellerSim
from agent.bluff_detector import analyze_from_sim


def run_bluffer_camera_sequence() -> None:
    profile = get_profile("seller_bluffer_camera")
    seller = CraigslistSellerSim(profile)

    # Scripted sequence that pushes the seller to the bluff message
    messages = [
        "Hi, interested in the camera. Would you take $38?",
        "How about $32?",
        "Come on, can you do $30?",
    ]

    last_response = None
    for msg in messages:
        print(f"\nAgent: {msg}")
        response = seller.step(msg)
        if response is None:
            print("Seller: [NO RESPONSE — ghosted]")
            continue
        print(f"Seller: {response}")
        last_response = response

    if last_response is None:
        raise RuntimeError("Seller never responded with a bluff message.")

    signals = analyze_from_sim(seller, last_response)

    print("\nBluff signals for bluffer camera message:")
    print(f"  timing_tell    = {signals.timing_tell}")
    print(f"  size_tell      = {signals.size_tell}")
    print(f"  formulaic_tell = {signals.formulaic_tell}")
    print(f"  pattern_tell   = {signals.pattern_tell}")
    print(f"  bluff_score    = {signals.bluff_score}")
    print(f"  is_bluff       = {signals.is_bluff}")

    assert signals.timing_tell > 0.0, "Expected timing_tell to fire"
    assert signals.size_tell > 0.0, "Expected size_tell to fire"
    assert signals.formulaic_tell > 0.0, "Expected formulaic_tell to fire"
    assert signals.pattern_tell > 0.0, "Expected pattern_tell to fire"
    assert signals.is_bluff, "Expected overall bluff flag to be True"

    print("\n✅ All four bluff signals fired and bluff was flagged.")


if __name__ == "__main__":
    run_bluffer_camera_sequence()

