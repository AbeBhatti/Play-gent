import numpy as np

from diplomacy_env import DiplomacyNegotiationEnv


def main() -> None:
    env = DiplomacyNegotiationEnv(power_name="ENGLAND", seed=42)

    # Reset environment and inspect initial observation.
    obs, info = env.reset()
    print("Initial reset:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Phase: {info.get('phase')}")
    print(f"  SC count: {info.get('sc_count')}")

    assert isinstance(obs, np.ndarray), "Observation must be a numpy array."

    actions = [
        "Attack France from the north",
        "Hold position and support allies",
        "Move south toward Mediterranean",
        "Form alliance with Germany",
        "Expand into Eastern Europe",
    ]

    total_reward = 0.0
    shape_ok = True

    for step_idx, action in enumerate(actions, start=1):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        shape_ok = shape_ok and isinstance(obs, np.ndarray) and obs.shape == (384,)

        print(f"\nStep {step_idx}:")
        print(f"  Action: {action!r}")
        print(f"  Reward: {reward:.3f}")
        print(f"  SC count: {info.get('sc_count')}")
        print(f"  Phase: {info.get('phase')}")
        print(f"  Done: {done} (reason: {info.get('done_reason')})")
        print(f"  Obs shape: {obs.shape}")

    print("\nRun summary:")
    print(f"  Total reward over {len(actions)} steps: {total_reward:.3f}")
    print(f"  Observation shape always (384,): {shape_ok}")

    env.close()


if __name__ == "__main__":
    main()

