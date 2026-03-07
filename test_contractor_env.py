from contractor_env import ContractorNegotiationEnv

env = ContractorNegotiationEnv(n_contractors=5, budget=10000, seed=42)
obs, info = env.reset()

print("=" * 60)
print("INITIAL STATE")
print("=" * 60)
env.render()
print(f"Initial best offer: ${info['best_offer']:,}")

actions = [
    "I have a competing offer at $8,500 can you beat it",
    "My budget is tight I need you to come down further",
    "Another contractor just offered me $7,800",
    "I can sign today if you match $7,500",
    "Accept the best current offer"
]

total_reward = 0.0

for i, action in enumerate(actions):
    print("\n" + "=" * 60)
    print(f"STEP {i+1}: {action}")
    print("=" * 60)
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Reward:      {reward:.3f}")
    print(f"Best offer:  ${info['best_offer']:,}")
    print(f"Savings:     ${info['savings']:,} ({info['savings_pct']:.1f}%)")
    total_reward += reward
    if done:
        print("\n>>> Episode done!")
        break

print("\n" + "=" * 60)
print(f"FINAL TOTAL REWARD: {total_reward:.3f}")
print(f"FINAL SAVINGS:      ${info['savings']:,} ({info['savings_pct']:.1f}% off budget)")
print("=" * 60)