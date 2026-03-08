import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("training/checkpoints/phase2_final/checkpoint-200/trainer_state.json") as f:
    state = json.load(f)

steps = [e["step"] for e in state["log_history"]]
rewards = [e["reward"] for e in state["log_history"]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, rewards, color="#4C72B0", linewidth=2.5, marker="o", markersize=4)
ax.axhline(y=rewards[0], color="gray", linestyle="--", alpha=0.5, label=f"Start: {rewards[0]:.3f}")
ax.axhline(y=rewards[-1], color="#2ca02c", linestyle="--", alpha=0.5, label=f"End: {rewards[-1]:.3f}")
ax.fill_between(steps, rewards, rewards[0], alpha=0.1, color="#4C72B0")
ax.set_xlabel("Training Step", fontsize=13)
ax.set_ylabel("Mean Reward", fontsize=13)
ax.set_title("ArbitrAgent Phase 2 GRPO Training\nContractor Curriculum (Human Imitation)", fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training/phase2_reward_curve.png", dpi=150)
print(f"Saved. Reward: {rewards[0]:.3f} → {rewards[-1]:.3f} over {steps[-1]} steps")
