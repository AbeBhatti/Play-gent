---
title: ArbitrAgent
emoji: 🤝
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: deploy/hf_spaces_app.py
pinned: false
---

# ArbitrAgent

**A curriculum-trained negotiation agent that autonomously executes multi-route arbitrage on simulated Craigslist-style markets.**

ArbitrAgent starts with a cash budget (e.g. **$20**), identifies high-value items, opens negotiations across multiple buy candidates and downstream trade targets, and commits capital only once a confirmed profitable route is locked. It combines skills from **Diplomacy** (multi-party coalition sequencing, strategic reveals, concession planning) and **Poker-style bluff detection** (reading when a stated floor is real vs. a bluff) — trained on public game data because real negotiation data is private and will never exist at scale.

Built for the **OpenEnv Hackathon, March 7–8 2026** (Shack15, San Francisco).

---

## What It Is

- **Agent:** Five-phase orchestration loop (Scout → Route Mapping → Pressure & Confirm → Route Scoring → Execute) that manages 9–12 simultaneous counterpart "sellers" and trade targets.
- **Policy:** TinyLlama 1.1B fine-tuned with **GRPO** (Group Relative Policy Optimization) in a curriculum over three environments.
- **Reward model:** DistilBERT fine-tuned on labeled Diplomacy game outcomes (validation loss 0.102).
- **Demo:** Simulated Craigslist-style sellers (four archetypes: Motivated, Bluffer, Ghoster, Trade-Curious) with deterministic bluff injection; the agent detects bluffs and deploys coalition pressure instead of accepting "final offer" at face value.

The thesis: *negotiation judgment can be learned from games* — Diplomacy for coalition strategy, contractor/poker scenarios for bluff detection — and transferred to an agent that decides when to move, when to wait, and when the other side is lying about their floor.

---

## Curriculum Training Pipeline

Training is **curriculum-ordered** to avoid catastrophic forgetting:

| Phase | Environment | Data / Setup | Goal |
|-------|-------------|--------------|------|
| **Phase 1** | `DiplomacyNegotiationEnv` | 211,278 labeled Diplomacy game states | Multi-party strategy, coalition language, long-horizon planning. **Done:** reward curve −0.35 → +0.63 over 200 steps. |
| **Phase 2** | `HumanImitationEnv` | Same Diplomacy states, human-outcome reward | Continue from Phase 1 checkpoint; human-imitation + bluff/pressure vocabulary. Saves `training/checkpoints/phase2_final` and `training/phase2_reward_curve.png`. |
| **Phase 3** (post-MVP) | `PokerNegotiationEnv` | IRC Poker–style scenarios | Bluff/fold signals, EV reasoning; continues from Phase 2. |

- **Phase 1** trains on `DiplomacyNegotiationEnv` (OpenEnv 0.2.1); checkpoint: `grpo_output/checkpoint-200`.
- **Phase 2** continues from that checkpoint on `HumanImitationEnv` (no weight reinit); 200 steps, reward curve and checkpoint under `training/`.
- Model: **TinyLlama 1.1B** throughout (chosen for training speed and demo inference).

---

## The Three Environments

All are **OpenEnv 0.2.1** compatible (inherit from `openenv.env.Env`).

| Environment | File | Role |
|-------------|------|------|
| **DiplomacyNegotiationEnv** | `envs/diplomacy_env.py` | Phase 1. Wraps `diplomacy.Game`; observation = 384-dim MiniLM embedding of game state; action = free-form text. Rewards track supply-center outcomes and strategic behavior. |
| **ContractorNegotiationEnv** | `envs/contractor_env.py` | Contractor bidding: agent negotiates with multiple contractors to achieve lowest price. Trains false-floor detection, pressure calibration, timing tells — same cognitive mechanics as bluff detection in negotiation. |
| **HumanImitationEnv** | `envs/human_imitation_env.py` | Phase 2. Samples real human Diplomacy states from `selfplay_states.json`; agent must mimic human decisions; reward = alignment with recorded human outcome. |

Smoke test all three:

```bash
python test_all_envs.py
```

---

## BluffDetector

The **BluffDetector** (`agent/bluff_detector.py`) extracts four behavioral signals from seller messages and thread history:

| Signal | Meaning |
|--------|--------|
| **Timing tell** | Response in under one turn → prepared script, not genuine constraint. |
| **Size tell** | Concession is a round number → anchoring, not real floor. |
| **Formulaic tell** | Canned phrases: "lowest I can go", "final offer", "can't go lower". |
| **Pattern tell** | Behavior inconsistent with earlier thread history. |

- Output: `BluffSignals` (per-signal scores + weighted `bluff_score`, plus `is_bluff` when `bluff_score > 0.6`).
- Used in the demo to highlight when the Bluffer seller hits the canonical "final offer" line and to drive coalition-pressure messaging instead of accepting the stated floor.

Run the bluff-detector test (scripted sequence through the camera Bluffer to the canonical bluff message):

```bash
python test_bluff_detector.py
```

---

## The $20 → $52 Demo Story

The demo is designed to show the trained behavior in ~90 seconds:

1. **Setup:** Agent starts with **$20**; scenario seeds three sellers (Motivated, Bluffer [vintage camera], Ghoster) and four trade targets.
2. **Scout:** Agent scores buy candidates (resale demand, trade liquidity, bluff probability) and opens soft inquiries with the top three.
3. **Route mapping:** For each candidate, 2–3 trade targets in the $35–$80 range; route graph built with soft/confirmed/dead edges.
4. **Pressure & confirm:** At ~60 seconds, the **Bluffer** seller says: *"look i really cant go lower than $30, thats my final offer. been getting a lot of interest so"* — this triggers **all four** bluff tells. The agent **flags the bluff**, shows reasoning, and deploys coalition pressure: *"I have a trade offer from another seller that makes this less urgent for me — can you do $22?"* Seller concedes to **$24**.
5. **Execute:** Agent commits on the best confirmed route. **Final value ≈ $52** on $24 deployed → **~2.2× return**.

**Baseline LLaMA accepts the $30 "final offer". The trained agent does not.** That gap is the proof of the curriculum.

Run the Rich terminal demo (with optional JSON log):

```bash
python demo/run_demo.py --budget 20 --sleep 0.7
python demo/run_demo.py --budget 20 --log-path demo_log.json
```

---

## Reward Curves

- **Phase 1 (Diplomacy GRPO):** Reward moves from **−0.35 to +0.63** over 200 steps. Checkpoint: `grpo_output/checkpoint-200` (TinyLlama 1.1B).
- **Phase 2 (Human Imitation):** Training script logs rewards and writes **`training/phase2_reward_curve.png`** and checkpoint to `training/checkpoints/phase2_final`.

(Phase 1 curve can be reproduced from the Phase 1 training run; Phase 2 curve is generated by `training/train_phase2.py`.)

---

## Tracks Hit

| Track | How ArbitrAgent addresses it |
|-------|------------------------------|
| **Multi-Agent** | Agent coordinates 9–12 simultaneous counterpart LLMs (sellers + trade targets). |
| **Long-Horizon** | Route-confirmation arc spans multiple rounds with full state tracking across threads. |
| **Self-Improvement** | Curriculum RL: two-phase (MVP) measurable reward improvement (Phase 1 → Phase 2). |
| **Wild Card** | Autonomous capital deployment via confirmed-route arbitrage ($20 → execution → final value). |
| **Halluminate $10k** | Agent managing multiple actors to discover and achieve the task. |
| **Fleet AI $10k** | Bluff detection as oversight: scoring counterpart behavior and triggering coalition pressure. |

---

## How to Run

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd Play-gent
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Additional deps for training (GRPO, OpenEnv, Diplomacy, etc.) — install as needed:

```bash
pip install openenv diplomacy transformers trl datasets matplotlib
```

### 3. Smoke-test the three environments

```bash
python test_all_envs.py
```

### 4. Test the BluffDetector (camera Bluffer to canonical bluff message)

```bash
python test_bluff_detector.py
```

### 5. Test seller simulation (all four archetypes)

```bash
python test_seller_sim.py
```

### 6. Run the agent loop (headless, no Rich UI)

```bash
PYTHONPATH=. python -m agent.arbitragent
```

### 7. Run the full Rich demo ($20 budget, ~90s with default sleep)

```bash
PYTHONPATH=. python demo/run_demo.py --budget 20 --sleep 0.7
```

Optional: save structured JSON log:

```bash
PYTHONPATH=. python demo/run_demo.py --budget 20 --sleep 0.7 --log-path demo_log.json
```

### 8. Phase 2 training (from Phase 1 checkpoint)

Requires Phase 1 checkpoint at `grpo_output/checkpoint-200` and data at `training/data/selfplay_states.json`:

```bash
PYTHONPATH=. python training/train_phase2.py
```

Output: `training/checkpoints/phase2_final/`, `training/phase2_reward_curve.png`.

---

## Colab Notebook

End-to-end curriculum training (Phase 1 + Phase 2 GRPO, reward curves, side-by-side inference, BluffDetector) in one notebook:

- **`training/arbitragent_colab.ipynb`** — Run on Colab (e.g. T4 GPU); clone repo, install deps, run all cells.

---

## Repository Layout

```
agent/           # ArbitrAgent loop, route graph, bluff detector
demo/            # run_demo.py (Rich demo), display.py
envs/            # DiplomacyNegotiationEnv, ContractorNegotiationEnv, HumanImitationEnv
simulation/      # Seller profiles, CraigslistSellerSim, scenario (get_scenario)
training/        # train_phase2.py, arbitragent_colab.ipynb, data/, checkpoints/
test_*.py        # test_all_envs, test_bluff_detector, test_seller_sim, etc.
requirements.txt
README.md
```

---

## License & Acknowledgments

- Diplomacy game engine: [diplomacy](https://github.com/diplomacy/diplomacy).
- OpenEnv: [OpenEnv](https://github.com/openenv) for the environment interface.
- Built for the OpenEnv Hackathon, March 7–8 2026.
