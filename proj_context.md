# ArbitrAgent — Project Context
**Read this file at the start of every session. Do not modify it.**
**After completing your session, update `session_progress.md` with your session number and what you built.**

---

## What We Are Building

**ArbitrAgent** is a curriculum-trained negotiation agent that autonomously executes multi-route arbitrage on simulated Craigslist-style markets. It starts with a cash budget ($20), identifies high-value items, simultaneously opens negotiations across multiple buy candidates and downstream trade targets, and only commits capital once a confirmed profitable route is locked.

Built for the **OpenEnv Hackathon, March 7-8 2026** at Shack15, San Francisco.

**Submission deadline: Sunday March 8, 1:00 PM sharp.**

---

## ✅ Already Built — Do Not Rebuild

A teammate completed the following before the hackathon started. Every session must read this before touching any ML or environment code.

| Component | Details |
|-----------|---------|
| `/home/rayyan/Desktop/Play-gent/reward_model.pt` | DistilBERT fine-tuned on Diplomacy data, val loss 0.102 |
| `DiplomacyNegotiationEnv` | OpenEnv 0.2.1 compliant, inherits from real Env base class |
| `ContractorNegotiationEnv` | OpenEnv 0.2.1 compliant, inherits from real Env base class |
| `/home/rayyan/Desktop/Play-gent/selfplay_states.json` | 211,278 labeled Diplomacy game states |
| `/home/rayyan/Desktop/Play-gent/grpo_output/checkpoint-200/model.safetensors` | TinyLlama 1.1B, GRPO Phase 1 trained, reward curve -0.35 → +0.63 over 200 steps |

**Saturday only requires:** Phase 2 GRPO training (~1.5 hrs), agent loop, seller sims, and demo UI. The hard ML work is done.

---

Real negotiation data is private and will never exist as training data. We extract negotiation judgment from two games that together cover the complete negotiation skill surface:

- **Diplomacy** → multi-party coalition sequencing, strategic information reveals, long-horizon concession planning, stopping policy
- **Poker** → bluff detection, behavioral pattern reading, pressure calibration, EV reasoning, clean exits

**The combined skill neither game alone produces:** detecting a bluff AND immediately deploying coalition pressure at exactly that moment. That is the demo's proof of training.

The training pipeline implements this in three phases: Diplomacy (Phase 1, ✅ complete), Contractor negotiation as an intermediate bluff-detection layer (Phase 2, 🔲 MVP), and full Poker training on the IRC Poker dataset (Phase 3, 🔲 post-MVP). The pitch is true at MVP and becomes fully implemented at Phase 3.

---

## Repository Structure

```
arbitragent/
├── proj_context.md              # This file — never modify
├── session_progress.md          # Updated by each session
├── envs/
│   ├── diplomacy_env.py         # ✅ BUILT — DiplomacyNegotiationEnv (OpenEnv 0.2.1)
│   ├── contractor_env.py        # ✅ BUILT — ContractorNegotiationEnv (OpenEnv 0.2.1)
│   └── poker_env.py             # 🔲 POST-MVP — PokerNegotiationEnv (OpenEnv 0.2.1)
├── training/
│   ├── reward_model.py          # ✅ BUILT — DistilBERT reward model (val loss 0.102)
│   ├── checkpoints/             # 🔲 TODO — optional future consolidation of checkpoints
│   │   ├── phase2_final.pt      # 🔲 TODO — after Session B2
│   │   └── phase3_final.pt      # 🔲 POST-MVP — after Session B3
│   ├── data/                    # 🔲 TODO — optional future data subfolder
│   │   └── (see root-level files for existing data artifacts)
│   ├── train_phase1.py          # ✅ BUILT — GRPO on Diplomacy env (done, -0.35→+0.63)
│   ├── train_phase2.py          # 🔲 TODO — GRPO on Contractor env (Session B2)
│   ├── train_phase3.py          # 🔲 POST-MVP — GRPO on Poker env (Session B3)
│   └── arbitragent_colab.ipynb  # 🔲 TODO — End-to-end Colab notebook (Session B2)
├── agent/
│   ├── arbitragent.py           # Main agent orchestration loop (5 phases)
│   ├── route_graph.py           # Route graph: confirmed/soft/dead edges + scoring
│   └── bluff_detector.py        # Signal extraction: timing/size/formulaic/pattern tells
├── simulation/
│   ├── seller_sim.py            # CraigslistSellerSim — LLM-backed seller counterparts
│   ├── seller_profiles.py       # All 4 archetype profiles + listing library
│   └── scenario.py              # Demo scenario: which seller ghosts, when bluff triggers
├── demo/
│   ├── run_demo.py              # Entry point — takes budget, runs full agent loop
│   └── display.py               # Rich terminal output showing live negotiation threads
└── deploy/
    └── hf_spaces_app.py         # HuggingFace Spaces deployment wrapper
```

---

## Training Architecture

### MVP (Submit This)

```
Phase 1: Diplomacy Training                         ✅ COMPLETE
211,278 labeled Diplomacy game states
→ Reward model (DistilBERT) trained, val loss 0.102
→ GRPO training on TinyLlama 1.1B: 200 steps
→ Reward curve: -0.35 → +0.63
→ Checkpoint saved: `/home/rayyan/Desktop/Play-gent/grpo_output/checkpoint-200/model.safetensors`

Phase 2: Contractor Curriculum Training             🔲 TODO — Session B2
Contractor negotiation scenarios (false-floor, pressure calibration, timing tells)
→ Continue GRPO from phase1_final.pt — do NOT reinitialize weights
→ 200 additional steps
→ Bluff detection accuracy must improve on held-out test set
→ Save checkpoint: training/checkpoints/phase2_final.pt

MVP Model: TinyLlama 1.1B, Diplomacy + Contractor trained
```

### Post-MVP (If Time Allows — Phase 3)

```
Phase 3: Poker Curriculum Training                  🔲 POST-MVP — Session B3
IRC Poker Database (free, 10M+ hands, no collection needed)
→ Replay hands as negotiation scenarios
→ Map bet sizing → negotiation pressure
→ Map bluff/fold signals → position authenticity reads
→ Continue GRPO from phase2_final.pt — do NOT reinitialize weights
→ 200 additional steps
→ Reward: EV of outcome vs. EV of folding
→ Save checkpoint: training/checkpoints/phase3_final.pt

Full Model: TinyLlama 1.1B, Diplomacy + Contractor + Poker trained
```

**Build Phase 3 only after:** Phase 2 is complete, demo is running end-to-end, and submission checklist is green. Phase 3 makes the implementation match the pitch exactly — the story becomes true all the way down. Estimated time: ~2 hours to build PokerNegotiationEnv + ~1.5 hours training on DGX.

**Why curriculum order matters:** Diplomacy builds the multi-party strategic foundation. Contractor adds false-floor detection on top of that. Poker sharpens the bluff-reading layer with pure behavioral signal. Each phase builds on the last. Running them simultaneously or out of order causes catastrophic forgetting.

**Why TinyLlama 1.1B and not LLaMA 3.1 8B:** Training time. 8B on the DGX Spark would take 17–24 hours for two phases alone — the entire hackathon gone on training. TinyLlama 1.1B completes all three phases in ~5 hours total, with Phase 1 already done. Do not switch to 8B.

---

## Tech Stack (LOCKED)

| Component | Technology | Status |
|-----------|-----------|--------|
| Agent LLM | TinyLlama 1.1B (trained policy) | ✅ Phase 1 trained |
| Phase 1 Env | DiplomacyNegotiationEnv (OpenEnv 0.2.1) | ✅ Built |
| Phase 2 Env | ContractorNegotiationEnv (OpenEnv 0.2.1) | ✅ Built |
| Phase 3 Env | PokerNegotiationEnv (OpenEnv 0.2.1) | 🔲 Post-MVP |
| Poker Data | IRC Poker Database (free, 10M+ hands) | 🔲 Post-MVP |
| Reward Model | DistilBERT, val loss 0.102 | ✅ Built |
| RL Framework | TRL + GRPO | ✅ Phase 1 complete |
| Training Data | `/home/rayyan/Desktop/Play-gent/selfplay_states.json`, 211,278 states | ✅ Built |
| Seller Simulation | TinyLlama 1.1B with archetype system prompts | 🔲 Session C1 |
| Route Graph | NetworkX or custom dict-based | 🔲 Session A2 |
| Agent Loop | 5-phase orchestration | 🔲 Session A2 |
| Bluff Detector | 4-signal extractor | 🔲 Session A3 |
| Demo UI | Rich terminal display | 🔲 Session A4 |
| Experiment Tracking | Weights & Biases | ✅ Active |
| Deployment | HuggingFace Spaces + HF Model Hub | 🔲 Session A4 |
| Hardware | DGX Spark (all training + inference) | ✅ Available |
| Colab Notebook | End-to-end training script | 🔲 Session B2 |

---

## The Five-Phase Agent Loop

### Phase 1: Scout
- Query simulated listings for $15–$25 items
- Score each on: resale demand, trade liquidity, seller bluff probability
- Select top 3 buy candidates
- Open soft-inquiry negotiations with all 3 simultaneously

### Phase 2: Route Mapping
- For each candidate, identify 2-3 trade targets in $35–$80 range
- Open parallel trade-interest threads
- Build route graph — edges: Confirmed / Soft / Dead

### Phase 3: Pressure and Confirm
- Use downstream confirmations as upstream leverage
- Run bluff detection on seller responses
- Lock soft commits before committing capital
- Kill routes below confirmation probability threshold

### Phase 4: Route Scoring
```python
route_score = (confirmed_exit_value - entry_cost)
              × route_confirmation_probability
              × seller_reliability_score
# Kill if route_score < minimum_threshold
```

### Phase 5: Execute
- Pull trigger on highest scored confirmed route
- Complete downstream trade
- Log final value vs. starting budget

---

## The Four Seller Archetypes

| Archetype | Response Prob | Floor Behavior | Trade Openness | Demo Purpose |
|-----------|--------------|----------------|----------------|--------------|
| Motivated Seller | 0.90 | Real floor, honest | High | Shows clean close |
| Bluffer | 0.85 | Says "firm" with 30% room left | Medium | Shows poker layer catching tell |
| Ghoster | 0.35 | Never reaches floor | Low | Shows agent detecting dead route, pivoting |
| Trade-Curious | 0.80 | Cash-resistant, trade-open | Very High | Shows agent switching offer type |

### Bluff Detection Signals (all four must be checked)
1. **Timing tell** — response came in under 1 turn (prepared script, not genuine constraint)
2. **Size tell** — concession is a round number (anchoring, not real floor)
3. **Formulaic tell** — canned phrasing: "lowest I can go", "final offer", "can't go lower"
4. **Pattern tell** — behavior inconsistent with their earlier thread history

### The Critical Demo Inject
At ~60 seconds into the demo, the Bluffer seller says "this is my final offer" on the vintage camera at $30. This response contains all four tells. The trained model flags it, shows reasoning trace, and deploys coalition pressure: *"I have a trade offer from another seller that makes this less urgent for me — can you do $22?"* Seller concedes to $24. Route executes. Final value: $52 on $24 deployed = 2.2x.

**Baseline LLaMA accepts the $30 "final offer" at face value. The trained model doesn't. That gap is the proof.**

---

## Seller Profile Schema

```python
{
    "id": "seller_001",
    "item": "vintage film camera",
    "listing_price": 45,
    "floor": 28,              # hidden from agent
    "archetype": "bluffer",
    "bluff_room": 0.30,       # still has 30% room when says "final offer"
    "response_prob": 0.85,
    "response_speed": "fast", # fast | slow | flaky
    "trade_openness": 0.6,
    "personality": "Casual seller, slightly impatient. Texts in short bursts.",
    "tells": ["round numbers", "formulaic language", "too-fast response"]
}
```

### Response Turn Simulation
```python
RESPONSE_PROFILES = {
    "fast":  {"turns_to_respond": 1, "ghost_prob": 0.10},
    "slow":  {"turns_to_respond": 3, "ghost_prob": 0.30},
    "flaky": {"turns_to_respond": 2, "ghost_prob": 0.60},
}
```

---

## Hackathon Tracks Hit

| Track | How |
|-------|-----|
| Statement 1: Multi-Agent | Agent manages 9-12 simultaneous counterpart LLMs |
| Statement 2: Long-Horizon | Route-confirmation arc spans multiple rounds with full state tracking |
| Statement 4: Self-Improvement | Curriculum RL loop, two-phase measurable reward improvement |
| Statement 5: Wild Card | Autonomous capital deployment via confirmed route arbitrage |
| Halluminate $10k bonus | Agent managing multiple actors to discover and achieve the task |
| Fleet AI $10k bonus | Bluff detection layer as oversight agent scoring counterpart behavior |

---

## The Pitch (memorize this)

> "The most important negotiations of your life happen once. The person across the table has done it hundreds of times. The data to train AI on these conversations is sealed by law and will never exist. We found where that judgment already lives at massive scale: in Diplomacy, where millions of humans practiced multi-party coalition strategy, and in Poker, where millions more learned to read when someone's stated position is real versus a bluff. We trained on both — curriculum style — and built an agent that doesn't just know negotiation theory. It has internalized when to move, when to wait, and when the other side is lying about their floor. Then we gave it $20 and let it run."

---

## Judge Q&A (have these ready)

**"Couldn't you just prompt GPT-4 to do this?"**
GPT-4 knows negotiation tactics abstractly. It has no learned behavioral policy about *when* to deploy them. It hasn't lost thousands of negotiations by revealing coalition pressure too early. Our model has — and the reward curves are the proof.

**"Does game training actually transfer to real negotiation?"**
The structural isomorphism is direct. Coalition sequencing in Diplomacy is mechanically identical to sequential offer reveals in any multi-party negotiation. Bluff detection in contractor bidding scenarios — reading whether a contractor's stated floor is real — is mechanically identical to the same skill in any negotiation. We're not claiming domain transfer — we're claiming the cognitive mechanics are identical across surface vocabulary.

**"Why simulate instead of real Craigslist?"**
Craigslist has 6-hour response latency, no API, and one ghost kills a live demo. Our parameterized LLM counterparts replicate the four real seller archetypes we identified from Craigslist interaction patterns. The agent reads behavioral signals in real time exactly as it would with real sellers.

**"Why GRPO instead of PPO?"**
GRPO is more sample-efficient for language model fine-tuning and produces more stable training. It's the same algorithm DeepSeek-R1 used. Our Phase 1 reward curve — -0.35 to +0.63 over 200 steps — is the evidence it works.

---

## Submission Requirements (do not miss any)

- [x] Reward model on HF Model Hub — **already built, just needs uploading**
- [x] Phase 1 reward curves (Diplomacy GRPO, -0.35 → +0.63) — **already exists, needs clean plot**
- [ ] Both envs live on HuggingFace Spaces (OpenEnv 0.2.1)
- [ ] Phase 2 reward curves (Contractor GRPO, climbing over 200 steps)
- [ ] Colab notebook: full curriculum training loop, runs in one click
- [ ] Side-by-side: trained vs baseline on same negotiation
- [ ] Full ArbitrAgent demo: $20 → autonomous route execution → final value
- [ ] 1-minute YouTube demo video (live agent run, no slides)
- [ ] Public GitHub repo with README
- [ ] Submit at cerebralvalley.ai by Sunday 1:00 PM

---

*This file is the ground truth for the project. If anything in session_progress.md conflicts with this file, this file wins on architecture and thesis. session_progress.md wins on what has already been built.*

**Handoff:** For a full breakdown of what has been built and what remains, give Claude both this file and `session_progress.md` (see the "Handoff for Claude" section at the end of session_progress.md).
