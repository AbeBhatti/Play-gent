---
title: ArbitrAgent — OpenEnv 0.2.1 Environments
emoji: 🤝
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0"
app_file: hf_spaces_app.py
python_version: "3.10"
short_description: Diplomacy, Contractor, and Human Imitation negotiation envs (OpenEnv 0.2.1).
tags:
  - openenv
  - reinforcement-learning
  - negotiation
  - diplomacy
  - gradio
---

# ArbitrAgent — OpenEnv 0.2.1 Demo

Three **OpenEnv 0.2.1**-compatible negotiation environments in one Gradio app:

| Tab | Environment | Description |
|-----|-------------|-------------|
| **DiplomacyNegotiationEnv** | Multi-party Diplomacy game state; supply centers, phases, strategic position. |
| **ContractorNegotiationEnv** | Contractor bidding: budget, multiple contractors, urgency; natural-language negotiation. |
| **HumanImitationEnv** | Real human Diplomacy game states; reward aligned with human outcomes (Phase 2 training). |

## How to use

1. Open a tab for the environment you want.
2. Click **Reset** to start an episode (initial state appears).
3. Type a **natural-language action** in the text box.
4. Click **Submit** to step the environment; **Reward / Info** and the updated state are shown.
5. Repeat until the episode is done, then click **Reset** to start again.

## Requirements

- **OpenEnv 0.2.1** API: `reset()` → (observation, info), `step(action)` → (observation, reward, done, info), `render()` for state text.
- **HumanImitationEnv** needs `training/data/selfplay_states.json` (or `selfplay_states_test.json`) in the repo; if you deploy only the `deploy/` folder, include a data file or that tab will show an error until data is provided.

## Deploying this Space

- **Space root = this folder (`deploy/`)**: Keep `app_file: hf_spaces_app.py` and use `deploy/requirements.txt`. For HumanImitationEnv you must add `training/data/` (or a sample JSON) into the Space.
- **Space root = full repo**: Set `app_file: deploy/hf_spaces_app.py`, add `sys.path` or install the repo as a package so `envs.diplomacy_env`, `envs.contractor_env`, and `envs.human_imitation_env` are importable; include `training/data/selfplay_states.json` for HumanImitationEnv.

Part of **ArbitrAgent** for the OpenEnv Hackathon (March 7–8 2026).
