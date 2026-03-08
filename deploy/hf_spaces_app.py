"""
HuggingFace Spaces deployment — Gradio app for all three OpenEnv 0.2.1 environments.

Tabs: DiplomacyNegotiationEnv, ContractorNegotiationEnv, HumanImitationEnv.
Each tab: current state, action input, submit button, reward output.
"""

import sys
from pathlib import Path

# Add repo root so we can import envs
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from envs.diplomacy_env import DiplomacyNegotiationEnv
from envs.contractor_env import ContractorNegotiationEnv
from envs.human_imitation_env import HumanImitationEnv


# ---------- Diplomacy tab ----------
def diplomacy_reset(state):
    if state is None or state.get("env") is None:
        env = DiplomacyNegotiationEnv(seed=42)
        state = {"env": env, "state_text": "", "last_reward": None, "last_done": False, "last_info": None}
    env = state["env"]
    obs, info = env.reset()
    state_text = env.render()
    state["state_text"] = state_text
    state["last_reward"] = None
    state["last_done"] = False
    state["last_info"] = info
    info_str = " | ".join(f"{k}={v}" for k, v in info.items())
    return state, state_text, f"Reset. Info: {info_str}", ""


def diplomacy_step(state, action):
    if state is None or state.get("env") is None:
        return state, "Click **Reset** to start an episode.", "No env. Click Reset.", ""
    env = state["env"]
    action = action or "(no action)"
    obs, reward, done, info = env.step(action)
    state_text = env.render()
    state["state_text"] = state_text
    state["last_reward"] = reward
    state["last_done"] = done
    state["last_info"] = info
    info_str = " | ".join(f"{k}={v}" for k, v in info.items())
    reward_str = f"Reward: {reward:.3f}\nDone: {done}\nInfo: {info_str}"
    return state, state_text, reward_str, ""


# ---------- Contractor tab ----------
def contractor_reset(state):
    if state is None or state.get("env") is None:
        env = ContractorNegotiationEnv(seed=42)
        state = {"env": env, "state_text": "", "last_reward": None, "last_done": False, "last_info": None}
    env = state["env"]
    obs, info = env.reset()
    state_text = env.render()
    state["state_text"] = state_text
    state["last_reward"] = None
    state["last_done"] = False
    state["last_info"] = info
    info_str = " | ".join(f"{k}={v}" for k, v in info.items())
    return state, state_text, f"Reset. Info: {info_str}", ""


def contractor_step(state, action):
    if state is None or state.get("env") is None:
        return state, "Click **Reset** to start an episode.", "No env. Click Reset.", ""
    env = state["env"]
    action = action or "(no action)"
    obs, reward, done, info = env.step(action)
    state_text = env.render()
    state["state_text"] = state_text
    state["last_reward"] = reward
    state["last_done"] = done
    state["last_info"] = info
    info_str = " | ".join(f"{k}={v}" for k, v in info.items())
    reward_str = f"Reward: {reward:.3f}\nDone: {done}\nInfo: {info_str}"
    return state, state_text, reward_str, ""


# ---------- Human Imitation tab ----------
def _human_imitation_env():
    data_path = ROOT / "training" / "data" / "selfplay_states.json"
    if not data_path.exists():
        data_path = ROOT / "training" / "data" / "selfplay_states_test.json"
    return HumanImitationEnv(data_path=str(data_path), seed=42)


def human_imitation_reset(state):
    try:
        if state is None or state.get("env") is None:
            env = _human_imitation_env()
            state = {"env": env, "state_text": "", "last_reward": None, "last_done": False, "last_info": None}
        env = state["env"]
        obs, info = env.reset()
        state_text = env.render()
        state["state_text"] = state_text
        state["last_reward"] = None
        state["last_done"] = False
        state["last_info"] = info
        info_str = " | ".join(f"{k}={v}" for k, v in info.items())
        return state, state_text, f"Reset. Info: {info_str}", ""
    except FileNotFoundError as e:
        return state, f"Data file not found: {e}. Ensure `training/data/selfplay_states.json` (or selfplay_states_test.json) exists.", "", ""
    except Exception as e:
        return state, f"Error: {e}", "", ""


def human_imitation_step(state, action):
    if state is None or state.get("env") is None:
        return state, "Click **Reset** to start an episode.", "No env. Click Reset.", ""
    env = state["env"]
    action = action or "(no action)"
    obs, reward, done, info = env.step(action)
    state_text = env.render()
    state["state_text"] = state_text
    state["last_reward"] = reward
    state["last_done"] = done
    state["last_info"] = info
    info_str = " | ".join(f"{k}={v}" for k, v in info.items())
    reward_str = f"Reward: {reward:.3f}\nDone: {done}\nInfo: {info_str}"
    return state, state_text, reward_str, ""


# ---------- Gradio UI ----------
def build_ui():
    with gr.Blocks(title="ArbitrAgent — OpenEnv 0.2.1 Environments", theme=gr.themes.Soft()) as app:
        gr.Markdown("# ArbitrAgent — OpenEnv 0.2.1 Demo\nThree negotiation environments. Use **Reset** to start, type an action, then **Submit**.")

        with gr.Tabs():
            # ---- Diplomacy ----
            with gr.Tab("DiplomacyNegotiationEnv"):
                dip_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=2):
                        dip_state_display = gr.Textbox(
                            label="Current state",
                            value="Click Reset to start.",
                            lines=18,
                            max_lines=25,
                            interactive=False,
                        )
                        dip_action = gr.Textbox(
                            label="Action (natural language strategic intent)",
                            placeholder="e.g. Propose alliance with France and move fleet to support attack on Germany.",
                            lines=2,
                        )
                        with gr.Row():
                            dip_submit_btn = gr.Button("Submit", variant="primary")
                            dip_reset_btn = gr.Button("Reset")
                    with gr.Column(scale=1):
                        dip_reward_display = gr.Textbox(
                            label="Reward / Info",
                            value="",
                            lines=10,
                            interactive=False,
                        )
                dip_reset_btn.click(
                    diplomacy_reset,
                    inputs=[dip_state],
                    outputs=[dip_state, dip_state_display, dip_reward_display, dip_action],
                )
                dip_submit_btn.click(
                    diplomacy_step,
                    inputs=[dip_state, dip_action],
                    outputs=[dip_state, dip_state_display, dip_reward_display, dip_action],
                )

            # ---- Contractor ----
            with gr.Tab("ContractorNegotiationEnv"):
                con_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=2):
                        con_state_display = gr.Textbox(
                            label="Current state",
                            value="Click Reset to start.",
                            lines=18,
                            max_lines=25,
                            interactive=False,
                        )
                        con_action = gr.Textbox(
                            label="Action (natural language negotiation move)",
                            placeholder="e.g. I have competing offers; can you beat $8,000?",
                            lines=2,
                        )
                        with gr.Row():
                            con_submit_btn = gr.Button("Submit", variant="primary")
                            con_reset_btn = gr.Button("Reset")
                    with gr.Column(scale=1):
                        con_reward_display = gr.Textbox(
                            label="Reward / Info",
                            value="",
                            lines=10,
                            interactive=False,
                        )
                con_reset_btn.click(
                    contractor_reset,
                    inputs=[con_state],
                    outputs=[con_state, con_state_display, con_reward_display, con_action],
                )
                con_submit_btn.click(
                    contractor_step,
                    inputs=[con_state, con_action],
                    outputs=[con_state, con_state_display, con_reward_display, con_action],
                )

            # ---- Human Imitation ----
            with gr.Tab("HumanImitationEnv"):
                hum_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=2):
                        hum_state_display = gr.Textbox(
                            label="Current state",
                            value="Click Reset to start.",
                            lines=18,
                            max_lines=25,
                            interactive=False,
                        )
                        hum_action = gr.Textbox(
                            label="Action (natural language move + reasoning)",
                            placeholder="e.g. I will support France in the north and hold my southern centers.",
                            lines=2,
                        )
                        with gr.Row():
                            hum_submit_btn = gr.Button("Submit", variant="primary")
                            hum_reset_btn = gr.Button("Reset")
                    with gr.Column(scale=1):
                        hum_reward_display = gr.Textbox(
                            label="Reward / Info",
                            value="",
                            lines=10,
                            interactive=False,
                        )
                hum_reset_btn.click(
                    human_imitation_reset,
                    inputs=[hum_state],
                    outputs=[hum_state, hum_state_display, hum_reward_display, hum_action],
                )
                hum_submit_btn.click(
                    human_imitation_step,
                    inputs=[hum_state, hum_action],
                    outputs=[hum_state, hum_state_display, hum_reward_display, hum_action],
                )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch()
