"""
HuggingFace Spaces deployment — Gradio app for ArbitrAgent.

Tab 1: ArbitrAgentEnv (unified env) — state, reward breakdown (accuracy / outcome / bluff), action, submit/reset.
Tab 2: Live Demo — Run Demo button streams run_demo.py output to textbox.
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr


# ---------- ArbitrAgentEnv (unified) ----------
def _unified_env():
    from envs.arbitragent_env import ArbitrAgentEnv
    data_path = ROOT / "training" / "data" / "selfplay_states.json"
    if not data_path.exists():
        data_path = ROOT / "training" / "data" / "selfplay_states_test.json"
    return ArbitrAgentEnv(data_path=str(data_path), seed=42)


def unified_reset(state):
    try:
        if state is None or state.get("env") is None:
            env = _unified_env()
            state = {"env": env, "state_text": "", "last_info": None}
        env = state["env"]
        obs, info = env.reset()
        state_text = env.render()
        state["state_text"] = state_text
        state["last_info"] = info
        breakdown = "accuracy: —  |  outcome: —  |  bluff: —"
        return state, state_text, breakdown, ""
    except FileNotFoundError as e:
        return state or {}, f"Data file not found: {e}", "Error", ""
    except Exception as e:
        return state or {}, f"Error: {e}", "Error", ""


def unified_step(state, action):
    try:
        if state is None or state.get("env") is None:
            return state, "Click **Reset** to start an episode.", "No env. Click Reset.", ""
        env = state["env"]
        action = action or "(no action)"
        obs, reward, done, info = env.step(action)
        state_text = env.render()
        state["state_text"] = state_text
        state["last_info"] = info
        acc = info.get("accuracy", 0)
        out = info.get("outcome", 0)
        blf = info.get("bluff", 0)
        total = info.get("total", reward)
        breakdown = f"accuracy: {acc:.3f}  |  outcome: {out:.3f}  |  bluff: {blf:.3f}  |  total: {total:.3f}\nDone: {done}"
        return state, state_text, breakdown, ""
    except Exception as e:
        return state, state.get("state_text", ""), f"Error: {e}", ""


def run_demo_cmd():
    """Run demo/run_demo.py and return combined stdout+stderr."""
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "demo" / "run_demo.py"), "--budget", "20", "--scenario", "standard_demo", "--sleep", "0.3"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=90,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out if out.strip() else "Demo finished (no output captured)."
    except subprocess.TimeoutExpired:
        return "Demo timed out after 90 seconds."
    except Exception as e:
        return f"Error running demo: {e}"


# ---------- Gradio UI ----------
def build_ui():
    with gr.Blocks(title="ArbitrAgent — Unified Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("# ArbitrAgent — Unified Negotiation Environment\nReset, then type an action and Submit. Reward breakdown: accuracy / outcome / bluff.")

        with gr.Tabs():
            with gr.Tab("ArbitrAgentEnv — Unified Negotiation Environment"):
                uni_state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=2):
                        uni_state_display = gr.Textbox(
                            label="Current state",
                            value="Click Reset to start.",
                            lines=18,
                            max_lines=25,
                            interactive=False,
                        )
                        uni_action = gr.Textbox(
                            label="Action (natural language move + reasoning)",
                            placeholder="e.g. I have a trade offer from another seller — can you do $26?",
                            lines=2,
                        )
                        with gr.Row():
                            uni_submit_btn = gr.Button("Submit", variant="primary")
                            uni_reset_btn = gr.Button("Reset")
                    with gr.Column(scale=1):
                        uni_reward_display = gr.Textbox(
                            label="Reward breakdown (accuracy / outcome / bluff)",
                            value="",
                            lines=10,
                            interactive=False,
                        )
                uni_reset_btn.click(
                    unified_reset,
                    inputs=[uni_state],
                    outputs=[uni_state, uni_state_display, uni_reward_display, uni_action],
                )
                uni_submit_btn.click(
                    unified_step,
                    inputs=[uni_state, uni_action],
                    outputs=[uni_state, uni_state_display, uni_reward_display, uni_action],
                )

            with gr.Tab("Live Demo"):
                gr.Markdown("Run the full 5-phase agent loop (budget $20, standard scenario). Output streams below (may take up to 90s).")
                demo_run_btn = gr.Button("Run Demo", variant="primary")
                demo_output = gr.Textbox(
                    label="Demo output",
                    value="",
                    lines=24,
                    interactive=False,
                )

                def run_and_show():
                    return run_demo_cmd()

                demo_run_btn.click(
                    run_and_show,
                    inputs=[],
                    outputs=[demo_output],
                )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
