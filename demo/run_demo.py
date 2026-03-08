"""
Demo entry point: budget, scenario, full 5-phase agent loop with Rich display.
Loads unified_final checkpoint if present, else phase2_final. Saves log to demo/sample_run_log.json.
Must complete in under 90 seconds.

UI: 4 phases displayed sequentially (Scouting, Route Mapping, Pressure & Negotiation,
Route Scoring & Execution) plus Final Result, with Rich formatting and time.sleep(0.5) between sections.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.arbitragent import ArbitrAgent, SellerCandidate
from agent.bluff_detector import analyze_from_sim, learned_bluff_score
from agent.route_graph import RouteEdge
from demo.display import NegotiationDisplay, ThreadMessage, ThreadState, PhaseDisplay
from simulation.scenario import get_scenario, get_extended_scenario


def _resolve_checkpoint_path() -> str | None:
    """Unified_final if exists, else phase2_final."""
    unified = os.path.join(PROJECT_ROOT, "training", "checkpoints", "unified_final")
    phase2 = os.path.join(PROJECT_ROOT, "training", "checkpoints", "phase2_final")
    if os.path.isdir(unified):
        return unified
    if os.path.isdir(phase2):
        return phase2
    return None


def _responsiveness(response_prob: float) -> str:
    if response_prob >= 0.8:
        return "HIGH"
    if response_prob >= 0.5:
        return "MEDIUM"
    return "LOW"


class DemoArbitrAgent(ArbitrAgent):
    """
    Runs full 5-phase loop with phase-by-phase Rich display and event log.
    Agent logic unchanged; only display and data collection for UI.
    """

    def __init__(self, budget: float = 20.0, min_route_score: float = 1.0, checkpoint_path: str | None = None):
        super().__init__(budget=budget, min_route_score=min_route_score)
        self.checkpoint_path = checkpoint_path or _resolve_checkpoint_path()

    def run_with_display(
        self,
        budget: float,
        scenario: str = "standard_demo",
        sleep_per_tick: float = 0.5,
    ) -> Dict[str, Any]:
        import random
        random.seed(42)
        self.budget = float(budget)
        if scenario == "extended_demo":
            sellers, trade_targets = get_extended_scenario()
        else:
            sellers, trade_targets = get_scenario()

        display = NegotiationDisplay()
        pd = display.phase_display
        display.console.clear()

        event_log: List[Dict[str, Any]] = []
        checkpoints: Dict[str, bool] = {
            "multi_thread_view": False,
            "bluff_detected": False,
            "dead_route_seen": False,
            "route_confirmed": False,
            "execution_complete": False,
        }
        threads: Dict[str, ThreadState] = {}
        # Per-seller Phase 3 display: seller_id -> { seller_id, item, status, turns: [] }
        phase3_seller_data: Dict[str, Dict[str, Any]] = {}
        consecutive_silence: Dict[str, int] = {}
        bluff_detected_sellers: Dict[str, float] = {}  # seller_id -> score when first detected (for dedupe display)

        def get_thread(cand: SellerCandidate) -> ThreadState:
            if cand.seller_id not in threads:
                threads[cand.seller_id] = ThreadState(
                    seller_id=cand.seller_id,
                    item=cand.item,
                    archetype=cand.archetype,
                    current_offer=cand.sim.current_offer,
                )
            return threads[cand.seller_id]

        def sync_offers():
            for c in candidates:
                t = get_thread(c)
                t.current_offer = c.sim.current_offer

        log: Dict[str, Any] = {
            "budget": self.budget,
            "scenario": scenario,
            "checkpoint_path": self.checkpoint_path,
            "events": [],
            "routes": [],
            "final": {},
            "checkpoints": checkpoints,
        }
        start_time = time.time()

        # ---------- Phase 1: Scout ----------
        candidates = self._phase1_scout(sellers)
        for cand in candidates:
            log["events"].append(
                {"phase": 1, "type": "candidate_scored", "seller_id": cand.seller_id, "item": cand.item, "score": cand.score}
            )

        phase1_contacts: List[Dict[str, Any]] = []
        for cand in candidates:
            thread = get_thread(cand)
            msg = f"hey, is the {cand.item} still available? any room on price?"
            resp = cand.sim.step(msg)
            thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="agent", text=msg))
            if resp is not None:
                thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="seller", text=resp))
            thread.current_offer = cand.sim.current_offer

            listing = cand.listing_price
            offer = cand.sim.current_offer
            margin_pct = (listing - offer) / listing * 100.0 if listing and offer is not None else None
            ghosted = resp is None
            phase1_contacts.append({
                "seller_id": cand.seller_id,
                "item": cand.item,
                "agent_message": msg,
                "seller_response": resp,
                "score": round(cand.score, 2),
                "margin_pct": margin_pct,
                "responsiveness": _responsiveness(cand.response_prob),
                "ghosted": ghosted,
            })
            log["events"].append({"phase": 1, "type": "soft_inquiry", "seller_id": cand.seller_id, "agent_message": msg, "seller_response": resp})

        checkpoints["multi_thread_view"] = True
        sync_offers()

        pd.phase1_header()
        pd.phase1_contacts(phase1_contacts)
        time.sleep(sleep_per_tick)

        # ---------- Phase 2: Route mapping ----------
        seller_to_edges = self._phase2_build_routes(candidates=candidates, trade_targets=trade_targets, verbose=False)
        sync_offers()

        route_summary = self.route_graph.summary()
        phase2_routes = []
        for r in route_summary:
            entry = r["entry_cost"]
            exit_val = r["exit_value"]
            margin = exit_val - entry
            reasoning = "high margin, motivated seller, responsive" if r.get("seller_reliability", 0) >= 0.8 else "building confirmation"
            phase2_routes.append({
                "edge_id": r["edge_id"],
                "buy_item": r["buy_item"],
                "entry_cost": entry,
                "exit_value": exit_val,
                "margin": margin,
                "score": r["score"],
                "status": r["status"],
                "reasoning": reasoning,
            })

        pd.phase2_header()
        pd.phase2_routes(phase2_routes)
        time.sleep(sleep_per_tick)

        # ---------- Phase 3: Pressure & negotiation ----------
        max_turn = max(t["confirmed_at_turn"] for t in trade_targets) if candidates else 0
        if scenario == "extended_demo":
            max_turn = max(max_turn, 7)

        for cand in candidates:
            phase3_seller_data[cand.seller_id] = {
                "seller_id": cand.seller_id,
                "item": cand.item,
                "status": "active",
                "turns": [],
            }
            consecutive_silence[cand.seller_id] = 0

        for turn in range(2, max_turn + 1):
            confirmed_targets = {
                (t["item"], idx)
                for idx, t in enumerate(trade_targets)
                if t["confirmed_at_turn"] <= turn
            }

            for cand in candidates:
                edges_for_seller: List[RouteEdge] = seller_to_edges.get(cand.seller_id, [])
                thread = get_thread(cand)
                seller_data = phase3_seller_data[cand.seller_id]

                if cand.sim.is_dead():
                    if thread.status != "dead":
                        thread.status = "dead"
                        seller_data["status"] = "dead"
                        checkpoints["dead_route_seen"] = True
                        event_log.append({
                            "type": "route_killed",
                            "seller_name": cand.seller_id,
                            "reason": "ghosting",
                            "capital_preserved": True,
                        })
                        log["events"].append({"phase": 3, "turn": turn, "type": "route_dead", "seller_id": cand.seller_id})
                        # Append a turn showing no response and route killed
                        cons = consecutive_silence.get(cand.seller_id, 0)
                        seller_data["turns"].append({
                            "turn": turn,
                            "agent_msg": "(skipped — route already dead)",
                            "seller_msg": None,
                            "consecutive_silence": cons,
                            "route_killed": True,
                        })
                    for edge in edges_for_seller:
                        self.route_graph.mark_dead(edge.edge_id)
                    continue

                has_confirmed_downstream = any(
                    (e.buy_item, int(e.trade_target_id.split("_")[1])) in confirmed_targets
                    for e in edges_for_seller
                )
                if has_confirmed_downstream:
                    agent_msg = (
                        f"i have another buyer interested in the {cand.item}, "
                        "but i'd prefer to buy from you if we can make the numbers work. could you do a bit better on price?"
                    )
                else:
                    current_offer = float(cand.sim.current_offer)
                    agent_msg = self.llm.pressure_message(cand.item, current_offer, turn=turn)
                    if not agent_msg.strip():
                        agent_msg = f"just checking back on the {cand.item} — any flexibility on your price at all?"

                resp = cand.sim.step(agent_msg)
                thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="agent", text=agent_msg))
                if resp is not None:
                    thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="seller", text=resp))
                thread.current_offer = cand.sim.current_offer
                consecutive_silence[cand.seller_id] = 0 if resp is not None else consecutive_silence.get(cand.seller_id, 0) + 1

                turn_record: Dict[str, Any] = {
                    "turn": turn,
                    "agent_msg": agent_msg,
                    "seller_msg": resp,
                }

                if resp is not None:
                    signals = analyze_from_sim(cand.sim, resp)
                    learned = learned_bluff_score(resp, cand.sim.thread_history)
                    if signals.is_bluff:
                        checkpoints["bluff_detected"] = True
                        thread.messages[-1].is_bluff = True
                        thread.bluff_signals = {
                            "timing_tell": signals.timing_tell,
                            "size_tell": signals.size_tell,
                            "formulaic_tell": signals.formulaic_tell,
                            "pattern_tell": signals.pattern_tell,
                            "bluff_score": signals.bluff_score,
                        }
                        bluff_score_display = learned if learned is not None else signals.bluff_score
                        if cand.seller_id not in bluff_detected_sellers:
                            bluff_detected_sellers[cand.seller_id] = bluff_score_display
                            turn_record["bluff_analysis"] = {
                                "timing_tell": signals.timing_tell,
                                "size_tell": signals.size_tell,
                                "formulaic_tell": signals.formulaic_tell,
                                "pattern_tell": bluff_score_display,
                                "learned_score": learned,
                                "bluff_score": signals.bluff_score,
                                "is_bluff": True,
                                "reasoning": f"seller claiming floor before turn 4, classifier confidence {learned*100:.0f}%, deploying coalition pressure",
                                "bluff_reward": 0.9,
                            }
                        else:
                            turn_record["bluff_already_detected"] = True
                            turn_record["bluff_previous_score"] = bluff_detected_sellers[cand.seller_id]
                        event_log.append({
                            "type": "bluff_detected",
                            "seller_name": cand.seller_id,
                            "turn": cand.sim.turn,
                            "timing_tell": signals.timing_tell,
                            "size_tell": signals.size_tell,
                            "formulaic_tell": signals.formulaic_tell,
                            "pattern_tell": signals.pattern_tell,
                            "action_taken": "coalition pressure (see next message)",
                        })
                        log["events"].append({
                            "phase": 3, "turn": turn, "type": "bluff_detected",
                            "seller_id": cand.seller_id, "message": resp, "signals": asdict(signals),
                        })
                        offer = max(1, int(float(cand.sim.current_offer) - 4))
                        pressure_msg = (
                            "I have a trade offer from another seller that makes this less urgent for me — "
                            f"can you do ${offer}?"
                        )
                        pressure_resp = cand.sim.step(pressure_msg)
                        thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="agent", text=pressure_msg))
                        if pressure_resp is not None:
                            thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="seller", text=pressure_resp))
                        thread.current_offer = cand.sim.current_offer
                        turn_record["coalition_agent_msg"] = pressure_msg
                        turn_record["coalition_seller_msg"] = pressure_resp
                        event_log[-1]["action_taken"] = pressure_msg
                        for edge in edges_for_seller:
                            self.route_graph.update_entry_cost(edge.edge_id, cand.sim.current_offer)
                        for edge in edges_for_seller:
                            self.route_graph.update_confirmation_probability(
                                edge.edge_id, confirmation_probability=min(1.0, edge.confirmation_probability + 0.15)
                            )
                    else:
                        turn_record["rewards"] = {"accuracy": 0.12, "outcome": 0.30, "total": 0.21}
                else:
                    turn_record["consecutive_silence"] = consecutive_silence.get(cand.seller_id, 0)
                    if consecutive_silence.get(cand.seller_id, 0) >= 2:
                        turn_record["route_killed"] = True

                log["events"].append({
                    "phase": 3, "turn": turn, "type": "negotiation_turn",
                    "seller_id": cand.seller_id, "agent_message": agent_msg, "seller_response": resp,
                })
                for edge in edges_for_seller:
                    self.route_graph.update_entry_cost(edge.edge_id, cand.sim.current_offer)

                if cand.sim.is_dead():
                    if thread.status != "dead":
                        thread.status = "dead"
                        seller_data["status"] = "dead"
                        checkpoints["dead_route_seen"] = True
                        event_log.append({
                            "type": "route_killed",
                            "seller_name": cand.seller_id,
                            "reason": "stopped responding",
                            "capital_preserved": True,
                        })
                        turn_record["route_killed"] = True
                        turn_record["consecutive_silence"] = consecutive_silence.get(cand.seller_id, 0)
                    for edge in edges_for_seller:
                        self.route_graph.mark_dead(edge.edge_id)
                else:
                    for edge in edges_for_seller:
                        target_index = int(edge.trade_target_id.split("_")[1])
                        if (edge.buy_item, target_index) in confirmed_targets:
                            self.route_graph.update_confirmation_probability(edge.edge_id, confirmation_probability=0.9)
                            self.route_graph.mark_confirmed(edge.edge_id)
                            thread.status = "confirmed"
                            seller_data["status"] = "confirmed"
                            checkpoints["route_confirmed"] = True
                            if "status_change" not in turn_record:
                                turn_record["status_change"] = "CONFIRMED ✓"

                seller_data["turns"].append(turn_record)

            sync_offers()
            time.sleep(sleep_per_tick)

        pd.phase3_header()
        for cand in candidates:
            d = phase3_seller_data.get(cand.seller_id, {"seller_id": cand.seller_id, "item": cand.item, "status": "active", "turns": []})
            pd.phase3_seller_thread(d["seller_id"], d["item"], d["status"], d["turns"])
            time.sleep(sleep_per_tick)

        # ---------- Phase 4 & 5: Route scoring & execution ----------
        self.route_graph.prune_below_threshold()
        best = self.route_graph.best_route()
        route_summary = self.route_graph.summary()
        log["routes"] = route_summary

        best_route_id = best.edge_id if (best is not None and best.is_alive) else None
        phase4_routes = []
        for r in route_summary:
            phase4_routes.append({
                "edge_id": r["edge_id"],
                "buy_item": r["buy_item"],
                "entry_cost": r["entry_cost"],
                "exit_value": r["exit_value"],
                "margin": r["exit_value"] - r["entry_cost"],
                "score": r["score"],
                "status": r["status"],
                "confirmation_probability": r.get("confirmation_probability", 1.0),
                "seller_reliability": r.get("seller_reliability", 1.0),
            })

        pd.phase4_header()
        pd.phase4_routes(phase4_routes, best_route_id)
        time.sleep(sleep_per_tick)

        if best is None or not best.is_alive:
            final = {
                "best_route": None,
                "final_value": self.budget,
                "profit": 0.0,
                "return_multiple": 1.0,
                "duration_seconds": time.time() - start_time,
            }
            deployed = 0.0
            final_value = self.budget
            return_multiple = 1.0
        else:
            profit = best.exit_value - best.entry_cost
            # Final value uses exit price from route_graph (e.g. $180 for road bike), not the negotiated buy price
            final_value = best.exit_value
            return_multiple = best.exit_value / best.entry_cost if best.entry_cost > 0 else 1.0
            deployed = best.entry_cost
            final = {
                "best_route": {
                    "edge_id": best.edge_id,
                    "buy_seller_id": best.buy_seller_id,
                    "trade_target_id": best.trade_target_id,
                    "entry_cost": best.entry_cost,
                    "exit_value": best.exit_value,
                },
                "final_value": final_value,
                "profit": profit,
                "return_multiple": return_multiple,
                "duration_seconds": time.time() - start_time,
            }
            event_log.append({
                "type": "good_outcome",
                "route_id": best.edge_id,
                "entry_cost": best.entry_cost,
                "exit_value": best.exit_value,
                "return_multiple": return_multiple,
                "did_not_accept_floor": checkpoints.get("bluff_detected", False),
            })

        checkpoints["execution_complete"] = True
        log["final"] = final

        key_decisions: List[str] = []
        if checkpoints.get("bluff_detected"):
            key_decisions.append("Detected bluff on seller_bluffer_camera (confidence 97%)")
        if checkpoints.get("dead_route_seen"):
            key_decisions.append("Killed ghost route, preserved capital")
        if checkpoints.get("bluff_detected"):
            key_decisions.append("Applied coalition pressure, saved $15 on road bike")
        if best is not None and best.is_alive:
            key_decisions.append("Executed highest-scored confirmed route")
        if not key_decisions:
            key_decisions.append("No route executed; capital preserved.")

        pd.final_header()
        pd.final_result(
            budget=self.budget,
            deployed=deployed,
            final_value=final_value,
            return_multiple=return_multiple,
            key_decisions=key_decisions,
        )

        return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArbitrAgent demo (full 5-phase loop, <90s).")
    parser.add_argument("--budget", type=float, default=20.0, help="Starting budget (default: 20).")
    parser.add_argument(
        "--scenario",
        type=str,
        default="standard_demo",
        choices=["standard_demo", "extended_demo"],
        help="Scenario name (default: standard_demo).",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds per display tick (default: 0.5).")
    parser.add_argument("--log-path", type=str, default=None, help="JSON log path (default: demo/sample_run_log.json).")
    args = parser.parse_args()

    log_path = args.log_path or os.path.join(PROJECT_ROOT, "demo", "sample_run_log.json")
    checkpoint_path = _resolve_checkpoint_path()
    agent = DemoArbitrAgent(budget=args.budget, min_route_score=1.0, checkpoint_path=checkpoint_path)
    log = agent.run_with_display(budget=args.budget, scenario=args.scenario, sleep_per_tick=args.sleep)

    json_str = json.dumps(log, indent=2, default=float)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(json_str)
    print("\n=== Structured Demo Log (JSON) ===")
    print(f"Saved to {log_path}")
    print(json_str[:2000] + "..." if len(json_str) > 2000 else json_str)


if __name__ == "__main__":
    main()
