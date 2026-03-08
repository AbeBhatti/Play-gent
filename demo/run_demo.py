"""
Demo entry point: budget, scenario, full 5-phase agent loop with Rich display.
Loads unified_final checkpoint if present, else phase2_final. Saves log to demo/sample_run_log.json.
Must complete in under 90 seconds.
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
from agent.bluff_detector import analyze_from_sim
from agent.route_graph import RouteEdge
from demo.display import NegotiationDisplay, ThreadMessage, ThreadState
from simulation.scenario import get_scenario


def _resolve_checkpoint_path() -> str | None:
    """Unified_final if exists, else phase2_final."""
    unified = os.path.join(PROJECT_ROOT, "training", "checkpoints", "unified_final")
    phase2 = os.path.join(PROJECT_ROOT, "training", "checkpoints", "phase2_final")
    if os.path.isdir(unified):
        return unified
    if os.path.isdir(phase2):
        return phase2
    return None


class DemoArbitrAgent(ArbitrAgent):
    """
    Runs full 5-phase loop with display and event log.
    Uses checkpoint path for future model loading; currently heuristic agent.
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
        random.seed(42)  # deterministic demo so all 5 checkpoints (including bluff) hit
        self.budget = float(budget)
        sellers, trade_targets = get_scenario()
        display = NegotiationDisplay()
        event_log: List[Dict[str, Any]] = []

        checkpoints: Dict[str, bool] = {
            "multi_thread_view": False,
            "bluff_detected": False,
            "dead_route_seen": False,
            "route_confirmed": False,
            "execution_complete": False,
        }

        threads: Dict[str, ThreadState] = {}

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

        # Phase 1
        candidates = self._phase1_scout(sellers)
        for cand in candidates:
            log["events"].append(
                {"phase": 1, "type": "candidate_scored", "seller_id": cand.seller_id, "item": cand.item, "score": cand.score}
            )

        for cand in candidates:
            thread = get_thread(cand)
            msg = f"hey, is the {cand.item} still available? any room on price?"
            resp = cand.sim.step(msg)
            thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="agent", text=msg))
            if resp is not None:
                thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="seller", text=resp))
            thread.current_offer = cand.sim.current_offer
            log["events"].append({"phase": 1, "type": "soft_inquiry", "seller_id": cand.seller_id, "agent_message": msg, "seller_response": resp})

        checkpoints["multi_thread_view"] = True
        sync_offers()
        display.render(
            threads=list(threads.values()),
            route_summaries=self.route_graph.summary(),
            budget=self.budget,
            event_log=event_log,
            final_metrics=None,
            checkpoints=checkpoints,
        )
        time.sleep(sleep_per_tick)

        # Phase 2
        seller_to_edges = self._phase2_build_routes(candidates=candidates, trade_targets=trade_targets, verbose=False)
        sync_offers()
        display.render(
            threads=list(threads.values()),
            route_summaries=self.route_graph.summary(),
            budget=self.budget,
            event_log=event_log,
            final_metrics=None,
            checkpoints=checkpoints,
        )
        time.sleep(sleep_per_tick)

        # Phase 3
        max_turn = max(t["confirmed_at_turn"] for t in trade_targets) if candidates else 0
        for turn in range(2, max_turn + 1):
            confirmed_targets = {
                (t["item"], idx)
                for idx, t in enumerate(trade_targets)
                if t["confirmed_at_turn"] <= turn
            }

            for cand in candidates:
                edges_for_seller: List[RouteEdge] = seller_to_edges.get(cand.seller_id, [])
                thread = get_thread(cand)

                if cand.sim.is_dead():
                    if thread.status != "dead":
                        thread.status = "dead"
                        checkpoints["dead_route_seen"] = True
                        event_log.append({
                            "type": "route_killed",
                            "seller_name": cand.seller_id,
                            "reason": "ghosting",
                            "capital_preserved": True,
                        })
                        log["events"].append({"phase": 3, "turn": turn, "type": "route_dead", "seller_id": cand.seller_id})
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
                    agent_msg = f"just checking back on the {cand.item} — any flexibility on your price at all?"

                resp = cand.sim.step(agent_msg)
                thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="agent", text=agent_msg))
                if resp is not None:
                    thread.messages.append(ThreadMessage(turn=cand.sim.turn, sender="seller", text=resp))
                thread.current_offer = cand.sim.current_offer

                if resp is not None:
                    signals = analyze_from_sim(cand.sim, resp)
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
                        # Coalition pressure: floor - 4
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
                        event_log[-1]["action_taken"] = pressure_msg
                        for edge in edges_for_seller:
                            self.route_graph.update_entry_cost(edge.edge_id, cand.sim.current_offer)
                        for edge in edges_for_seller:
                            self.route_graph.update_confirmation_probability(
                                edge.edge_id, confirmation_probability=min(1.0, edge.confirmation_probability + 0.15)
                            )

                log["events"].append({
                    "phase": 3, "turn": turn, "type": "negotiation_turn",
                    "seller_id": cand.seller_id, "agent_message": agent_msg, "seller_response": resp,
                })
                for edge in edges_for_seller:
                    self.route_graph.update_entry_cost(edge.edge_id, cand.sim.current_offer)

                if cand.sim.is_dead():
                    if thread.status != "dead":
                        thread.status = "dead"
                        checkpoints["dead_route_seen"] = True
                        event_log.append({
                            "type": "route_killed",
                            "seller_name": cand.seller_id,
                            "reason": "stopped responding",
                            "capital_preserved": True,
                        })
                    for edge in edges_for_seller:
                        self.route_graph.mark_dead(edge.edge_id)
                    continue

                for edge in edges_for_seller:
                    target_index = int(edge.trade_target_id.split("_")[1])
                    if (edge.buy_item, target_index) in confirmed_targets:
                        self.route_graph.update_confirmation_probability(edge.edge_id, confirmation_probability=0.9)
                        self.route_graph.mark_confirmed(edge.edge_id)
                        thread.status = "confirmed"
                        checkpoints["route_confirmed"] = True

            sync_offers()
            display.render(
                threads=list(threads.values()),
                route_summaries=self.route_graph.summary(),
                budget=self.budget,
                event_log=event_log,
                final_metrics=None,
                checkpoints=checkpoints,
            )
            time.sleep(sleep_per_tick)

        # Phase 4 & 5
        self.route_graph.prune_below_threshold()
        best = self.route_graph.best_route()
        route_summary = self.route_graph.summary()
        log["routes"] = route_summary

        if best is None or not best.is_alive:
            final = {
                "best_route": None,
                "final_value": self.budget,
                "profit": 0.0,
                "return_multiple": 1.0,
                "duration_seconds": time.time() - start_time,
            }
            final_metrics_display = None
        else:
            profit = best.exit_value - best.entry_cost
            final_value = self.budget - best.entry_cost + best.exit_value
            route_multiple = best.exit_value / best.entry_cost if best.entry_cost > 0 else 0.0
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
                "return_multiple": route_multiple,
                "duration_seconds": time.time() - start_time,
            }
            event_log.append({
                "type": "good_outcome",
                "route_id": best.edge_id,
                "entry_cost": best.entry_cost,
                "exit_value": best.exit_value,
                "return_multiple": route_multiple,
                "did_not_accept_floor": checkpoints.get("bluff_detected", False),
            })
            final_metrics_display = {
                "entry_cost": best.entry_cost,
                "exit_value": best.exit_value,
                "return_multiple": route_multiple,
                "route_id": best.edge_id,
                "why": "best scored confirmed route (bluff detected and pressure applied)" if checkpoints.get("bluff_detected") else "best scored confirmed route",
            }

        checkpoints["execution_complete"] = True
        log["final"] = final

        display.render(
            threads=list(threads.values()),
            route_summaries=route_summary,
            budget=self.budget,
            event_log=event_log,
            final_metrics=final_metrics_display,
            checkpoints=checkpoints,
        )
        return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArbitrAgent demo (full 5-phase loop, <90s).")
    parser.add_argument("--budget", type=float, default=20.0, help="Starting budget (default: 20).")
    parser.add_argument("--scenario", type=str, default="standard_demo", help="Scenario name (default: standard_demo).")
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
