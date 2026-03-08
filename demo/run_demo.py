from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List

# Ensure project root is on sys.path when run as `python demo/run_demo.py`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.arbitragent import ArbitrAgent, SellerCandidate  # type: ignore
from agent.bluff_detector import analyze_from_sim
from agent.route_graph import RouteEdge
from demo.display import NegotiationDisplay, ThreadMessage, ThreadState
from simulation.scenario import get_scenario


class DemoArbitrAgent(ArbitrAgent):
    """
    Thin wrapper around ArbitrAgent that:
    - Drives the existing five-phase loop.
    - Streams state into the Rich-based NegotiationDisplay.
    - Builds a structured JSON log of the entire episode.
    """

    def run_with_display(
        self,
        budget: float,
        sleep_per_tick: float = 0.7,
    ) -> Dict[str, Any]:
        self.budget = float(budget)

        sellers, trade_targets = get_scenario()
        display = NegotiationDisplay()

        # Checkpoint flags for the demo.
        checkpoints: Dict[str, bool] = {
            "multi_thread_view": False,
            "bluff_detected": False,
            "dead_route_seen": False,
            "route_confirmed": False,
            "execution_complete": False,
        }

        # Thread state tracking per seller.
        threads: Dict[str, ThreadState] = {}

        def get_thread_for_candidate(cand: SellerCandidate) -> ThreadState:
            if cand.seller_id not in threads:
                threads[cand.seller_id] = ThreadState(
                    seller_id=cand.seller_id,
                    item=cand.item,
                    archetype=cand.archetype,
                )
            return threads[cand.seller_id]

        # Structured log scaffold.
        log: Dict[str, Any] = {
            "budget": self.budget,
            "events": [],
            "routes": [],
            "final": {},
            "checkpoints": checkpoints,
        }

        start_time = time.time()

        # -----------------------------
        # Phase 1: Scout + soft inquiry
        # -----------------------------
        candidates = self._phase1_scout(sellers)

        for cand in candidates:
            log["events"].append(
                {
                    "phase": 1,
                    "type": "candidate_scored",
                    "seller_id": cand.seller_id,
                    "item": cand.item,
                    "score": cand.score,
                    "listing_price": cand.listing_price,
                    "resale_value": cand.resale_value,
                }
            )

        # Open soft inquiries and populate initial threads.
        for cand in candidates:
            thread = get_thread_for_candidate(cand)
            msg = f"hey, is the {cand.item} still available? any room on price?"
            resp = cand.sim.step(msg)

            thread.messages.append(
                ThreadMessage(turn=cand.sim.turn, sender="agent", text=msg)
            )
            if resp is not None:
                thread.messages.append(
                    ThreadMessage(turn=cand.sim.turn, sender="seller", text=resp)
                )

            log["events"].append(
                {
                    "phase": 1,
                    "type": "soft_inquiry",
                    "seller_id": cand.seller_id,
                    "agent_message": msg,
                    "seller_response": resp,
                }
            )

        # Initial multi-thread view.
        checkpoints["multi_thread_view"] = True
        display.render(
            threads=list(threads.values()),
            route_summaries=self.route_graph.summary(),
            budget=self.budget,
            final_metrics=None,
            checkpoints=checkpoints,
        )
        time.sleep(sleep_per_tick)

        # -----------------------------
        # Phase 2: Route Mapping
        # -----------------------------
        seller_to_edges = self._phase2_build_routes(
            candidates=candidates,
            trade_targets=trade_targets,
            verbose=False,
        )

        # Render after routes created (still soft).
        display.render(
            threads=list(threads.values()),
            route_summaries=self.route_graph.summary(),
            budget=self.budget,
            final_metrics=None,
            checkpoints=checkpoints,
        )
        time.sleep(sleep_per_tick)

        # -----------------------------
        # Phase 3: Pressure & Confirm
        # -----------------------------
        if candidates:
            max_turn = max(t["confirmed_at_turn"] for t in trade_targets)
        else:
            max_turn = 0

        for turn in range(2, max_turn + 1):
            # Which downstream trade targets are confirmed by this turn?
            confirmed_targets = {
                (t["item"], idx)
                for idx, t in enumerate(trade_targets)
                if t["confirmed_at_turn"] <= turn
            }

            for cand in candidates:
                edges_for_seller: List[RouteEdge] = seller_to_edges.get(
                    cand.seller_id, []
                )

                # Track threads even if seller has no explicit route edges (e.g., ghoster).
                thread = get_thread_for_candidate(cand)

                # Death / ghosting.
                if cand.sim.is_dead():
                    if thread.status != "dead":
                        thread.status = "dead"
                        checkpoints["dead_route_seen"] = True
                        log["events"].append(
                            {
                                "phase": 3,
                                "turn": turn,
                                "type": "route_dead",
                                "seller_id": cand.seller_id,
                            }
                        )
                    # If there are edges, mark them dead in the graph.
                    for edge in edges_for_seller:
                        self.route_graph.mark_dead(edge.edge_id)
                    continue

                # Do we have a confirmed downstream target by this turn?
                has_confirmed_downstream = any(
                    (edge.buy_item, int(edge.trade_target_id.split("_")[1]))
                    in confirmed_targets
                    for edge in edges_for_seller
                )

                if has_confirmed_downstream:
                    agent_msg = (
                        f"i have another buyer interested in the {cand.item}, "
                        "but i'd prefer to buy from you if we can make the numbers work. "
                        "could you do a bit better on price?"
                    )
                else:
                    agent_msg = (
                        f"just checking back on the {cand.item} — any flexibility on your price at all?"
                    )

                resp = cand.sim.step(agent_msg)

                # Log messages into thread.
                thread.messages.append(
                    ThreadMessage(turn=cand.sim.turn, sender="agent", text=agent_msg)
                )
                if resp is not None:
                    thread.messages.append(
                        ThreadMessage(turn=cand.sim.turn, sender="seller", text=resp)
                    )

                # Bluff analysis if we got a response.
                if resp is not None:
                    signals = analyze_from_sim(cand.sim, resp)
                    if signals.is_bluff:
                        checkpoints["bluff_detected"] = True
                        # Mark the most recent seller message as bluff-highlighted.
                        thread.messages[-1].is_bluff = True
                        thread.bluff_signals = {
                            "timing_tell": signals.timing_tell,
                            "size_tell": signals.size_tell,
                            "formulaic_tell": signals.formulaic_tell,
                            "pattern_tell": signals.pattern_tell,
                            "bluff_score": signals.bluff_score,
                        }
                        log["events"].append(
                            {
                                "phase": 3,
                                "turn": turn,
                                "type": "bluff_detected",
                                "seller_id": cand.seller_id,
                                "message": resp,
                                "signals": asdict(signals),
                            }
                        )

                log["events"].append(
                    {
                        "phase": 3,
                        "turn": turn,
                        "type": "negotiation_turn",
                        "seller_id": cand.seller_id,
                        "agent_message": agent_msg,
                        "seller_response": resp,
                    }
                )

                # Update entry cost with latest offer.
                for edge in edges_for_seller:
                    self.route_graph.update_entry_cost(edge.edge_id, cand.sim.current_offer)

                # If seller ghosted after this message, mark dead.
                if cand.sim.is_dead():
                    if thread.status != "dead":
                        thread.status = "dead"
                        checkpoints["dead_route_seen"] = True
                    for edge in edges_for_seller:
                        self.route_graph.mark_dead(edge.edge_id)
                    continue

                # Upgrade confirmation probability when downstream target has confirmed.
                for edge in edges_for_seller:
                    target_index = int(edge.trade_target_id.split("_")[1])
                    if (edge.buy_item, target_index) in confirmed_targets:
                        self.route_graph.update_confirmation_probability(
                            edge.edge_id, confirmation_probability=0.9
                        )
                        self.route_graph.mark_confirmed(edge.edge_id)
                        thread.status = "confirmed"
                        checkpoints["route_confirmed"] = True

            # Render this turn.
            display.render(
                threads=list(threads.values()),
                route_summaries=self.route_graph.summary(),
                budget=self.budget,
                final_metrics=None,
                checkpoints=checkpoints,
            )
            time.sleep(sleep_per_tick)

        # -----------------------------
        # Phase 4: Route Scoring
        # -----------------------------
        self.route_graph.prune_below_threshold()

        # -----------------------------
        # Phase 5: Execute
        # -----------------------------
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
        else:
            profit = best.exit_value - best.entry_cost
            final_value = self.budget - best.entry_cost + best.exit_value
            route_multiple = (
                best.exit_value / best.entry_cost if best.entry_cost > 0 else 0.0
            )
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

        checkpoints["execution_complete"] = True
        log["final"] = final

        # Final render with ROI panel filled.
        final_route = None
        if final["best_route"] is not None:
            final_route = final["best_route"]
        display.render(
            threads=list(threads.values()),
            route_summaries=route_summary,
            budget=self.budget,
            final_metrics={
                "entry_cost": final_route["entry_cost"] if final_route else None,
                "exit_value": final_route["exit_value"] if final_route else None,
                "return_multiple": final["return_multiple"],
            },
            checkpoints=checkpoints,
        )

        return log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ArbitrAgent Rich demo (90-second negotiation walkthrough)."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=20.0,
        help="Starting cash budget for the agent (default: 20.0).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=15.0,
        help="Seconds to pause between display updates (default: 15.0, ~90s total demo).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional path to write the structured JSON log. If omitted, prints to stdout only.",
    )
    args = parser.parse_args()

    agent = DemoArbitrAgent(budget=args.budget, min_route_score=1.0)
    log = agent.run_with_display(budget=args.budget, sleep_per_tick=args.sleep)

    json_str = json.dumps(log, indent=2, default=float)
    if args.log_path:
        with open(args.log_path, "w") as f:
            f.write(json_str)
    print("\n=== Structured Demo Log (JSON) ===")
    print(json_str)


if __name__ == "__main__":
    main()

