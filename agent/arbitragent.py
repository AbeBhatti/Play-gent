"""
arbitragent.py — Five-phase ArbitrAgent loop.

This module wires together:
- Seller simulations from `simulation.scenario`
- The lightweight `RouteGraph` from `agent.route_graph`

It runs the full 5-phase loop end-to-end with mocked sellers:
    Phase 1: Scout
    Phase 2: Route Mapping
    Phase 3: Pressure and Confirm
    Phase 4: Route Scoring
    Phase 5: Execute

No real LLM or RL policy is required here; the goal for Session A2 is a
deterministic, testable orchestration loop that future sessions can plug
policies and bluff detectors into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.route_graph import RouteGraph, RouteEdge
from agent.bluff_detector import analyze_from_sim
from agent.agent_llm import AgentLLM
from simulation.scenario import get_scenario
from simulation.seller_profiles import LISTINGS


@dataclass
class SellerCandidate:
    seller_id: str
    item: str
    listing_price: float
    resale_value: float
    trade_openness: float
    archetype: str
    response_prob: float
    score: float
    sim: Any  # CraigslistSellerSim instance


class ArbitrAgent:
    """
    Minimal ArbitrAgent implementation for the hackathon demo.

    It is intentionally heuristic and deterministic — the *shape* of the
    5-phase loop is correct, even though the policy/bluff detector are
    stubbed out for now.
    """

    def __init__(self, budget: float = 20.0, min_route_score: float = 1.0):
        self.budget = float(budget)
        self.route_graph = RouteGraph(minimum_threshold=min_route_score)
        self.llm = AgentLLM()
        # Structured event log for downstream inspection / demo UIs.
        self._structured_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run_once(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single end-to-end arbitrage episode on the standard scenario.

        Returns a dict with:
        - "best_route": RouteEdge | None
        - "final_value": float
        - "profit": float
        - "route_graph_summary": list[dict]
        - "structured_log": list[dict]  # structured event log for this run
        """
        # Reset per-episode structured log.
        self._structured_log = []
        sellers, trade_targets = get_scenario()

        # Phase 1: Scout
        candidates = self._phase1_scout(sellers)
        if verbose:
            print("=== Phase 1: Scout ===")
            for c in candidates:
                print(
                    f"- {c.seller_id} ({c.item}): score={c.score:.3f}, "
                    f"listing=${c.listing_price}, resale≈${c.resale_value}"
                )
            print()

        # Phase 1b: Open soft-inquiry negotiations
        if verbose:
            print("Opening soft inquiries with top candidates...")
        self._open_soft_inquiries(candidates, verbose=verbose)
        print()

        # Phase 2: Route Mapping
        if verbose:
            print("=== Phase 2: Route Mapping ===")
        seller_to_edges = self._phase2_build_routes(candidates, trade_targets, verbose=verbose)

        # Phase 3: Pressure and Confirm (stubbed but structurally correct)
        if verbose:
            print("\n=== Phase 3: Pressure & Confirm ===")
        self._phase3_pressure_and_confirm(candidates, trade_targets, seller_to_edges, verbose=verbose)

        # Phase 4: Route Scoring
        if verbose:
            print("\n=== Phase 4: Route Scoring ===")
        self.route_graph.prune_below_threshold()
        if verbose:
            for row in self.route_graph.summary():
                print(
                    f"{row['edge_id']}: {row['buy_seller_id']} -> {row['trade_target_id']} | "
                    f"status={row['status']} | score={row['score']:.2f} | "
                    f"entry=${row['entry_cost']:.2f} exit=${row['exit_value']:.2f}"
                )

        # Phase 5: Execute
        if verbose:
            print("\n=== Phase 5: Execute ===")
        best = self.route_graph.best_route()
        if best is None or not best.is_alive:
            if verbose:
                print("No viable route found. Holding cash.")
            return {
                "best_route": None,
                "final_value": self.budget,
                "profit": 0.0,
                "route_graph_summary": self.route_graph.summary(),
                "structured_log": self._structured_log,
            }

        profit = best.exit_value - best.entry_cost
        final_value = self.budget - best.entry_cost + best.exit_value
        if verbose:
            print(
                f"Executing route {best.edge_id}: buy from {best.buy_seller_id} at "
                f"${best.entry_cost:.2f}, exit at ${best.exit_value:.2f} "
                f"(score={best.score():.2f})"
            )
            print(f"Final value: ${final_value:.2f} on ${best.entry_cost:.2f} deployed.")

        return {
            "best_route": best,
            "final_value": final_value,
            "profit": profit,
            "route_graph_summary": self.route_graph.summary(),
            "structured_log": self._structured_log,
        }

    # ------------------------------------------------------------------
    # Phase 1: Scout
    # ------------------------------------------------------------------
    def _phase1_scout(self, sellers: List[Any], top_k: int = 3) -> List[SellerCandidate]:
        """
        Score sellers on:
        - resale demand
        - trade liquidity
        - bluff probability (higher means more upside if we can detect it)
        """
        listing_lookup = {l["item"]: l for l in LISTINGS}

        # Heuristic maximum for normalization
        max_resale = max(l["resale_value"] for l in LISTINGS)

        candidates: List[SellerCandidate] = []
        for s in sellers:
            p = s.profile
            item = p["item"]
            listing = listing_lookup.get(item)
            resale_value = float(listing["resale_value"]) if listing else float(p["listing_price"])

            resale_demand = resale_value / max_resale if max_resale else 0.0
            trade_liquidity = float(p.get("trade_openness", 0.5))
            bluff_prob = self._archetype_bluff_probability(p["archetype"])

            # Weighted sum; tuned for demo, not correctness.
            score = 0.5 * resale_demand + 0.3 * trade_liquidity + 0.2 * bluff_prob

            candidates.append(
                SellerCandidate(
                    seller_id=p["id"],
                    item=item,
                    listing_price=float(p["listing_price"]),
                    resale_value=resale_value,
                    trade_openness=trade_liquidity,
                    archetype=p["archetype"],
                    response_prob=float(p["response_prob"]),
                    score=score,
                    sim=s,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def _open_soft_inquiries(self, candidates: List[SellerCandidate], verbose: bool = True) -> None:
        for c in candidates:
            msg = self.llm.scout_message(c.item, c.listing_price)
            resp = c.sim.step(msg)
            if verbose:
                print(f"[to {c.seller_id}] {msg}")
                print(f"[from {c.seller_id}] {resp if resp is not None else '…no response'}")

    # ------------------------------------------------------------------
    # Phase 2: Route Mapping
    # ------------------------------------------------------------------
    def _phase2_build_routes(
        self,
        candidates: List[SellerCandidate],
        trade_targets: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, List[RouteEdge]]:
        """
        Build the route graph edges between buy candidates and trade targets.

        Returns:
            Mapping from seller_id -> list[RouteEdge]
        """
        seller_to_edges: Dict[str, List[RouteEdge]] = {}

        for c in candidates:
            for idx, target in enumerate(trade_targets):
                if target["item"] != c.item:
                    continue

                trade_target_id = f"buyer_{idx}_{target['item'].replace(' ', '_')}"

                # Initial confirmation probability is low and increases in Phase 3
                base_conf_prob = 0.3

                edge = self.route_graph.add_route(
                    buy_seller_id=c.seller_id,
                    buy_item=c.item,
                    trade_target_id=trade_target_id,
                    entry_cost=c.sim.current_offer,
                    exit_value=float(target["buyer_price"]),
                    status="soft",
                    confirmation_probability=base_conf_prob,
                    seller_reliability=c.response_prob,
                )

                seller_to_edges.setdefault(c.seller_id, []).append(edge)

                if verbose:
                    print(
                        f"Route {edge.edge_id}: {c.seller_id} ({c.item}) "
                        f"-> {trade_target_id} at "
                        f"entry≈${edge.entry_cost:.2f}, exit≈${edge.exit_value:.2f}"
                    )

        if not seller_to_edges and verbose:
            print("No matching trade targets found for current candidates.")

        return seller_to_edges

    # ------------------------------------------------------------------
    # Phase 3: Pressure and Confirm (stub)
    # ------------------------------------------------------------------
    def _phase3_pressure_and_confirm(
        self,
        candidates: List[SellerCandidate],
        trade_targets: List[Dict[str, Any]],
        seller_to_edges: Dict[str, List[RouteEdge]],
        verbose: bool = True,
    ) -> None:
        """
        Apply simple, deterministic pressure and confirmation logic:
        - Downstream trade targets "confirm" based on their configured turn.
        - Once confirmed, we use them as leverage with the upstream seller.
        - Ghosted sellers mark their routes dead.
        - Bluffer responses trigger bluff detection via `agent.bluff_detector`.
        """
        if not candidates:
            return

        max_turn = max(t["confirmed_at_turn"] for t in trade_targets)

        for turn in range(2, max_turn + 1):
            if verbose:
                print(f"\n-- Negotiation turn {turn} --")

            # Track which trade targets are considered confirmed by this turn.
            confirmed_targets = {
                (t["item"], idx)
                for idx, t in enumerate(trade_targets)
                if t["confirmed_at_turn"] <= turn
            }

            for c in candidates:
                # Skip sellers with no routes.
                edges = seller_to_edges.get(c.seller_id, [])
                if not edges:
                    continue

                # Check death/ghosting first.
                if c.sim.is_dead():
                    if verbose:
                        print(f"{c.seller_id} route is dead due to ghosting.")
                    for edge in edges:
                        self.route_graph.mark_dead(edge.edge_id)
                    continue

                current_offer = float(c.sim.current_offer)
                msg = self.llm.pressure_message(c.item, current_offer, turn=turn)

                resp = c.sim.step(msg)
                if verbose:
                    print(f"[to {c.seller_id}] {msg}")
                    print(f"[from {c.seller_id}] {resp if resp is not None else '…no response'}")

                # Update entry cost based on latest seller offer.
                for edge in edges:
                    self.route_graph.update_entry_cost(edge.edge_id, c.sim.current_offer)

                # If seller ghosted this turn, mark routes dead.
                if c.sim.is_dead():
                    if verbose:
                        print(f"{c.seller_id} stopped responding; killing all routes.")
                    for edge in edges:
                        self.route_graph.mark_dead(edge.edge_id)
                    continue

                # Bluff detection: inspect full thread via BluffDetector.
                signals = analyze_from_sim(c.sim, resp or "")

                # Unverified floor claim: formulaic language present but not flagged as full bluff.
                formulaic_present = signals.formulaic_tell > 0

                # Log full bluff reasoning: turn, seller_id, bluff_score, signals dict, action_taken.
                action_taken = msg  # the agent message we just sent before this response
                self._structured_log.append(
                    {
                        "event": "bluff_analysis",
                        "phase": 3,
                        "turn": c.sim.turn,
                        "seller_id": c.seller_id,
                        "item": c.item,
                        "bluff_score": signals.bluff_score,
                        "signals": {
                            "timing_tell": signals.timing_tell,
                            "size_tell": signals.size_tell,
                            "formulaic_tell": signals.formulaic_tell,
                            "pattern_tell": signals.pattern_tell,
                            "bluff_score": signals.bluff_score,
                            "is_bluff": signals.is_bluff,
                        },
                        "action_taken": action_taken,
                        "seller_message": resp,
                    }
                )

                if not signals.is_bluff and formulaic_present:
                    self._structured_log.append(
                        {
                            "event": "unverified_floor_claim",
                            "phase": 3,
                            "turn": c.sim.turn,
                            "seller_id": c.seller_id,
                            "seller_message": resp,
                        }
                    )

                if verbose:
                    print(
                        f"[bluff_analysis {c.seller_id}] "
                        f"timing={signals.timing_tell:.2f}, "
                        f"size={signals.size_tell:.2f}, "
                        f"formulaic={signals.formulaic_tell:.2f}, "
                        f"pattern={signals.pattern_tell:.2f}, "
                        f"score={signals.bluff_score:.2f}, "
                        f"is_bluff={signals.is_bluff}"
                    )

                # When a bluff is detected, deploy coalition pressure: floor - 4.
                if signals.is_bluff:
                    current_offer = float(c.sim.current_offer)
                    offer = max(1, int(current_offer - 4))
                    pressure_msg = self.llm.coalition_message(c.item, offer)
                    pressure_resp = c.sim.step(pressure_msg)
                    if verbose:
                        print(f"[to {c.seller_id}] {pressure_msg}")
                        print(
                            f"[from {c.seller_id}] "
                            f"{pressure_resp if pressure_resp is not None else '…no response'}"
                        )

                    self._structured_log.append(
                        {
                            "event": "coalition_pressure",
                            "phase": 3,
                            "seller_id": c.seller_id,
                            "item": c.item,
                            "turn": c.sim.turn,
                            "pressure_message": pressure_msg,
                            "response": pressure_resp,
                            "counter_offer": offer,
                        }
                    )

                    for edge in edges:
                        self.route_graph.update_entry_cost(edge.edge_id, c.sim.current_offer)
                    # Bluff means seller has room — update confirmation probability upward.
                    for edge in edges:
                        self.route_graph.update_confirmation_probability(
                            edge.edge_id,
                            confirmation_probability=min(1.0, edge.confirmation_probability + 0.15),
                        )

                for edge in edges:
                    target_index = int(edge.trade_target_id.split("_")[1])
                    if (edge.buy_item, target_index) in confirmed_targets:
                        self.route_graph.update_confirmation_probability(
                            edge.edge_id, confirmation_probability=0.9
                        )
                        self.route_graph.mark_confirmed(edge.edge_id)

                    new_reliability = min(
                        1.0, edge.seller_reliability + 0.1 * float(signals.bluff_score)
                    )
                    self.route_graph.update_seller_reliability(
                        edge.edge_id, seller_reliability=new_reliability
                    )

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------
    @staticmethod
    def _archetype_bluff_probability(archetype: str) -> float:
        if archetype == "bluffer":
            return 0.9
        if archetype == "motivated":
            return 0.2
        if archetype == "ghoster":
            return 0.3
        if archetype == "trade_curious":
            return 0.4
        return 0.5

    @staticmethod
    def _bluff_heuristic(response: str, candidate: SellerCandidate) -> float:
        """
        Very lightweight bluff detector for Session A2:
        - Looks for formulaic "final offer" style language.
        - Gives higher score when the seller archetype is "bluffer".
        """
        if not response:
            return 0.0

        lower = response.lower()
        formulaic_phrases = [
            "final offer",
            "cant go lower",
            "can't go lower",
            "lowest i can do",
            "lowest i can go",
        ]
        has_formulaic = any(p in lower for p in formulaic_phrases)

        base = 0.0
        if has_formulaic:
            base += 0.6
        if candidate.archetype == "bluffer":
            base += 0.3

        return min(1.0, base)


def run_demo() -> None:
    """
    Small harness to run the ArbitrAgent loop end-to-end.

    This is intentionally simple so tests (or humans) can call:
        python -m agent.arbitragent
    or:
        from agent.arbitragent import run_demo
        run_demo()
    """
    agent = ArbitrAgent(budget=20.0, min_route_score=1.0)
    agent.run_once(verbose=True)


if __name__ == "__main__":
    run_demo()

