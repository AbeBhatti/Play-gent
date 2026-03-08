from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple


EdgeStatus = Literal["soft", "confirmed", "dead"]


@dataclass
class RouteEdge:
    """
    A single arbitrage route between a buy seller and a downstream trade target.

    This is intentionally lightweight and does not depend on NetworkX so that
    it can be used inside training loops and demos without extra dependencies.
    """

    edge_id: str
    buy_seller_id: str
    buy_item: str
    trade_target_id: str
    entry_cost: float
    exit_value: float
    status: EdgeStatus = "soft"
    confirmation_probability: float = 0.0
    seller_reliability: float = 1.0

    def score(self) -> float:
        """
        Compute the scalar route score used in Phase 4.

        Implements the project-level scoring formula:

            route_score = (confirmed_exit_value - entry_cost)
                          × route_confirmation_probability
                          × seller_reliability_score

        Notes:
        - When the exit value is not yet fully confirmed, callers should encode
          that uncertainty into `confirmation_probability`.
        - Negative margins are clamped to zero; unprofitable routes should be
          killed by the graph.
        """
        margin = self.exit_value - self.entry_cost
        if margin <= 0:
            return 0.0
        return float(margin * self.confirmation_probability * self.seller_reliability)

    @property
    def is_alive(self) -> bool:
        return self.status != "dead"


class RouteGraph:
    """
    Minimal route graph for ArbitrAgent.

    - Nodes (implicit): buy sellers and trade targets.
    - Edges: `RouteEdge` instances with status ∈ {soft, confirmed, dead}.
    - Scoring: uses the global project scoring formula.

    The graph is intentionally simple: it behaves like a scored set of
    candidate routes rather than a full general-purpose graph library.
    """

    def __init__(self, minimum_threshold: float = 0.0):
        """
        Args:
            minimum_threshold: Any route whose score drops below this value
                will be marked as dead when `prune_below_threshold` is called.
        """
        self._edges: Dict[str, RouteEdge] = {}
        self._next_id: int = 1
        self.minimum_threshold: float = float(minimum_threshold)

    # ------------------------------------------------------------------
    # Edge creation and updates
    # ------------------------------------------------------------------
    def add_route(
        self,
        buy_seller_id: str,
        buy_item: str,
        trade_target_id: str,
        entry_cost: float,
        exit_value: float,
        *,
        status: EdgeStatus = "soft",
        confirmation_probability: float = 0.0,
        seller_reliability: float = 1.0,
    ) -> RouteEdge:
        """
        Register a new route edge in the graph.
        """
        edge_id = f"route_{self._next_id}"
        self._next_id += 1

        edge = RouteEdge(
            edge_id=edge_id,
            buy_seller_id=buy_seller_id,
            buy_item=buy_item,
            trade_target_id=trade_target_id,
            entry_cost=float(entry_cost),
            exit_value=float(exit_value),
            status=status,
            confirmation_probability=float(confirmation_probability),
            seller_reliability=float(seller_reliability),
        )
        self._edges[edge_id] = edge
        return edge

    def get(self, edge_id: str) -> Optional[RouteEdge]:
        return self._edges.get(edge_id)

    def edges(self) -> Iterable[RouteEdge]:
        return self._edges.values()

    def alive_edges(self) -> List[RouteEdge]:
        return [e for e in self._edges.values() if e.is_alive]

    def soft_edges(self) -> List[RouteEdge]:
        return [e for e in self._edges.values() if e.status == "soft"]

    def confirmed_edges(self) -> List[RouteEdge]:
        return [e for e in self._edges.values() if e.status == "confirmed"]

    def mark_confirmed(self, edge_id: str) -> None:
        edge = self._require_edge(edge_id)
        edge.status = "confirmed"
        # Once confirmed, treat probability as 1.0 unless caller overrides.
        if edge.confirmation_probability < 1.0:
            edge.confirmation_probability = 1.0

    def mark_soft(self, edge_id: str) -> None:
        edge = self._require_edge(edge_id)
        if edge.status != "dead":
            edge.status = "soft"

    def mark_dead(self, edge_id: str) -> None:
        edge = self._require_edge(edge_id)
        edge.status = "dead"

    def update_entry_cost(self, edge_id: str, entry_cost: float) -> None:
        edge = self._require_edge(edge_id)
        edge.entry_cost = float(entry_cost)

    def update_exit_value(self, edge_id: str, exit_value: float) -> None:
        edge = self._require_edge(edge_id)
        edge.exit_value = float(exit_value)

    def update_confirmation_probability(
        self, edge_id: str, confirmation_probability: float
    ) -> None:
        edge = self._require_edge(edge_id)
        edge.confirmation_probability = float(
            max(0.0, min(1.0, confirmation_probability))
        )

    def update_seller_reliability(
        self, edge_id: str, seller_reliability: float
    ) -> None:
        edge = self._require_edge(edge_id)
        edge.seller_reliability = float(max(0.0, min(1.0, seller_reliability)))

    # ------------------------------------------------------------------
    # Scoring and pruning
    # ------------------------------------------------------------------
    def scored_routes(self) -> List[Tuple[RouteEdge, float]]:
        """
        Return all (edge, score) pairs, sorted by descending score.
        Dead routes are still included for inspection.
        """
        scored = [(edge, edge.score()) for edge in self._edges.values()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def best_route(self) -> Optional[RouteEdge]:
        """
        Return the highest-scoring *alive* route, or None if none exist.
        """
        best_edge: Optional[RouteEdge] = None
        best_score: float = float("-inf")
        for edge in self._edges.values():
            if not edge.is_alive:
                continue
            score = edge.score()
            if score > best_score:
                best_score = score
                best_edge = edge
        return best_edge

    def prune_below_threshold(self, threshold: Optional[float] = None) -> None:
        """
        Mark as dead any alive route whose score is below the given threshold.

        If `threshold` is None, uses `self.minimum_threshold`.
        """
        cutoff = float(self.minimum_threshold if threshold is None else threshold)
        for edge in self.alive_edges():
            if edge.score() < cutoff:
                edge.status = "dead"

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def summary(self) -> List[dict]:
        """
        Lightweight serializable view of the current graph for logging/printing.
        """
        return [
            {
                "edge_id": e.edge_id,
                "buy_seller_id": e.buy_seller_id,
                "buy_item": e.buy_item,
                "trade_target_id": e.trade_target_id,
                "entry_cost": e.entry_cost,
                "exit_value": e.exit_value,
                "status": e.status,
                "confirmation_probability": e.confirmation_probability,
                "seller_reliability": e.seller_reliability,
                "score": e.score(),
            }
            for e in self._edges.values()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_edge(self, edge_id: str) -> RouteEdge:
        try:
            return self._edges[edge_id]
        except KeyError:
            raise KeyError(f"Route edge {edge_id!r} not found in RouteGraph") from None


__all__ = ["RouteGraph", "RouteEdge", "EdgeStatus"]

