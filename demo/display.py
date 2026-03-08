from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class ThreadMessage:
    turn: int
    sender: str  # "agent" or "seller"
    text: str
    is_bluff: bool = False


@dataclass
class ThreadState:
    seller_id: str
    item: str
    archetype: str
    status: str = "active"  # "active" | "dead" | "confirmed"
    messages: List[ThreadMessage] = field(default_factory=list)
    bluff_signals: Optional[Dict[str, float]] = None


class NegotiationDisplay:
    """
    Rich-based terminal display for the ArbitrAgent demo.

    Responsibilities:
    - Show all active negotiation threads as side-by-side panels.
    - Highlight bluff detection in yellow with individual signals.
    - Use red for dead routes / threads and green for confirmed routes.
    - Render a final panel with budget → entry cost → exit value → return multiple.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def render(
        self,
        threads: List[ThreadState],
        route_summaries: List[Dict[str, Any]],
        budget: float,
        final_metrics: Optional[Dict[str, Any]] = None,
        checkpoints: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Render the full demo view."""
        self.console.clear()

        thread_panels = [self._build_thread_panel(t) for t in threads]
        if thread_panels:
            self.console.print(Columns(thread_panels, expand=True, equal=True))

        # Routes + ROI panel at the bottom
        summary_panel = self._build_summary_panel(
            route_summaries=route_summaries,
            budget=budget,
            final_metrics=final_metrics,
            checkpoints=checkpoints or {},
        )
        self.console.print()
        self.console.print(summary_panel)

    # ------------------------------------------------------------------ #
    # Panel builders
    # ------------------------------------------------------------------ #
    def _build_thread_panel(self, thread: ThreadState) -> Panel:
        # Border colors by status
        border_style = "bright_white"
        if thread.status == "dead":
            border_style = "red"
        elif thread.status == "confirmed":
            border_style = "green"

        title = f"{thread.seller_id} • {thread.item} • {thread.archetype}"

        table = Table.grid(padding=(0, 1))
        table.expand = True
        table.add_column("Speaker", style="bold", no_wrap=True)
        table.add_column("Text", overflow="fold")

        # Only show the last few turns to keep panels readable.
        for msg in thread.messages[-8:]:
            speaker = "you" if msg.sender == "agent" else "seller"
            style = "cyan" if msg.sender == "agent" else "white"
            text = Text(msg.text, style=style)
            if msg.is_bluff:
                # Yellow highlight for bluff detection.
                text.stylize("black on yellow")
            table.add_row(speaker, text)

        # Bluff signal breakdown, if present.
        if thread.bluff_signals:
            sig = thread.bluff_signals
            sig_table = Table.grid(padding=(0, 1))
            sig_table.add_column(justify="left", no_wrap=True)
            sig_table.add_column(justify="right", no_wrap=True)
            sig_table.add_row(
                "[bold yellow]Bluff detected[/bold yellow]",
                f"[yellow]score={sig.get('bluff_score', 0.0):.2f}[/yellow]",
            )
            for key in ("timing_tell", "size_tell", "formulaic_tell", "pattern_tell"):
                if key in sig:
                    label = key.replace("_", " ")
                    sig_table.add_row(label, f"{sig[key]:.2f}")

            table.add_row("", sig_table)

        return Panel(
            table,
            title=title,
            border_style=border_style,
            padding=(1, 1),
        )

    def _build_summary_panel(
        self,
        route_summaries: List[Dict[str, Any]],
        budget: float,
        final_metrics: Optional[Dict[str, Any]],
        checkpoints: Dict[str, bool],
    ) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.expand = True

        # Left: route statuses
        routes_sub = Table(
            show_header=True,
            header_style="bold",
            title="Route Graph",
            title_style="bold",
        )
        routes_sub.add_column("Route", no_wrap=True)
        routes_sub.add_column("Status", no_wrap=True)
        routes_sub.add_column("Δ", justify="right", no_wrap=True)
        routes_sub.add_column("Score", justify="right", no_wrap=True)

        for row in route_summaries:
            margin = row["exit_value"] - row["entry_cost"]
            status = row["status"]
            status_style = {
                "dead": "red",
                "confirmed": "green",
                "soft": "yellow",
            }.get(status, "white")
            routes_sub.add_row(
                row["edge_id"],
                f"[{status_style}]{status}[/{status_style}]",
                f"{margin:.2f}",
                f"{row['score']:.2f}",
            )

        # Right: ROI + checkpoints
        roi_sub = Table(
            show_header=False,
            box=None,
            title="Capital Deployment",
            title_style="bold",
        )
        roi_sub.add_column("Label", no_wrap=True)
        roi_sub.add_column("Value", no_wrap=True)

        entry_cost = None
        exit_value = None
        return_multiple = None

        if final_metrics is not None:
            entry_cost = final_metrics.get("entry_cost")
            exit_value = final_metrics.get("exit_value")
            return_multiple = final_metrics.get("return_multiple")

        roi_sub.add_row("Budget", f"$ {budget:.2f}")
        if entry_cost is not None:
            roi_sub.add_row("Entry cost", f"$ {entry_cost:.2f}")
        if exit_value is not None:
            roi_sub.add_row("Exit value", f"$ {exit_value:.2f}")
        if return_multiple is not None:
            roi_sub.add_row("Return multiple", f"{return_multiple:.2f}x")

        # Checkpoints list
        checkpoints_sub = Table(
            show_header=False,
            box=None,
            title="Demo Checkpoints",
            title_style="bold",
        )
        checkpoints_sub.add_column("State", no_wrap=True)

        labels = [
            ("multi_thread_view", "Threads visible"),
            ("bluff_detected", "Bluff flagged"),
            ("dead_route_seen", "Dead route surfaced"),
            ("route_confirmed", "Route confirmed"),
            ("execution_complete", "Executed & logged"),
        ]
        for key, label in labels:
            done = checkpoints.get(key, False)
            style = "green" if done else "dim"
            marker = "●" if done else "○"
            checkpoints_sub.add_row(f"[{style}]{marker} {label}[/{style}]")

        table.add_row(routes_sub, roi_sub, checkpoints_sub)
        return Panel(
            table,
            title="ArbitrAgent — $20 → Multi-Route Arbitrage",
            border_style="cyan",
            padding=(1, 1),
        )


__all__ = ["NegotiationDisplay", "ThreadState", "ThreadMessage"]

