"""
Rich terminal UI for the ArbitrAgent demo.

Panel 1: NEGOTIATION THREADS — one row per seller (name, item, current offer, status).
Panel 2: LIVE EVENT LOG — scrolling [BLUFF DETECTED], [GOOD OUTCOME], [HUMAN-ALIGNED MOVE], [ROUTE KILLED].
Panel 3: ROUTE GRAPH — route_id, entry, exit, score, status.
Panel 4: FINAL RESULT — Budget → Deployed → Final Value → Return, route and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class ThreadMessage:
    turn: int
    sender: str
    text: str
    is_bluff: bool = False


@dataclass
class ThreadState:
    seller_id: str
    item: str
    archetype: str
    status: str = "active"  # "active" | "pending" | "confirmed" | "dead"
    current_offer: Optional[float] = None
    messages: List[ThreadMessage] = field(default_factory=list)
    bluff_signals: Optional[Dict[str, float]] = None


# Event types for the live event log
BluffDetectedEvent = Dict[str, Any]  # seller_name, turn, timing_tell, size_tell, formulaic_tell, pattern_tell, action_taken
GoodOutcomeEvent = Dict[str, Any]   # route_id, entry_cost, exit_value, return_multiple, did_not_accept_floor
HumanAlignedEvent = Dict[str, Any]  # phase_name, action_taken, similarity_pct
RouteKilledEvent = Dict[str, Any]   # seller_name, reason, capital_preserved


def _status_style(status: str) -> str:
    if status == "confirmed":
        return "green"
    if status == "active":
        return "yellow"
    if status == "dead":
        return "red"
    return "white"  # pending


class NegotiationDisplay:
    """
    Live terminal UI: negotiation threads, event log, route graph, final result.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def render(
        self,
        threads: List[ThreadState],
        route_summaries: List[Dict[str, Any]],
        budget: float,
        event_log: Optional[List[Dict[str, Any]]] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        checkpoints: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.console.clear()

        # Panel 1 — NEGOTIATION THREADS
        threads_table = Table(
            show_header=True,
            header_style="bold",
            title="NEGOTIATION THREADS",
            title_style="bold",
        )
        threads_table.add_column("Seller", no_wrap=True)
        threads_table.add_column("Item", no_wrap=True)
        threads_table.add_column("Current offer", justify="right", no_wrap=True)
        threads_table.add_column("Status", no_wrap=True)
        for t in threads:
            offer_str = f"${t.current_offer:.2f}" if t.current_offer is not None else "—"
            style = _status_style(t.status)
            threads_table.add_row(
                t.seller_id,
                t.item,
                offer_str,
                f"[{style}]{t.status}[/{style}]",
            )
        self.console.print(Panel(threads_table, border_style="cyan", padding=(0, 1)))
        self.console.print()

        # Panel 2 — LIVE EVENT LOG (scrolling, last N events)
        events = event_log or []
        log_lines: List[Any] = []
        for ev in events[-30:]:
            kind = ev.get("type") or ev.get("event")
            if kind == "bluff_detected":
                log_lines.append(Text("[BLUFF DETECTED]", style="bold yellow"))
                log_lines.append(Text(f"  {ev.get('seller_name', ev.get('seller_id', ''))}, turn {ev.get('turn', '')}"))
                log_lines.append(Text(f"  ✦ timing tell: {ev.get('timing_tell', 0):.2f}"))
                log_lines.append(Text(f"  ✦ size tell: {ev.get('size_tell', 0):.2f}"))
                log_lines.append(Text(f"  ✦ formulaic tell: {ev.get('formulaic_tell', 0):.2f}"))
                log_lines.append(Text(f"  ✦ pattern tell: {ev.get('pattern_tell', 0):.2f}"))
                log_lines.append(Text(f"  → action taken: {ev.get('action_taken', '')[:80]}..."))
                log_lines.append(Text(""))
            elif kind == "good_outcome":
                log_lines.append(Text("[GOOD OUTCOME]", style="bold green"))
                log_lines.append(Text(f"  route {ev.get('route_id', '')}, entry ${ev.get('entry_cost', 0):.2f}, exit ${ev.get('exit_value', 0):.2f}, return {ev.get('return_multiple', 0):.2f}x"))
                log_lines.append(Text("  ✦ did not accept stated floor"))
                log_lines.append(Text(""))
            elif kind == "human_aligned":
                log_lines.append(Text("[HUMAN-ALIGNED MOVE]", style="bold blue"))
                log_lines.append(Text(f"  {ev.get('phase_name', '')}: {str(ev.get('action_taken', ''))[:60]}..."))
                log_lines.append(Text(f"  ✦ matches human Diplomacy pattern: {ev.get('similarity_pct', 0):.0f}% similarity"))
                log_lines.append(Text(""))
            elif kind == "route_killed":
                log_lines.append(Text("[ROUTE KILLED]", style="bold red"))
                log_lines.append(Text(f"  {ev.get('seller_name', ev.get('seller_id', ''))}, {ev.get('reason', '')}"))
                log_lines.append(Text("  ✦ capital preserved, pivoting"))
                log_lines.append(Text(""))

        if log_lines:
            log_content = Text()
            for line in log_lines:
                log_content.append_text(line)
                log_content.append("\n")
            self.console.print(Panel(log_content, title="LIVE EVENT LOG", border_style="dim", padding=(0, 1), height=14))
        else:
            self.console.print(Panel("(no events yet)", title="LIVE EVENT LOG", border_style="dim", padding=(0, 1), height=6))
        self.console.print()

        # Panel 3 — ROUTE GRAPH
        route_table = Table(
            show_header=True,
            header_style="bold",
            title="ROUTE GRAPH",
            title_style="bold",
        )
        route_table.add_column("route_id", no_wrap=True)
        route_table.add_column("entry", justify="right", no_wrap=True)
        route_table.add_column("exit", justify="right", no_wrap=True)
        route_table.add_column("score", justify="right", no_wrap=True)
        route_table.add_column("status", no_wrap=True)
        for row in route_summaries:
            st = row.get("status", "soft")
            route_table.add_row(
                row.get("edge_id", ""),
                f"${row.get('entry_cost', 0):.2f}",
                f"${row.get('exit_value', 0):.2f}",
                f"{row.get('score', 0):.2f}",
                f"[{_status_style(st)}]{st}[/{_status_style(st)}]",
            )
        self.console.print(Panel(route_table, border_style="cyan", padding=(0, 1)))
        self.console.print()

        # Panel 4 — FINAL RESULT (when available)
        if final_metrics is not None:
            entry = final_metrics.get("entry_cost")
            exit_val = final_metrics.get("exit_value")
            ret = final_metrics.get("return_multiple")
            route_id = final_metrics.get("route_id", "")
            why = final_metrics.get("why", "best scored confirmed route")
            line1 = f"Budget: ${budget:.1f}  →  Deployed: ${entry:.2f}  →  Final Value: ${exit_val:.2f}  →  Return: {ret:.2f}x"
            line2 = f"Route: {route_id} — {why}"
            self.console.print(Panel(f"[bold]{line1}[/bold]\n\n{line2}", title="FINAL RESULT", border_style="green", padding=(1, 2)))
        elif checkpoints and checkpoints.get("execution_complete"):
            self.console.print(Panel("No route executed. Capital preserved.", title="FINAL RESULT", border_style="yellow", padding=(1, 2)))

    # Legacy API: build thread panel per thread (for side-by-side thread view if needed)
    def _build_thread_panel(self, thread: ThreadState) -> Panel:
        border_style = _status_style(thread.status)
        title = f"{thread.seller_id} • {thread.item}"
        table = Table.grid(padding=(0, 1))
        table.add_column("Speaker", style="bold", no_wrap=True)
        table.add_column("Text", overflow="fold")
        for msg in thread.messages[-6:]:
            speaker = "you" if msg.sender == "agent" else "seller"
            style = "cyan" if msg.sender == "agent" else "white"
            text = Text(msg.text, style=style)
            if msg.is_bluff:
                text.stylize("black on yellow")
            table.add_row(speaker, text)
        if thread.bluff_signals:
            table.add_row("", f"[yellow]bluff_score={thread.bluff_signals.get('bluff_score', 0):.2f}[/yellow]")
        return Panel(table, title=title, border_style=border_style, padding=(0, 1))


__all__ = ["NegotiationDisplay", "ThreadState", "ThreadMessage"]
