"""
Rich terminal UI for the ArbitrAgent demo.

Sequential 4-phase display with clear headers:
  Phase 1: Scouting (seller contacts, scores, margin, responsiveness)
  Phase 2: Route mapping (route graph with reasoning)
  Phase 3: Pressure & negotiation (full threads, bluff analysis)
  Phase 4: Route scoring & execution
  Final result: budget, deployed, return, key decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Legacy / shared data structures (used by run_demo and for backward compat)
# ---------------------------------------------------------------------------

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
    status: str = "active"
    current_offer: Optional[float] = None
    messages: List[ThreadMessage] = field(default_factory=list)
    bluff_signals: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Phase display data (passed from run_demo)
# ---------------------------------------------------------------------------

# Phase 1: list of { seller_id, item, agent_message, seller_response, score, margin_pct, responsiveness, ghosted }
# Phase 2: list of { edge_id, buy_item, entry_cost, exit_value, margin, score, status, reasoning }
# Phase 3: list of { seller_id, item, status, turns: [ { turn, agent_msg, seller_msg, bluff_analysis?, rewards?, status_change?, consecutive_silence? } ] }
# Phase 4: routes list + best_route_id
# Final: budget, deployed, final_value, return_multiple, key_decisions

SEP = "─────────────────────────────────────────"


def _status_style(status: str) -> str:
    if status == "confirmed":
        return "green"
    if status == "active" or status == "soft":
        return "yellow"
    if status == "dead":
        return "red"
    return "white"


class PhaseDisplay:
    """
    Renders the 4-phase demo UI sequentially with Rich panels and color.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def phase1_header(self) -> None:
        self.console.print()
        self.console.print(SEP, style="cyan")
        self.console.print("PHASE 1: SCOUTING", style="bold cyan")
        self.console.print(SEP, style="cyan")
        self.console.print()

    def phase1_contacts(
        self,
        contacts: List[Dict[str, Any]],
    ) -> None:
        for c in contacts:
            seller_id = c.get("seller_id", "")
            item = c.get("item", "")
            agent_msg = c.get("agent_message", "")
            seller_resp = c.get("seller_response")
            score = c.get("score", 0.0)
            margin_pct = c.get("margin_pct")
            responsiveness = c.get("responsiveness", "—")
            ghosted = c.get("ghosted", False)

            self.console.print(f"  [bold cyan]→[/bold cyan] Agent to [bold]{seller_id}[/bold]: \"{agent_msg}\"")
            if seller_resp is None or ghosted:
                self.console.print(f"  [bold cyan]←[/bold cyan] [dim]{seller_id}: [red][NO RESPONSE][/red][/dim]")
                self.console.print(
                    f"     [dim]Score: {score:.2f}  |  ghosted: routing to low priority[/dim]"
                )
            else:
                self.console.print(f"  [bold cyan]←[/bold cyan] {seller_id}: \"{seller_resp}\"")
                margin_str = f"margin: {margin_pct:.0f}%" if margin_pct is not None else "margin: —"
                self.console.print(
                    f"     [dim]Score: {score:.2f}  |  {margin_str}  |  responsiveness: {responsiveness}[/dim]"
                )
            self.console.print()

    def phase2_header(self) -> None:
        self.console.print(SEP, style="cyan")
        self.console.print("PHASE 2: ROUTE MAPPING", style="bold cyan")
        self.console.print(SEP, style="cyan")
        self.console.print()

    def phase2_routes(self, routes: List[Dict[str, Any]]) -> None:
        for r in routes:
            edge_id = r.get("edge_id", "")
            buy_item = r.get("buy_item", "")
            entry = r.get("entry_cost", 0)
            exit_val = r.get("exit_value", 0)
            margin = r.get("margin", exit_val - entry)
            score = r.get("score", 0.0)
            status = r.get("status", "soft")
            reasoning = r.get("reasoning", "")
            st = _status_style(status)
            self.console.print(
                f"  [bold]{edge_id}[/bold]: buy {buy_item} @ ${entry:.0f} → resell @ ${exit_val:.0f}"
            )
            self.console.print(
                f"           entry=${entry:.2f}  exit=${exit_val:.2f}  margin=${margin:.2f}  "
                f"score={score:.2f}  status=[{st}]{status.upper()}[/{st}]"
            )
            if reasoning:
                self.console.print(f"           [dim]reasoning: {reasoning}[/dim]")
            self.console.print()

    def phase3_header(self) -> None:
        self.console.print(SEP, style="cyan")
        self.console.print("PHASE 3: PRESSURE & NEGOTIATION", style="bold cyan")
        self.console.print(SEP, style="cyan")
        self.console.print()

    def phase3_seller_thread(
        self,
        seller_id: str,
        item: str,
        status: str,
        turns: List[Dict[str, Any]],
    ) -> None:
        st = _status_style(status)
        self.console.print(f"  [bold]── {seller_id} ({item}) ──[/bold]")
        for t in turns:
            turn_num = t.get("turn", 0)
            agent_msg = t.get("agent_msg", "")
            seller_msg = t.get("seller_msg")
            bluff = t.get("bluff_analysis")
            bluff_already_detected = t.get("bluff_already_detected", False)
            bluff_previous_score = t.get("bluff_previous_score")
            rewards = t.get("rewards")
            status_change = t.get("status_change")
            consecutive_silence = t.get("consecutive_silence")
            route_killed = t.get("route_killed", False)

            self.console.print(f"  [bold]Turn {turn_num}:[/bold]")
            self.console.print(f"    [cyan]→[/cyan] Agent: \"{agent_msg}\"")
            if seller_msg is None and (consecutive_silence or route_killed):
                self.console.print(f"    [cyan]←[/cyan] Seller: [red][NO RESPONSE][/red]")
                if consecutive_silence is not None and route_killed:
                    self.console.print(
                        f"       [red]consecutive_silence: {consecutive_silence}  →  ROUTE KILLED, capital preserved[/red]"
                    )
            elif seller_msg is not None:
                self.console.print(f"    [cyan]←[/cyan] Seller: \"{seller_msg}\"")
                if rewards is not None:
                    acc = rewards.get("accuracy", 0)
                    out = rewards.get("outcome", 0)
                    tot = rewards.get("total", 0)
                    self.console.print(
                        f"       [dim]accuracy_reward: {acc:.2f}  |  outcome_reward: {out:.2f}  |  total: {tot:.2f}[/dim]"
                    )
                if status_change:
                    self.console.print(f"       [green]status → {status_change}[/green]")

                if bluff_already_detected and bluff_previous_score is not None:
                    self.console.print(
                        f"       [yellow]⚡ Bluff previously detected (score: {bluff_previous_score:.2f}) — maintaining pressure[/yellow]"
                    )
                elif bluff is not None:
                    self.console.print()
                    self.console.print(
                        Panel(
                            self._bluff_analysis_content(bluff),
                            title="[bold yellow]⚡ BLUFF ANALYSIS[/bold yellow]",
                            border_style="yellow",
                            padding=(0, 1),
                        )
                    )
                    reasoning = bluff.get("reasoning", "")
                    if reasoning:
                        self.console.print(f"       [dim]reasoning: {reasoning}[/dim]")
                    bluff_reward = bluff.get("bluff_reward")
                    if bluff_reward is not None:
                        self.console.print(f"       [green]bluff_reward: +{bluff_reward:.2f}[/green]")
                    # Coalition pressure message is the next turn's agent message; show inline if present
                    coalition_msg = t.get("coalition_agent_msg")
                    if coalition_msg:
                        self.console.print(f"    [cyan]→[/cyan] Agent: \"{coalition_msg}\"")
                        coalition_resp = t.get("coalition_seller_msg")
                        if coalition_resp is not None:
                            self.console.print(f"    [cyan]←[/cyan] Seller: \"{coalition_resp}\"")
                        self.console.print(f"       [green]status → CONFIRMED ✓[/green]")
            self.console.print()
        self.console.print()

    def _bluff_analysis_content(self, bluff: Dict[str, Any]) -> Text:
        t = Text()
        timing = bluff.get("timing_tell", 0)
        size = bluff.get("size_tell", 0)
        formulaic = bluff.get("formulaic_tell", 0)
        learned = bluff.get("learned_score")
        # pattern_tell and learned_score are the same signal; display learned_score for both
        pattern_display = learned if learned is not None else bluff.get("pattern_tell", 0)
        score = bluff.get("bluff_score", 0)
        is_bluff = bluff.get("is_bluff", score > 0.6)
        t.append(f"timing_tell:    {timing:.2f}  (aggressive early floor claim)\n", style="white")
        t.append(f"size_tell:      {size:.2f}  (floor claim vs listing)\n", style="white")
        t.append(f"formulaic_tell: {formulaic:.2f}  (matches known bluff pattern)\n", style="white")
        t.append(f"pattern_tell:   {pattern_display:.2f}  (learned classifier)\n", style="white")
        t.append(f"bluff_score:    {score:.2f}  →  ", style="white")
        t.append("IS BLUFF" if is_bluff else "no bluff", style="bold red" if is_bluff else "green")
        return t

    def phase4_header(self) -> None:
        self.console.print(SEP, style="cyan")
        self.console.print("PHASE 4: ROUTE SCORING & EXECUTION", style="bold cyan")
        self.console.print(SEP, style="cyan")
        self.console.print()

    def phase4_routes(
        self,
        routes: List[Dict[str, Any]],
        best_route_id: Optional[str],
    ) -> None:
        for r in routes:
            edge_id = r.get("edge_id", "")
            buy_item = r.get("buy_item", "")
            entry = r.get("entry_cost", 0)
            exit_val = r.get("exit_value", 0)
            margin = r.get("margin", exit_val - entry)
            score = r.get("score", 0.0)
            status = r.get("status", "soft")
            conf = r.get("confirmation_probability", 1.0)
            reliability = r.get("seller_reliability", 1.0)
            st = _status_style(status)
            is_best = edge_id == best_route_id and status == "confirmed"
            self.console.print(f"  [bold]{edge_id}[/bold]: {buy_item}")
            self.console.print(
                f"    entry=${entry:.2f}  exit=${exit_val:.2f}  raw_margin=${margin:.2f}"
            )
            self.console.print(
                f"    score = margin(${margin:.2f}) × responsiveness({reliability:.2f}) × bluff_bonus({conf:.2f}) = {score:.2f}"
            )
            if is_best:
                self.console.print(f"    status: [{st}]{status.upper()} ✓  [bold green]← EXECUTING THIS ROUTE[/bold green][/{st}]")
            elif status == "confirmed":
                self.console.print(f"    status: [{st}]{status.upper()} ✓  (not selected, lower score)[/{st}]")
            else:
                self.console.print(f"    status: [{st}]{status.upper()}[/{st}]")
            self.console.print()

    def final_header(self) -> None:
        self.console.print(SEP, style="green")
        self.console.print("FINAL RESULT", style="bold green")
        self.console.print(SEP, style="green")
        self.console.print()

    def final_result(
        self,
        budget: float,
        deployed: float,
        final_value: float,
        return_multiple: float,
        key_decisions: List[str],
    ) -> None:
        self.console.print(f"  Budget:      ${budget:.2f}")
        self.console.print(f"  Deployed:    ${deployed:.2f}")
        self.console.print(f"  Final Value: ${final_value:.2f}")
        self.console.print(f"  Return:      {return_multiple:.2f}x")
        self.console.print()
        self.console.print("  [bold]Key decisions:[/bold]")
        for k in key_decisions:
            self.console.print(f"  [green]✓[/green] {k}")
        self.console.print()


# ---------------------------------------------------------------------------
# Legacy NegotiationDisplay (kept for backward compat; delegates to PhaseDisplay when given phase data)
# ---------------------------------------------------------------------------

class NegotiationDisplay:
    """
    Demo UI: supports both legacy render(threads, route_summaries, ...) and
    the new phase-by-phase flow via PhaseDisplay.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.phase_display = PhaseDisplay(console=self.console)

    def render(
        self,
        threads: List[ThreadState],
        route_summaries: List[Dict[str, Any]],
        budget: float,
        event_log: Optional[List[Dict[str, Any]]] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        checkpoints: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Legacy single-panel render (used if run_demo uses old flow)."""
        self.console.clear()
        threads_table = Table(show_header=True, header_style="bold", title="NEGOTIATION THREADS", title_style="bold")
        threads_table.add_column("Seller", no_wrap=True)
        threads_table.add_column("Item", no_wrap=True)
        threads_table.add_column("Current offer", justify="right", no_wrap=True)
        threads_table.add_column("Status", no_wrap=True)
        for t in threads:
            offer_str = f"${t.current_offer:.2f}" if t.current_offer is not None else "—"
            style = _status_style(t.status)
            threads_table.add_row(t.seller_id, t.item, offer_str, f"[{style}]{t.status}[/{style}]")
        self.console.print(Panel(threads_table, border_style="cyan", padding=(0, 1)))
        self.console.print()
        route_table = Table(show_header=True, header_style="bold", title="ROUTE GRAPH", title_style="bold")
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


__all__ = ["NegotiationDisplay", "PhaseDisplay", "ThreadState", "ThreadMessage"]
