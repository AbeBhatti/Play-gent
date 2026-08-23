"""
eval_multiagent.py — Multi-agent trading simulation
5 agents (base, sft, diplomacy, poker, giant) all loaded simultaneously.
Each agent starts with $500 cash and one item (random true value $100–$400).
Goal: maximize final portfolio value (cash + item value) over 20 rounds.

Run:
  python eval_multiagent.py 2>&1 | tee logs/multiagent_$(date +%Y%m%d_%H%M%S).log
"""

import json
import re
import random
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Paths ──────────────────────────────────────────────────────────────────────

_HF_CACHE = Path.home() / ".cache/huggingface/hub"
_LLAMA_SNAP = (
    _HF_CACHE
    / "models--meta-llama--Llama-3.1-8B-Instruct"
    / "snapshots"
    / "0e9e39f249a16976918f6564b8830bc894c89659"
)

MODEL_PATHS = {
    "base":      "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "sft":       "checkpoints/sft/",
    "diplomacy": "checkpoints/grpo_diplomacy/",
    "poker":     "checkpoints/grpo_poker/",
    "giant":     str(_LLAMA_SNAP),
}

ITEM_NAMES = {
    "base":      "Artifact-A",
    "sft":       "Artifact-B",
    "diplomacy": "Artifact-C",
    "poker":     "Artifact-D",
    "giant":     "Artifact-E",
}

NUM_ROUNDS          = 20
STARTING_CASH       = 500.0
FLOOR_PCT           = 0.80
VALUE_MIN           = 100.0
VALUE_MAX           = 400.0
MAX_NEW_TOKENS_OFFER    = 120
MAX_NEW_TOKENS_RESPONSE = 80
TEMPERATURE         = 0.7
LOGS_DIR            = Path("logs")


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(ts: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / f"multiagent_{ts}.log"),
            logging.StreamHandler(),
        ],
    )


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(name: str, path: str, dtype) -> tuple:
    logging.info("Loading %-10s from %s ...", name, path)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="auto")
    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logging.info("  %-10s loaded  (%.2fB params)", name, param_count)
    return model, tok


# ── Agent State ───────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, name: str, item_true_value: float) -> None:
        self.name            = name
        self.cash            = STARTING_CASH
        self.has_item        = True
        self.item_name       = ITEM_NAMES[name]
        self.item_true_value = item_true_value
        self.item_cost       = item_true_value
        self.floor_price     = item_true_value * FLOOR_PCT
        self.deals_made      = 0
        self.deals_rejected  = 0

    def portfolio_value(self) -> float:
        return self.cash + (self.item_true_value if self.has_item else 0.0)

    def to_dict(self) -> dict:
        return {
            "name":            self.name,
            "cash":            round(self.cash, 2),
            "has_item":        self.has_item,
            "item_name":       self.item_name if self.has_item else None,
            "item_true_value": round(self.item_true_value, 2) if self.has_item else None,
            "item_cost":       round(self.item_cost, 2) if self.has_item else None,
            "floor_price":     round(self.floor_price, 2) if self.has_item else None,
            "portfolio_value": round(self.portfolio_value(), 2),
            "deals_made":      self.deals_made,
            "deals_rejected":  self.deals_rejected,
        }


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_offer_prompt(
    offerer: Agent, target: Agent, pair_history: list, round_num: int
) -> list:
    rounds_left = NUM_ROUNDS - round_num + 1

    if offerer.has_item:
        status_text = (
            f"- Cash: ${offerer.cash:.0f}\n"
            f"- Item: {offerer.item_name} (you paid ${offerer.item_cost:.0f})\n"
            f"- Floor price: ${offerer.floor_price:.0f} — never sell below this\n"
        )
        actions_text = (
            "Choose ONE action:\n"
            f"  1. SELL your item ({offerer.item_name}) to {target.name} at your chosen price\n"
            f"  2. BUY {target.name}'s item at a price you offer\n"
        )
    else:
        status_text = (
            f"- Cash: ${offerer.cash:.0f}\n"
            "- Item: none\n"
        )
        actions_text = (
            "Choose ONE action:\n"
            f"  1. BUY {target.name}'s item at a price you offer\n"
            "  (You have no item to sell)\n"
        )

    target_info = (
        f"{target.name} currently holds an item."
        if target.has_item
        else f"{target.name} has no item."
    )

    if pair_history:
        history_lines = [
            f"  [R{h['round']}] {h['offerer']}→{h['target']}: "
            f"{h['action']} ${h['price']:.0f} → {h['outcome']}"
            for h in pair_history[-6:]
        ]
        history_text = (
            f"Your trade history with {target.name}:\n" + "\n".join(history_lines)
        )
    else:
        history_text = f"No prior trades with {target.name} this session."

    user_content = (
        f"Round {round_num}/{NUM_ROUNDS}  ({rounds_left} rounds remaining)\n\n"
        f"Your status:\n{status_text}\n"
        f"{target_info}\n\n"
        f"{history_text}\n\n"
        f"{actions_text}\n"
        "Respond EXACTLY in this format:\n"
        "ACTION: SELL or BUY\n"
        "PRICE: [number only]\n"
        "MESSAGE: [one sentence to the other agent]\n"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a strategic trading agent. Your goal is to maximize your final "
                "portfolio value (cash + item value) at the end of 20 rounds. "
                "Sell high, buy undervalued items, and never sell your item below its floor price."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def build_response_prompt(
    target: Agent, offerer: Agent, action: str, price: float, message: str, round_num: int
) -> list:
    rounds_left = NUM_ROUNDS - round_num + 1

    if target.has_item:
        status_text = (
            f"- Cash: ${target.cash:.0f}\n"
            f"- Item: {target.item_name} (you paid ${target.item_cost:.0f})\n"
            f"- Floor price: ${target.floor_price:.0f}\n"
        )
    else:
        status_text = f"- Cash: ${target.cash:.0f}\n- Item: none\n"

    if action == "SELL":
        offer_desc = (
            f"{offerer.name} wants to SELL you their item for ${price:.0f}.\n"
            f"  You would pay ${price:.0f} (you have ${target.cash:.0f} cash)."
        )
    else:
        if target.has_item:
            offer_desc = (
                f"{offerer.name} wants to BUY your item ({target.item_name}) for ${price:.0f}.\n"
                f"  You would receive ${price:.0f} cash (your floor: ${target.floor_price:.0f})."
            )
        else:
            offer_desc = (
                f"{offerer.name} offers ${price:.0f} for your item, but you have no item to sell."
            )

    user_content = (
        f"Round {round_num}/{NUM_ROUNDS}  ({rounds_left} rounds remaining)\n\n"
        f"Your status:\n{status_text}\n"
        f"Offer from {offerer.name}:\n  {offer_desc}\n"
        f"  Their message: \"{message}\"\n\n"
        "Respond EXACTLY in this format:\n"
        "DECISION: ACCEPT or REJECT\n"
        "REASON: [one sentence]\n"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a strategic trading agent. Your goal is to maximize your final "
                "portfolio value (cash + item value) at the end of 20 rounds. "
                "Accept deals that improve your position; reject bad ones."
            ),
        },
        {"role": "user", "content": user_content},
    ]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, messages: list, max_new_tokens: int) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = (
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            + "\nASSISTANT:"
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_offer(text: str, offerer: Agent) -> tuple:
    """Returns (action, price).  action in {'SELL','BUY'}."""
    action = None
    price  = None

    action_m = re.search(r'ACTION\s*:\s*(SELL|BUY)', text, re.IGNORECASE)
    price_m  = re.search(r'PRICE\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)

    if action_m:
        action = action_m.group(1).upper()
    if price_m:
        price = float(price_m.group(1))

    if action is None:
        if re.search(r'\bsell\b', text, re.IGNORECASE):
            action = "SELL"
        elif re.search(r'\bbuy\b', text, re.IGNORECASE):
            action = "BUY"
        else:
            action = "SELL" if offerer.has_item else "BUY"

    if price is None:
        nums = re.findall(r'\$?\s*([1-9][0-9]*(?:\.[0-9]+)?)', text)
        if nums:
            price = float(nums[-1])
        else:
            price = offerer.floor_price * 1.2 if offerer.has_item else 150.0

    price = max(1.0, min(price, 9999.0))
    return action, round(price, 2)


def parse_decision(text: str) -> str:
    """Returns 'ACCEPT' or 'REJECT'."""
    dec_m = re.search(r'DECISION\s*:\s*(ACCEPT|REJECT)', text, re.IGNORECASE)
    if dec_m:
        return dec_m.group(1).upper()
    if re.search(r'\baccept\b', text, re.IGNORECASE):
        return "ACCEPT"
    return "REJECT"


def extract_message(text: str) -> str:
    msg_m = re.search(r'MESSAGE\s*:\s*(.+)', text, re.IGNORECASE)
    if msg_m:
        return msg_m.group(1).strip()[:120]
    return text.split(".")[0].strip()[:120] or "(no message)"


# ── Trade Execution ───────────────────────────────────────────────────────────

def execute_trade(
    offerer: Agent, target: Agent, action: str, price: float
) -> tuple:
    """Returns (success: bool, reason: str).  Hard-enforces floor and cash limits."""
    if action == "SELL":
        if not offerer.has_item:
            return False, "offerer has no item"
        if price < offerer.floor_price:
            return False, f"${price:.0f} below offerer floor ${offerer.floor_price:.0f}"
        if target.cash < price:
            return False, f"target cash ${target.cash:.0f} < price ${price:.0f}"
        # Transfer item to target
        old_name, old_val = offerer.item_name, offerer.item_true_value
        offerer.cash    += price
        offerer.has_item = False
        target.cash     -= price
        target.has_item  = True
        target.item_name       = old_name
        target.item_true_value = old_val
        target.item_cost       = price
        target.floor_price     = old_val * FLOOR_PCT
        return True, "deal closed"

    else:  # BUY — offerer pays price for target's item
        if not target.has_item:
            return False, "target has no item"
        if price < target.floor_price:
            return False, f"${price:.0f} below target floor ${target.floor_price:.0f}"
        if offerer.cash < price:
            return False, f"offerer cash ${offerer.cash:.0f} < price ${price:.0f}"
        old_name, old_val = target.item_name, target.item_true_value
        offerer.cash    -= price
        offerer.has_item = True
        offerer.item_name       = old_name
        offerer.item_true_value = old_val
        offerer.item_cost       = price
        offerer.floor_price     = old_val * FLOOR_PCT
        target.cash     += price
        target.has_item  = False
        return True, "deal closed"


# ── Main Simulation ───────────────────────────────────────────────────────────

def run_simulation() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(ts)

    logging.info("=" * 65)
    logging.info("MULTI-AGENT TRADING SIMULATION  —  %s", ts)
    logging.info("Agents: %s", list(MODEL_PATHS.keys()))
    logging.info("Rounds: %d  |  Starting cash: $%.0f  |  Item range: $%.0f–$%.0f",
                 NUM_ROUNDS, STARTING_CASH, VALUE_MIN, VALUE_MAX)
    logging.info("=" * 65)

    # ── Verify paths before loading ────────────────────────────────────────────
    for name, path in MODEL_PATHS.items():
        p = Path(path)
        if p.exists():
            logging.info("  ✓ %-10s  %s", name, path)
        else:
            logging.info("  ~ %-10s  %s  (HuggingFace hub ID)", name, path)

    # ── Load all 5 models ──────────────────────────────────────────────────────
    dtype = torch.bfloat16
    models: dict     = {}
    tokenizers: dict = {}
    for name, path in MODEL_PATHS.items():
        models[name], tokenizers[name] = load_model(name, path, dtype)
    logging.info("All 5 models loaded successfully.")

    # ── Initialize agents ──────────────────────────────────────────────────────
    rng         = random.Random(42)
    agent_names = list(MODEL_PATHS.keys())
    agents: dict = {}

    for name in agent_names:
        true_val       = round(rng.uniform(VALUE_MIN, VALUE_MAX), 2)
        agents[name]   = Agent(name, true_val)
        a = agents[name]
        logging.info(
            "  Agent %-10s  cash=$%.0f  item=%s  true_value=$%.0f  floor=$%.0f",
            name, a.cash, a.item_name, a.item_true_value, a.floor_price,
        )

    # ── Per-pair conversation histories ────────────────────────────────────────
    pair_histories: dict = {}

    def pair_key(a: str, b: str) -> tuple:
        return tuple(sorted([a, b]))

    # ── Round log ─────────────────────────────────────────────────────────────
    round_log: list = []

    # ── Rounds ────────────────────────────────────────────────────────────────
    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info("─── Round %d/%d ───────────────────────────────────────", round_num, NUM_ROUNDS)

        turn_order = agent_names[:]
        rng.shuffle(turn_order)
        round_events: list = []

        for offerer_name in turn_order:
            others      = [n for n in agent_names if n != offerer_name]
            target_name = rng.choice(others)
            offerer     = agents[offerer_name]
            target      = agents[target_name]
            pk          = pair_key(offerer_name, target_name)
            history     = pair_histories.get(pk, [])

            # ── Generate offer ─────────────────────────────────────────────────
            offer_msgs = build_offer_prompt(offerer, target, history, round_num)
            offer_raw  = generate(
                models[offerer_name], tokenizers[offerer_name],
                offer_msgs, MAX_NEW_TOKENS_OFFER,
            )
            logging.info("  %s→%s OFFER_RAW: %s", offerer_name, target_name, repr(offer_raw[:100]))

            action, price = parse_offer(offer_raw, offerer)
            message       = extract_message(offer_raw)
            logging.info(
                "  %s→%s | %s $%.0f | \"%s\"",
                offerer_name, target_name, action, price, message[:60],
            )

            # ── Generate target response ───────────────────────────────────────
            resp_msgs = build_response_prompt(target, offerer, action, price, message, round_num)
            resp_raw  = generate(
                models[target_name], tokenizers[target_name],
                resp_msgs, MAX_NEW_TOKENS_RESPONSE,
            )
            logging.info("  %s RESP_RAW: %s", target_name, repr(resp_raw[:80]))

            decision = parse_decision(resp_raw)
            logging.info("  %s decision: %s", target_name, decision)

            # ── Apply trade ────────────────────────────────────────────────────
            trade_success = False
            trade_reason  = "rejected by target"

            if decision == "ACCEPT":
                trade_success, trade_reason = execute_trade(offerer, target, action, price)
                if trade_success:
                    offerer.deals_made  += 1
                    target.deals_made   += 1
                else:
                    offerer.deals_rejected += 1
            else:
                offerer.deals_rejected += 1

            outcome_str = "DEAL" if trade_success else f"NO_DEAL ({trade_reason})"
            logging.info("  → %s", outcome_str)

            # ── Record event ───────────────────────────────────────────────────
            event = {
                "round":       round_num,
                "offerer":     offerer_name,
                "target":      target_name,
                "action":      action,
                "price":       price,
                "message":     message,
                "decision":    decision,
                "outcome":     outcome_str,
                "offer_raw":   offer_raw[:300],
                "resp_raw":    resp_raw[:200],
            }
            pair_histories.setdefault(pk, []).append(event)
            round_events.append(event)

        # ── Portfolio snapshot after round ─────────────────────────────────────
        snapshot = {name: agents[name].to_dict() for name in agent_names}
        round_log.append({"round": round_num, "events": round_events, "portfolios": snapshot})

        logging.info("  Portfolios after round %d:", round_num)
        for name in agent_names:
            a = agents[name]
            logging.info(
                "    %-10s  cash=$%6.0f  %s (v=$%.0f)  → $%.0f",
                name, a.cash,
                f"{a.item_name}" if a.has_item else "no item    ",
                a.item_true_value if a.has_item else 0,
                a.portfolio_value(),
            )

    # ── Final Leaderboard ──────────────────────────────────────────────────────
    ranked = sorted(agents.values(), key=lambda a: a.portfolio_value(), reverse=True)

    print("\n" + "=" * 72)
    print("FINAL LEADERBOARD  —  Multi-Agent Trading Simulation")
    print("=" * 72)
    print(
        f"{'Rank':<5} {'Agent':<12} {'Cash':>8} {'Item Held':>12} "
        f"{'True Val':>9} {'Portfolio':>10} {'Deals':>6} {'Rejctd':>7}"
    )
    print("-" * 72)
    for rank, a in enumerate(ranked, 1):
        item_str = a.item_name       if a.has_item else "none"
        item_val = f"${a.item_true_value:.0f}" if a.has_item else "$0"
        print(
            f"{rank:<5} {a.name:<12} ${a.cash:>7.0f} {item_str:>12} "
            f"{item_val:>9} ${a.portfolio_value():>9.0f} "
            f"{a.deals_made:>6} {a.deals_rejected:>7}"
        )
    print("=" * 72)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_path = LOGS_DIR / f"multiagent_{ts}.json"
    results = {
        "timestamp":      ts,
        "num_rounds":     NUM_ROUNDS,
        "starting_cash":  STARTING_CASH,
        "floor_pct":      FLOOR_PCT,
        "final_rankings": [a.to_dict() for a in ranked],
        "round_log":      round_log,
    }
    out_path.write_text(json.dumps(results, indent=2))
    logging.info("Results saved → %s", out_path)


if __name__ == "__main__":
    run_simulation()
