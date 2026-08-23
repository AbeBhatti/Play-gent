"""
eval_marketplace.py — Step 7
Buyer: TinyLlama from checkpoints/grpo_poker/
Seller: Llama-3.1-8B-Instruct (adversarial LLM seller)

Run in tmux:
  tmux new-session -d -s eval_marketplace \
    "python eval_marketplace.py 2>&1 | tee logs/eval_marketplace_$(date +%Y%m%d_%H%M%S).log"
"""

import argparse
import json
import re
import random
import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf 
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ─────────────────────────────────────────────────────────────────────

def load_cfg(path: str = "config.yaml") -> OmegaConf:
    with open(path) as f:
        return OmegaConf.create(yaml.safe_load(f))


def setup_logging(cfg) -> None:
    log_dir = Path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"eval_marketplace_{ts}.log"),
            logging.StreamHandler(),
        ],
    )


# ── Response Parser (buyer) ────────────────────────────────────────────────────

def _extract_offer(text: str, floor: float, budget: float, asking: float, ev) -> tuple:
    """
    Returns (offer_price, parse_method) using priority order:
    1. explicit_pct  — "25% of the range / gap / negotiation range", "offer 25%"
    2. bare_float    — standalone 0.xx value treated as fraction of [floor, budget]
    3. dollar_amount — raw $X fallback (original behavior)
    Returns (None, "no_parse") when nothing valid is found.
    """
    # Priority 1: any explicit percentage
    pct_m = re.search(r'(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if pct_m:
        pct = float(pct_m.group(1)) / 100.0
        if 0.0 <= pct <= 1.0:
            return floor + pct * (budget - floor), "explicit_pct"

    # Priority 2: bare float in (0, 1) — 0.xx pattern only to avoid false positives
    float_m = re.search(r'(?<!\d)(0\.\d+)(?!\d)', text)
    if float_m:
        pct = float(float_m.group(1))
        if 0.0 < pct < 1.0:
            return floor + pct * (budget - floor), "bare_float"

    # Priority 3: dollar amount (original behavior)
    cleaned = re.sub(r'\$\s*X\b', '', text, flags=re.IGNORECASE)
    prices = re.findall(r'\$?\s*([1-9]\d*(?:\.\d{1,2})?)', cleaned)
    if prices:
        budget_val = float(budget)
        candidates = [float(p) for p in prices
                      if not (ev.budget_filter
                              and abs(float(p) - budget_val) / budget_val <= ev.budget_tolerance)]
        if candidates:
            return candidates[-1], "dollar_amount"
        return asking * ev.anchor_pct, "dollar_anchor"

    return None, "no_parse"


def parse_buyer_response(text: str, cfg, asking: float,
                         floor: float = 0.0, budget: float = 0.0) -> tuple:
    """
    Returns (action, offer_price, parse_method).
    action : "fold" | "call" | "counter"
    Prints raw response. Never raises.
    """
    ev = cfg.eval_marketplace
    print(f"  BUYER RAW: {repr(text[:120])}")
    try:
        low = text.lower()

        for kw in cfg.poker.parse_keywords.fold:
            if kw in low:
                return "fold", None, "keyword_fold"

        for kw in cfg.poker.parse_keywords.call:
            if kw in low:
                return "call", None, "keyword_call"

        for kw in cfg.poker.parse_keywords.counter:
            if kw in low:
                offer, method = _extract_offer(text, floor, budget, asking, ev)
                return "counter", offer, method

        if ev.price_as_raise:
            offer, method = _extract_offer(text, floor, budget, asking, ev)
            if offer is not None:
                return "counter", offer, method

    except Exception:
        pass

    return ev.default_action, None, "no_parse"


# ── Seller Response Parser ─────────────────────────────────────────────────────

def parse_seller_response(text: str, accept_keywords: list) -> tuple:
    """
    Returns (accepted: bool, counter_price: float | None).
    accepted=True means the seller closed the deal.
    Never raises.
    """
    try:
        low = text.lower()
        for kw in accept_keywords:
            if kw in low:
                return True, None
        prices = re.findall(r"\$?\s*(\d+(?:\.\d{1,2})?)", text)
        if prices:
            return False, float(prices[-1])
    except Exception:
        pass
    return False, None


# ── LLM Seller ─────────────────────────────────────────────────────────────────

class LLMSeller:
    def __init__(self, model, tokenizer, asking: float, floor: float, cfg) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.asking = asking
        self.floor = floor
        self.cfg = cfg
        self.current_price = float(asking)

        ev = cfg.eval_marketplace
        self.system_prompt = (
            ev.seller_system_prompt
            .replace("{asking_price}", f"${asking:.0f}")
            .replace("{floor_price}", f"${floor:.0f}")
        )

    def _generate(self, user_text: str) -> str:
        ev = self.cfg.eval_marketplace
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=ev.seller_max_new_tokens,
                do_sample=True,
                temperature=ev.seller_temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        resp_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(resp_ids, skip_special_tokens=True).strip()

    def opening(self) -> str:
        ev = self.cfg.eval_marketplace
        user_text = ev.seller_opening_template.replace("{asking_price}", f"${self.asking:.0f}")
        return self._generate(user_text)

    def respond(self, buyer_raw_text: str, buyer_offer: float | None) -> tuple:
        """
        Returns (accepted: bool, response_text: str, deal_price: float | None).
        Hard floor enforcement applied after LLM output regardless of what Llama said.
        """
        ev = self.cfg.eval_marketplace
        user_msg = f"Buyer says: {buyer_raw_text.strip()[:ev.buyer_context_max_chars]}"
        resp = self._generate(user_msg)
        print(f"  SELLER RAW: {repr(resp[:120])}")

        accepted, counter_price = parse_seller_response(resp, list(ev.seller_accept_keywords))

        # ── Hard floor enforcement ─────────────────────────────────────────────
        if ev.floor_enforcement:
            if accepted:
                deal_price = buyer_offer if (buyer_offer and buyer_offer > 0) else self.current_price
                if deal_price < self.floor:
                    # LLM accepted below floor — override with rejection
                    override_msg = (
                        ev.seller_min_response
                        .replace("{floor_price}", f"${self.floor:.0f}")
                    )
                    self.current_price = self.floor
                    return False, override_msg, None
                return True, resp, deal_price

            if counter_price and counter_price > 0:
                # Clamp counter to [floor, asking]
                counter_price = max(counter_price, self.floor)
                counter_price = min(counter_price, self.asking)
                self.current_price = counter_price
            else:
                # No price in response — concede slightly but stay at or above floor
                self.current_price = max(
                    self.current_price * (1.0 - ev.seller_concession_pct),
                    self.floor,
                )
            return False, resp, None

        # floor_enforcement disabled — original logic
        if accepted:
            deal_price = buyer_offer if buyer_offer else self.current_price
            return True, resp, deal_price
        if counter_price and counter_price > 0:
            self.current_price = counter_price
        return False, resp, None


# ── Buyer Prompt ───────────────────────────────────────────────────────────────

def build_buyer_prompt(asking: float, budget: float, seller_response: str,
                       round_num: int, max_rounds: int,
                       action_history: str, cfg,
                       floor: float = 0.0, seller_ask: float = 0.0) -> str:
    ev = cfg.eval_marketplace
    resp_snippet = seller_response.strip()[:ev.seller_response_max_chars]
    return (
        ev.prompt_template
        .replace("{asking}",          str(int(asking)))
        .replace("{budget}",          str(int(budget)))
        .replace("{floor}",           str(int(floor)))
        .replace("{seller_ask}",      str(int(seller_ask)))
        .replace("{seller_response}", resp_snippet)
        .replace("{round}",           str(round_num))
        .replace("{max_rounds}",      str(max_rounds))
        .replace("{action_history}",  action_history or "(none)")
    )


# ── Single Scenario ────────────────────────────────────────────────────────────

def run_scenario(buyer_model, buyer_tokenizer,
                 seller_model, seller_tokenizer,
                 cfg, rng: random.Random) -> dict:
    ev = cfg.eval_marketplace
    asking      = rng.uniform(ev.asking_price_min, ev.asking_price_max)
    floor_pct   = rng.uniform(ev.floor_pct_min, ev.floor_pct_max)
    floor       = asking * floor_pct

    seller = LLMSeller(seller_model, seller_tokenizer, asking, floor, cfg)

    # Seller opens the conversation
    seller_response = seller.opening()
    logging.info("Seller opening: %s", seller_response[:80])

    deal_price    = None
    closed        = False
    walked        = False
    action_history = ""
    parse_log     = {}

    for round_num in range(1, ev.max_rounds + 1):
        # ── Buyer turn ──────────────────────────────────────────────────────
        prompt = build_buyer_prompt(
            asking, ev.starting_budget, seller_response,
            round_num, ev.max_rounds, action_history, cfg,
            floor=floor, seller_ask=seller.current_price,
        )
        inputs = buyer_tokenizer(prompt, return_tensors="pt").to(buyer_model.device)
        with torch.no_grad():
            out = buyer_model.generate(
                **inputs,
                max_new_tokens=cfg.grpo_poker.max_completion_length,
                do_sample=False,
                pad_token_id=buyer_tokenizer.eos_token_id,
            )
        buyer_raw = buyer_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        action, offer, parse_method = parse_buyer_response(
            buyer_raw, cfg, asking, floor=floor, budget=ev.starting_budget
        )
        parse_log[parse_method] = parse_log.get(parse_method, 0) + 1
        action_history += f"R{round_num}:{action} "
        logging.info("Round %d | buyer action=%s offer=%s parse=%s",
                     round_num, action, offer, parse_method)

        if action == "fold":
            walked = True
            break

        if action == "call":
            # Buyer accepts seller's current price
            closed = True
            deal_price = seller.current_price
            break

        # action == "counter" — send buyer's raw response to seller
        accepted, seller_response, dp = seller.respond(buyer_raw, offer)
        logging.info("Round %d | seller accepted=%s response: %s",
                     round_num, accepted, seller_response[:60])

        if accepted:
            closed = True
            deal_price = dp
            break

        # threshold close — prices have converged within accept_threshold even
        # though the seller didn't explicitly accept; close at buyer's offer
        if (offer is not None
                and seller.current_price > 0
                and abs(offer - seller.current_price) / seller.current_price
                    <= ev.accept_threshold):
            closed = True
            deal_price = offer
            parse_log["threshold_close"] = parse_log.get("threshold_close", 0) + 1
            logging.info("Round %d | threshold close at $%.0f (gap %.1f%%)",
                         round_num, offer,
                         100 * abs(offer - seller.current_price) / seller.current_price)
            break

    capital_return = (
        (asking / deal_price)
        if (closed and deal_price and deal_price > 0)
        else None
    )
    return {
        "asking":         asking,
        "floor":          floor,
        "deal_price":     deal_price,
        "capital_return": capital_return,
        "closed":         closed,
        "walked":         walked,
        "rounds_used":    round_num,
        "parse_log":      parse_log,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marketplace ablation eval")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/grpo_diplomacy/",
        help="Buyer model checkpoint path or HuggingFace model ID",
    )
    p.add_argument(
        "--tag",
        default="none",
        help="Label for this run — included in the saved results filename",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_cfg()
    setup_logging(cfg)
    ev = cfg.eval_marketplace
    dtype = torch.bfloat16 if cfg.grpo_poker.bf16 else torch.float32

    logging.info("Loading buyer model from %s", args.checkpoint)
    buyer_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if buyer_tokenizer.pad_token is None:
        buyer_tokenizer.pad_token = buyer_tokenizer.eos_token
    buyer_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=dtype, device_map="auto"
    )
    buyer_model.eval()
    logging.info("Buyer model loaded.")

    logging.info("Loading seller model from %s", ev.seller_model)
    seller_tokenizer = AutoTokenizer.from_pretrained(ev.seller_model)
    if seller_tokenizer.pad_token is None:
        seller_tokenizer.pad_token = seller_tokenizer.eos_token
    seller_model = AutoModelForCausalLM.from_pretrained(
        ev.seller_model, dtype=dtype, device_map="auto"
    )
    seller_model.eval()
    logging.info("Seller model loaded.")

    rng = random.Random(ev.random_seed)
    results = []

    for i in range(ev.num_scenarios):
        logging.info("─── Scenario %d/%d ───", i + 1, ev.num_scenarios)
        result = run_scenario(
            buyer_model, buyer_tokenizer,
            seller_model, seller_tokenizer,
            cfg, rng,
        )
        results.append(result)
        status = "CLOSED" if result["closed"] else ("WALKED" if result["walked"] else "NO_DEAL")
        cr = f"{result['capital_return']:.3f}" if result["capital_return"] else "—"
        logging.info(
            "Scenario %3d/%d  %-8s  asking=$%.0f  deal=$%s  return=%s",
            i + 1, ev.num_scenarios, status,
            result["asking"],
            f"{result['deal_price']:.0f}" if result["deal_price"] else "—",
            cr,
        )

    # ── Report ─────────────────────────────────────────────────────────────────
    n = len(results)
    closed = [r for r in results if r["closed"]]
    walked = [r for r in results if r["walked"]]
    returns = [r["capital_return"] for r in closed if r["capital_return"]]

    avg_return  = sum(returns) / len(returns) if returns else None
    best_return = max(returns) if returns else None
    worst_return = min(returns) if returns else None

    print("\n" + "=" * 50)
    print("MARKETPLACE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Tag                 : {args.tag}")
    print(f"Checkpoint          : {args.checkpoint}")
    print(f"Scenarios run       : {n}")
    print(f"Deals closed        : {len(closed):>4}  ({100*len(closed)/n:.1f}%)")
    print(f"Agent walked away   : {len(walked):>4}  ({100*len(walked)/n:.1f}%)")
    print(f"No deal (timeout)   : {n-len(closed)-len(walked):>4}  "
          f"({100*(n-len(closed)-len(walked))/n:.1f}%)")
    if returns:
        print(f"\nCapital return (closed deals only):")
        print(f"  Average  : {avg_return:.3f}x")
        print(f"  Best     : {best_return:.3f}x")
        print(f"  Worst    : {worst_return:.3f}x")
    else:
        print("\nNo closed deals.")

    # ── Parse method breakdown ──────────────────────────────────────────────
    totals: dict = {}
    for r in results:
        for method, count in r["parse_log"].items():
            totals[method] = totals.get(method, 0) + count
    grand = sum(totals.values()) or 1
    print(f"\nParse method breakdown ({grand} buyer turns total):")
    for method, count in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"  {method:<20} {count:>4}  ({100*count/grand:.1f}%)")
    print("=" * 50)

    # ── Save tagged results JSON ────────────────────────────────────────────
    log_dir = Path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = log_dir / f"ablation_{args.tag}_{ts}.json"
    summary = {
        "tag":          args.tag,
        "checkpoint":   args.checkpoint,
        "n":            n,
        "closed":       len(closed),
        "walked":       len(walked),
        "no_deal":      n - len(closed) - len(walked),
        "closure_pct":  100 * len(closed) / n,
        "walk_pct":     100 * len(walked) / n,
        "avg_return":   avg_return,
        "best_return":  best_return,
        "worst_return": worst_return,
        "parse_totals": totals,
        "scenarios":    results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    logging.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
