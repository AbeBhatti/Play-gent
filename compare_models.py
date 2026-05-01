"""
compare_models.py
Runs eval_marketplace against two buyer checkpoints with identical scenarios.
Seller (Llama 8B) loaded once and reused for both runs.
"""

import random
import logging
import yaml
import torch
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_marketplace import (
    run_scenario, setup_logging, load_cfg,
)


def run_eval(buyer_ckpt, buyer_label, seller_model, seller_tokenizer,
             cfg, seed=42):
    dtype = torch.bfloat16 if cfg.grpo_poker.bf16 else torch.float32

    logging.info("Loading buyer: %s", buyer_label)
    buyer_tokenizer = AutoTokenizer.from_pretrained(buyer_ckpt)
    if buyer_tokenizer.pad_token is None:
        buyer_tokenizer.pad_token = buyer_tokenizer.eos_token
    buyer_model = AutoModelForCausalLM.from_pretrained(
        buyer_ckpt, dtype=dtype, device_map="auto"
    )
    buyer_model.eval()

    rng = random.Random(seed)
    results = []
    ev = cfg.eval_marketplace

    for i in range(ev.num_scenarios):
        result = run_scenario(
            buyer_model, buyer_tokenizer,
            seller_model, seller_tokenizer,
            cfg, rng,
        )
        results.append(result)
        status = "CLOSED" if result["closed"] else ("WALKED" if result["walked"] else "NO_DEAL")
        cr = f"{result['capital_return']:.3f}" if result["capital_return"] else "—"
        logging.info("%s | Scenario %2d/%d  %-8s  asking=$%.0f  return=%s",
                     buyer_label, i+1, ev.num_scenarios, status,
                     result["asking"], cr)

    del buyer_model
    torch.cuda.empty_cache()
    return results


def summarize(results, label):
    n = len(results)
    closed = [r for r in results if r["closed"]]
    walked = [r for r in results if r["walked"]]
    returns = [r["capital_return"] for r in closed if r["capital_return"]]
    return {
        "label":       label,
        "n":           n,
        "closed_pct":  100 * len(closed) / n,
        "walked_pct":  100 * len(walked) / n,
        "avg_return":  sum(returns) / len(returns) if returns else None,
        "best":        max(returns) if returns else None,
        "worst":       min(returns) if returns else None,
    }


def print_table(a, b):
    print("\n" + "=" * 62)
    print(f"{'Metric':<28} {'Base TinyLlama':>15} {'Diplomacy GRPO':>15}")
    print("=" * 62)

    def row(label, ka, kb, fmt="{:.1f}%"):
        va = fmt.format(a[ka]) if a[ka] is not None else "—"
        vb = fmt.format(b[kb]) if b[kb] is not None else "—"
        print(f"{label:<28} {va:>15} {vb:>15}")

    row("Deals closed",    "closed_pct", "closed_pct")
    row("Agent walked",    "walked_pct", "walked_pct")
    row("Avg capital return", "avg_return", "avg_return", "{:.3f}x")
    row("Best return",     "best",       "best",       "{:.3f}x")
    row("Worst return",    "worst",      "worst",      "{:.3f}x")
    print("=" * 62)


def main():
    cfg = load_cfg()
    setup_logging(cfg)
    dtype = torch.bfloat16 if cfg.grpo_poker.bf16 else torch.float32
    ev = cfg.eval_marketplace

    logging.info("Loading seller model: %s", ev.seller_model)
    seller_tokenizer = AutoTokenizer.from_pretrained(ev.seller_model)
    if seller_tokenizer.pad_token is None:
        seller_tokenizer.pad_token = seller_tokenizer.eos_token
    seller_model = AutoModelForCausalLM.from_pretrained(
        ev.seller_model, dtype=dtype, device_map="auto"
    )
    seller_model.eval()
    logging.info("Seller loaded.")

    SEED = 42
    BUYERS = [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Base TinyLlama"),
        ("checkpoints/grpo_diplomacy",           "Diplomacy GRPO"),
    ]

    summaries = []
    for ckpt, label in BUYERS:
        results = run_eval(ckpt, label, seller_model, seller_tokenizer, cfg, seed=SEED)
        summaries.append(summarize(results, label))

    print_table(summaries[0], summaries[1])


if __name__ == "__main__":
    main()
