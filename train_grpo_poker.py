
"""
train_grpo_poker.py — Phase 3
GRPO fine-tuning on IRC holdem hand histories.
Loads from checkpoints/grpo_diplomacy/, saves to checkpoints/grpo_poker/.

Run via run_all.sh or directly:
  python train_grpo_poker.py
"""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

# Make CUDA errors synchronous so try/except can catch them
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ─────────────────────────────────────────────────────────────────────

def load_cfg(path: str = "config.yaml") -> OmegaConf:
    with open(path) as f:
        return OmegaConf.create(yaml.safe_load(f))


def setup_logging(cfg) -> None:
    Path(cfg.paths.logs).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(Path(cfg.paths.logs) / f"grpo_poker_{ts}.log"),
            logging.StreamHandler(),
        ],
    )


# ── IRC Data Parsing ───────────────────────────────────────────────────────────

STREETS = ["preflop", "flop", "turn", "river"]


def parse_hdb_line(line: str) -> dict | None:
    """
    hdb format:
      timestamp table hand num_players  preflop/pot flop/pot turn/pot river/pot  [cards...]
    Returns dict with pot sizes per street, or None on parse failure.
    """
    try:
        parts = line.strip().split()
        if len(parts) < 8:
            return None
        pots = {}
        for i, street in enumerate(STREETS):
            field = parts[4 + i]          # e.g. "2/60" or "0/0"
            pot_str = field.split("/")[1]
            pots[street] = int(pot_str)
        community = parts[8:] if len(parts) > 8 else []
        return {
            "timestamp": parts[0],
            "table":     parts[1],
            "hand_id":   parts[2],
            "num_players": int(parts[3]),
            "pots":      pots,
            "community": community,
        }
    except Exception:
        return None


def parse_pdb_line(line: str) -> dict | None:
    """
    pdb format (11 fields, no separate total-actions column):
      player timestamp table seat preflop flop turn river bankroll winnings net
    Returns dict, or None on parse failure.
    """
    try:
        parts = line.strip().split()
        if len(parts) < 11:
            return None
        return {
            "player":    parts[0],
            "timestamp": parts[1],
            "table":     parts[2],
            "seat":      int(parts[3]),
            "preflop":   parts[4],
            "flop":      parts[5],
            "turn":      parts[6],
            "river":     parts[7],
            "bankroll":  int(parts[8]),
            "winnings":  int(parts[9]),
            "net":       int(parts[10]),
        }
    except Exception:
        return None


def load_hands(cfg) -> list:
    """
    Returns list of hand dicts:
      { hdb: {...}, players: [{pdb fields, street, action_char, pot_size}, ...] }
    One training example per player-action-point per hand.
    Only holdem data (path already points to holdem/).
    """
    poker_dir = Path(cfg.paths.poker_data)
    examples = []
    max_hands = cfg.grpo_poker.max_hands

    for month_dir in sorted(poker_dir.iterdir()):
        if len(examples) >= max_hands:
            break

        hdb_path = month_dir / "hdb"
        pdb_dir  = month_dir / "pdb"
        if not hdb_path.exists() or not pdb_dir.exists():
            continue

        # Index hdb by timestamp
        hdb_index = {}
        with open(hdb_path) as f:
            for line in f:
                h = parse_hdb_line(line)
                if h:
                    hdb_index[h["timestamp"]] = h

        # Walk pdb files, match to hdb
        for pdb_file in sorted(pdb_dir.iterdir()):
            if len(examples) >= max_hands:
                break
            try:
                with open(pdb_file) as f:
                    for line in f:
                        if len(examples) >= max_hands:
                            break
                        p = parse_pdb_line(line)
                        if not p:
                            continue
                        hdb = hdb_index.get(p["timestamp"])
                        if not hdb:
                            continue

                        # One example per player per hand at their first non-blind action
                        # preflop field holds all preflop actions; strip blinds to find voluntary action
                        action_seq = p["preflop"].replace("B", "").replace("b", "")
                        if not action_seq or action_seq[0] == "-":
                            continue

                        # Determine street of first voluntary action
                        street, pot = "preflop", hdb["pots"]["preflop"]
                        for s in STREETS:
                            if p[s] and p[s] != "-":
                                street = s
                                pot = hdb["pots"][s]
                                break

                        examples.append({
                            "street":        street,
                            "pot":           pot,
                            "seat":          p["seat"],
                            "num_players":   hdb["num_players"],
                            "actions_so_far": p["preflop"],
                            "net":           p["net"],
                        })
            except Exception:
                continue

    return examples


# ── Prompt & Response Parsing ──────────────────────────────────────────────────

def build_prompt(ex: dict) -> str:
    return (
        f"You are playing poker.\n"
        f"Street: {ex['street']}\n"
        f"Pot: {ex['pot']}\n"
        f"Position: seat {ex['seat']} of {ex['num_players']}\n"
        f"Actions so far: {ex['actions_so_far']}\n"
        f"Your action (fold/call/raise):\n"
    )


def parse_action(text: str, cfg) -> str:
    """
    Returns "fold", "call", or "raise".
    Defaults to cfg.poker.default_action on unparseable output. Never raises.
    """
    try:
        low = text.lower()
        for kw in cfg.poker.parse_keywords.fold:
            if kw in low:
                return "fold"
        for kw in cfg.poker.parse_keywords.call:
            if kw in low:
                return "call"
        for kw in cfg.poker.parse_keywords.counter:
            if kw in low:
                return "raise"
    except Exception:
        pass
    return cfg.poker.default_action


# ── GRPO Core (same structure as diplomacy) ────────────────────────────────────

def sequence_log_probs(
    model, input_ids: torch.Tensor, gen_ids: torch.Tensor
) -> torch.Tensor:
    """[B] scalar log prob sum over generated tokens."""
    full = torch.cat([input_ids, gen_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    shift = input_ids.shape[1] - 1
    gen_logits = logits[:, shift : shift + gen_ids.shape[1], :]
    lp = F.log_softmax(gen_logits, dim=-1)
    return lp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).sum(-1)


def grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    epsilon: float,
    kl_coeff: float,
) -> torch.Tensor:
    advantages = rewards - rewards.mean()
    if advantages.std() > 1e-8:
        advantages = advantages / (advantages.std() + 1e-8)
    # Clamp log ratio before exp to prevent Inf overflow when policy drifts from reference
    log_ratio = (log_probs - ref_log_probs.detach()).clamp(-5.0, 5.0)
    ratios = torch.exp(log_ratio)
    clipped = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
    pg = -torch.min(ratios * advantages, clipped * advantages).mean()
    kl = log_ratio.mean()
    return pg + kl_coeff * kl


# ── Training Loop ──────────────────────────────────────────────────────────────

def find_resume_checkpoint(ckpt_dir: Path) -> tuple:
    """
    Scans ckpt_dir for hand_N subdirs, returns (path, hand_index) of the
    highest N found, or (None, 0) if none exist.
    """
    best_n = 0
    best_path = None
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("hand_"):
            try:
                n = int(d.name.split("_")[1])
                if n > best_n:
                    
                    best_n = n
                    best_path = d
            except (IndexError, ValueError):
                pass
    return best_path, best_n


def main() -> None:
    cfg = load_cfg()
    setup_logging(cfg)
    gcfg = cfg.grpo_poker

    logging.info("Parsing IRC holdem hands from %s", cfg.paths.poker_data)
    examples = load_hands(cfg)
    logging.info("Loaded %d hand examples (max %d)", len(examples), gcfg.max_hands)

    # Shuffle with fixed seed so order is identical across resume runs
    rng = random.Random(42)
    rng.shuffle(examples)

    ckpt_dir = Path(cfg.paths.checkpoints.grpo_poker)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume from highest saved hand checkpoint if one exists
    resume_ckpt, resume_hand = find_resume_checkpoint(ckpt_dir)
    if resume_ckpt:
        load_from = str(resume_ckpt)
        logging.info("Resuming from checkpoint %s (skipping first %d hands)",
                     resume_ckpt, resume_hand)
    else:
        load_from = cfg.paths.checkpoints.grpo_diplomacy
        logging.info("No resume checkpoint found — loading from %s", load_from)

    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if gcfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        load_from, dtype=dtype, device_map="auto"
    )
    # Reference policy always stays as the diplomacy checkpoint
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.paths.checkpoints.grpo_diplomacy, dtype=dtype, device_map="auto"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=gcfg.learning_rate)

    optimizer.zero_grad()
    accum_steps = 0

    vocab_size = model.config.vocab_size

    for hand_idx, ex in enumerate(examples):
        # Skip hands already trained on in a previous run
        if hand_idx < resume_hand:
            continue

        # Skip neutral hands (net == 0) if configured — no reward signal
        if gcfg.skip_neutral_hands and ex["net"] == 0:
            continue

        try:
            prompt = build_prompt(ex)
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = enc["input_ids"]                          # [1, prompt_len]
            input_batch = input_ids.repeat(gcfg.num_generations, 1)  # [N, prompt_len]

            # Generate N responses with top_p to prevent degenerate distributions
            with torch.no_grad():
                gen_out = model.generate(
                    input_batch,
                    max_new_tokens=gcfg.max_completion_length,
                    do_sample=True,
                    temperature=gcfg.temperature,
                    top_p=gcfg.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = gen_out[:, input_ids.shape[1]:]             # [N, gen_len]

            # Clamp to valid vocab range — prevents device-side assert in gather
            gen_ids = gen_ids.clamp(0, vocab_size - 1)

            # Reward from historical net outcome — same for all N generations
            net = ex["net"]
            if net > 0:
                reward = gcfg.reward_win
            elif net < 0:
                reward = gcfg.reward_loss
            else:
                reward = gcfg.reward_neutral

            rewards_t = torch.full(
                (gcfg.num_generations,), reward, dtype=torch.float32, device=model.device
            )

            # Current policy log probs (with grad)
            full = torch.cat([input_batch, gen_ids], dim=1)
            logits = model(full).logits
            shift = input_ids.shape[1] - 1
            gen_logits = logits[:, shift : shift + gen_ids.shape[1], :]
            lp_current = F.log_softmax(gen_logits, dim=-1)
            lp_current = lp_current.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).sum(-1)

            # Reference policy log probs (no grad)
            with torch.no_grad():
                lp_ref = sequence_log_probs(ref_model, input_batch, gen_ids)

            loss = grpo_loss(lp_current, lp_ref, rewards_t, gcfg.epsilon, gcfg.kl_coeff)

            # Skip step if loss is NaN/Inf — prevents corrupted gradients
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN/Inf loss at hand %d, skipping step", hand_idx + 1)
                optimizer.zero_grad()
                continue

            (loss / gcfg.gradient_accumulation_steps).backward()
            accum_steps += 1

            if accum_steps % gcfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gcfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if (hand_idx + 1) % gcfg.logging_steps == 0:
                first_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                action = parse_action(first_text, cfg)
                logging.info(
                    "Hand %d/%d  reward=%.1f  parsed_action=%s",
                    hand_idx + 1, len(examples), reward, action,
                )

            if (hand_idx + 1) % gcfg.save_steps == 0:
                step_ckpt = ckpt_dir / f"hand_{hand_idx + 1}"
                model.save_pretrained(step_ckpt)
                tokenizer.save_pretrained(step_ckpt)
                logging.info("Saved checkpoint at hand %d", hand_idx + 1)

        except Exception as e:
            logging.warning("Skipping hand %d due to error: %s", hand_idx + 1, e)
            optimizer.zero_grad()
            continue

    # Final flush
    if accum_steps % gcfg.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gcfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logging.info("Saved final checkpoint to %s", ckpt_dir)


if __name__ == "__main__":
    main()
