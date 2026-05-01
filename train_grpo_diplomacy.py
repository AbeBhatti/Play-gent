"""
train_grpo_diplomacy.py — Phase 2
GRPO fine-tuning on the Diplomacy board game engine.
Loads from checkpoints/sft/, saves to checkpoints/grpo_diplomacy/.

Run via run_all.sh or directly:
  python train_grpo_diplomacy.py
"""

import copy
import logging
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from diplomacy import Game
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
            logging.FileHandler(Path(cfg.paths.logs) / f"grpo_diplomacy_{ts}.log"),
            logging.StreamHandler(),
        ],
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

def build_prompt(game: Game, power: str) -> str:
    phase = game.get_current_phase()
    power_obj = game.powers[power]
    units = sorted(power_obj.units)
    centers = sorted(power_obj.centers)
    all_centers = {p: sorted(game.powers[p].centers) for p in game.powers}
    all_centers_str = " | ".join(f"{p}: {cs}" for p, cs in sorted(all_centers.items()))

    return (
        f"You are playing Diplomacy as {power}.\n"
        f"Phase: {phase}\n"
        f"Your units: {units}\n"
        f"Your supply centers: {centers}\n"
        f"All powers supply centers: {all_centers_str}\n"
        f"Submit your orders one per line:\n"
    )


# ── Order Parsing ──────────────────────────────────────────────────────────────

def parse_orders(text: str, game: Game, power: str) -> list:
    """
    Extract order strings from model output.
    Validates each order against game.get_all_possible_orders().
    Invalid orders silently replaced with HOLD. Never raises.
    """
    try:
        possible = game.get_all_possible_orders()
        orderable = set(game.get_orderable_locations(power))
        valid_orders = {o for loc in orderable for o in possible.get(loc, [])}

        parsed = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line in valid_orders:
                parsed.append(line)
            else:
                # Try prefix match (model may add punctuation)
                match = next((o for o in valid_orders if line.startswith(o[:6])), None)
                if match:
                    parsed.append(match)

        # Fill any unordered locations with HOLD
        ordered_units = {o.split()[1] for o in parsed if len(o.split()) > 1}
        for loc in orderable:
            unit_key = next(
                (u.split()[-1] for u in game.powers[power].units if u.endswith(loc)),
                None,
            )
            if unit_key and unit_key not in ordered_units:
                hold = next((o for o in possible.get(loc, []) if "H" in o), None)
                if hold:
                    parsed.append(hold)

        return parsed
    except Exception:
        return []


# ── GRPO Core ──────────────────────────────────────────────────────────────────

def sequence_log_probs(
    model, input_ids: torch.Tensor, gen_ids: torch.Tensor
) -> torch.Tensor:
    """
    Returns scalar sum of log_probs over generated tokens for each item in batch.
    input_ids : [B, prompt_len]
    gen_ids   : [B, gen_len]
    returns   : [B]
    """
    full = torch.cat([input_ids, gen_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits  # [B, full_len, vocab]
    # Logits at position i predict token i+1; shift so logits align with gen_ids
    shift = input_ids.shape[1] - 1
    gen_logits = logits[:, shift : shift + gen_ids.shape[1], :]
    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_lp = log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # [B, gen_len]
    return token_lp.sum(dim=-1)  # [B]


def grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    epsilon: float,
    kl_coeff: float,
) -> torch.Tensor:
    advantages = rewards - rewards.mean()
    # Normalize advantages for stable training
    if advantages.std() > 1e-8:
        advantages = advantages / (advantages.std() + 1e-8)

    ratios = torch.exp(log_probs - ref_log_probs.detach())
    clipped = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
    pg = -torch.min(ratios * advantages, clipped * advantages).mean()
    kl = (log_probs - ref_log_probs.detach()).mean()
    return pg + kl_coeff * kl


# ── Episode ────────────────────────────────────────────────────────────────────

def run_episode(model, ref_model, tokenizer, cfg, optimizer, rng: random.Random) -> float:
    """
    Plays one full game as a randomly chosen power.
    Returns mean reward across all phases.
    """
    gcfg = cfg.grpo_diplomacy
    game = Game()
    power = rng.choice(list(game.powers.keys()))

    episode_rewards = []
    phase_count = 0
    optimizer.zero_grad()
    accum_loss = torch.tensor(0.0, device=model.device)
    accum_steps = 0

    while not game.is_game_done and phase_count < gcfg.max_phases_per_episode:
        if game.powers[power].is_eliminated():
            break

        prompt = build_prompt(game, power)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]  # [1, prompt_len]

        # Generate N responses
        input_batch = input_ids.repeat(gcfg.num_generations, 1)
        with torch.no_grad():
            gen_out = model.generate(
                input_batch,
                max_new_tokens=gcfg.max_completion_length,
                do_sample=True,
                temperature=gcfg.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = gen_out[:, input_ids.shape[1]:]  # [N, gen_len]

        # Submit first generation's parsed orders to advance the game
        first_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        orders = parse_orders(first_text, game, power)
        try:
            game.set_orders(power, orders)
            # Other powers play randomly
            possible = game.get_all_possible_orders()
            for other in game.powers:
                if other != power and not game.powers[other].is_eliminated():
                    other_orders = [
                        rng.choice(possible[loc])
                        for loc in game.get_orderable_locations(other)
                        if loc in possible and possible[loc]
                    ]
                    game.set_orders(other, other_orders)
            game.process()
        except Exception:
            pass

        # Reward based on game state after this phase
        if game.is_game_done:
            sc_counts = {p: len(game.powers[p].centers) for p in game.powers}
            max_sc = max(sc_counts.values())
            reward = gcfg.reward_win if sc_counts[power] == max_sc else gcfg.reward_loss
        elif game.powers[power].is_eliminated():
            reward = gcfg.reward_loss
        else:
            reward = gcfg.reward_draw

        episode_rewards.append(reward)

        # GRPO update over the N generations
        rewards_t = torch.full(
            (gcfg.num_generations,), reward, dtype=torch.float32, device=model.device
        )

        # Recompute log_probs with grad for current policy
        full = torch.cat([input_batch, gen_ids], dim=1)
        logits = model(full).logits
        shift = input_ids.shape[1] - 1
        gen_logits = logits[:, shift : shift + gen_ids.shape[1], :]
        lp_current = F.log_softmax(gen_logits, dim=-1)
        lp_current = lp_current.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).sum(-1)

        with torch.no_grad():
            lp_ref = sequence_log_probs(ref_model, input_batch, gen_ids)

        loss = grpo_loss(lp_current, lp_ref, rewards_t, gcfg.epsilon, gcfg.kl_coeff)
        (loss / gcfg.gradient_accumulation_steps).backward()
        accum_loss = accum_loss + loss.detach()
        accum_steps += 1

        if accum_steps % gcfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gcfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        phase_count += 1

    # Final grad flush
    if accum_steps % gcfg.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gcfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return sum(episode_rewards) / max(len(episode_rewards), 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_cfg()
    setup_logging(cfg)
    gcfg = cfg.grpo_diplomacy

    logging.info("Loading SFT checkpoint from %s", cfg.paths.checkpoints.sft)
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.checkpoints.sft)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if gcfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.paths.checkpoints.sft, dtype=dtype, device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.paths.checkpoints.sft, dtype=dtype, device_map="auto"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=gcfg.learning_rate)
    rng = random.Random(42)
    ckpt_dir = Path(cfg.paths.checkpoints.grpo_diplomacy)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, gcfg.num_episodes + 1):
        mean_reward = run_episode(model, ref_model, tokenizer, cfg, optimizer, rng)
        if episode % gcfg.logging_steps == 0:
            logging.info("Episode %d/%d  mean_reward=%.4f", episode, gcfg.num_episodes, mean_reward)
        if episode % gcfg.save_steps == 0:
            step_ckpt = ckpt_dir / f"episode_{episode}"
            model.save_pretrained(step_ckpt)
            tokenizer.save_pretrained(step_ckpt)
            logging.info("Saved checkpoint at episode %d", episode)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logging.info("Saved final checkpoint to %s", ckpt_dir)


if __name__ == "__main__":
    main()
