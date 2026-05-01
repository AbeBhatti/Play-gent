"""
verify.py — Step 8
Runs all pre-training checks. Reports issues only. Does not train.

Run in tmux:
  tmux new-session -d -s verify \
    "python verify.py 2>&1 | tee logs/verify_$(date +%Y%m%d_%H%M%S).log"
"""

import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def load_cfg(path: str = "config.yaml") -> OmegaConf:
    with open(path) as f:
        return OmegaConf.create(yaml.safe_load(f))


PASS = "  OK  "
FAIL = " FAIL "
WARN = " WARN "


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    line = f"[{tag}] {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return ok


def main() -> None:
    issues = 0

    # ── 1. Config ──────────────────────────────────────────────────────────────
    print("\n── 1. Config ─────────────────────────────────────────────────────")
    try:
        cfg = load_cfg()
        check("config.yaml loads", True)
    except Exception as e:
        check("config.yaml loads", False, str(e))
        print("\nCannot continue without config. Aborting.")
        sys.exit(1)

    # ── 2. Cicero data ─────────────────────────────────────────────────────────
    print("\n── 2. Cicero data ────────────────────────────────────────────────")
    cicero_dir = Path(cfg.paths.cicero_data)
    if not cicero_dir.exists():
        issues += check("Cicero data directory exists", False,
                        f"{cicero_dir} not found — run: "
                        "git clone --depth=1 https://github.com/facebookresearch/diplomacy_cicero "
                        "&& cp -r diplomacy_cicero/data/cicero_redacted_games training/data/cicero_games")
    else:
        games = list(cicero_dir.glob("*.json"))
        ok = len(games) > 0
        issues += not check("Cicero games present", ok,
                            f"{len(games)} JSON files" if ok else "directory is empty")

        if ok:
            # Count training examples (messages per phase across all games)
            import json
            total_examples = 0
            for g in games:
                try:
                    data = json.loads(g.read_text())
                    for phase in data.get("phases", []):
                        msgs = phase.get("messages", [])
                        if msgs:
                            total_examples += len(msgs)
                except Exception:
                    pass
            check("Cicero training examples countable", True,
                  f"{total_examples:,} message examples across {len(games)} games")

    # ── 3. Poker data ──────────────────────────────────────────────────────────
    print("\n── 3. Poker data ─────────────────────────────────────────────────")
    poker_dir = Path(cfg.paths.poker_data)
    if not poker_dir.exists():
        issues += check("Poker data directory exists", False,
                        f"{poker_dir} not found — check symlink at training/data/poker")
    else:
        check("Poker data directory exists", True, str(poker_dir))
        total_hands = 0
        months_found = 0
        for month_dir in sorted(poker_dir.iterdir()):
            hdb = month_dir / "hdb"
            if hdb.exists():
                with open(hdb) as f:
                    count = sum(1 for line in f if line.strip())
                total_hands += count
                months_found += 1
        ok = total_hands > 0
        issues += not check("Poker hands available", ok,
                            f"{total_hands:,} hands across {months_found} month(s)")

    # ── 4. Diplomacy engine ────────────────────────────────────────────────────
    print("\n── 4. Diplomacy engine ───────────────────────────────────────────")
    try:
        from diplomacy import Game
        import random as _random

        game = Game()
        rng = _random.Random(0)
        phase_count = 0

        while not game.is_game_done and phase_count < cfg.diplomacy.max_phases:
            possible = game.get_all_possible_orders()
            for pname in game.powers:
                if not game.powers[pname].is_eliminated():
                    orders = [
                        rng.choice(possible[loc])
                        for loc in game.get_orderable_locations(pname)
                        if loc in possible and possible[loc]
                    ]
                    game.set_orders(pname, orders)
            game.process()
            phase_count += 1

        winner = max(game.powers.items(), key=lambda x: len(x[1].centers))
        check("Diplomacy engine runs full game", True,
              f"{phase_count} phases, winner={winner[0]} ({len(winner[1].centers)} SCs), "
              f"done={game.is_game_done}")
    except ImportError:
        issues += check("Diplomacy engine runs full game", False,
                        "diplomacy not installed — run: pip install diplomacy")
    except Exception as e:
        issues += check("Diplomacy engine runs full game", False, str(e))

    # ── 5. SFT forward pass ────────────────────────────────────────────────────
    print("\n── 5. SFT forward pass ───────────────────────────────────────────")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Try SFT checkpoint first; fall back to base model if not yet trained.
        # A directory existing is not sufficient — it must contain model weights.
        sft_ckpt = Path(cfg.paths.checkpoints.sft)
        sft_has_weights = sft_ckpt.is_dir() and any(
            sft_ckpt.glob("*.safetensors")
        ) or any(sft_ckpt.glob("pytorch_model*.bin")) if sft_ckpt.is_dir() else False
        model_id = str(sft_ckpt) if sft_has_weights else cfg.model.name
        source = "SFT checkpoint" if sft_has_weights else f"base model ({cfg.model.name})"

        if not sft_has_weights:
            print(f"  [{WARN}] SFT checkpoint not found — using base model for forward pass check")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if cfg.sft.bf16 else torch.float32,
            device_map="auto",
        )
        model.eval()

        probe = "Negotiate a deal for a vintage camera listed at $45."
        inputs = tokenizer(probe, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        ok = out.logits is not None and out.logits.shape[0] == 1
        check(f"SFT forward pass ({source})", ok,
              f"logits shape {tuple(out.logits.shape)}")

        # ── 6. Parameter count ──────────────────────────────────────────────
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        check("Model parameter count", True,
              f"total={total_params/1e6:.1f}M  trainable={trainable/1e6:.1f}M")

    except ImportError as e:
        issues += not check("SFT forward pass", False,
                            f"Missing dependency: {e} — run: pip install transformers torch")
    except Exception as e:
        issues += not check("SFT forward pass", False, str(e))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if issues == 0:
        print("All checks passed. Ready for training.")
        print("Run training only after explicit confirmation.")
    else:
        print(f"{issues} issue(s) found. Resolve before training.")
    print("=" * 55 + "\n")

    sys.exit(0 if issues == 0 else 1)


if __name__ == "__main__":
    main()
