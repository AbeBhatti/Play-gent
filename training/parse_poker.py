"""
Parse IRC poker pdb files and produce labeled bluff examples.

Line format: player_name  timestamp  num_players  position  preflop  flop  turn  river  bankroll  won  won2  [cards]
Action codes: f=fold, c=call, r=raise, b=bet, k=check, B=blind, -=no action
Cards at end of line = player went to showdown.

BLUFF = True: preflop has 'r' or 'b', hand ends in fold (last non-dash action ends in 'f'), no cards at end.
BLUFF = False: cards at end (showdown) OR folded with no aggression.
"""

import json
import os
import re
from pathlib import Path

BASE_POKER = Path(__file__).resolve().parent / "data" / "poker"
PDB_DIR = BASE_POKER / "IRCdata" / "holdem" / "199901" / "pdb"
OUT_PATH = BASE_POKER / "bluff_labels.json"
MAX_EXAMPLES = 50_000

CARD_PATTERN = re.compile(r"^[2-9TJKQA][cdhs]$", re.IGNORECASE)


def _is_card_token(s: str) -> bool:
    return bool(s and CARD_PATTERN.match(s.strip()))


def _has_cards_at_end(tokens: list) -> bool:
    """True if line ends with card tokens (showdown)."""
    if len(tokens) <= 11:
        return False
    # Last 1 or 2 tokens can be cards (e.g. "Ks Kh" or single card)
    tail = tokens[11:]
    return all(_is_card_token(t) for t in tail) and len(tail) >= 1


def _last_non_dash_ends_in_f(preflop: str, flop: str, turn: str, river: str) -> bool:
    """Last non-dash action field ends in 'f' (fold)."""
    for s in (river, turn, flop, preflop):
        if s and s != "-":
            return s.strip().endswith("f")
    return False


def _preflop_aggressive(preflop: str) -> bool:
    """Preflop contains raise or bet."""
    return "r" in (preflop or "") or "b" in (preflop or "")


def parse_line(line: str) -> dict | None:
    """
    Returns {"text": str, "is_bluff": bool} or None if line invalid.
    """
    line = line.strip()
    if not line:
        return None
    tokens = line.split()
    if len(tokens) < 11:
        return None
    player_name = tokens[0]
    timestamp = tokens[1]
    num_players = tokens[2]
    position = tokens[3]
    preflop = tokens[4]
    flop = tokens[5]
    turn = tokens[6]
    river = tokens[7]
    bankroll = tokens[8]
    won = tokens[9]
    won2 = tokens[10]
    try:
        pot = abs(int(won))
    except ValueError:
        pot = 0

    has_cards = _has_cards_at_end(tokens)
    ends_in_fold = _last_non_dash_ends_in_f(preflop, flop, turn, river)
    aggressive = _preflop_aggressive(preflop)

    # BLUFF = True: aggressive preflop, ended in fold, no showdown
    is_bluff = aggressive and ends_in_fold and not has_cards
    # BLUFF = False: showdown OR fold with no aggression
    if has_cards:
        is_bluff = False
    elif not aggressive and ends_in_fold:
        is_bluff = False

    text = (
        f"Position {position} of {num_players}. "
        f"Preflop: {preflop}. Flop: {flop}. Turn: {turn}. River: {river}. Pot: {pot}."
    )
    return {"text": text, "is_bluff": is_bluff}


def main():
    os.makedirs(OUT_PATH.parent, exist_ok=True)
    examples = []
    # Files are named pdb.^, pdb.A2k, etc. (not *.pdb)
    if PDB_DIR.exists():
        pdb_files = [f for f in PDB_DIR.iterdir() if f.is_file() and f.name.startswith("pdb.")]
    else:
        pdb_files = []
    if not pdb_files:
        for d in BASE_POKER.rglob("pdb"):
            if d.is_dir():
                pdb_files.extend([f for f in d.iterdir() if f.is_file() and f.name.startswith("pdb.")])
                if pdb_files:
                    break
    if not pdb_files:
        print(f"ERROR: No pdb files in {PDB_DIR} or under {BASE_POKER}")
        return
    for pdb_path in pdb_files:
        if len(examples) >= MAX_EXAMPLES:
            break
        try:
            with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if len(examples) >= MAX_EXAMPLES:
                        break
                    rec = parse_line(line)
                    if rec is not None:
                        examples.append(rec)
        except Exception as e:
            print(f"Warning: {pdb_path}: {e}")
    with open(OUT_PATH, "w") as f:
        json.dump(examples, f, indent=0)
    n = len(examples)
    n_bluff = sum(1 for e in examples if e["is_bluff"])
    print(f"Total examples: {n}")
    print(f"Class balance: is_bluff=True {n_bluff}, is_bluff=False {n - n_bluff}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
