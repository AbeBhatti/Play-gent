#!/bin/bash
# run_all.sh — Chains all 4 training stages in a single tmux session.
#
# Usage:
#   bash run_all.sh            # launches tmux session and detaches
#   tmux attach -t playgent_training   # reattach anytime
#   tail -f logs/playgent_training_<ts>.log  # follow log from outside tmux

set -e

mkdir -p logs

# ── Launch into tmux (first call, not already inside the session) ──────────────
if [ "$1" != "--run" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="logs/playgent_training_${TS}.log"
    SESSION="playgent_training"
    SCRIPT=$(realpath "$0")

    tmux new-session -d -s "$SESSION" \
        "bash \"$SCRIPT\" --run 2>&1 | tee \"$LOG\"; echo; echo '=== All stages finished ==='"

    echo ""
    echo "  Session : $SESSION"
    echo "  Log     : $LOG"
    echo ""
    echo "  Attach  : tmux attach -t $SESSION"
    echo "  Detach  : Ctrl-b d"
    echo "  Tail log: tail -f $LOG"
    echo ""
    exit 0
fi

# ── Pipeline (runs inside tmux) ────────────────────────────────────────────────
# Always run from the directory this script lives in
cd "$(dirname "$(realpath "$0")")"
source venv/bin/activate

echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo ""

echo "=== Stage 1: SFT ==="
python train_sft.py
echo "=== SFT complete ==="
echo ""

echo "=== Stage 2: GRPO Diplomacy ==="
python train_grpo_diplomacy.py
echo "=== GRPO Diplomacy complete ==="
echo ""

echo "=== Stage 3: GRPO Poker ==="
python train_grpo_poker.py
echo "=== GRPO Poker complete ==="
echo ""

echo "=== Stage 4: Eval ==="
python eval_marketplace.py
echo "=== Eval complete ==="
echo ""

echo "All stages complete: $(date)"
