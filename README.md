# Playgent

Curriculum-trained RL negotiation agent. Work in progress.

## What it is

TinyLlama 1.1B trained on game data to negotiate in real-world marketplace scenarios. No marketplace data used in training — the model learns strategic behavior from Diplomacy and poker, then transfers zero-shot to price negotiation.

## Training pipeline

1. SFT on 40 Cicero Diplomacy games (5,924 examples)
2. GRPO self-play on Diplomacy (win/loss/draw reward)
3. GRPO self-play on poker (net profit/loss reward)

## Results

Evaluated zero-shot against an adversarial Llama-3.1-8B seller with hard floor enforcement.

- 98% deal closure rate
- 1.302x average capital return
- 1.664x best return
- Base TinyLlama baseline: 0% closure

## Stack

PyTorch, HuggingFace Transformers, TRL

## Status

Active development. Poker GRPO currently causes over-folding behavior in marketplace eval — reward signal redesign in progress.
