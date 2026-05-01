"""
train_sft.py — Phase 1
Fine-tunes TinyLlama on 40 Cicero game JSONs using next-token prediction.

Run via run_all.sh or directly:
  python train_sft.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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
            logging.FileHandler(log_dir / f"sft_{ts}.log"),
            logging.StreamHandler(),
        ],
    )


# ── Data ───────────────────────────────────────────────────────────────────────

def format_example(phase: dict, msg_idx: int, cfg) -> str:
    name = phase.get("name", "")
    state = phase.get("state", {})
    centers = state.get("centers", {})
    units = state.get("units", {})
    messages = phase.get("messages", [])

    target = messages[msg_idx]
    start = max(0, msg_idx - cfg.sft.context_messages)
    prev = messages[start:msg_idx]

    centers_str = " | ".join(
        f"{p}: {sorted(cs)}" for p, cs in sorted(centers.items())
    )
    units_str = " | ".join(
        f"{p}: {sorted(us)}" for p, us in sorted(units.items())
    )
    prev_str = "\n".join(
        f"[{m['sender']} -> {m['recipient']}]: {m['message']}" for m in prev
    ) or "(none)"
    target_str = f"[{target['sender']} -> {target['recipient']}]: {target['message']}"

    return (
        f"Phase: {name}\n"
        f"Centers: {centers_str}\n"
        f"Units: {units_str}\n"
        f"Previous messages this phase:\n{prev_str}\n"
        f"Generate: {target_str}"
    )


def load_examples(cfg) -> list:
    cicero_dir = Path(cfg.paths.cicero_data)
    examples = []
    for game_file in sorted(cicero_dir.glob("*.json")):
        try:
            data = json.loads(game_file.read_text())
        except Exception as e:
            logging.warning("Skipping %s: %s", game_file.name, e)
            continue
        for phase in data.get("phases", []):
            msgs = phase.get("messages", [])
            for i in range(len(msgs)):
                examples.append(format_example(phase, i, cfg))
    return examples


class SFTDataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.examples[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


# ── Training ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, cfg, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    steps = 0
    if train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        if train:
            (loss / cfg.sft.gradient_accumulation_steps).backward()
            if (step + 1) % cfg.sft.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        steps += 1

        if train and step % cfg.sft.logging_steps == 0:
            logging.info("  step %d  loss %.4f", step, total_loss / steps)

    return total_loss / max(steps, 1)


def main() -> None:
    cfg = load_cfg()
    setup_logging(cfg)

    logging.info("Loading Cicero examples...")
    examples = load_examples(cfg)
    logging.info("Total training examples: %d", len(examples))

    split = int(len(examples) * cfg.sft.train_val_split)
    train_ex, val_ex = examples[:split], examples[split:]
    logging.info("Train: %d  Val: %d", len(train_ex), len(val_ex))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=torch.bfloat16 if cfg.sft.bf16 else torch.float32,
        device_map="auto",
    )

    train_loader = DataLoader(
        SFTDataset(train_ex, tokenizer, cfg.sft.max_length),
        batch_size=cfg.sft.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SFTDataset(val_ex, tokenizer, cfg.sft.max_length),
        batch_size=cfg.sft.batch_size,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.sft.learning_rate,
        weight_decay=cfg.sft.weight_decay,
    )

    for epoch in range(cfg.sft.num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, cfg, train=True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, optimizer, cfg, train=False)
        logging.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch + 1, cfg.sft.num_epochs, train_loss, val_loss,
        )

    ckpt = Path(cfg.paths.checkpoints.sft)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    logging.info("Saved SFT checkpoint to %s", ckpt)


if __name__ == "__main__":
    main()
