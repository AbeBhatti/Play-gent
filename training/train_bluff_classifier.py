"""
Train DistilBERT binary classifier on bluff labels.

Default data: training/data/poker/bluff_labels.json
Model: distilbert-base-uncased + linear 768→2
80/20 train/val stratified, 3 epochs, lr 2e-5, batch 32
Saves: training/checkpoints/bluff_classifier.pt, bluff_classifier_tokenizer/

Use --data to point at negotiation_bluff_labels.json and --output to choose
an alternative checkpoint path.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "poker" / "bluff_labels.json"
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_MODEL_PT = DEFAULT_CHECKPOINT_DIR / "bluff_classifier.pt"
TOKENIZER_DIR = DEFAULT_CHECKPOINT_DIR / "bluff_classifier_tokenizer"
MAX_LENGTH = 128
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 32


class BluffClassifier(nn.Module):
    """DistilBERT + linear head 768 → 2 (binary: not_bluff, bluff)."""

    def __init__(self, base_model: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.head(pooled)


class BluffDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Train bluff classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help=(
            "Path to JSON bluff label file "
            '(default: training/data/poker/bluff_labels.json)'
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_MODEL_PT),
        help=(
            "Path to save model checkpoint "
            "(default: training/checkpoints/bluff_classifier.pt)"
        ),
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    model_pt = Path(args.output)
    checkpoint_dir = model_pt.parent

    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        return
    with data_path.open() as f:
        data = json.load(f)
    texts = [x["text"] for x in data]
    labels = [1 if x["is_bluff"] else 0 for x in data]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = BluffDataset(X_train, y_train, tokenizer)
    val_ds = BluffDataset(X_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BluffClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(out, batch["labels"].to(device))
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                pred = out.argmax(dim=1)
                correct += (pred == batch["labels"].to(device)).sum().item()
                total += pred.size(0)
                all_pred.extend(pred.cpu().tolist())
                all_true.extend(batch["labels"].tolist())
        acc = correct / total if total else 0

        # F1 binary: bluff=1
        tp = sum(1 for p, t in zip(all_pred, all_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(all_pred, all_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(all_pred, all_true) if p == 0 and t == 1)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        print(f"Epoch {epoch + 1}/{EPOCHS}  Val accuracy: {acc:.4f}  Val F1: {f1:.4f}")

    if acc < 0.65:
        print(f"WARNING: Val accuracy {acc:.4f} < 0.65 (target). Consider more data or epochs.")
    torch.save(model.state_dict(), model_pt)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"Saved model to {model_pt}, tokenizer to {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
