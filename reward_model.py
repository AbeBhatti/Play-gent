import json
import random
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class DiplomacyRewardModel(nn.Module):
    """
    DistilBERT-based reward regressor for Diplomacy self-play states.

    Input: state_text (encoded by distilbert-base-uncased)
    Output: raw scalar (regression target), no sigmoid.
    """

    def __init__(self, base_model: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Ignore token_type_ids and other unused fields if present.
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.head(pooled).squeeze(-1)

    def score(
        self,
        state_text: str,
        action_text: str,
        tokenizer: AutoTokenizer,
        device: torch.device,
    ) -> float:
        """
        Convenience helper: encode (state, action) text pair and return scalar reward.
        """
        self.eval()
        combined = f"STATE: {state_text}\nACTION: {action_text}"
        with torch.no_grad():
            enc = tokenizer(
                combined,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            pred = self(**enc)
        return float(pred.item())


class StatesDataset(Dataset):
    """Simple dataset wrapping tokenized state_text and scalar rewards."""

    def __init__(self, texts: List[str], rewards: List[float], tokenizer: AutoTokenizer, max_length: int = 128):
        self.texts = texts
        self.rewards = rewards
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        reward = self.rewards[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "reward": torch.tensor(reward, dtype=torch.float32),
        }
        return item


def score_state(model: DiplomacyRewardModel, tokenizer: AutoTokenizer, state_text: str, device: torch.device) -> float:
    """Helper to score a single state_text with the trained model."""
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            state_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        pred = model(**enc)
    return float(pred.item())


def train(
    data_path: str = "selfplay_states.json",
    output_model_path: str = "reward_model.pt",
    loss_plot_path: str = "reward_model_loss.png",
    epochs: int = 2,
    batch_size: int = 128,
    lr: float = 2e-5,
) -> None:
    # Device setup
    use_cuda = torch.cuda.is_available()
    print("torch.cuda.is_available():", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("Using GPU:", torch.cuda.get_device_name(0))

    # Load data
    print(f"Loading self-play states from {data_path}...")
    with open(data_path, "r") as f:
        data = json.load(f)

    # Limit to 50k random examples for faster training.
    random.shuffle(data)
    data = data[:50000]

    texts: List[str] = [ex.get("state_text", "") for ex in data]
    rewards: List[float] = [float(ex.get("reward", 0.0)) for ex in data]

    print(f"Total examples: {len(texts)}")

    # Tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = StatesDataset(texts, rewards, tokenizer, max_length=128)

    # 90/10 train/val split
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"Train examples: {n_train} | Val examples: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, optimizer, loss
    model = DiplomacyRewardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        # Train epoch
        model.train()
        running_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["reward"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * input_ids.size(0)

        avg_train_loss = running_train_loss / n_train

        # Validation epoch
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} - val"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["reward"].to(device)

                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(preds, targets)
                running_val_loss += loss.item() * input_ids.size(0)

        avg_val_loss = running_val_loss / n_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  -> New best val loss. Model saved to {output_model_path}")

    # Plot loss curves
    epochs_axis = np.arange(1, epochs + 1)
    plt.figure()
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Reward Model Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curves saved to {loss_plot_path}")


if __name__ == "__main__":
    train()

