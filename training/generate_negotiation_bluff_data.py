"""
Generate synthetic negotiation bluff dataset for the bluff classifier.

Output: training/data/negotiation_bluff_labels.json
Format: [{"text": "...", "is_bluff": true/false}, ...]

We generate:
- 500 bluff examples (is_bluff = True) — seller asserting a fake floor.
- 4500 non-bluff examples (is_bluff = False) — neutral / exploratory / accepting messages.
"""

import json
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "negotiation_bluff_labels.json"


BLUFF_TEMPLATES = [
    "look i really cant go lower than ${price}, thats my final offer",
    "i really cant go any lower than ${price}, thats my absolute floor",
    "thats the lowest i can go, been getting a lot of interest at ${price}",
    "im firm on ${price}, i have other buyers lined up",
    "cant do it for less than ${price}, final price",
    "honestly ${price} is my bottom line, cant go lower",
    "got a lot of messages already, ${price} is the best i can do",
]


NON_BLUFF_TEMPLATES = [
    "hey is this still available?",
    "can you do ${price}?",
    "i have a trade offer from another seller, can you do ${price}?",
    "just checking back, any flexibility on the price?",
    "ok ${price} works for me",
    "ill take it at ${price}",
    "i have another buyer offering more, can you match ${price}?",
    "thanks for the info, im thinking about ${price}",
    "if you can do ${price} i can pick up today",
]


def _sample_price() -> int:
    """Sample a realistic small-item price in the $15–$200 range."""
    return random.randint(15, 200)


def _fill_template(template: str) -> str:
    price = _sample_price()
    text = template.replace("${price}", str(price))
    # Light stylistic variation: optional punctuation and casing tweaks.
    if random.random() < 0.3:
        text = text.replace("i ", "I ")
    if random.random() < 0.2:
        text = text + "!"
    return text


def generate_examples(num_bluff: int = 500, num_non_bluff: int = 4500):
    random.seed(42)

    examples = []

    # Bluff examples
    for _ in range(num_bluff):
        template = random.choice(BLUFF_TEMPLATES)
        text = _fill_template(template)
        examples.append({"text": text, "is_bluff": True})

    # Non-bluff examples
    for _ in range(num_non_bluff):
        template = random.choice(NON_BLUFF_TEMPLATES)
        text = _fill_template(template)
        examples.append({"text": text, "is_bluff": False})

    random.shuffle(examples)
    return examples


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    examples = generate_examples()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    num_bluff = sum(1 for ex in examples if ex["is_bluff"])
    num_non_bluff = len(examples) - num_bluff
    print(
        f"Wrote {len(examples)} examples to {OUTPUT_PATH} "
        f"({num_bluff} bluff, {num_non_bluff} non-bluff)"
    )


if __name__ == "__main__":
    main()

