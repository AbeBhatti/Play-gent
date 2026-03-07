import json
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


def test_embeddings(path: str = "selfplay_states.json", n_examples: int = 10) -> None:
    """
    Quick sanity check for text embeddings:
      1. Load first n_examples states.
      2. Embed state_text using all-MiniLM-L6-v2.
      3. Print shape of each embedding (should be (384,)).
      4. Confirm embeddings are not all zeros or all identical.
      5. Print cosine similarity between first two embeddings.
    """
    print(f"Loading self-play states from {path}...")
    with open(path, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not data:
        print("Dataset is empty; cannot test embeddings.")
        return

    states = data[: min(n_examples, len(data))]
    texts = [ex.get("state_text", "") for ex in states]

    print(f"Loaded {len(texts)} examples for embedding test.")

    print("Loading sentence transformer model 'sentence-transformers/all-MiniLM-L6-v2'...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Model loaded.")

    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    print("\nEmbedding shapes:")
    for i, emb in enumerate(embeddings):
        print(f"  Example {i}: {emb.shape}")

    # Check for all-zero or identical embeddings.
    norms = np.linalg.norm(embeddings, axis=1)
    all_zero = np.allclose(norms, 0.0)
    if all_zero:
        print("[WARNING] All embeddings appear to be zero-vectors.")
    else:
        print("Embeddings have non-zero norms (good).")

    all_identical = all(
        np.allclose(embeddings[i], embeddings[0]) for i in range(1, len(embeddings))
    )
    if all_identical:
        print("[WARNING] All embeddings appear to be identical.")
    else:
        print("Embeddings are not all identical (good).")

    if len(embeddings) >= 2:
        v0 = embeddings[0]
        v1 = embeddings[1]
        denom = (np.linalg.norm(v0) * np.linalg.norm(v1)) or 1e-8
        cosine_sim = float(np.dot(v0, v1) / denom)
        print(f"\nCosine similarity between first two embeddings: {cosine_sim:.4f}")
        if cosine_sim >= 0.999:
            print("[WARNING] Cosine similarity is extremely close to 1.0; "
                  "embeddings may be too similar.")
    else:
        print("Not enough embeddings to compute cosine similarity.")


def main() -> None:
    test_embeddings()


if __name__ == "__main__":
    main()

