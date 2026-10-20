"""Zero-shot leave-self-out cosine retrieval over the pooled patient gallery."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def nearest(features: np.ndarray, ids: list[str], k: int):
    if not 0 < k < len(ids):
        raise ValueError("k must be between 1 and patient count minus one")
    norm = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    similarity = norm @ norm.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)[:, :k]
    return [{"query_id": ids[i], "rank": rank + 1, "neighbour_id": ids[j], "similarity": float(similarity[i, j])}
            for i in range(len(ids)) for rank, j in enumerate(order[i])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    with np.load(a.features) as file:
        rows = nearest(file["X"], file["case_id"].astype(str).tolist(), a.k)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()