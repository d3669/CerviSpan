"""Manifest loading and coordinate-preserving crop sampling."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def rows_from_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"case_id", "split", "npz_path"}.issubset(rows[0]):
        raise ValueError("Expected case_id,split,npz_path")
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patient case_id")
    for row in rows:
        path_to_data = Path(row["npz_path"])
        row["npz_path"] = str(path_to_data if path_to_data.is_absolute() else path.parent / path_to_data)
    return rows


def crop(array: np.ndarray, start: tuple[int, int, int], size: int) -> np.ndarray:
    output = np.zeros((size,) * 3, dtype=array.dtype)
    source = []
    target = []
    for axis, first in enumerate(start):
        lo, hi = max(first, 0), min(first + size, array.shape[axis])
        source.append(slice(lo, hi))
        target.append(slice(lo - first, hi - first))
    output[tuple(target)] = array[tuple(source)]
    return output


def centered(array: np.ndarray, size: int) -> np.ndarray:
    return crop(array, tuple((n - size) // 2 for n in array.shape), size)


class PretrainDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], global_size: int = 160, local_size: int = 48,
                 n_local: int = 4, seed: int = 42):
        self.rows, self.global_size, self.local_size, self.n_local, self.seed = rows, global_size, local_size, n_local, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.rows[index]["npz_path"]) as file:
            volume = file["ct"].astype(np.float32)
        # Torch worker RNG changes every call/epoch while remaining seedable by DataLoader.
        nonce = int(torch.randint(0, 2**62, ()).item())
        rng = np.random.default_rng(self.seed + nonce + index)
        global_views, global_boxes, local_views, local_boxes = [], [], [], []
        for _ in range(2):
            start = tuple(int(rng.integers(-(max(0, self.global_size - n)), max(0, n - self.global_size) + 1))
                          for n in volume.shape)
            global_views.append(crop(volume, start, self.global_size))
            global_boxes.append(np.array([start, tuple(x + self.global_size for x in start)], dtype=np.float32))
        for _ in range(self.n_local):
            parent = int(rng.integers(0, 2))
            start = tuple(int(global_boxes[parent][0, axis] + rng.integers(0, self.global_size - self.local_size + 1))
                          for axis in range(3))
            local_views.append(crop(volume, start, self.local_size))
            local_boxes.append(np.array([start, tuple(x + self.local_size for x in start)], dtype=np.float32))
        return (torch.from_numpy(np.stack(global_views)[:, None]), torch.from_numpy(np.stack(local_views)[:, None]),
                torch.from_numpy(np.stack(global_boxes)), torch.from_numpy(np.stack(local_boxes)))