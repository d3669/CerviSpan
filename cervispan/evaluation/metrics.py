"""Patient-level percentile bootstrap for manuscript-style 95% intervals."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(count: int, statistic, replicates: int = 20000, seed: int = 42):
    if replicates <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sample = rng.integers(0, count, count)
        try:
            value = float(statistic(sample))
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            values.append(value)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None