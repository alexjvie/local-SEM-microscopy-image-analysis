from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def skeleton_length_pixels(skel: np.ndarray) -> float:
    """Connection-length with diagonal sqrt(2) weighting."""
    s = skel.astype(bool)
    if s.sum() == 0:
        return 0.0

    right = s[:, :-1] & s[:, 1:]
    down = s[:-1, :] & s[1:, :]

    dr = s[:-1, :-1] & s[1:, 1:]
    dl = s[:-1, 1:] & s[1:, :-1]

    hv = float(right.sum() + down.sum())
    diag = float(dr.sum() + dl.sum())
    return hv * 1.0 + diag * math.sqrt(2.0)


def stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def hist(xs: List[float], bins: int = 18) -> Optional[Dict[str, Any]]:
    if not xs:
        return None
    x = np.asarray(xs, dtype=np.float64)
    if float(x.max() - x.min()) <= 1e-12:
        return {"bin_edges": [float(x.min()), float(x.max())], "counts": [int(x.size)]}
    counts, edges = np.histogram(x, bins=bins)
    return {
        "bin_edges": [float(round(e, 6)) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }
