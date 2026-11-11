"""Active learning: what to ask the checker (plan §5.5).

Query by committee (Seung, Opper & Sompolinsky 1992): the version space V is every
surviving candidate within L* + Delta bits. Ask about the input cell where the
posterior-weighted predictions of V disagree most (maximum entropy). When V has
collapsed there is no disagreement left, so the query becomes a lock check: a fresh
cell the candidate has not yet been tested on (plan §5.6 condition 2).
"""
from __future__ import annotations

import numpy as np

from .mdl import Scored


def prediction_entropy(sc: Scored, vs: np.ndarray, n_values: int) -> np.ndarray:
    """Entropy (bits) of the weighted vote of V at every cell."""
    w = sc.posterior(vs)
    S = sc.sem[vs].astype(np.int64)                      # (|V|, C)
    C = S.shape[1]
    probs = np.zeros((n_values, C))
    for v in range(n_values):
        probs[v] = ((S == v) * w[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(probs > 0, probs * np.log2(probs), 0.0)).sum(axis=0)
    return h


def choose_query(sc: Scored, vs: np.ndarray, known: set[int], n_values: int,
                 rng: np.random.Generator, allowed: np.ndarray | None = None) -> tuple[int, float] | None:
    """Return (cell, entropy) for the next checker query, or None if no cell is left."""
    C = sc.sem.shape[1]
    free = np.ones(C, dtype=bool)
    if known:
        free[np.fromiter(known, dtype=np.int64)] = False
    if allowed is not None:
        free &= allowed
    if not free.any():
        return None
    h = prediction_entropy(sc, vs, n_values) if len(vs) > 1 else np.zeros(C)
    h = np.where(free, h, -1.0)
    top = h.max()
    ties = np.flatnonzero(np.isclose(h, top) & free)
    cell = int(ties[int(rng.integers(len(ties)))])
    return cell, float(max(top, 0.0))
