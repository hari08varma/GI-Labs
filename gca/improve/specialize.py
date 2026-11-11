"""Specialization, the out-of-scope fix (plan §7.5): Ohlsson's constraint-based
specialization, as formalised in ICARUS by Choi & Ohlsson (2010).

Search the Boolean part of the learner's language for the cheapest condition b with
b = true on the column's validated inputs and b = false on the failing inputs (MDL
again: the cheapest such b in bits).
"""
from __future__ import annotations

import numpy as np


def find_condition(pool, good: list[int], bad: list[int], max_bits: float = 40.0):
    """Return (expr, sem, bits) or None."""
    sem, cost, exprs, _ = pool.flat("B")
    if len(sem) == 0 or not bad:
        return None
    g = np.array(sorted(set(good)), dtype=np.int64)
    b = np.array(sorted(set(bad)), dtype=np.int64)
    best = None
    for neg in (False, True):                     # b, or not-b at one extra bit
        S = 1 - sem if neg else sem
        ok = np.ones(len(sem), dtype=bool)
        if len(g):
            ok &= (S[:, g] == 1).all(axis=1)
        ok &= (S[:, b] == 0).all(axis=1)
        c = cost + (1.0 if neg else 0.0)
        idx = np.flatnonzero(ok & (c <= max_bits))
        if len(idx) == 0:
            continue
        i = int(idx[np.argmin(c[idx])])
        if best is None or c[i] < best[2]:
            best = (("not", exprs[i]) if neg else exprs[i], S[i].astype(np.int64), float(c[i]))
    return best
