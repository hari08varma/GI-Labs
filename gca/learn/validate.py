"""Validation and locking (plan §5.6, P7).

A candidate column locks when all of these hold:
1. version space collapse: every other candidate within Delta bits is refuted or
   agrees with it on the whole domain (rows of a Scored are distinct functions, so
   this means |V| = 1 and V = {the column});
2. at least K_LOCK checker-verified correct predictions on inputs not in experience
   (or, once the budget is gone, an SPRT accepting H1 over later observations);
3. a compression gain of at least delta_sprout;
4. no unresolved conflict in mismatch involving the column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..memory.columns import Column


@dataclass
class LockCheck:
    collapsed: bool
    confirmed: int
    gain_ok: bool
    no_conflict: bool
    sprt_ok: bool = False

    def ok(self, k_lock: int) -> bool:
        return self.collapsed and self.gain_ok and self.no_conflict and (
            self.confirmed >= k_lock or self.sprt_ok)

    def as_dict(self) -> dict:
        return {"collapsed": self.collapsed, "confirmed": self.confirmed, "gain_ok": self.gain_ok,
                "no_conflict": self.no_conflict, "sprt_ok": self.sprt_ok}


def collapsed(sem_vs: np.ndarray, col_sem: np.ndarray) -> bool:
    return len(sem_vs) >= 1 and all(np.array_equal(r, col_sem) for r in sem_vs)


def lock_check(col: Column, sem_vs: np.ndarray, gain: float, delta: float, conflicts: int,
               sprt_accept: float, budget_left: int) -> LockCheck:
    return LockCheck(collapsed=collapsed(sem_vs, col.sem), confirmed=len(col.stats.confirmed),
                     gain_ok=gain >= delta, no_conflict=conflicts == 0,
                     sprt_ok=budget_left <= 0 and col.stats.sprt >= sprt_accept)
