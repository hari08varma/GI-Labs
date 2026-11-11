"""Invented, pronounceable tokens. Hidden-world module (learners must not import)."""
from __future__ import annotations

import numpy as np

_C = list("bdfgklmnprstvz")
_V = list("aeiou")
_END = list("kmnrstx")


def make_token(rng: np.random.Generator) -> str:
    shape = rng.integers(0, 3)
    c = lambda: _C[rng.integers(len(_C))]
    v = lambda: _V[rng.integers(len(_V))]
    e = lambda: _END[rng.integers(len(_END))]
    if shape == 0:
        return c() + v() + e()                 # dax
    if shape == 1:
        return c() + v() + c() + v()           # tolu
    return c() + v() + c() + v() + e()         # tolim


class TokenPool:
    """Issues tokens unique within a system; can deliberately reuse earlier tokens."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.issued: list[str] = []            # every token ever issued, in order
        self._all: set[str] = set()

    def fresh(self, taken: set[str]) -> str:
        while True:
            t = make_token(self.rng)
            if t not in self._all and t not in taken and not t.startswith("#"):
                self._all.add(t)
                self.issued.append(t)
                return t

    def reuse(self, taken: set[str]) -> str | None:
        """Return an earlier-issued token not used in this system (crowding test)."""
        cand = [t for t in self.issued if t not in taken]
        if not cand:
            return None
        return cand[self.rng.integers(len(cand))]
