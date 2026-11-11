"""Resonance router (plan §6.2): ART-style, non-parametric, no trainable weights.

The query key is built like the column keys (memory/columns.py). Candidates are the
columns sorted by state (locked first) and then by cosine similarity. Resonance needs
cos(q, key) >= rho. If nothing resonates the subgoal is novel and goes to induction.

Design note (build log): with 4 equally weighted fields, cos ~ m/4 when m fields match,
so the plan's rho = 0.6 also admits a different frame of the same context (cos 0.75).
rho = 0.8 makes a subgoal resonate only with columns of its own frame (cos 1.0).
"""
from __future__ import annotations

import numpy as np

from ..memory.columns import RANK, Column, column_key
from ..memory.vsa import VSA


class Router:
    def __init__(self, vsa: VSA, vigilance: float = 0.8):
        self.vsa = vsa
        self.rho = vigilance
        self.calls = 0
        self.novel = 0

    def key(self, kind: str, signature: str, ctx: str | None, frame: str) -> np.ndarray:
        return column_key(self.vsa, kind, signature, ctx, frame)

    def select(self, q: np.ndarray, columns: list[Column]) -> list[tuple[Column, float]]:
        self.calls += 1
        out = []
        for c in columns:
            if c.key is None or not c.active:
                continue
            s = VSA.sim(q, c.key)
            if s >= self.rho:
                out.append((c, s))
        if not out:
            self.novel += 1
        return sorted(out, key=lambda t: (RANK.get(t[0].state, 9), -t[1], t[0].provenance.created_at))
