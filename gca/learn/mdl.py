"""Scoring: two-part minimum description length (plan §5.3) and the version space (§5.5).

L(H)     : prior bits of the hypothesis (grammar + parameters, or schema + holes).
L(D | H) : sum over observations of -log2(1 - eps) when H predicts it, and
           -log2(eps / (V - 1)) when it does not (V = number of possible values).
Checker facts are noiseless: a candidate that contradicts one is refuted outright.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameData:
    """Everything known about one frame's input-output map, on canonical cells."""
    arity: int
    n_values: int                     # m for object frames, 2 for boolean frames
    cells: np.ndarray                 # (k,) observed cell indices (one row per distinct (cell, out))
    outs: np.ndarray                  # (k,) observed outputs
    weights: np.ndarray               # (k,) number of observations behind each row
    pos_cells: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # checker: true
    pos_outs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    neg_cells: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # checker: false
    neg_outs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def n_obs(self) -> int:
        return int(self.weights.sum())

    def known_cells(self) -> set[int]:
        return set(self.cells.tolist()) | set(self.pos_cells.tolist())


def obs_bits(eps: float, n_values: int) -> tuple[float, float]:
    """(bits for a predicted observation, bits for a mispredicted one)."""
    eps = min(max(eps, 1e-6), 0.5)
    return -math.log2(1.0 - eps), -math.log2(eps / max(n_values - 1, 1))


def data_bits(sem: np.ndarray, d: FrameData, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """L(D|H) for every row of sem (N, C), and the weighted mismatch count."""
    hit, miss = obs_bits(eps, d.n_values)
    if len(d.cells) == 0:
        return np.zeros(len(sem)), np.zeros(len(sem))
    wrong = (sem[:, d.cells] != d.outs[None, :]).astype(np.float64) @ d.weights.astype(np.float64)
    right = d.weights.sum() - wrong
    return right * hit + wrong * miss, wrong


def refuted(sem: np.ndarray, d: FrameData) -> np.ndarray:
    bad = np.zeros(len(sem), dtype=bool)
    if len(d.pos_cells):
        bad |= (sem[:, d.pos_cells] != d.pos_outs[None, :]).any(axis=1)
    if len(d.neg_cells):
        bad |= (sem[:, d.neg_cells] == d.neg_outs[None, :]).any(axis=1)
    return bad


def adequate(wrong: float, n: float, eta: float, alpha: float = 1e-3) -> bool:
    """Is the hypothesis' mismatch count plausible as noise? One-sided binomial tail
    P(Bin(n, eta) >= wrong) >= alpha. A failing fit means the hypothesis space lacks the
    concept (yet), so a 'collapsed' version space is not evidence and queries are wasted."""
    w, n = int(round(wrong)), int(round(n))
    if w <= 0 or n <= 0:
        return True
    eta = min(max(eta, 1e-4), 0.5)
    tail = sum(math.comb(n, k) * eta ** k * (1 - eta) ** (n - k) for k in range(w, n + 1))
    return tail >= alpha


def null_bits(d: FrameData) -> float:
    """The uniform code: |D| log2 V bits."""
    return d.n_obs * math.log2(max(d.n_values, 2))


def sprout_threshold(system_index: int, start: float = 16.0, end: float = 8.0, over: int = 5) -> float:
    """delta_sprout falls linearly from `start` to `end` over the first `over` systems."""
    if system_index >= over:
        return end
    return start + (end - start) * system_index / over


@dataclass
class Scored:
    """Candidates for one frame, scored. Rows are distinct full-domain functions."""
    sem: np.ndarray
    prior: np.ndarray                 # L(H)
    data: np.ndarray                  # L(D|H)
    wrong: np.ndarray                 # weighted mismatches
    exprs: list
    origin: list                      # "pool" | "split" | "schema:<id>" | "analogy:<ctx>"
    alive: np.ndarray                 # not refuted by a checker fact

    @property
    def total(self) -> np.ndarray:
        return np.where(self.alive, self.prior + self.data, np.inf)

    def best(self) -> int:
        return int(np.argmin(self.total))

    def version_space(self, delta: float) -> np.ndarray:
        t = self.total
        if not np.isfinite(t).any():
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(t <= t.min() + delta)

    def posterior(self, idx: np.ndarray) -> np.ndarray:
        t = self.total[idx]
        w = np.exp2(-(t - t.min()))
        return w / w.sum()
