"""Hidden generator grammar: operator definitions over Z_n.

PRIVATE to the world. Its language is deliberately different from the learner's
(the learner never sees integers, polynomials or modular arithmetic).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np


@dataclass(frozen=True)
class OpSpec:
    kind: str            # "succ" | "affine" | "poly" | "max" | "min" | "lt"
    params: tuple = ()

    @property
    def arity(self) -> int:
        return 1 if self.kind in ("succ", "affine") else 2

    @property
    def boolean(self) -> bool:
        return self.kind == "lt"


def table(spec: OpSpec, n: int) -> np.ndarray:
    """Full truth table over hidden integers. Unary: shape (n,), binary: (n, n).
    Boolean ops return 0/1."""
    x = np.arange(n)
    if spec.kind == "succ":
        return (x + 1) % n
    if spec.kind == "affine":
        a, b = spec.params
        return (a * x + b) % n
    X, Y = np.meshgrid(x, x, indexing="ij")
    if spec.kind == "poly":
        a, b, c, d, e, f = spec.params   # a x^2 + b y^2 + c xy + d x + e y + f
        return (a * X * X + b * Y * Y + c * X * Y + d * X + e * Y + f) % n
    if spec.kind == "max":
        return np.maximum(X, Y)
    if spec.kind == "min":
        return np.minimum(X, Y)
    if spec.kind == "lt":
        return (X < Y).astype(np.int64)
    if spec.kind == "terms":
        # sum of coef * x^i * y^j, params flattened as (coef, i, j, coef, i, j, ...)
        p = spec.params
        out = np.zeros_like(X)
        for k in range(0, len(p), 3):
            out = out + p[k] * X ** p[k + 1] * Y ** p[k + 2]
        return out % n
    raise ValueError(spec.kind)


def parse(s: str) -> OpSpec:
    """Config strings: succ | affine:a,b | lin:a,b,c | poly:a,b,c,d,e,f | max | min | lt |
    terms:c*i*j,... (sum of c x^i y^j). lin:a,b,c means a*x + b*y + c."""
    if s.startswith("terms:"):
        vals = []
        for t in s[6:].split(","):
            vals += [int(v) for v in t.split("*")]
        return OpSpec("terms", tuple(vals))
    if ":" in s:
        k, p = s.split(":")
        vals = tuple(int(v) for v in p.split(","))
        if k == "lin":
            a, b, c = vals
            return OpSpec("poly", (0, 0, 0, a, b, c))
        return OpSpec(k, vals)
    return OpSpec(s)


def is_quasigroup(t: np.ndarray) -> bool:
    n = t.shape[0]
    return all(len(set(t[i, :])) == n for i in range(n)) and all(len(set(t[:, j])) == n for j in range(n))


def family_tables(spec: OpSpec, n: int) -> np.ndarray:
    """Alternative hypotheses in the generator's own grammar, for the
    identifiability check. Returns stacked flattened tables (k, cells)."""
    return _family(spec.arity, spec.boolean, n)


@lru_cache(maxsize=None)
def _family(arity: int, boolean: bool, n: int) -> np.ndarray:
    spec = OpSpec("succ") if arity == 1 else OpSpec("lt") if boolean else OpSpec("max")
    if spec.arity == 1:
        tabs = [table(OpSpec("affine", (a, b)), n) for a in range(n) for b in range(n)]
        return np.stack(tabs)
    if spec.boolean:
        x = np.arange(n)
        X, Y = np.meshgrid(x, x, indexing="ij")
        tabs = [X < Y, X > Y, X <= Y, X >= Y, X == Y, X != Y]
        return np.stack([t.astype(np.int64).ravel() for t in tabs])
    tabs = []
    for a, b, c in product(range(n), repeat=3):          # full linear family
        tabs.append(table(OpSpec("poly", (0, 0, 0, a, b, c)), n).ravel())
    for co in product(range(4), repeat=6):                # small-coefficient polynomials
        tabs.append(table(OpSpec("poly", co), n).ravel())
    tabs.append(table(OpSpec("max"), n).ravel())
    tabs.append(table(OpSpec("min"), n).ravel())
    # cubic terms (block F): x^2y, xy^2, x^3, y^3 with small coefficients, plus a small linear part
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cub = [X * X * Y, X * Y * Y, X ** 3, Y ** 3]
    for cs in product(range(3), repeat=4):
        if not any(cs):
            continue
        base = sum(c * t for c, t in zip(cs, cub))
        for d, e, f in product(range(3), repeat=3):
            for g in range(3):
                tabs.append(((base + g * X * Y + d * X + e * Y + f) % n).ravel())
    return np.unique(np.stack(tabs), axis=0)
