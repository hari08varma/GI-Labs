"""Search: bottom-up enumeration with observational-equivalence pruning (plan §5.4).

The carrier is finite, so every candidate is evaluated on ALL input cells and two
candidates with the same output vector are equivalent; only the cheaper (in bits) is
kept. The pool is built once per library state (primitives + priors) and is then
scored against any frame's data with vectorised operations (build log, "Learner").

Limits (plan §5.4): size <= s_enum, caps per (type, size), a combination budget per
production and a total node budget. Hitting any limit sets `incomplete`.
"""
from __future__ import annotations

import math

import numpy as np

from . import hypothesis as H
from .priors import Priors

TYPES = ("O", "N", "B")
_HASH = np.random.default_rng(12345).integers(-2**62, 2**62, size=4096, dtype=np.int64) | 1


def row_hash(sem: np.ndarray) -> np.ndarray:
    """Exact 64-bit hash of each row: sum_j (v_j + 3) * R_j with wrap-around integer
    arithmetic, accumulated column by column (exact, unlike a float BLAS product whose
    rounding can differ between rows). Distinct rows collide with negligible probability."""
    sem = np.asarray(sem)
    h = np.zeros(sem.shape[0], dtype=np.int64)
    with np.errstate(over="ignore"):
        for j in range(sem.shape[1]):
            h += (sem[:, j].astype(np.int64) + 3) * _HASH[j]
    return h


class Pool:
    def __init__(self, prims: H.Prims, arity: int, priors: Priors, *, s_enum: int = 8,
                 caps: dict | None = None, combo_budget: int = 60_000, node_budget: int = 2_000_000,
                 exclude: frozenset = frozenset(), chunk: int = 16_384):
        self.P = prims
        self.m = prims.m
        self.arity = arity
        self.priors = priors
        self.s_enum = s_enum
        self.caps = caps or {"O": 3000, "N": 1000, "B": 1500}
        self.combo_budget = combo_budget
        self.node_budget = node_budget
        self.exclude = exclude
        self.chunk = chunk
        self.X, self.Y = H.grid(self.m, arity)
        self.C = self.X.size
        # per type, per size: sem (k, C) int8, cost (k,), desc list -> expr list
        self.sem = {t: {} for t in TYPES}
        self.cost = {t: {} for t in TYPES}
        self.expr = {t: {} for t in TYPES}
        self.seen = {t: {} for t in TYPES}           # size -> hashes of committed rows
        self._sorted: dict = {}
        self.nodes = 0
        self.incomplete = False
        self._flat: dict = {}
        self._combos: list = []                      # (prod, params, child types, child sizes)
        self._build()

    # ---- production instances --------------------------------------------
    def _refs(self, kind: str) -> list[str]:
        return [r for r in self.P.kinds(kind) if r not in self.exclude]

    def _pbits(self, typ: str, prod: str, kinds: tuple[str, ...]) -> float:
        av = self.priors.available(typ, self.P, self.arity)
        b = self.priors.prod_bits(typ, prod, av)
        return b + sum(self.priors.param_bits(k, self.P) for k in kinds)

    def _instances(self):
        """(out type, prod, params, child types, bits, evaluator)"""
        P, out = self.P, []
        for r in self._refs("U"):
            T = P.unary[r]
            out.append(("O", "U", (r,), ("O",), self._pbits("O", "U", ("U",)), lambda a, T=T: T[a]))
        for r in self._refs("B"):
            T = P.binary[r]
            out.append(("O", "B", (r,), ("O", "O"), self._pbits("O", "B", ("B",)), lambda a, b, T=T: T[a, b]))
        for r in self._refs("P"):
            T = P.powers[r]
            out.append(("O", "iter", (r,), ("N", "O"), self._pbits("O", "iter", ("P",)),
                        lambda k, a, T=T: T[k, a]))
        for r in self._refs("B"):
            Q = P.cubes[r]
            out.append(("O", "iterc", (r,), ("O", "N", "O"), self._pbits("O", "iterc", ("B",)),
                        lambda p, k, a, Q=Q: Q[p, k, a]))
        out.append(("O", "if", (), ("B", "O", "O"), self._pbits("O", "if", ()),
                    lambda c, a, b: np.where(c < 0, H.UNDEF, np.where(c == 1, a, b))))
        for r in self._refs("C"):
            for o in range(self.m):
                T = P.chains[r][o]
                out.append(("N", "pos", (r, o), ("O",), self._pbits("N", "pos", ("C", "E")),
                            lambda a, T=T: T[a]))
        out.append(("B", "lt", (), ("N", "N"), self._pbits("B", "lt", ()),
                    lambda a, b: np.where((a >= 0) & (b >= 0), (a < b).astype(np.int64), H.UNDEF)))
        for r in self._refs("R"):
            T = P.rel[r]
            out.append(("B", "R", (r,), ("O", "O"), self._pbits("B", "R", ("R",)), lambda a, b, T=T: T[a, b]))
        out.append(("B", "eq", (), ("O", "O"), self._pbits("B", "eq", ()),
                    lambda a, b: np.where((a >= 0) & (b >= 0), (a == b).astype(np.int64), H.UNDEF)))
        return out

    # ---- building --------------------------------------------------------
    def _build(self) -> None:
        leaves = [(("x",), self.X)]
        if self.arity == 2:
            leaves.append((("y",), self.Y))
        lb = {p: self._pbits("O", p, ()) for p in ("x", "y")}
        cb = self._pbits("O", "c", ("E",))
        rows, costs, exprs = [], [], []
        for e, v in leaves:
            rows.append(v)
            costs.append(lb[e[0]])
            exprs.append(e)
        for c in range(self.m):
            rows.append(np.full(self.C, c))
            costs.append(cb)
            exprs.append(("c", c))
        self._commit("O", 1, np.array(rows, dtype=np.int8), np.array(costs), exprs)
        for t in ("N", "B"):
            self._commit(t, 1, np.zeros((0, self.C), np.int8), np.zeros(0), [])
        inst = self._instances()
        for s in range(2, self.s_enum + 1):
            front = {t: None for t in TYPES}
            for (typ, prod, prm, ctypes, bits, fn) in inst:
                for parts in _partitions(s - 1, len(ctypes)):
                    sets = [(self.sem[ct].get(sz), self.cost[ct].get(sz)) for ct, sz in zip(ctypes, parts)]
                    if any(x[0] is None or len(x[0]) == 0 for x in sets):
                        continue
                    front[typ] = self._combine(front[typ], typ, prod, prm, ctypes, parts, bits, fn, sets)
                    if self.nodes >= self.node_budget:
                        self.incomplete = True
                        break
            for t in TYPES:
                f = front[t]
                if f is None:
                    self._commit(t, s, np.zeros((0, self.C), np.int8), np.zeros(0), [])
                else:
                    sem, cost, _h, cids, kids = f
                    self._commit(t, s, sem, cost, [self._mk(int(c), k) for c, k in zip(cids, kids)])
            if self.nodes >= self.node_budget:
                for s2 in range(s + 1, self.s_enum + 1):
                    for t in TYPES:
                        self._commit(t, s2, np.zeros((0, self.C), np.int8), np.zeros(0), [])
                break

    def _mk(self, cid: int, kid_idx) -> tuple:
        prod, prm, ctypes, parts = self._combos[cid]
        ch = [self.expr[t][sz][int(i)] for t, sz, i in zip(ctypes, parts, kid_idx)]
        if prod in ("U", "B", "iter", "iterc", "R"):
            return (prod, prm[0], *ch)
        if prod == "pos":
            return ("pos", prm[0], prm[1], ch[0])
        return (prod, *ch)

    def _combine(self, front, typ, prod, prm, ctypes, parts, bits, fn, sets):
        cid = len(self._combos)
        self._combos.append((prod, prm, ctypes, parts))
        sizes = [len(x[0]) for x in sets]
        total = math.prod(sizes)
        if total > self.combo_budget:
            self.incomplete = True
            f = (self.combo_budget / total) ** (1.0 / len(sizes))
            sizes = [max(1, int(n * f)) for n in sizes]
        grids = np.meshgrid(*[np.arange(n) for n in sizes], indexing="ij")
        idx = np.stack([g.ravel() for g in grids], axis=1).astype(np.int32)
        N = idx.shape[0]
        self.nodes += N
        sh, sloc = self._seen_sorted(typ)
        for lo in range(0, N, self.chunk):
            sl = idx[lo:lo + self.chunk]
            args = [sets[j][0][sl[:, j]] for j in range(len(sets))]
            sem = fn(*args)
            cost = bits + sum(sets[j][1][sl[:, j]] for j in range(len(sets)))
            ok = (sem >= 0).all(axis=1)
            if not ok.any():
                continue
            sem, cost, sl = sem[ok].astype(np.int8, copy=False), cost[ok], sl[ok]
            h = row_hash(sem)
            if len(sh):
                pos = np.minimum(np.searchsorted(sh, h), len(sh) - 1)
                dup = sh[pos] == h
            else:
                pos, dup = np.zeros(len(h), dtype=np.int64), np.zeros(len(h), dtype=bool)
            for k in np.flatnonzero(dup):
                sz, i = sloc[pos[k]]
                if cost[k] < self.cost[typ][sz][i] - 1e-9:      # cheaper duplicate: replace in place
                    self.cost[typ][sz][i] = cost[k]
                    self.expr[typ][sz][i] = self._mk(cid, sl[k])
                    self._flat.pop(typ, None)
            keep = ~dup
            if not keep.any():
                continue
            pad = np.full((int(keep.sum()), 3), -1, dtype=np.int32)
            pad[:, : sl.shape[1]] = sl[keep]
            cids = np.full(len(pad), cid, dtype=np.int32)
            front = _merge(front, (sem[keep], cost[keep], h[keep], cids, pad), self.caps[typ])
        return front

    def _commit(self, t, s, sem, cost, exprs) -> None:
        self.sem[t][s] = sem
        self.cost[t][s] = np.asarray(cost, dtype=np.float64).copy()
        self.expr[t][s] = list(exprs)
        if len(exprs):
            self.seen[t][s] = row_hash(sem)
        self._sorted.pop(t, None)
        self._flat.pop(t, None)

    def _seen_sorted(self, t):
        """Sorted hashes of committed rows of type t, with their (size, index)."""
        got = self._sorted.get(t)
        if got is None:
            hs, loc = [], []
            for s, h in self.seen[t].items():
                hs.append(h)
                loc += [(s, i) for i in range(len(h))]
            h = np.concatenate(hs) if hs else np.zeros(0)
            o = np.argsort(h, kind="stable")
            got = (h[o], [loc[i] for i in o])
            self._sorted[t] = got
        return got

    # ---- access ----------------------------------------------------------
    def flat(self, t: str):
        """All candidates of type t: (sem (N, C), cost (N,), exprs, sizes)."""
        f = self._flat.get(t)
        if f is None:
            ss = sorted(self.sem[t])
            sem = np.concatenate([self.sem[t][s] for s in ss]) if ss else np.zeros((0, self.C), np.int8)
            cost = np.concatenate([self.cost[t][s] for s in ss]) if ss else np.zeros(0)
            exprs = [e for s in ss for e in self.expr[t][s]]
            sizes = np.concatenate([np.full(len(self.expr[t][s]), s) for s in ss]) if ss else np.zeros(0)
            f = (sem, cost, exprs, sizes)
            self._flat[t] = f
        return f

    def __len__(self) -> int:
        return sum(len(self.expr[t][s]) for t in TYPES for s in self.expr[t])


def _partitions(total: int, k: int):
    """Ordered k-tuples of positive ints summing to total."""
    if k == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - k + 2):
        for rest in _partitions(total - first, k - 1):
            yield (first,) + rest


def _merge(front, new, cap: int):
    """Union of two candidate batches; dedupe by hash keeping the cheaper; keep top `cap`."""
    parts = new if front is None else tuple(np.concatenate([a, b]) for a, b in zip(front, new))
    sem, cost, h, cids, kids = parts
    order = np.lexsort((cost, h))
    h_s = h[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = h_s[1:] != h_s[:-1]
    sel = order[first]
    if len(sel) > cap:
        sel = sel[np.argpartition(cost[sel], cap - 1)[:cap]]
    sel = sel[np.argsort(cost[sel], kind="stable")]
    return sem[sel], cost[sel], h[sel], cids[sel], kids[sel]
