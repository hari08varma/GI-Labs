"""Induction for one frame: generate candidates, then score them by MDL.

Candidate sources:
  pool      bottom-up enumeration (enumerate.py)
  split     one top-down step through an invertible learned primitive: if r is a Latin
            row (b -> r(a, b) is a bijection), then f = r(A, B) fixes B's values on the
            observed cells once A is chosen, so B is looked up by hash. Likewise
            f = U_r(A) fixes A when U_r is a bijection. This is the witness-function idea
            of top-down synthesis (FlashMeta), used to reach expressions larger than the
            pool's size bound. Candidates stay exact: their full-domain semantics are
            computed from the parts.
  schema    library schemas instantiated by filling their holes (build log, "Schemas")
  analogy   tables pulled back through an isomorphism (passed in by the caller)
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from . import hypothesis as H
from .enumerate import Pool, row_hash
from .mdl import FrameData, Scored, data_bits, refuted
from .priors import Priors


@dataclass
class Extra:
    sem: np.ndarray          # (k, C)
    prior: np.ndarray        # (k,)
    exprs: list
    origin: list


def majority_target(d: FrameData) -> tuple[np.ndarray, np.ndarray]:
    """One output per observed cell (majority; checker facts override)."""
    best: dict[int, tuple[float, int]] = {}
    for c, o, w in zip(d.cells.tolist(), d.outs.tolist(), d.weights.tolist()):
        if c not in best or w > best[c][0]:
            best[c] = (w, o)
    for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist()):
        best[c] = (1e9, o)
    cells = np.array(sorted(best), dtype=np.int64)
    return cells, np.array([best[c][1] for c in cells.tolist()], dtype=np.int64)


def _row_inverse(T: np.ndarray) -> np.ndarray | None:
    """L[a, y] = b with T[a, b] = y, if every row of T is a bijection, else None."""
    m = T.shape[0]
    L = np.full((m + 1, m + 1), H.UNDEF, dtype=np.int64)
    for a in range(m):
        row = T[a]
        if len(set(row.tolist())) != m:
            return None
        L[a, row] = np.arange(m)
    return L


def splits(pool: Pool, d: FrameData, priors: Priors, *, tol: int, max_pairs: int = 200_000,
           groups: int = 3, keep_top: int = 3000) -> Extra:
    """Top-down split candidates for an object-valued frame (see module docstring)."""
    P = pool.P
    sem, cost, exprs, sizes = pool.flat("O")
    empty = Extra(np.zeros((0, pool.C), np.int8), np.zeros(0), [], [])
    cells, tgt = majority_target(d)
    k = len(cells)
    if k < 6 or len(sem) == 0 or not P.binary:
        return empty
    S = sem[:, cells].astype(np.int64)
    groups = 1 if k < 15 else (2 if k < 30 else groups)
    if groups == 1:
        tol = 0
    rng = np.random.default_rng(k * 7919 + int(tgt.sum()))
    perm = rng.permutation(k)
    grp = [np.sort(perm[g::groups]) for g in range(groups)]
    gh = [row_hash(S[:, g]) for g in grp]
    order = [np.argsort(h, kind="stable") for h in gh]
    sorted_h = [h[o] for h, o in zip(gh, order)]
    av = priors.available("O", P, pool.arity)
    fresh = priors.fresh_bits(pool.arity, "O")
    max_size = 2 * pool.s_enum + 2

    # targets: f itself, and U_r^-1(f) for every bijective unary r (one wrapping level)
    targets = [(tgt, None)]
    for r in pool._refs("U"):
        t = P.raw[r]
        if len(set(t.tolist())) == P.m:
            inv = np.empty(P.m, dtype=np.int64)
            inv[t] = np.arange(P.m)
            targets.append((inv[tgt], r))
    ubits = priors.prod_bits("O", "U", av) + priors.param_bits("U", P) if P.unary else 0.0
    bbits = priors.prod_bits("O", "B", av) + priors.param_bits("B", P)
    rows = []          # (full sem, cost, A, B, ref, wrap)
    for r in pool._refs("B"):
        T = P.raw[r]
        for side in (0, 1):                       # 0: A is the left argument, solve the right
            L = _row_inverse(T if side == 0 else T.T)
            if L is None:
                continue
            for target, wrap in targets:
                need = L[S, target[None, :]]      # (N, k) required values of the other part
                okrow = (need >= 0).all(axis=1)
                found = []
                for g in range(groups):
                    hn = row_hash(need[:, grp[g]])
                    lo = np.searchsorted(sorted_h[g], hn, "left")
                    hi = np.searchsorted(sorted_h[g], hn, "right")
                    cnt = np.where(okrow, hi - lo, 0)
                    tot = int(cnt.sum())
                    if tot == 0 or tot > max_pairs:     # nothing, or too ambiguous on this group
                        continue
                    I = np.repeat(np.arange(len(cnt)), cnt)
                    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    J = order[g][np.repeat(lo, cnt) + off]
                    found.append(I * len(sem) + J)
                if not found:
                    continue
                pa = np.unique(np.concatenate(found))
                Ia, Ib = pa // len(sem), pa % len(sem)
                A, Bp = (Ia, Ib) if side == 0 else (Ib, Ia)
                sz = 1 + sizes[A] + sizes[Bp] + (wrap is not None)
                ok = sz <= max_size
                A, Bp = A[ok], Bp[ok]
                if len(A) == 0:
                    continue
                full = T[sem[A].astype(np.int64), sem[Bp].astype(np.int64)]
                if wrap is not None:
                    full = P.raw[wrap][full]
                miss = (full[:, cells] != tgt[None, :]).sum(axis=1)
                keep = miss <= tol
                if not keep.any():
                    continue
                c = bbits + cost[A] + cost[Bp] + (ubits if wrap is not None else 0.0) + fresh
                rows.append((full[keep].astype(np.int8), c[keep], A[keep], Bp[keep], r, wrap))
    if not rows:
        return empty
    full = np.concatenate([x[0] for x in rows])
    c = np.concatenate([x[1] for x in rows])
    which = np.concatenate([np.full(len(x[1]), i) for i, x in enumerate(rows)])
    A = np.concatenate([x[2] for x in rows])
    Bp = np.concatenate([x[3] for x in rows])
    h = row_hash(full)
    o = np.lexsort((c, h))
    first = np.ones(len(o), dtype=bool)
    first[1:] = h[o][1:] != h[o][:-1]
    sel = o[first]
    sel = sel[np.argsort(c[sel], kind="stable")][:keep_top]
    out_expr = []
    for q in sel.tolist():
        _, _, _, _, r, wrap = rows[int(which[q])]
        e = ("B", r, exprs[A[q]], exprs[Bp[q]])
        out_expr.append(("U", wrap, e) if wrap is not None else e)
    return Extra(full[sel], c[sel], out_expr, ["split"] * len(out_expr))


def schema_instances(P: H.Prims, arity: int, out_type: str, priors: Priors, *, exclude=frozenset(),
                     cap: int = 4000) -> Extra:
    X, Y = H.grid(P.m, arity)
    sems, costs, exprs, origin = [], [], [], []
    for s in priors.library(arity, out_type):
        holes = sorted(s.kinds)
        opts = [[o for o in H.hole_options(s.kinds[h], P, arity)
                 if not (isinstance(o, str) and o in exclude)] for h in holes]
        if any(not o for o in opts):
            continue
        n = math.prod(len(o) for o in opts)
        if n > cap:
            continue
        bits = priors.schema_bits(s) + sum(math.log2(len(o)) for o in opts)
        for combo in itertools.product(*opts):
            e = H.instantiate(s.body, dict(zip(holes, combo)))
            if exclude and H.refs_used(e) & set(exclude):
                continue
            v = H.evaluate(e, X, Y, P)
            if (v < 0).any():
                continue
            sems.append(v.astype(np.int8))
            costs.append(bits)
            exprs.append(e)
            origin.append("schema:" + s.sid)
    if not sems:
        return Extra(np.zeros((0, X.size), np.int8), np.zeros(0), [], [])
    return Extra(np.stack(sems), np.array(costs), exprs, origin)


def score(d: FrameData, eps: float, parts: list[Extra]) -> Scored:
    """Merge candidate sources (dedupe by full-domain semantics, keep the cheapest prior)
    and score every distinct function against the frame's data."""
    parts = [p for p in parts if len(p.exprs)]
    if not parts:
        z = np.zeros(0)
        return Scored(np.zeros((0, 1), np.int8), z, z, z, [], [], np.zeros(0, bool))
    sem = np.concatenate([p.sem for p in parts])
    prior = np.concatenate([p.prior for p in parts])
    exprs = [e for p in parts for e in p.exprs]
    origin = [o for p in parts for o in p.origin]
    h = row_hash(sem)
    order = np.lexsort((prior, h))
    first = np.ones(len(order), dtype=bool)
    first[1:] = h[order][1:] != h[order][:-1]
    sel = order[first]
    sem, prior = sem[sel], prior[sel]
    exprs = [exprs[i] for i in sel]
    origin = [origin[i] for i in sel]
    db, wrong = data_bits(sem, d, eps)
    return Scored(sem, prior, db, wrong, exprs, origin, ~refuted(sem, d))


def pool_part(pool: Pool, out_type: str, priors: Priors) -> Extra:
    sem, cost, exprs, _ = pool.flat(out_type)
    fresh = priors.fresh_bits(pool.arity, out_type)
    return Extra(sem, cost + fresh, exprs, ["pool"] * len(exprs))
