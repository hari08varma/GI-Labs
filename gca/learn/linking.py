"""Linking earlier submodels (Harinath's linked equal-size submodels, Vishal example).

A trusted concept from earlier worlds is instantiated in THIS world as a callable part:
its holes are filled with this world's own validated primitives (its successor chain, its
locked rules such as addition) and every entity as anchor. Multiplication, for example, is
rebuilt from this world's own addition. New rules can then CALL these linked parts, up to two
levels deep, so a rule that is too large to find from the innate core can be built from
earlier knowledge. Whatever is proposed is scored by the same MDL and must pass the same
checker validation; nothing is trusted because it was linked.

Bounds (resource limits): at most 96 linked parts per world (concepts that call this world's own
validated rules first), argument
fillers {x, y, u(x), u(y)} for the world's unary primitives, two composition levels, and an
optional unary wrap at the top. Concepts are linked only if their trust >= 0.6 and they are
bound in at least one other world.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from . import hypothesis as H
from .enumerate import row_hash
from .induce import Extra, majority_target


def _args(P: H.Prims, arity: int):
    X, Y = H.grid(P.m, arity)
    out = [(("x",), X), (("y",), Y)]
    for r in sorted(P.unary):
        out.append((("U", r, ("x",)), P.raw[r][X]))
        out.append((("U", r, ("y",)), P.raw[r][Y]))
    return out


def derive(cl, P: H.Prims, max_parts: int = 96) -> list[tuple[str, np.ndarray, float, str]]:
    """Linked parts for this world: (ref, table (m, m), bits, description)."""
    a = cl.a
    m = P.m
    X, Y = H.grid(m, 2)
    local = {r: P.raw[r].reshape(m, m) for r in P.binary}
    seen = {t.tobytes() for t in local.values()}
    concepts = [s for s in a.priors.library(2, "O")
                if s.trust >= 0.6 and any(cid != cl.cid for cid, _, _ in getattr(s, "bindings", []))]
    out: list = []

    def fills(s, Pk: H.Prims, b_first: list[str]):
        holes = sorted(s.kinds)
        opts = []
        for h in holes:
            k = s.kinds[h]
            if "T" in k:
                opts.append([("x",), ("y",)])
            elif "E" in k:
                opts.append(list(range(m)))
            elif "B" in k:
                opts.append(b_first + [r for r in H.hole_options(k, Pk) if r not in b_first])
            else:
                opts.append(H.hole_options(k, Pk))
        if any(not o for o in opts):
            return []
        return [dict(zip(holes, c)) for c in itertools.product(*opts)]

    def add_parts(s, Pk, b_first, quota):
        bits = a.priors.schema_bits(s)
        fs = fills(s, Pk, b_first)
        for f in fs:
            if len(out) >= quota:
                return
            e = H.instantiate(s.body, f)
            try:
                v = H.evaluate(e, X, Y, Pk)
            except KeyError:
                continue
            if (v < 0).any():
                continue
            t = v.reshape(m, m).astype(np.int64)
            if (t == t[:, :1]).all() or (t == t[:1, :]).all():
                continue                              # ignores an argument: not a binary part
            key = t.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((f"link:{len(out)}", t, bits + math.log2(max(len(fs), 1)),
                        f"{s.sid}{dict(sorted((k, v) for k, v in f.items() if not isinstance(v, tuple)))}"))

    with_b = [s for s in concepts if any("B" in k for k in s.kinds.values())]
    no_b = [s for s in concepts if s not in with_b]
    if local:
        # first: concepts that CALL a binary rule, built on this world's own validated rules
        for s in with_b:
            add_parts(s, P, sorted(local), max_parts)
        for s in no_b:
            add_parts(s, P, [], max_parts)
    else:
        # no validated binary rule here: link a binary-free concept first, then build on it
        for s in no_b:
            add_parts(s, P, [], max_parts // 2)
        if out:
            Pk = H.Prims(m)
            for r, t in P.raw.items():
                if r in P.binary:
                    Pk.add_binary(r, t.reshape(m, m))
                elif r in P.rel:
                    Pk.add_rel(r, t.reshape(m, m))
                else:
                    Pk.add_unary(r, t)
            for ref, t, _, _ in list(out):
                Pk.add_binary(ref, t)
            for s in with_b:
                add_parts(s, Pk, [r for r, _, _, _ in out], max_parts)
    return out


def linked_part(cl, F: str, d, P: H.Prims, cap_rows: int = 400_000) -> Extra:
    """Candidates that call linked parts: level 1 p(a, b), level 2 p(q(a, b), c) or p(c, q(a, b)),
    optionally wrapped in one unary step. Matched on the observed cells by vectorised tests,
    then evaluated on the whole domain."""
    m = P.m
    C = m * m
    empty = Extra(np.zeros((0, C), np.int8), np.zeros(0), [], [])
    key = (P.signature(), cl.a.priors.version)
    if cl.linked_cache.get("key") != key:
        cl.linked_cache = {"key": key, "parts": derive(cl, P)}
        for ref, t, _, desc in cl.linked_cache["parts"]:
            cl.linked_names[ref] = desc
    parts = cl.linked_cache["parts"]
    if not parts:
        return empty
    cells, tgt = majority_target(d)
    if len(cells) < 8:
        return empty
    args = _args(P, 2)
    tables = [(ref, t, bits) for ref, t, bits, _ in parts] + \
             [(r, P.raw[r].reshape(m, m), 4.0) for r in sorted(P.binary)]
    arg_bits = math.log2(len(args))
    # level 1 rows
    l1_sem, l1_expr, l1_cost = [], [], []
    for (ref, t, bits) in tables:
        for (ea, va), (eb, vb) in itertools.product(args, args):
            l1_sem.append(t[va, vb])
            l1_expr.append(("B", ref, ea, eb))
            l1_cost.append(1 + bits + 2 * arg_bits)
    L1 = np.array(l1_sem, dtype=np.int64)
    h = row_hash(L1)
    _, first = np.unique(h, return_index=True)
    first.sort()
    L1, l1_expr, l1_cost = L1[first], [l1_expr[i] for i in first], np.array(l1_cost)[first]
    targets = [(tgt, None)]
    for r in sorted(P.unary):
        t = P.raw[r]
        if len(set(t.tolist())) == m:
            inv = np.empty(m, dtype=np.int64)
            inv[t] = np.arange(m)
            targets.append((inv[tgt], r))
    found_sem, found_expr, found_cost = [], [], []
    for target, wrap in targets:
        hit = np.flatnonzero((L1[:, cells] == target[None, :]).all(axis=1))
        for i in hit.tolist():
            e = l1_expr[i] if wrap is None else ("U", wrap, l1_expr[i])
            row = L1[i] if wrap is None else P.raw[wrap][L1[i]]
            found_sem.append(row)
            found_expr.append(e)
            found_cost.append(l1_cost[i] + (1 if wrap else 0))
        R = L1[:, cells]
        rows = 0
        for (ref, t, bits) in tables:
            for (ec, vc) in args:
                for side in (0, 1):
                    vals = t[R, vc[cells][None, :]] if side == 0 else t[vc[cells][None, :], R]
                    rows += len(R)
                    hit = np.flatnonzero((vals == target[None, :]).all(axis=1))
                    for i in hit.tolist():
                        inner = l1_expr[i]
                        e = ("B", ref, inner, ec) if side == 0 else ("B", ref, ec, inner)
                        full = t[L1[i], vc] if side == 0 else t[vc, L1[i]]
                        if wrap is not None:
                            e, full = ("U", wrap, e), P.raw[wrap][full]
                        found_sem.append(full)
                        found_expr.append(e)
                        found_cost.append(1 + bits + arg_bits + l1_cost[i] + (1 if wrap else 0))
                if rows > cap_rows:
                    break
            if rows > cap_rows:
                break
    if not found_sem:
        return empty
    fresh = cl.a.priors.fresh_bits(2, "O")
    return Extra(np.array(found_sem, dtype=np.int8), np.array(found_cost) + fresh, found_expr,
                 ["linked"] * len(found_expr))
