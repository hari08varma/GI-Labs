"""Analogy and transfer (plan §6.6, build log "Analogy").

For a context whose frames are not yet explained, look for an isomorphism phi between
its observed partial tables and the locked tables of an earlier context of the same size
(structure mapping, Gentner 1983, done as constraint satisfaction: backtracking with
propagation). Candidate source contexts are ranked by a VSA system signature.

On success the source's columns are LINKED into the new context through phi (pulled
back), never copied or merged: matched frames get candidate tables that compete in the
ordinary MDL scoring and must pass the checker like any other hypothesis; unmatched source
columns (e.g. the source's successor table) become candidate primitives of the new context.
Automorphisms give identical induced tables for the matched frames, so distinct induced
tables, not distinct maps, form the analogy's version space.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from ..learn import reading as R
from ..learn.induce import Extra
from ..memory.vsa import VSA


def solve_maps(m: int, constraints: list, max_maps: int = 64, node_limit: int = 200_000) -> list[list[int]]:
    """Bijections phi (new entity -> source entity) such that every fact (a, b, c) of every
    constraint (facts, T, boolean) satisfies T[phi a, phi b] = phi c (or = c if boolean)."""
    phi, inv = [-1] * m, [-1] * m
    by_ent: dict[int, list] = defaultdict(list)
    rowinv, colinv = [], []
    for ci, (facts, T, boolean) in enumerate(constraints):
        ri, cj = defaultdict(list), defaultdict(list)
        if not boolean:
            for a in range(m):
                for b in range(m):
                    ri[(a, int(T[a, b]))].append(b)
                    cj[(b, int(T[a, b]))].append(a)
        rowinv.append(ri)
        colinv.append(cj)
        for f in facts:
            for x in set(f[:2] if boolean else f):
                by_ent[x].append((ci, f))
    degree = sorted(range(m), key=lambda x: (-len(by_ent[x]), x))
    results: list[list[int]] = []
    trail: list[int] = []
    nodes = [0]

    def assign(x: int, v: int) -> bool:
        if phi[x] == v:
            return True
        if phi[x] != -1 or inv[v] != -1:
            return False
        phi[x], inv[v] = v, x
        trail.append(x)
        return True

    def propagate(queue: list[int]) -> bool:
        while queue:
            x = queue.pop()
            for ci, f in by_ent[x]:
                _, T, boolean = constraints[ci]
                a, b, c = f
                pa, pb = phi[a], phi[b]
                if boolean:
                    if pa >= 0 and pb >= 0 and int(T[pa, pb]) != c:
                        return False
                    continue
                pc = phi[c]
                if pa >= 0 and pb >= 0:
                    v = int(T[pa, pb])
                    if pc == -1:
                        if not assign(c, v):
                            return False
                        queue.append(c)
                    elif pc != v:
                        return False
                elif pa >= 0 and pc >= 0:
                    opts = [o for o in rowinv[ci].get((pa, pc), []) if inv[o] == -1 or inv[o] == b]
                    if not opts:
                        return False
                    if len(opts) == 1 and phi[b] == -1:
                        if not assign(b, opts[0]):
                            return False
                        queue.append(b)
                elif pb >= 0 and pc >= 0:
                    opts = [o for o in colinv[ci].get((pb, pc), []) if inv[o] == -1 or inv[o] == a]
                    if not opts:
                        return False
                    if len(opts) == 1 and phi[a] == -1:
                        if not assign(a, opts[0]):
                            return False
                        queue.append(a)
        return True

    def undo(mark: int) -> None:
        while len(trail) > mark:
            x = trail.pop()
            inv[phi[x]] = -1
            phi[x] = -1

    def search() -> None:
        nodes[0] += 1
        if nodes[0] > node_limit or len(results) >= max_maps:
            return
        x = next((y for y in degree if phi[y] == -1), None)
        if x is None:
            results.append(list(phi))
            return
        for v in range(m):
            if inv[v] != -1:
                continue
            mark = len(trail)
            if assign(x, v) and propagate([x]):
                search()
            undo(mark)
            if len(results) >= max_maps or nodes[0] > node_limit:
                return

    search()
    return results


class Analogy:
    def __init__(self, agent):
        self.a = agent
        self.tried: set = set()
        self.found = 0
        self.registry: dict = {}             # map id -> (source learner, phi, matched column ids)

    def signature(self, cl) -> np.ndarray:
        vsa: VSA = self.a.vsa
        feats = [f"m={cl.ctx.m}"]
        for F, info in cl.ctx.frames.items():
            if cl.ctx.words_of(F):
                feats.append(f"frame:{info['arity']}:{'B' if info['bool'] else 'O'}")
        P = cl.prims()
        feats.append(f"chain:{bool(P.chains)}")
        return vsa.key_sum([vsa.hv("sig:" + f) for f in feats])

    def _binary_frames(self, cl) -> list[str]:
        return [F for F, info in cl.ctx.frames.items() if info["arity"] == 2 and cl.ctx.words_of(F)]

    def attempt(self, cl, checker) -> bool:
        a, ctx = self.a, cl.ctx
        m = ctx.m
        frames = self._binary_frames(cl)
        open_ = [F for F in frames if (c := cl.active_col(F)) is None or c.state == "plastic"]
        if not open_:
            return False
        key = (m, sum(len(v) for v in ctx.events.values()))
        if key in self.tried:
            return False
        self.tried.add(key)
        sources = []
        for sid, src in a.learners.items():
            if src is cl or src.ctx.m != m:
                continue
            cols = [c for c in a.spaces.in_context(sid) if c.state == "locked" and c.kind == "operator"
                    and c.scope is None and c.arity == 2 and c.sem.size == m * m]
            if cols:
                sources.append((src, cols))
        if not sources:
            return False
        q = self.signature(cl)
        sources.sort(key=lambda t: (-VSA.sim(q, self.signature(t[0])), t[0].cid))
        facts = {}
        for F in frames:
            d = R.frame_data(ctx, F)
            fs = [(c // m, c % m, int(o)) for c, o in zip(d.cells.tolist(), d.outs.tolist())]
            fs += [(c // m, c % m, int(o)) for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist())]
            if fs:
                facts[F] = sorted(set(fs))
        frames = [F for F in frames if F in facts]
        for src, cols in sources[: a.hp.analogy_top]:
            got = self._match(cl, src, cols, frames, facts)
            if got:
                a.priors.observe("iso:yes")
                self.found += 1
                return True
        a.priors.observe("iso:no")
        return False

    def _match(self, cl, src, cols, frames, facts) -> bool:
        """Try frame assignments and role permutations: a frame may match a source frame
        with its roles in another order (x - y = z is x + y = z with roles permuted).
        Every consistent assignment contributes candidates: the analogy's version space."""
        a, ctx, m = self.a, cl.ctx, cl.ctx.m
        open_ = {F for F in frames if (c := cl.active_col(F)) is None or c.state == "plastic"}
        options = []
        for F in frames:
            boolean = ctx.frames[F]["bool"]
            perms = [p + (2,) for p in itertools.permutations(range(2))] if boolean else                 list(itertools.permutations(range(3)))
            ot = "B" if boolean else "O"
            options.append([(c, p) for c in cols if c.out_type == ot for p in perms] + [None])
        tried, found = 0, []
        for assign in itertools.product(*options):
            used = [x[0].id for x in assign if x is not None]
            if not used or len(set(used)) != len(used):
                continue
            if not any(F in open_ for F, x in zip(frames, assign) if x is not None):
                continue
            tried += 1
            if tried > 200:
                break
            cons = [([tuple(f[i] for i in x[1]) for f in facts[F]], x[0].sem.reshape(m, m), x[0].out_type == "B")
                    for F, x in zip(frames, assign) if x is not None]
            maps = solve_maps(m, cons, a.hp.analogy_max_maps, a.hp.analogy_node_limit)
            if not maps:
                continue
            if len(maps) >= a.hp.analogy_max_maps:
                # under-determined: the induced tables are not all enumerated, so their version
                # space would be incomplete and a "collapse" would not be evidence. Wait for data.
                a.log.log("analogy_wait", ctx=cl.cid, src=src.cid, maps=len(maps))
                return False
            found.append((assign, maps))
        if not found:
            return False
        p_iso = a.priors.rate("iso:yes", ("iso:yes", "iso:no"), prior=0.5)
        per_frame: dict[str, dict] = {}
        for assign, maps in found:
            matched = {F: x for F, x in zip(frames, assign) if x is not None}
            for F, (c, p) in matched.items():
                if F not in open_:
                    continue
                tabs = {}
                for phi in maps:
                    t = self._pull(c, phi, m, p)
                    if t is not None:
                        tabs.setdefault(t.tobytes(), (t, phi))
                if not tabs:
                    continue
                bits = -math.log2(p_iso) + math.log2(len(found) + 1) + math.log2(len(tabs))
                for k, (t, phi) in tabs.items():
                    cur = per_frame.setdefault(F, {}).get(k)
                    if cur is None or bits < cur[1]:
                        per_frame[F][k] = (t, bits, c, phi, set(x[0].id for x in matched.values()))
        if not per_frame:
            return False
        for F, tabs in per_frame.items():
            rows = list(tabs.values())
            ids = []
            for r in rows:                       # every candidate table remembers its own phi
                rid = len(self.registry)
                self.registry[rid] = (src, r[3], r[4])
                ids.append(rid)
            cl.analog[F] = Extra(np.stack([r[0].astype(np.int8) for r in rows]), np.array([r[1] for r in rows]),
                                 [("map", r[2].id, rid) for r, rid in zip(rows, ids)], ["analogy:" + src.cid] * len(rows))
        a.log.log("analogy", ctx=cl.cid, src=src.cid, assignments=len(found), tried=tried,
                  candidates={F: len(t) for F, t in per_frame.items()})
        return True

    def on_lock(self, cl, col) -> None:
        """An analogy table was validated: link the source's other columns through its phi."""
        got = self.registry.get(col.body[2]) if col.body[0] == "map" else None
        if got is None:
            return
        src, phi, matched = got
        self._link(cl, src, phi, matched)

    @staticmethod
    def _pull(col, phi: list[int], m: int, perm=None) -> np.ndarray | None:
        """The source table seen in the new context through phi (and a role permutation):
        the new frame's tuple t is in the relation iff phi(permuted t) is in the source's."""
        phi = np.array(phi, dtype=np.int64)
        inv = np.empty(m, dtype=np.int64)
        inv[phi] = np.arange(m)
        if col.arity == 1:
            return inv[col.sem[phi]]
        T = col.sem.reshape(m, m)
        if col.out_type == "B":
            p = perm or (0, 1, 2)
            A, B = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
            args = (A, B)
            return T[phi[args[p[0]]], phi[args[p[1]]]].ravel()
        if perm is None or tuple(perm) == (0, 1, 2):
            return inv[T[np.ix_(phi, phi)]].ravel()
        # general role order: for each (a, b) find the unique c with phi(perm(a, b, c)) in R_src
        a_, b_, c_ = np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij")
        tup = (a_, b_, c_)
        q = [None, None, None]
        for i in range(3):          # source position i holds new canonical position perm[i]
            q[i] = phi[tup[perm[i]]]
        hold = T[q[0], q[1]] == q[2]
        cnt = hold.sum(axis=2)
        if not (cnt == 1).all():
            return None
        return hold.argmax(axis=2).ravel()

    def _link(self, cl, src, phi: list[int], matched: set) -> None:
        """Unmatched source operator columns become candidate primitives here, through phi."""
        a, ctx, m = self.a, cl.ctx, cl.ctx.m
        for c in a.spaces.in_context(src.cid):
            if c.state != "locked" or c.kind != "operator" or c.id in matched or c.scope is not None:
                continue
            if c.sem.size not in (m, m * m):
                continue
            words = src.ctx.words_of(c.frame)
            F = f"~{src.cid}:{words[0] if words else c.frame}"
            if F in ctx.frames:
                continue
            ctx.frames[F] = {"arity": c.arity, "bool": c.out_type == "B"}
            sem = self._pull(c, phi, m)
            a.new_column(cl, F, "operator", ("map", c.id, 0), sem, 8.0, origin="analogy-link:" + src.cid,
                         state="candidate", parent=c.id)
