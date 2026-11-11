"""Rest and play mode (plan §7.8): runs between systems, with no external task.

1. Abstraction (in the spirit of DreamCoder and Stitch): anti-unify pairs of library
   schemas. Differing parameters become new holes, differing object subtrees become
   subtree holes. Keep an abstraction only if
   (a) the library's description length falls by >= 8 bits:
       L(s1) + L(s2) - [L(abs) + fill(s1) + fill(s2)] >= 8, and
   (b) a replay of the finished system needs fewer search nodes with it (its instances
       in that context are fewer than the enumeration nodes the system needed).
   Otherwise the proposal goes to mismatch.
2. Conjectures: commutative, associative, identity element, idempotent and latin are
   checked internally over the whole domain of every locked binary column. They are
   derived from locked columns, so they need no checker (build log: rest has no checker).
3. Consolidation: duplicate merging is done by the lifecycle; priors are updated by the agent.
"""
from __future__ import annotations

import math

import numpy as np

from ..learn import hypothesis as H

STRUCT_BITS = 3.0          # bits per node of a schema skeleton (8 productions for objects)
HOLE_BITS = 2.0            # bits to fill one extra parameter hole (about log2 of a few refs)


def struct_bits(e) -> float:
    return STRUCT_BITS * H.size(e) + len(H.params(e))


def anti_unify(a, b):
    """Least general generalization of two schema bodies (params and object subtrees)."""
    pairs: dict = {}

    def hole(pa, pb, prefix):
        key = (prefix, pa, pb)
        if key not in pairs:            # fresh names cannot collide with the inputs' ?r/?e holes
            pairs[key] = f"?{prefix}g{len(pairs)}"
        return pairs[key]

    def go(x, y):
        if x == y:
            return x
        tx, ty = x[0], y[0]
        if tx == ty and tx not in ("table", "map", "H"):
            lst = [tx] + [None] * (len(x) - 1)
            for i, k in H.PARAM_SLOTS.get(tx, ()):
                lst[i] = x[i] if x[i] == y[i] else hole(x[i], y[i], "e" if k == "E" else "r")
            for i in H.CHILD_SLOTS[tx]:
                g = go(x[i], y[i])
                if g is None:
                    return None
                lst[i] = g
            return tuple(lst)
        if H.OUT_TYPE.get(tx) == "O" and H.OUT_TYPE.get(ty) == "O":
            return ("H", hole(x, y, "t"))
        return None

    g = go(a, b)
    return g


def _rename(e):
    """Normalise hole names (refs ?r, entities ?e, subtrees ?t)."""
    sch, kinds = H.to_schema(e)
    return sch, kinds


class Play:
    def __init__(self, agent):
        self.a = agent
        self.proposed = 0
        self.kept = 0

    def rest(self, cl) -> None:
        self.conjectures(cl)
        if self.a.hp.flags.library:
            self.abstraction(cl)

    # ---- conjectures ----------------------------------------------------------------
    def conjectures(self, cl) -> list[tuple[str, str]]:
        a, m = self.a, cl.ctx.m
        found = []
        for col in list(a.spaces.in_context(cl.cid)):
            if col.state != "locked" or col.kind != "operator" or col.arity != 2 or col.out_type != "O":
                continue
            if col.sem.size != m * m:
                continue
            T = col.sem.reshape(m, m)
            props = {
                "commutative": bool((T == T.T).all()),
                "associative": _assoc(T),
                "idempotent": bool((np.diag(T) == np.arange(m)).all()),
                "latin": all(len(set(r.tolist())) == m for r in T) and all(len(set(c.tolist())) == m for c in T.T),
            }
            ident = [e for e in range(m) if (T[e, :] == np.arange(m)).all() and (T[:, e] == np.arange(m)).all()]
            props["identity"] = bool(ident)
            for name, holds in props.items():
                a.priors.observe(f"prop:{name}:{'yes' if holds else 'no'}")
                if holds:
                    param = ident[0] if name == "identity" else None
                    p = a.new_column(cl, col.frame, "property", ("prop", name, param), np.zeros(0), 0.0,
                                     origin="derived", state="locked", parent=col.id)
                    p.note = f"{name} of {col.id}"
                    found.append((col.id, name))
            a.log.log("conjecture", col=col.id, props=props, identity=ident)
        return found

    # ---- abstraction ------------------------------------------------------------------
    def abstraction(self, cl) -> int:
        a = self.a
        lib = [s for s in a.priors.schemas.values() if not s.abstraction]
        kept = 0
        P = cl.prims()
        for i, s1 in enumerate(lib):
            for s2 in lib[i + 1:]:
                if (s1.arity, s1.out_type) != (s2.arity, s2.out_type):
                    continue
                g = anti_unify(s1.body, s2.body)
                if g is None or g in (s1.body, s2.body) or g[0] == "H":
                    continue
                sch, kinds = _rename(g)
                key = (sch, s1.arity, s1.out_type)
                if key in a.priors.schemas:
                    continue
                self.proposed += 1
                extra = len(kinds) - (len(s1.kinds) + len(s2.kinds)) / 2.0
                fill = 0.0
                for h, k in kinds.items():
                    fill += STRUCT_BITS if "T" in k else HOLE_BITS
                saving = struct_bits(s1.body) + struct_bits(s2.body) - (struct_bits(sch) + fill)
                opts = [len(H.hole_options(kinds[h], P, s1.arity)) for h in sorted(kinds)]
                instances = math.prod(max(o, 1) for o in opts)
                replay_ok = instances < max(a.stats["search_nodes"] // max(a.stats["pools_built"], 1), 1)
                ok = saving >= a.hp.play_min_saving and replay_ok and H.size(sch) <= 12
                a.log.log("abstraction", s1=s1.sid, s2=s2.sid, body=H.show(sch), saving=round(saving, 2),
                          instances=instances, replay_ok=replay_ok, kept=ok, extra_holes=extra)
                if ok:
                    new = a.priors.add_schema(sch, s1.arity, s1.out_type, f"{s1.sid}+{s2.sid}", abstraction=True)
                    new.count = s1.count + s2.count
                    kept += 1
                else:
                    a.spaces.note("abstraction-rejected", cl.cid, None, a.log.seq, body=H.show(sch),
                                  saving=saving)
        self.kept += kept
        return kept


def _assoc(T: np.ndarray) -> bool:
    m = T.shape[0]
    x = np.arange(m)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return bool((T[T[X, Y], Z] == T[X, T[Y, Z]]).all())
