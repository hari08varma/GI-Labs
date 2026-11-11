"""Shared concepts across worlds: extend first, link, propagate (Harinath's rule, 9 Nov).

A concept is a library schema (a context-free rule body with holes) together with its
BINDINGS: the world columns that instantiate it. When a rule locks in a world:
  1. if it has the same form as an existing concept, it becomes one more binding of it;
  2. otherwise the learner tries to EXTEND a related concept by generalizing it
     (anti-unification). The extension is accepted only if every rule already bound to the
     concept is still an exact instance of the generalized rule (so everything those rules
     proved still holds: learning without forgetting) and it fits the size budget S_MAX;
  3. only if no related concept can be extended does it create a new concept.
Old versions stay in `versions` for audit.

When a bound rule is shown wrong in one world, the lesson propagates: the concept's trust
drops (so later worlds verify it more), and every other bound world re-checks its rule against
its own stored facts at no cost. A contradiction with a checker-verified fact there supersedes
that rule (evidence from its own world); a contradiction with an observation only marks it a
suspect. Evidence from one world never overwrites another world's validated rule.
"""
from __future__ import annotations

from ..learn import hypothesis as H
from ..memory.columns import S_MAX


def match(schema, expr, fill: dict | None = None) -> dict | None:
    """Fill of the schema's holes that turns it into expr exactly, or None."""
    fill = {} if fill is None else fill
    if schema[0] == "H":
        h = schema[1]
        if h in fill:
            return fill if fill[h] == expr else None
        if H.OUT_TYPE.get(expr[0], "O") != "O":
            return None
        fill[h] = expr
        return fill
    if schema[0] != expr[0] or len(schema) != len(expr):
        return None
    if schema[0] in ("table", "map"):
        return fill if schema == expr else None
    for i, _k in H.PARAM_SLOTS.get(schema[0], ()):
        p, v = schema[i], expr[i]
        if H.is_hole(p):
            if p in fill and fill[p] != v:
                return None
            fill[p] = v
        elif p != v:
            return None
    for i in H.CHILD_SLOTS[schema[0]]:
        if match(schema[i], expr[i], fill) is None:
            return None
    return fill


def concrete_share(g) -> float:
    """Fraction of a schema's nodes that are not subtree holes (how much structure is shared)."""
    tot = H.size(g)
    holes = len(H.subtree_holes(g))
    return (tot - holes) / max(tot, 1)


class Concepts:
    def __init__(self, agent):
        self.a = agent
        self.extended = 0
        self.created = 0
        self.joined = 0

    def bindings(self, s) -> list:
        return getattr(s, "bindings", [])

    def on_lock(self, cl, col):
        """Bind a newly locked rule to a concept: join, extend, or create (in that order)."""
        a, pr = self.a, self.a.priors
        sch, kinds = H.to_schema(col.body)
        key = (sch, col.arity, col.out_type)
        s = pr.schemas.get(key)
        if s is not None:                                   # 1. same form: one more binding
            s.count += 1
            s.sources.append(col.id)
            s.bindings.append((cl.cid, col.frame, col.id))
            pr.version += 1
            self.joined += 1
            a.log.log("concept", how="join", concept=s.sid, col=col.id, count=s.count)
            self.record_history(cl, col, s)
            return s
        for k2, s2 in sorted(pr.schemas.items(), key=lambda kv: (-kv[1].count, kv[1].sid)):
            if (s2.arity, s2.out_type) != (col.arity, col.out_type) or s2.abstraction:
                continue
            if s2.body[0] != sch[0]:
                continue                                    # not related: different root rule
            g = _anti_unify(s2.body, sch)
            if g is None or H.size(g) > S_MAX or concrete_share(g) < 0.6:
                continue
            g, gk = H.to_schema(g)
            bound = [a.spaces.columns.get(cid) for _, _, cid in s2.bindings]
            bodies = [c.body for c in bound if c is not None] + [col.body]
            if any(match(g, b) is None for b in bodies):   # 2. every proven rule must still hold
                continue
            pr.schemas.pop(k2)
            s2.versions.append(s2.body)
            s2.body, s2.kinds = g, gk
            s2.count += 1
            s2.sources.append(col.id)
            s2.bindings.append((cl.cid, col.frame, col.id))
            new_key = (g, col.arity, col.out_type)
            if new_key in pr.schemas:                       # merged into an equal concept
                other = pr.schemas[new_key]
                other.count += s2.count
                other.bindings += s2.bindings
                other.versions += s2.versions
                s2 = other
            pr.schemas[new_key] = s2
            pr.version += 1
            self.extended += 1
            a.log.log("concept", how="extend", concept=s2.sid, col=col.id, body=H.show(g),
                      versions=len(s2.versions), bindings=len(s2.bindings))
            self.record_history(cl, col, s2)
            return s2
        s = pr.add_schema(col.body, col.arity, col.out_type, col.id)    # 3. new concept
        s.bindings.append((cl.cid, col.frame, col.id))
        self.created += 1
        a.log.log("concept", how="create", concept=s.sid, col=col.id, body=H.show(s.body))
        self.record_history(cl, col, s)
        return s

    # ---- how it was learned: decisive checks and rivals -------------------------------
    @staticmethod
    def _frame_of_ref(s, fill, P):
        """(anchor entity, chain table) of a concept instance, if it has both."""
        ents = sorted(h for h, k in s.kinds.items() if "E" in k and h in fill)
        refs = sorted(h for h, k in s.kinds.items() if "C" in k and h in fill)
        if not ents or not refs or fill[refs[0]] not in P.chains:
            return None
        return int(fill[ents[0]]), P.chains[fill[refs[0]]]

    def record_history(self, cl, col, s) -> None:
        """At lock: which checker facts refuted which rivals. Checks are stored as offsets from
        the concept's anchor along its chain, so they mean the same thing in another world."""
        import numpy as np
        sc = cl.scored.get(col.frame)
        if sc is None or len(sc.exprs) == 0 or col.arity != 2:
            return
        d = cl.data(col.frame)
        pre = sc.prior + sc.data                     # before checker refutations
        ok = np.isfinite(pre)
        if not ok.any():
            return
        same = (sc.sem == col.sem.astype(sc.sem.dtype)[None, :]).all(axis=1)
        best = pre[same].min() if same.any() else pre[ok].min()
        rivals = np.flatnonzero(ok & ~same & (pre <= best + 2 * self.a.hp.delta_vs))
        fill = match(s.body, col.body) or {}
        fr = self._frame_of_ref(s, fill, cl.prims(col.frame))
        m = cl.ctx.m
        facts = [(c, o, True) for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist())] + \
                [(c, o, False) for c, o in zip(d.neg_cells.tolist(), d.neg_outs.tolist())]
        for r in rivals.tolist():
            orig = sc.origin[r]
            hit = [c for c, o, t in facts if (int(sc.sem[r][c]) != o) == t]
            if not hit:
                continue
            if orig.startswith("schema:"):
                sid = orig.split(":", 1)[1]
                s.rivals[sid] = s.rivals.get(sid, 0) + 1
            if fr is not None:
                anchor, C = fr
                c = hit[0]
                pat = (int(C[anchor][c // m]), int(C[anchor][c % m]))
                s.decisive[pat] = s.decisive.get(pat, 0) + 1
        if s.rivals or s.decisive:
            self.a.log.log("history", concept=s.sid, col=col.id, decisive=s.decisive, rivals=s.rivals)

    def history_cells(self, cl, col, s, sc, known: set) -> list[int]:
        """Cells to check first for a recognized concept: its remembered decisive checks, then a
        cell that separates it from a rival that fooled it before."""
        import numpy as np
        m = cl.ctx.m
        out: list[int] = []
        fill = match(s.body, col.body) or {}
        fr = self._frame_of_ref(s, fill, cl.prims(col.frame))
        if fr is not None:
            anchor, C = fr
            row = C[anchor][:m]
            for (ka, kb), _n in sorted(s.decisive.items(), key=lambda kv: -kv[1]):
                ea = np.flatnonzero(row == ka % m)
                eb = np.flatnonzero(row == kb % m)
                if len(ea) and len(eb):
                    c = int(ea[0]) * m + int(eb[0])
                    if c not in known and c not in out:
                        out.append(c)
        if sc is not None and len(sc.exprs):
            pre = sc.prior + sc.data
            for sid, _n in sorted(s.rivals.items(), key=lambda kv: -kv[1]):
                rows = [i for i, o in enumerate(sc.origin) if o == "schema:" + sid and np.isfinite(sc.total[i])
                        and not np.array_equal(sc.sem[i], col.sem.astype(sc.sem.dtype))]
                if not rows:
                    continue
                i = min(rows, key=lambda i: pre[i])
                diff = [int(c) for c in np.flatnonzero(sc.sem[i] != col.sem.astype(sc.sem.dtype)) if int(c) not in known]
                if diff and diff[0] not in out:
                    out.append(diff[0])
        return out

    def concept_of(self, col):
        for s in self.a.priors.schemas.values():
            if any(cid == col.id for _, _, cid in self.bindings(s)):
                return s
        return None

    def on_fault(self, cl, col, why: str) -> None:
        """A bound rule was shown wrong in one world: lower the concept's trust and re-check every
        other bound world against its own stored facts (free)."""
        s = self.concept_of(col)
        if s is None:
            return
        s.refuted += 1
        self.a.priors.version += 1
        self.a.log.log("concept", how="fault", concept=s.sid, col=col.id, why=why, trust=round(s.trust, 3))
        for cid, frame, col_id in list(s.bindings):
            other = self.a.spaces.columns.get(col_id)
            if other is None or other.id == col.id or not other.active or cid == cl.cid:
                continue
            ocl = self.a.learners.get(cid)
            if ocl is None:
                continue
            d = ocl.data(frame)
            vs = [(c, o) for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist())
                  if c < other.sem.size and int(other.sem[c]) != o]
            os_ = [(c, o) for c, o in zip(d.cells.tolist(), d.outs.tolist())
                   if c < other.sem.size and int(other.sem[c]) != o]
            if vs:
                self.a.replace(other, None, f"linked concept {s.sid} failed in {cl.cid}; its own checker fact disagrees")
                self.a.log.log("concept", how="propagate-supersede", concept=s.sid, col=other.id)
            elif os_:
                for c, o in os_:
                    ocl.suspects[other.id].append((c, o))
                self.a.spaces.note("suspect-from-link", cid, frame, self.a.log.seq, col=other.id, concept=s.sid,
                                   n=len(os_))
                self.a.log.log("concept", how="propagate-suspect", concept=s.sid, col=other.id, n=len(os_))
            else:
                self.a.log.log("concept", how="propagate-ok", concept=s.sid, col=other.id)


def _anti_unify(a, b):
    from .play import anti_unify
    return anti_unify(a, b)


def self_test(agent, cl) -> int:
    """Self-generated consistency test at rest (free, no checker): every newly bound rule is
    compared with the derived properties its concept shows in other worlds (commutative,
    associative, latin, identity). A property the concept always had but this rule lacks is
    a warning: the rule becomes a suspect for the next check in its world. Returns warnings."""
    import numpy as np

    from .play import _assoc
    warn = 0
    m = cl.ctx.m
    for s in agent.priors.schemas.values():
        here = [(cid, F, c) for cid, F, c in getattr(s, "bindings", []) if cid == cl.cid]
        there = [agent.spaces.columns.get(c) for cid, _, c in getattr(s, "bindings", []) if cid != cl.cid]
        there = [c for c in there if c is not None and c.arity == 2 and c.out_type == "O"]
        if not here or len(there) < 2:
            continue

        def props(col, n):
            T = col.sem.reshape(n, n)
            x = np.arange(n)
            return {"commutative": bool((T == T.T).all()), "associative": _assoc(T),
                    "latin": all(len(set(r.tolist())) == n for r in T) and all(len(set(c.tolist())) == n for c in T.T),
                    "identity": any((T[e] == x).all() and (T[:, e] == x).all() for e in range(n))}

        usual = None
        for c in there:
            n = int(round(c.sem.size ** 0.5))
            p = props(c, n)
            usual = p if usual is None else {k: usual[k] and p[k] for k in usual}
        for cid, F, col_id in here:
            col = agent.spaces.columns.get(col_id)
            if col is None or col.arity != 2 or col.sem.size != m * m:
                continue
            mine = props(col, m)
            missing = [k for k, v in usual.items() if v and not mine[k]]
            if missing:
                warn += 1
                agent.spaces.note("self-test", cl.cid, F, agent.log.seq, col=col_id, concept=s.sid, missing=missing)
                agent.log.log("self_test", ctx=cl.cid, col=col_id, concept=s.sid, missing=missing)
    return warn

