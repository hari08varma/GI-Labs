"""Thinking: parse a question into goal frames, then search for zero-energy
configurations (plan §6.1, §6.4).

Each application node is a subgoal on its frame: the canonical tuple of the frame is
(surface args, node value) permuted by the word's role order sigma. When the node's
value is the frame's canonical output it is computed forward. Otherwise (reversed
phrasing, or the unknown of a solve question) every value of the finite domain is
tried and only zero-violation values survive: reversal through roles, with no
separately learned inverse (build log, "Solver").

Answer rules (build log):
- `none` or `many` only when every candidate value's evaluation is defined;
- a unique solution when all candidates are defined, or when the operator has the
  derived `latin` property (then a found solution is the only one);
- otherwise `unknown` (abstain, P6).
Step sources, in priority order: locked column, candidate column, event-store
majority, plastic column.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..learn import reading as R
from ..world.api import MANY, NONE, UNKNOWN, Answer, Question
from .decoder import decode
from .energy import Weights, energy, weakest
from .workspace import Workspace, WorkspaceFull


@dataclass
class Node:
    id: int
    kind: str                       # obj | var | app
    value: int | None = None
    word: str | None = None
    kids: list = field(default_factory=list)
    frame: str | None = None
    sigma: tuple | None = None
    boolean: bool = False


@dataclass
class Step:
    word: str
    args: tuple
    out: str
    source: str                     # column id or "event"
    state: str                      # locked | candidate | event | plastic
    frame: str
    cell: int
    canon: tuple
    kind: str = "forward"           # forward | reversal | reading | composition | fact


STEP_KINDS = ("forward", "reversal", "reading", "composition", "fact")


def step_kind(n, forward: bool, solved_here: bool, sigma_identity: bool, nested: bool, state: str) -> str:
    """What kind of reasoning a step is, so failures can be counted by kind."""
    if state == "event":
        return "fact"
    if not forward or solved_here:
        return "reversal"
    if not sigma_identity:
        return "reading"
    if nested:
        return "composition"
    return "forward"


class Unreadable(Exception):
    pass


def is_latin(sem: np.ndarray, m: int) -> bool:
    if sem.size != m * m:
        return False
    t = sem.reshape(m, m)
    return all(len(set(r.tolist())) == m for r in t) and all(len(set(c.tolist())) == m for c in t.T)


class Solver:
    def __init__(self, agent):
        self.a = agent
        hp = agent.hp
        self.w = Weights(hp.w_locked, hp.w_candidate, hp.w_event, hp.w_plastic, hp.lam)

    # ---- parsing ----------------------------------------------------------------
    def parse(self, cl, expr, nodes: list[Node]) -> int:
        ctx = cl.ctx
        t = expr[0]
        if t == "obj":
            v = ctx.index.get(expr[1])
            if v is None:
                raise Unreadable(f"unknown token {expr[1]}")
            nodes.append(Node(len(nodes), "obj", value=v))
        elif t == "var":
            nodes.append(Node(len(nodes), "var"))
        else:
            e = ctx.lex.get(expr[1])
            if e is None:
                raise Unreadable(f"unknown word {expr[1]}")
            kids = [self.parse(cl, x, nodes) for x in expr[2:]]
            if len(kids) + 1 != len(e.sigma):
                raise Unreadable("arity mismatch")
            k = len(kids)
            boolean = ctx.frames[e.frame]["bool"] and e.sigma[k] == k
            nodes.append(Node(len(nodes), "app", word=expr[1], kids=kids, frame=e.frame, sigma=e.sigma,
                              boolean=boolean))
        return len(nodes) - 1

    # ---- sources ----------------------------------------------------------------
    def sources(self, cl, F: str):
        ctx = cl.ctx
        info = ctx.frames[F]
        sig = f"{info['arity']}->{'B' if info['bool'] else 'O'}"
        q = self.a.router.key("operator", sig, cl.cid, F)
        cols = [c for c, _s in self.a.router.select(q, self.a.spaces.in_context(cl.cid))
                if c.frame == F and c.kind == "operator"]
        size = ctx.m ** info["arity"]
        cols = [c for c in cols if c.sem.size == size]
        events = ctx.majority(F)
        return cols, events

    def lookup(self, cl, F: str, args: tuple, srcs):
        cols, events = srcs
        m = cl.ctx.m
        if any(a < 0 or a >= m for a in args):
            return None
        cell = R.cell_of(cl.ctx, args)
        order = [c for c in cols if c.state == "locked"] + [c for c in cols if c.state == "candidate"]
        for c in order:
            if self.a.scope_ok(cl, c, cell):
                return int(c.sem[cell]), c.id, c.state, c.prior_bits
        if args in events:
            return int(events[args]), "event", "event", 0.0
        for c in cols:
            if c.state == "plastic" and self.a.scope_ok(cl, c, cell):
                return int(c.sem[cell]), c.id, "plastic", c.prior_bits
        return None

    # ---- search -----------------------------------------------------------------
    def answer(self, cl, q: Question) -> Answer:
        nodes: list[Node] = []
        try:
            root = self.parse(cl, q.expr, nodes)
        except Unreadable as ex:
            return Answer(UNKNOWN, [("unreadable", str(ex))], "low")
        k = self.a.hp.workspace_k
        while True:
            try:
                return self._solve(cl, q, nodes, root, Workspace(k))
            except WorkspaceFull:
                if k >= self.a.hp.workspace_kmax:
                    return Answer(UNKNOWN, [("workspace", k)], "low")
                k += 1
                self.a.stats["workspace_grow"] += 1

    def _solve(self, cl, q: Question, nodes: list[Node], root: int, ws: Workspace) -> Answer:
        ctx = cl.ctx
        m = ctx.m
        srcs = {n.frame: self.sources(cl, n.frame) for n in nodes if n.kind == "app"}
        parent = {}
        for n in nodes:
            for kid in n.kids:
                parent[kid] = n.id
        var = next((n.id for n in nodes if n.kind == "var"), None)
        target = None
        if q.kind == "solve":
            rn = nodes[root]
            if q.target is None or var is None:
                return Answer(UNKNOWN, [("malformed",)], "low")
            if rn.boolean:
                target = 1 if q.target == "#T" else 0 if q.target == "#F" else None
            else:
                target = ctx.index.get(q.target)
            if target is None:
                return Answer(UNKNOWN, [("unreadable", q.target)], "low")
        ans_node = var if q.kind == "solve" else root
        vals = [None] * len(nodes)
        steps: list[Step] = []
        sols: dict[int, list[Step]] = {}
        state = {"undef": False, "nodes": 0}
        order = list(range(len(nodes)))          # parse order is post-order

        def dfs(i: int) -> None:
            state["nodes"] += 1
            if i == len(order):
                v = vals[ans_node]
                if v not in sols:
                    sols[v] = list(steps)
                return
            n = nodes[order[i]]
            if n.kind == "obj":
                vals[n.id] = n.value
                ws.bind(n.id, n.value, "given", "high")
                dfs(i + 1)
                ws.release(n.id)
                return
            if n.kind == "var":
                for v in range(m):
                    vals[n.id] = v
                    ws.bind(n.id, v, "search", "low")
                    dfs(i + 1)
                ws.release(n.id)
                return
            kv = tuple(vals[c] for c in n.kids)
            k = len(kv)
            forward = n.sigma[k] == k
            if n.id == root and target is not None:
                cand = [target]
            elif forward:
                cargs = [0] * k                       # surface args -> canonical args via sigma
                for j in range(k):
                    cargs[n.sigma[j]] = kv[j]
                pred = self.lookup(cl, n.frame, tuple(cargs), srcs[n.frame])
                if pred is None:
                    state["undef"] = True
                    return
                cand = [pred[0]]
            else:
                cand = range(2) if n.boolean else range(m)
            for v in cand:
                surf = kv + (v,)
                canon = [0] * (k + 1)
                for j, cp in enumerate(n.sigma):
                    canon[cp] = surf[j]
                pred = self.lookup(cl, n.frame, tuple(canon[:-1]), srcs[n.frame])
                if pred is None:
                    state["undef"] = True
                    continue
                if pred[0] != canon[-1]:
                    continue                          # violation: energy > 0, pruned
                vals[n.id] = v
                ws.bind(n.id, v, pred[1], pred[2])
                for c in n.kids:
                    if nodes[c].kind != "var":
                        ws.release(c)
                kind = step_kind(n, forward, n.id == root and target is not None,
                                 tuple(n.sigma) == tuple(range(k + 1)),
                                 any(nodes[c].kind == "app" for c in n.kids), pred[2])
                steps.append(Step(n.word, tuple(decode(ctx, x, False) for x in kv),
                                  decode(ctx, v, n.boolean), pred[1], pred[2], n.frame,
                                  R.cell_of(ctx, tuple(canon[:-1])), tuple(canon), kind))
                dfs(i + 1)
                steps.pop()
                for c in n.kids:
                    if nodes[c].kind != "var" and vals[c] is not None:
                        ws.bind(c, vals[c], "restore", "high")
                ws.release(n.id)

        dfs(0)
        self.a.stats["solver_nodes"] += state["nodes"]
        boolean_ans = nodes[ans_node].boolean
        vals_found = sorted(sols)
        if q.kind == "solve":
            pnode = nodes[parent[var]] if var in parent else None
            latin = pnode is not None and self.a.is_latin_frame(cl, pnode.frame)
            if state["undef"]:
                if len(vals_found) == 1 and latin:
                    return self._final(ctx, vals_found[0], boolean_ans, sols)
                return Answer(UNKNOWN, [("undefined",)], "low")
            if not vals_found:
                return Answer(NONE, [("no zero-energy value",)], "medium")
            if len(vals_found) > 1:
                return Answer(MANY, [("solutions", len(vals_found))], "medium")
            return self._final(ctx, vals_found[0], boolean_ans, sols)
        if state["undef"] or len(vals_found) != 1:
            return Answer(UNKNOWN, [("undefined" if state["undef"] else "ambiguous", len(vals_found))], "low")
        return self._final(ctx, vals_found[0], boolean_ans, sols)

    def _final(self, ctx, v: int, boolean: bool, sols) -> Answer:
        steps = sols[v]
        bits = sum(self.a.spaces.columns[s.source].prior_bits for s in steps if s.source in self.a.spaces.columns)
        e = energy([], bits, self.w)
        ans = Answer(decode(ctx, v, boolean), steps, weakest([s.state for s in steps]))
        ans.trace.append(("energy", round(e, 3)))
        return ans
