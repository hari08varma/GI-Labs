"""Lifecycle (plan §7.7): merge, retire, budget. Nothing is ever deleted.

Merge: two locked operator columns in one context with the same signature and
identical output vectors on the whole domain name the same function. The later frame's
words are unified into the earlier frame (identity role order), so each fact is stored
once, and the later column becomes a `merged` record pointing at the earlier one.
Retire: superseded and unused for 5 systems, or plastic and unused for 2 systems.
Budget: at most 2,000 active columns; above that the plastic columns with the lowest
uses x recency are retired. Locked columns are never retired by the budget.
"""
from __future__ import annotations

import numpy as np


class Lifecycle:
    def __init__(self, agent):
        self.a = agent

    def rest(self, cl) -> None:
        self.merge(cl)
        self.retire_old()
        self.budget()

    def merge(self, cl) -> int:
        a, ctx = self.a, cl.ctx
        locked = [c for c in a.spaces.in_context(cl.cid) if c.state == "locked" and c.kind == "operator"
                  and c.scope is None]
        n = 0
        for i, c1 in enumerate(locked):
            for c2 in locked[i + 1:]:
                if c2.state != "locked" or c1.signature() != c2.signature() or c1.frame == c2.frame:
                    continue
                if c1.sem.shape != c2.sem.shape or not np.array_equal(c1.sem, c2.sem):
                    continue
                words = ctx.words_of(c2.frame)
                if len(words) != 1 or ctx.lex[words[0]].sigma != tuple(range(len(ctx.lex[words[0]].sigma))):
                    continue
                ctx.unify(words[0], c1.frame, ctx.lex[words[0]].sigma, 0)
                c2.state = "merged"
                c2.key = None
                c2.replaced_by = c1.id
                a.log.log("merge", col=c2.id, into=c1.id, words=words)
                n += 1
        return n

    def retire_old(self) -> None:
        a = self.a
        now = a.sys_count
        for c in list(a.spaces.columns.values()):
            idle = now - c.stats.last_used if c.stats.uses else now - _created_sys(a, c)
            if c.state == "superseded" and idle >= a.hp.retire_superseded_after:
                c.state = "retired"
                a.log.log("retire", col=c.id, why="superseded and unused")
            elif c.state == "plastic" and idle >= a.hp.retire_plastic_after:
                a.retire(c, "plastic and unused")

    def budget(self) -> None:
        a = self.a
        active = [c for c in a.spaces.columns.values() if c.active]
        over = len(active) - a.hp.column_budget
        if over <= 0:
            return
        plastic = sorted((c for c in active if c.state == "plastic"),
                         key=lambda c: (c.stats.uses * (1 + c.stats.last_used), c.provenance.created_at))
        for c in plastic[:over]:
            a.retire(c, "column budget")


def _created_sys(agent, col) -> int:
    ids = list(agent.learners)
    return ids.index(col.ctx) if col.ctx in ids else 0
