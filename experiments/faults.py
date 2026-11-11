"""Fault injection (plan §9.4): diagnosis has ground truth because the fault is known.

Each injector corrupts one piece of a learner's knowledge right after the experience
phase of a system and returns the ground truth (component, cause). Practice questions
with feedback then trigger diagnosis; precision is whether the first diagnosis that
names the faulted component also names the right cause.

  out_of_scope   a locked column gets a wrong region that a cheap condition separates
                 from all its validated inputs (scope corruption)
  misread        a unified word's role order is swapped
  wrong_concept  a locked column's body is replaced by one that is right on ~80% of the
                 domain and wrong on at least one validated input
  slip           a burst of noisy observations contradicting locked columns
"""
from __future__ import annotations

import itertools

import numpy as np

from gca.learn import hypothesis as H
from gca.world.api import Observation


def _locked_binary(agent, cid):
    return [c for c in agent.spaces.in_context(cid) if c.state == "locked" and c.kind == "operator"
            and c.arity == 2 and c.out_type == "O" and c.scope is None and c.body[0] != "table"]


def inject_out_of_scope(agent, cid: str, rng) -> dict | None:
    cl = agent.learners[cid]
    cols = _locked_binary(agent, cid)
    if not cols:
        return None
    col = cols[int(rng.integers(len(cols)))]
    m = cl.ctx.m
    pool = cl.pool(2, col.frame)
    sem, cost, exprs, _ = pool.flat("B")
    V = np.array(sorted(col.validated), dtype=np.int64)
    ok = (sem[:, V] == 0).all(axis=1) & (sem.sum(axis=1) >= 3) if len(V) else sem.sum(axis=1) >= 3
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None
    i = int(idx[np.argmin(cost[idx])])
    region = sem[i] == 1
    bad = col.sem.copy()
    bad[region] = (bad[region] + 1) % m
    col.sem = bad                          # same id, same validated set: the scope is what broke
    col.note = f"FAULT out_of_scope region={H.show(exprs[i])}"
    return {"type": "out_of_scope", "component": col.id, "cause": "out_of_scope", "cells": int(region.sum())}


def inject_wrong_concept(agent, cid: str, rng) -> dict | None:
    cl = agent.learners[cid]
    cols = _locked_binary(agent, cid)
    if not cols:
        return None
    col = cols[int(rng.integers(len(cols)))]
    m = cl.ctx.m
    n = col.sem.size
    k = max(1, int(round(0.2 * n)))
    V = sorted(col.validated)
    wrong = set(rng.choice(n, k, replace=False).tolist())
    if V and not (wrong & set(V)):
        wrong.add(int(V[int(rng.integers(len(V)))]))
    bad = col.sem.copy()
    for c in wrong:
        bad[c] = (bad[c] + 1 + int(rng.integers(m - 1))) % m
    col.sem = bad
    col.note = "FAULT wrong_concept"
    return {"type": "wrong_concept", "component": col.id, "cause": "wrong_concept", "cells": len(wrong)}


def inject_misread(agent, cid: str, rng) -> dict | None:
    cl = agent.learners[cid]
    ctx = cl.ctx
    words = [w for w, e in ctx.lex.items() if len(ctx.words_of(e.frame)) > 1
             and e.sigma != tuple(range(len(e.sigma)))]
    if not words:
        return None
    w = words[int(rng.integers(len(words)))]
    e = ctx.lex[w]
    k = len(e.sigma) - 1
    perms = [p for p in itertools.permutations(range(k + 1)) if p != e.sigma
             and (not ctx.frames[e.frame]["bool"] or p[k] == k)]
    e.sigma = perms[int(rng.integers(len(perms)))]
    return {"type": "misread", "component": f"lex:{w}", "cause": "misread"}


def inject_slip(agent, cid: str, rng, sw, burst: int = 6) -> tuple[dict | None, list]:
    """Returns the ground truth and the noisy observations to feed to the learner."""
    cols = [c for c in agent.spaces.in_context(cid) if c.state == "locked" and c.kind == "operator"
            and c.scope is None]
    words = [w for w in sw.words if any(agent.learners[cid].ctx.lex.get(w.word) is not None
                                        and agent.learners[cid].ctx.lex[w.word].frame == c.frame for c in cols)]
    if not words:
        return None, []
    obs = []
    for _ in range(burst):
        w = words[int(rng.integers(len(words)))]
        k = sw.arity(w.word)
        args = tuple(int(a) for a in rng.integers(0, sw.n, k))
        v = sw.apply(w.word, args)
        v = 1 - v if sw.boolean(w.word) else (v + 1 + int(rng.integers(sw.n - 1))) % sw.n
        obs.append(Observation(sw.sid, w.word, tuple(sw.objs[a] for a in args), sw.tok(v, w.word)))
    return {"type": "slip", "component": "event", "cause": "slip", "n": len(obs)}, obs


INJECTORS = {"out_of_scope": inject_out_of_scope, "wrong_concept": inject_wrong_concept,
             "misread": inject_misread}


def judge(truth: dict, records: list[dict]) -> dict:
    """Did diagnosis name the faulted component, and the right cause?"""
    if truth["type"] == "slip":
        slips = [r for r in records if r.get("cause") == "slip"]
        wrong = [r for r in records if r.get("cause") in ("wrong_concept", "out_of_scope")]
        return {"triggered": bool(slips or wrong), "named": len(slips), "of": truth["n"],
                "correct": len(slips) > 0 and not wrong, "superseded_locked": len(wrong)}
    hits = [r for r in records if r.get("component") == truth["component"]
            or (truth["type"] != "misread" and str((r.get("first_wrong") or {}).get("source", "")) == truth["component"])]
    first = next((r for r in records if r.get("cause") not in (None, "unlocated")), None)
    return {"triggered": first is not None, "component_ok": bool(hits) and first in hits,
            "cause": first.get("cause") if first else None,
            "correct": first is not None and first in hits and first.get("cause") == truth["cause"]}
