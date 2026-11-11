"""World-side metrics the agent never sees: rule correctness against the hidden truth."""
from __future__ import annotations


def column_accuracy(agent, cl, sw, col) -> float | None:
    """Fraction of canonical cells where a column agrees with the world (None if untestable)."""
    ctx = cl.ctx
    words = [w for w in ctx.words_of(col.frame) if w in sw.tables]
    if not words or col.sem.size == 0:
        return None
    w = words[0]
    e = ctx.lex[w]
    m, k = ctx.m, col.arity
    ok = 0
    for c in range(col.sem.size):
        args = (c // m, c % m) if k == 2 else (c,)
        surf = ctx.surface(e, args + (int(col.sem[c]),))
        ok += sw.true_fact(w, tuple(ctx.tokens[i] for i in surf[:-1]), agent.token(ctx, col.frame, surf[-1]))
    return ok / col.sem.size


def locked_corruption(agent, systems) -> dict:
    """Locked operator rules that are wrong anywhere on the domain (should be 0)."""
    bad = tot = 0
    for sw in systems:
        cl = agent.learners.get(sw.sid)
        if cl is None:
            continue
        for c in agent.spaces.in_context(sw.sid):
            if c.state == "locked" and c.kind == "operator" and c.scope is None:
                acc = column_accuracy(agent, cl, sw, c)
                if acc is not None:
                    tot += 1
                    bad += acc < 1.0
    return {"locked": tot, "locked_wrong": bad}


def lock_costs(agent) -> list[dict]:
    """Per-rule cost to lock (fine measurement): observations of the rule's words plus checker
    questions about them, counted up to the moment the rule locked. One row per locked rule."""
    obs: dict = {}
    qs: dict = {}
    out = []
    order = {cid: i for i, cid in enumerate(agent.learners)}
    for r in agent.log.records:
        ev, ctx = r["ev"], r.get("ctx")
        if ev == "observe":
            obs[(ctx, r["word"])] = obs.get((ctx, r["word"]), 0) + 1
        elif ev == "query":
            key = (ctx, r.get("word"))
            qs[key] = qs.get(key, 0) + 1
        elif ev == "lock" and not str(r.get("rule", "")).startswith("lexicon"):
            col = agent.spaces.columns.get(r["col"])
            cl = agent.learners.get(ctx)
            if col is None or cl is None or col.kind != "operator":
                continue
            words = cl.ctx.words_of(col.frame)
            out.append({"sid": ctx, "index": order.get(ctx, -1), "words": words, "rule": r.get("rule"),
                        "origin": col.provenance.origin, "table": col.body == ("table",),
                        "obs": sum(obs.get((ctx, w), 0) for w in words),
                        "queries": sum(qs.get((ctx, w), 0) for w in words)})
    return out
