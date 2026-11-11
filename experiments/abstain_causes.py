"""Why did the learner abstain? (world-side analysis; the agent never sees this.) Usage:
    python -m experiments.abstain_causes --seed 0 --noise 0.2 --out results/dev/causes_s0_n0.2.json

Each abstained test question gets ONE cause, checked in this order:
  foundation    a fully observed unary frame of the system (its successor chain) is not locked
  budget        a frame the question needs has no usable rule and the checker budget was spent
                down to the reserve during learning
  rule:data     no usable rule, budget left, and the CLEAN observations (noise removed, using
                world truth) do identify the true operator in the generator's grammar: fixable
  rule:nodata   no usable rule, and even the clean observations do not identify it: abstaining
                is correct unless the learner queries for more
  solver        every frame has a rule, yet the answer was "unknown" (undefined/ambiguous)
Confident wrong answers are counted separately.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np

from experiments.harness import run_system
from experiments.run_stream import make
from gca.world import grammar as G
from gca.world.api import UNKNOWN
from gca.world.checker import score
from gca.world.generator import StreamBuilder, load_config


def words(expr) -> set:
    if expr[0] != "app":
        return set()
    out = {expr[1]}
    for e in expr[2:]:
        out |= words(e)
    return out


def clean_identifiable(sw, word: str) -> bool:
    w = sw.wdef(word)
    if w.foundation:
        return True
    clean = [ob for ob in sw.experience if ob.word == word and sw.true_fact(ob.word, ob.args, ob.out)]
    t = sw.tables[word]
    cells = [int(np.ravel_multi_index(tuple(sw.objs.index(a) for a in ob.args), t.shape)) for ob in clean]
    fam = G.family_tables(w.spec, sw.n)
    flat = t.ravel()
    if not cells:
        return False
    cons = fam[(fam[:, cells] == flat[cells]).all(axis=1)]
    return bool((cons == flat).all())


def analyse(seed: int, noise: float, cfg: dict) -> dict:
    sb = StreamBuilder(cfg, seed, noise)
    agent = make("gca", seed)
    causes, per_sys = Counter(), {}
    for sw in sb.build():
        r = run_system(agent, sw, sb.budget(sw), curve=False)
        cl = agent.learners[sw.sid]
        ctx = cl.ctx
        budget_spent = r.calls_learn >= r.budget - agent.hp.reserve
        broken = any(info["arity"] == 1 and not info["bool"] and ctx.words_of(F)
                     and (cl.active_col(F) is None or cl.active_col(F).state != "locked")
                     for F, info in ctx.frames.items())
        c = Counter()
        for qs in sw.questions.values():
            for q in qs:
                a = agent.answer(q)
                if score(sw, q, a.value):
                    continue
                if a.value != UNKNOWN:
                    c["wrong"] += 1
                    continue
                if broken:
                    c["foundation"] += 1
                    continue
                need = [w for w in words(q.expr) if w in ctx.lex]
                missing = [w for w in need if (col := cl.active_col(ctx.lex[w].frame)) is None
                           or col.state not in ("locked", "candidate")]
                if not missing:
                    c["solver"] += 1
                elif budget_spent:
                    c["budget"] += 1
                elif all(clean_identifiable(sw, w) for w in missing):
                    c["rule:data"] += 1
                else:
                    c["rule:nodata"] += 1
        per_sys[sw.sid] = {"acc": r.test_acc, "calls": r.calls_learn, "budget": r.budget, **c}
        causes.update(c)
    return {"seed": seed, "noise": noise, "causes": dict(causes), "systems": per_sys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--config", default="configs/stream_default.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = analyse(a.seed, a.noise, load_config(a.config))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    n = sum(out["causes"].values())
    print(f"seed={a.seed} noise={a.noise} failures={n} " +
          " ".join(f"{k}={v}" for k, v in sorted(out["causes"].items())), flush=True)


if __name__ == "__main__":
    main()
