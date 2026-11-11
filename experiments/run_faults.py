"""Fault injection experiment (plan §9.4). Usage:
    python -m experiments.run_faults --seed 10 --out results/faults_s10.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from experiments.faults import INJECTORS, inject_slip, judge
from experiments.harness import run_system
from gca.agent import GCAAgent
from gca.config import Hparams
from gca.world.api import leaves
from gca.world.generator import StreamBuilder, load_config

TYPES = ("out_of_scope", "wrong_concept", "misread", "slip")


def _words(expr) -> set:
    if expr[0] != "app":
        return set()
    out = {expr[1]}
    for e in expr[2:]:
        out |= _words(e)
    return out


def fault_stream(cfg: dict, seed: int, ftype: str, *, systems: list[str] | None = None,
                 hp: Hparams | None = None, max_extra: int = 30) -> list[dict]:
    sb = StreamBuilder(cfg, seed, 0.0)
    stream = sb.build()
    agent = GCAAgent(hp or Hparams(seed=seed))
    rng = np.random.default_rng([seed, 99, TYPES.index(ftype)])
    results = []
    for sw in stream:
        state = {}

        def hook(learner, sw_, checker):
            if systems is not None and sw_.sid not in systems:
                return []
            state["mark"] = len(agent.diagnoser.records)
            if ftype == "slip":
                truth, obs = inject_slip(agent, sw_.sid, rng, sw_)
                if truth is None:
                    return []
                agent.observe(obs)
                agent.learn(checker)
                state["truth"] = truth
                return []
            truth = INJECTORS[ftype](agent, sw_.sid, rng)
            if truth is None:
                return []
            state["truth"] = truth
            cl = agent.learners[sw_.sid]
            if ftype == "misread":
                fw = {truth["component"][4:]}
            else:
                col = agent.spaces.columns[truth["component"]]
                fw = set(cl.ctx.words_of(col.frame))
            qs = [q for sp in ("S1", "S2", "S4") for q in sw_.questions[sp] if _words(q.expr) & fw
                  and all(t in cl.ctx.index for t in leaves(q.expr))]
            return qs[:max_extra]

        r = run_system(agent, sw, sb.budget(sw), curve=False, hook=hook)
        if "truth" in state:
            recs = agent.diagnoser.records[state["mark"]:]
            j = judge(state["truth"], recs)
            results.append({"sid": sw.sid, "truth": state["truth"], "judge": j,
                            "diagnoses": [{k: v for k, v in x.items() if k in ("cause", "component", "checks", "via")}
                                          for x in recs], "test_acc": r.test_acc})
    return results


def summarize(results: list[dict]) -> dict:
    trig = [r for r in results if r["judge"]["triggered"]]
    ok = [r for r in trig if r["judge"]["correct"]]
    return {"injected": len(results), "triggered": len(trig), "correct": len(ok),
            "precision": (len(ok) / len(trig)) if trig else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--config", default="configs/stream_default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--types", default=",".join(TYPES))
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = {}
    for t in a.types.split(","):
        res = fault_stream(cfg, a.seed, t)
        out[t] = {"summary": summarize(res), "runs": res}
        print(t, out[t]["summary"], flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
