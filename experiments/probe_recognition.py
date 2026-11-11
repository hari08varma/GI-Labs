"""Probe: why does "I recognize this, quick check, done" not happen? (world-side analysis)

Runs the stream up to a target system, then steps through the target increment by increment
and, after every learn() call, reports for each operator frame:
  obs     observations of the frame's words so far
  best    the current MAP hypothesis and where it came from (pool / split / schema / analogy)
  true    rank, origin and bits of the TRUE function among the candidates (from world truth)
  gap     bits by which the true function trails the MAP (prior + data)
  post    posterior of the true function
  state   the column's state, and why the recognize-verify path did or did not fire
Usage: python -m experiments.probe_recognition --seed 0 --target S10
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.harness import run_system
from experiments.run_stream import make
from gca.learn.mdl import null_bits, sprout_threshold
from gca.world.checker import Checker
from gca.world.generator import StreamBuilder, load_config


def true_row(cl, sw, F):
    ctx, m = cl.ctx, cl.ctx.m
    w = ctx.words_of(F)[0]
    e = ctx.lex[w]
    k = ctx.frames[F]["arity"]
    row = np.zeros(m ** k, dtype=np.int64)
    for c in range(m ** k):
        args = (c // m, c % m) if k == 2 else (c,)
        # canonical args -> surface args of word w, then world truth, then back to canonical out
        for v in range(m):
            surf = ctx.surface(e, args + (v,))
            out = cl.a.token(ctx, F, surf[-1])
            if sw.true_fact(w, tuple(ctx.tokens[i] for i in surf[:-1]), out):
                row[c] = v
                break
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="S10")
    a = ap.parse_args()
    sb = StreamBuilder(load_config("configs/stream_default.yaml"), a.seed, 0.0)
    systems = sb.build()
    agent = make("gca", a.seed)
    for sw in systems:
        if sw.sid == a.target:
            break
        run_system(agent, sw, sb.budget(sw), curve=False)
    agent.start_system(sw.sid)
    checker = Checker(sw, sb.budget(sw))
    cl = agent.learners[sw.sid]
    print(f"{sw.sid} n={sw.n} budget={sb.budget(sw)} ops={[(w.word, w.spec.kind, w.spec.params) for w in sw.words]}")
    for i, inc in enumerate(sw.increments):
        agent.observe(inc)
        agent.learn(checker)
        for F, info in cl.ctx.frames.items():
            if not cl.ctx.words_of(F) or info["arity"] != 2:
                continue
            col = cl.active_col(F)
            d = cl.data(F)
            sc = cl.scored.get(F)
            line = f"inc {i:2d} calls={checker.calls_used():2d} {'/'.join(cl.ctx.words_of(F)):12s} obs={d.n_obs:3.0f}"
            if col is not None and col.state == "locked":
                print(line, f"LOCKED [{col.provenance.origin}] {cl.show(col.body)[:70]}")
                continue
            if sc is None or len(sc.exprs) == 0:
                print(line, "no candidates")
                continue
            tr = true_row(cl, sw, F)
            t = sc.total
            b = sc.best()
            same = np.flatnonzero((sc.sem == tr.astype(sc.sem.dtype)[None, :]).all(axis=1))
            delta = sprout_threshold(agent.sys_count, agent.hp.sprout_start, agent.hp.sprout_end, agent.hp.sprout_over)
            gain = null_bits(d) - t[b]
            if len(same) == 0:
                print(line, f"best=[{sc.origin[b]}] {cl.show(sc.exprs[b])[:50]} | TRUE NOT IN CANDIDATES | gain={gain:.1f}/{delta:.1f}")
                continue
            j = int(same[0])
            rank = int((t < t[j]).sum())
            w = np.exp2(-(t[np.isfinite(t)] - t[b]))
            post = float(np.exp2(-(t[j] - t[b])) / w.sum())
            rec = cl.recognized(col) if col is not None else None
            why = []
            if col is None:
                why.append(f"no column (gain {gain:.1f} < sprout {delta:.1f})" if gain < delta else "no column")
            else:
                if not col.provenance.origin.startswith("schema:"):
                    why.append(f"MAP origin {col.provenance.origin}, not a recognized concept")
                elif rec is None:
                    why.append("concept trust < 0.6")
                if not np.array_equal(col.sem, tr):
                    why.append("MAP is WRONG function")
                if post < 0.99:
                    why.append(f"posterior {post:.2f} < 0.99")
            print(line, f"best=[{sc.origin[b]}] {cl.show(sc.exprs[b])[:40]:40s} true: rank {rank} [{sc.origin[j]}] "
                        f"prior {sc.prior[j]:.1f}b (best {sc.prior[b]:.1f}b) gap {t[j] - t[b]:.1f}b post {post:.2f} | "
                        f"{col.state if col else '-'} | {'; '.join(why)}")


if __name__ == "__main__":
    main()
