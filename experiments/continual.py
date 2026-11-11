"""Continual-learning metrics over the 20-system stream (plan §9.2). Usage:
    python -m experiments.continual results/main results/ablations > results/continual.md

- ACC, BWT (Lopez-Paz & Ranzato 2017) from R[T, i] (re-test after the whole stream) and R[i, i].
- FWT as defined there needs accuracy on a system before training on it. Every system is a
  new context with new tokens, so that is 0 for every learner and FWT is 0 by construction.
  It is reported but carries no information here; learning-speed transfer replaces it.
- Learning-speed transfer: X_k = observations + checker calls before S1 (or S4) reaches 95%.
  X_k(full) / X_k(no_library) per system, and for each transfer copy X_k(copy) / X_k(original)
  (M7 target: <= 0.30).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(dirs: list[str]) -> dict:
    runs = defaultdict(dict)                     # (name, noise) -> seed -> run
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(f, encoding="utf-8"))
            name = r["learner"] if r.get("ablation", "full") == "full" else r["ablation"]
            runs[(name, r["noise"])][r["seed"]] = r
    return runs


def mean_ci(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return "n/a"
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return f"{m:.3f}"
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78}.get(len(xs) - 1, 2.0)
    return f"{m:.3f} ± {t * sd / math.sqrt(len(xs)):.3f}"


def xk(run, sid):
    for s in run["systems"]:
        if s["sid"] == sid:
            return s["x_crit"]
    return None


def main() -> None:
    dirs = sys.argv[1:] or ["results/main", "results/ablations"]
    runs = load(dirs)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "stream_default.yaml"), encoding="utf-8"))
    copies = {s["id"]: s["copy_of"] for s in cfg["systems"] if "copy_of" in s}
    print("## Continual metrics (noise 0, fresh seeds)\n")
    print("| learner | ACC | BWT | FWT (see note) | mean X_k | systems reaching criterion |")
    print("|---|---|---|---|---|---|")
    for (name, noise), by_seed in sorted(runs.items()):
        if noise != 0.0:
            continue
        acc, bwt, xm, reach = [], [], [], []
        for r in by_seed.values():
            sy = r["systems"]
            acc.append(sum(s["retest_acc"] for s in sy) / len(sy))
            bwt.append(sum(s["retest_acc"] - s["test_acc"] for s in sy[:-1]) / (len(sy) - 1))
            xs = [s["x_crit"] for s in sy if s["x_crit"] is not None]
            xm.append(sum(xs) / len(xs) if xs else None)
            reach.append(len(xs))
        print(f"| {name} | {mean_ci(acc)} | {mean_ci(bwt)} | 0 | {mean_ci(xm)} | {mean_ci(reach)} |")
    full, nolib = runs.get(("gca", 0.0), {}), runs.get(("no_library", 0.0), {})
    print("\n## Learning-speed transfer: X_k(full) / X_k(no_library), noise 0\n")
    print("| system | X_k full | X_k no_library | ratio |")
    print("|---|---|---|---|")
    ratios_by_block = defaultdict(list)
    sids = [s["sid"] for s in next(iter(full.values()))["systems"]] if full else []
    blocks = {s["id"]: s.get("block", "") for s in cfg["systems"]}
    for sid in sids:
        a = [xk(full[s], sid) for s in full if s in nolib]
        b = [xk(nolib[s], sid) for s in full if s in nolib]
        rs = [x / y for x, y in zip(a, b) if x is not None and y]
        ratios_by_block[blocks.get(sid, "")] += rs
        print(f"| {sid} | {mean_ci([x for x in a if x is not None])} | {mean_ci([x for x in b if x is not None])} "
              f"| {mean_ci(rs)} |")
    print("\nMean ratio by block: " + ", ".join(f"{k}: {mean_ci(v)}" for k, v in sorted(ratios_by_block.items())))
    print("\n## Transfer copies: X_k(copy) / X_k(original), full learner, noise 0 (M7 target <= 0.30)\n")
    print("| copy | original | X_k copy | X_k original | ratio |")
    print("|---|---|---|---|---|")
    for c, o in copies.items():
        xc = [xk(full[s], c) for s in full]
        xo = [xk(full[s], o) for s in full]
        rs = [x / y for x, y in zip(xc, xo) if x is not None and y]
        print(f"| {c} | {o} | {mean_ci([x for x in xc if x is not None])} | "
              f"{mean_ci([x for x in xo if x is not None])} | {mean_ci(rs)} |")


if __name__ == "__main__":
    main()
