"""Aggregate run JSONs into tables (plan §9.2). Usage:
    python -m experiments.report results/main > results/main_table.md
Every number is mean +- 95% CI over seeds (t-distribution)."""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

SPLITS = ("S1", "S2", "S3", "S4", "S4b")
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}


def ci(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "n/a"
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return f"{m:.3f}"
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = T95.get(len(xs) - 1, 1.96) * sd / math.sqrt(len(xs))
    return f"{m:.3f} ± {h:.3f}"


def per_run(d: dict) -> dict:
    sy = d["systems"]
    tot = {k: [0, 0, 0, 0] for k in SPLITS}           # n, correct, abstain, wrong
    for s in sy:
        for k, v in s["splits"].items():
            t = tot[k]
            t[0] += v["n"]
            t[1] += v["correct"]
            t[2] += v["abstain"]
            t[3] += v["wrong"]
    n = sum(t[0] for t in tot.values())
    out = {"acc": sum(s["test_acc"] for s in sy) / len(sy),
           "abstain": sum(t[2] for t in tot.values()) / n,
           "halluc": sum(t[3] for t in tot.values()) / n,
           "corrupt": d["corruption"]["locked_wrong"],
           "locked": d["corruption"]["locked"],
           "calls": sum(s["calls"] for s in sy) / len(sy),
           "solved": sum(1 for s in sy if s["test_acc"] >= 0.95),
           "retest": sum(s["retest_acc"] for s in sy) / len(sy)}
    for k, t in tot.items():
        out[k] = t[1] / t[0] if t[0] else None
    xs = [s["x_crit"] for s in sy if s["x_crit"] is not None]
    out["x_crit"] = sum(xs) / len(xs) if xs else None
    out["bwt"] = sum(s["retest_acc"] - s["test_acc"] for s in sy[:-1]) / max(len(sy) - 1, 1)
    return out


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "results/main"
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        name = d["learner"] if d.get("ablation", "full") == "full" else d["ablation"]
        runs[(name, d["noise"])].append((d["seed"], per_run(d)))
    seeds = sorted({s for v in runs.values() for s, _ in v})
    print(f"Seeds: {seeds}. Mean over the systems in the stream, then mean ± 95% CI over seeds.\n")
    print("| learner | noise | accuracy | systems ≥95% | S1 | S2 | S3 | S4 | S4b | abstain | hallucination "
          "| locked rules wrong | checker calls/system | BWT |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (name, noise) in sorted(runs, key=lambda k: (k[1], k[0])):
        rs = [r for _, r in runs[(name, noise)]]
        g = lambda k: ci([r[k] for r in rs])
        corrupt = f"{sum(r['corrupt'] for r in rs)}/{sum(r['locked'] for r in rs)}"
        print(f"| {name} | {noise} | {g('acc')} | {g('solved')} | {g('S1')} | {g('S2')} | {g('S3')} | {g('S4')} "
              f"| {g('S4b')} | {g('abstain')} | {g('halluc')} | {corrupt} | {g('calls')} | {g('bwt')} |")


if __name__ == "__main__":
    main()
