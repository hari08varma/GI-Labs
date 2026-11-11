"""Fine learning-speed measurement: examples + questions needed to lock each rule, one rule at a
time, over the stream (operator rules only; successor tables are excluded because they are
fully observed by design). Usage:
    python -m experiments.curves --runs "results/dev/base_gca_s*.json" --vs "results/dev/base_nolib_s*.json" --png results/dev/curve.png
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict


def rows(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f, encoding="utf-8"))
        for r in d.get("lock_costs", []):
            if not r["table"]:
                out.append({**r, "seed": d["seed"], "cost": r["obs"] + r["queries"]})
    return out


def by_index(rs: list[dict]) -> dict[int, list[float]]:
    g = defaultdict(list)
    for r in rs:
        g[r["index"]].append(r["cost"])
    return g


def slope(rs: list[dict]) -> float:
    xs = [r["index"] for r in rs]
    ys = [r["cost"] for r in rs]
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def summary(name: str, rs: list[dict]) -> str:
    early = [r["cost"] for r in rs if r["index"] < 7]
    late = [r["cost"] for r in rs if r["index"] >= 13]
    m = lambda v: sum(v) / len(v) if v else math.nan
    return (f"| {name} | {len(rs)} | {m([r['cost'] for r in rs]):.1f} | {m(early):.1f} | {m(late):.1f} | "
            f"{slope(rs):+.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--vs", default=None)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    sets = {"full": rows(a.runs)}
    if a.vs:
        sets["no_library"] = rows(a.vs)
    print("| learner | rules locked | mean cost to lock | systems 1-7 | systems 14-20 | slope per system |")
    print("|---|---|---|---|---|---|")
    for k, rs in sets.items():
        print(summary(k, rs))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for k, rs in sets.items():
            g = by_index(rs)
            xs = sorted(g)
            ax.plot([x + 1 for x in xs], [sum(g[x]) / len(g[x]) for x in xs], marker="o", label=k)
        ax.set_xlabel("system in the stream")
        ax.set_ylabel("examples + questions to lock a rule")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.png, dpi=130)


if __name__ == "__main__":
    main()
