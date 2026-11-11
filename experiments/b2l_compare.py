"""Does a learner do better when it keeps its earlier knowledge? (the question B2-L answers)

Per stream (seed, noise), per world: accuracy over that world's test questions. Groups:
  cumulative  block F, S21-S24 (rules built from earlier ones)
  copies      block C, S10-S14 (transfer copies of S02, S05, S06, S04, S08; new symbols)
  originals   S02, S05, S06, S04, S08 (the first time those systems were seen)
  all         all 24 worlds
Pairs, with vs without earlier knowledge:
  B2-L (Popper, earlier programs in its background knowledge)  vs  B2 (Popper, fresh each world)
  GCA (full)                                                   vs  GCA no_library (noise 0 only)
Only streams that have a B2-L result are used, for every learner, so all rows see the same worlds.
    python -m experiments.b2l_compare
"""
from __future__ import annotations

import json
import os
import re

GROUPS = {"cumulative (S21-S24)": ["S21", "S22", "S23", "S24"],
          "copies (S10-S14)": ["S10", "S11", "S12", "S13", "S14"],
          "their originals": ["S02", "S05", "S06", "S04", "S08"],
          "all 24 worlds": [f"S{i:02d}" for i in range(1, 25)]}
SEEDS, NOISE = (50, 51, 52), ("0.0", "0.1")
POP = "results/baselines/popper"


def pred_acc(pred_file: str, keys_file: str) -> dict:
    keys = json.load(open(keys_file, encoding="utf-8"))
    preds = json.load(open(pred_file, encoding="utf-8"))["predictions"]
    tot: dict = {}
    for qid, k in keys.items():
        t = tot.setdefault(qid.split("-")[0], [0, 0])
        t[0] += 1
        t[1] += preds.get(qid, "unknown") == k
    return {s: c / n for s, (n, c) in tot.items()}


def run_acc(run_file: str) -> dict | None:
    if not os.path.exists(run_file):
        return None
    return {s["sid"]: s["test_acc"] for s in json.load(open(run_file, encoding="utf-8"))["systems"]}


def lib_use(pred_file: str) -> dict:
    """Share of B2-L's learned clauses per world that call an earlier world's program."""
    out = {}
    for e in json.load(open(pred_file, encoding="utf-8"))["programs"]:
        n, u = out.get(e["system"], (0, 0))
        out[e["system"]] = (n + len(e["program"]), u + sum("lib_" in c for c in e["program"]))
    return out


def main() -> None:
    for n in NOISE:
        seeds = [s for s in SEEDS if os.path.exists(f"{POP}/b2l_stream_s{s}_n{n}.json")]
        if not seeds:
            continue
        rows = {"B2 Popper (fresh each world)": [], "B2-L Popper (keeps earlier programs)": [],
                "GCA no_library": [], "GCA (full)": []}
        use = {}
        for s in seeds:
            keys = f"baselines/data/keys_s{s}_n{n}.json"
            rows["B2 Popper (fresh each world)"].append(pred_acc(f"{POP}/b2_stream_s{s}_n{n}.json", keys))
            rows["B2-L Popper (keeps earlier programs)"].append(pred_acc(f"{POP}/b2l_stream_s{s}_n{n}.json", keys))
            rows["GCA (full)"].append(run_acc(f"results/final3/main/gca_s{s}_n{n}.json"))
            rows["GCA no_library"].append(run_acc(f"results/final3/ablations/no_library_s{s}_n{n}.json"))
            for sid, (tot, u) in lib_use(f"{POP}/b2l_stream_s{s}_n{n}.json").items():
                a, b = use.get(sid, (0, 0))
                use[sid] = (a + tot, b + u)
        print(f"\n### Noise {n}: seeds {seeds} (B2-L available for {len(seeds)} of 3 seeds)\n")
        print("| learner | " + " | ".join(GROUPS) + " |")
        print("|---|" + "---|" * len(GROUPS))
        for name, per_seed in rows.items():
            if any(x is None for x in per_seed):
                continue
            cells = []
            for sids in GROUPS.values():
                vals = [sum(acc[x] for x in sids) / len(sids) for acc in per_seed]
                cells.append(f"{sum(vals) / len(vals):.3f}")
            print(f"| {name} | " + " | ".join(cells) + " |")
        for g, sids in GROUPS.items():
            if g.startswith("all"):
                continue
            tot = sum(use.get(x, (0, 0))[0] for x in sids)
            u = sum(use.get(x, (0, 0))[1] for x in sids)
            print(f"- B2-L clauses that call an earlier world's program, {g}: {u} of {tot}")


if __name__ == "__main__":
    main()
