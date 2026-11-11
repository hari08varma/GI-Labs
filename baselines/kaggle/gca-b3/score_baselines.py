"""Score external baselines (Popper B2 / B2+core / B2-L, facts-only floor) with the same rule as
experiments/report.py: accuracy per system over all its questions, then the mean over the
systems of the stream, then mean +- 95% CI over seeds. Usage:
    python -m experiments.score_baselines --modes b2 b2c b2l b2_nolearn --seeds 50 51 52 --noise 0.0 0.1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

SPLITS = ("S1", "S2", "S3", "S4", "S4b")


def per_seed(pred_path: str, keys_path: str, part: str = "predictions") -> dict | None:
    """part: "predictions" (right after a system's own training) or "after_stream" (B3/B4
    sequential: every system re-asked at the end, which measures forgetting)."""
    if not os.path.exists(pred_path):
        return None
    keys = json.load(open(keys_path, encoding="utf-8"))
    data = json.load(open(pred_path, encoding="utf-8"))
    if part not in data:
        return None
    preds = data[part]
    sys_tot = defaultdict(lambda: [0, 0])
    split = defaultdict(lambda: [0, 0])
    abst = wrong = 0
    for qid, key in keys.items():
        sid, sp = qid.split("-")[0], qid.split("-")[1]
        a = preds.get(qid, "unknown")
        sys_tot[sid][0] += 1
        sys_tot[sid][1] += a == key
        split[sp][0] += 1
        split[sp][1] += a == key
        abst += a == "unknown"
        wrong += a not in ("unknown", key)
    n = len(keys)
    out = {"acc": sum(c / t for t, c in sys_tot.values()) / len(sys_tot),
           "abstain": abst / n, "wrong": wrong / n}
    out.update({s: split[s][1] / split[s][0] for s in SPLITS if split[s][0]})
    return out


def mean_ci(xs: list[float]) -> str:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return f"{m:.3f}"
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(xs), 2.0)
    return f"{m:.3f} ± {t * sd / math.sqrt(len(xs)):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["b2", "b2c", "b2l"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[50, 51, 52])
    ap.add_argument("--noise", nargs="+", default=["0.0", "0.1"])
    ap.add_argument("--preds", default="results/baselines/popper")
    ap.add_argument("--keys", default="baselines/data")
    ap.add_argument("--part", default="predictions", choices=["predictions", "after_stream"])
    a = ap.parse_args()
    cols = ["acc", *SPLITS, "abstain", "wrong"]
    print("| learner | noise | seeds | accuracy | " + " | ".join(cols[1:]) + " |")
    print("|" + "---|" * (len(cols) + 3))
    for mode in a.modes:
        for n in a.noise:
            rows = [per_seed(os.path.join(a.preds, f"{mode}_stream_s{s}_n{n}.json"),
                             os.path.join(a.keys, f"keys_s{s}_n{n}.json"), a.part) for s in a.seeds]
            rows = [r for r in rows if r]
            if not rows:
                continue
            cells = [mean_ci([r[c] for r in rows]) for c in cols]
            print(f"| {mode} | {n} | {len(rows)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
