"""Export the exact streams every learner sees, for the external baselines (Kaggle notebooks,
Popper). Same experience, same budget (as extra noise-free labels, plan §2.6), same questions.
Answer keys go to a separate file that only the scorer reads. Usage:
    python -m experiments.export_dataset --seeds 10 11 12 --noise 0.0 0.1 --out baselines/data
"""
from __future__ import annotations

import argparse
import json
import os

from gca.world.api import show
from gca.world.generator import StreamBuilder, load_config


def ob(o) -> dict:
    return {"word": o.word, "args": list(o.args), "out": o.out}


def export(cfg: dict, seed: int, noise: float, out_dir: str) -> str:
    sb = StreamBuilder(cfg, seed, noise)
    path = os.path.join(out_dir, f"stream_s{seed}_n{noise}.jsonl")
    keys = os.path.join(out_dir, f"keys_s{seed}_n{noise}.json")
    os.makedirs(out_dir, exist_ok=True)
    allkeys = {}
    with open(path, "w", encoding="utf-8") as f:
        for sw in sb.build():
            qs = []
            for split, items in sw.questions.items():
                for q in items:
                    qs.append({"qid": q.qid, "split": split, "kind": q.kind, "expr": q.expr,
                               "target": q.target, "text": show(q.expr) + (f" = {q.target}" if q.target else " = ?")})
                    allkeys[q.qid] = sw.keys[q.qid]
            rec = {"system": sw.sid, "budget": sb.budget(sw),
                   "experience": [ob(o) for o in sw.experience],
                   "extra_labels": [ob(o) for o in sw.extra_labels],
                   "practice": [{"qid": q.qid, "split": q.split, "kind": q.kind, "expr": q.expr, "target": q.target,
                                 "key": sw.keys[q.qid]} for q in sw.practice],
                   "questions": qs}
            f.write(json.dumps(rec) + "\n")
    with open(keys, "w", encoding="utf-8") as f:
        json.dump(allkeys, f)
    return path


def score(pred_path: str, keys_path: str, part: str | None = None) -> dict:
    """Score predictions {qid: answer} with the same rule as experiments/harness.py.
    `part` picks a sub-dict (e.g. "after_stream", "after_own_system", "predictions")."""
    keys = json.load(open(keys_path, encoding="utf-8"))
    preds = json.load(open(pred_path, encoding="utf-8"))
    if part is None:
        for k in ("predictions", "after_stream"):
            if k in preds and isinstance(preds[k], dict):
                part = k
                break
    if part is not None:
        preds = preds[part]
    by = {}
    for qid, key in keys.items():
        split = qid.split("-")[1]
        s = by.setdefault(split, [0, 0, 0])
        s[0] += 1
        a = preds.get(qid, "unknown")
        s[1] += a == key
        s[2] += a == "unknown"
    return {k: {"n": v[0], "acc": v[1] / v[0], "abstain": v[2] / v[0]} for k, v in sorted(by.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0])
    ap.add_argument("--config", default="configs/stream_default.yaml")
    ap.add_argument("--out", default="baselines/data")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for s in a.seeds:
        for n in a.noise:
            print(export(cfg, s, n, a.out))


if __name__ == "__main__":
    main()
