"""Run one learner over one stream and write a JSON summary. Usage:
    python -m experiments.run_stream --seed 10 --noise 0.1 --learner gca --out results/x.json
learner: gca | b1. --ablation: a name from gca.config.ABLATIONS (gca only)."""
from __future__ import annotations

import argparse
import json
import os
import time

from experiments.harness import run_stream
from experiments.metrics import lock_costs, locked_corruption
from gca.agent import GCAAgent
from gca.config import ABLATIONS, B1_FLAGS, Hparams
from gca.world.generator import StreamBuilder, load_config


def make(learner: str, seed: int, ablation: str = "full") -> GCAAgent:
    hp = Hparams(seed=seed)
    if learner == "b1":
        hp = Hparams(seed=seed, flags=B1_FLAGS)
    elif ablation != "full":
        hp = hp.with_flags(**ABLATIONS[ablation])
    a = GCAAgent(hp)
    a.name = learner if ablation == "full" else ablation
    return a


def run(cfg: dict, seed: int, noise: float, learner: str = "gca", ablation: str = "full",
        curve: bool = True) -> dict:
    sb = StreamBuilder(cfg, seed, noise)
    systems = sb.build()
    agent = make(learner, seed, ablation)
    t0 = time.perf_counter()
    res = run_stream(agent, systems, sb.budget, extra_labels=agent.hp.flags.extra_labels, curve=curve)
    out = {"learner": learner, "ablation": ablation, "seed": seed, "noise": noise,
           "wall": time.perf_counter() - t0, "systems": [r.as_dict() for r in res],
           "corruption": locked_corruption(agent, systems), "eta_hat": agent.eta_hat(),
           "stats": dict(agent.stats), "lock_costs": lock_costs(agent),
           "diagnoses": len(agent.diagnoser.records), "fixes": [f.__dict__ for f in agent.diagnoser.fixes]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--learner", default="gca", choices=["gca", "b1"])
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    ap.add_argument("--config", default="configs/stream_default.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = run(load_config(a.config), a.seed, a.noise, a.learner, a.ablation)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, default=str)
    accs = [s["test_acc"] for s in out["systems"]]
    print(f"{a.learner}/{a.ablation} seed={a.seed} noise={a.noise} mean_acc={sum(accs)/len(accs):.3f} "
          f"corruption={out['corruption']} wall={out['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
