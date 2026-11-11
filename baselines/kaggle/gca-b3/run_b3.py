"""B3 on Kaggle: depth sweep on dev seeds, then the test runs with the chosen depth.

Rule fixed before any result: depth in {2, 3, 4}; per_system models on dev seeds 0-2 (noise 0);
pick the highest mean accuracy (per system, then over seeds, same rule as experiments/report.py);
a depth within 0.005 of the best that has fewer layers wins (simpler model). Test seeds 50-52 are
never scored here: their keys are not in this dataset.

    python run_b3.py --src <dataset dir> --work /kaggle/working [--steps 3000] [--jobs 4]
Output: <work>/b3_results.zip with every prediction file, the sweep table and the log.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DEPTHS = (2, 3, 4)
DEV = (0, 1, 2)
TEST = (50, 51, 52)
NOISE = ("0.0", "0.1")


def gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def run_all(jobs: list[tuple[list[str], str]], n_par: int, log) -> None:
    """jobs: (command, name). Runs n_par at a time, spread over the GPUs."""
    ng = gpus()

    def one(i_job):
        i, (cmd, name) = i_job
        env = dict(os.environ)
        if ng:
            env["CUDA_VISIBLE_DEVICES"] = str(i % ng)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        msg = f"{name}: exit {r.returncode}, {time.time() - t0:.0f}s"
        if r.returncode != 0:
            msg += "\n" + r.stderr[-2000:]
        return msg

    with ThreadPoolExecutor(max_workers=n_par) as ex:
        for msg in ex.map(one, list(enumerate(jobs))):
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    src, work = a.src, a.work
    out = os.path.join(work, "results", "b3")
    os.makedirs(os.path.join(out, "sweep"), exist_ok=True)
    os.makedirs(os.path.join(out, "test"), exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")
    script = os.path.join(src, "transformer_scratch.py")
    sys.path.insert(0, src)
    from score_baselines import per_seed     # the same scoring rule as every other learner

    print("GPUs:", gpus(), flush=True)
    # 1. sweep: per_system, dev seeds, three depths
    jobs = []
    for L in DEPTHS:
        for s in DEV:
            o = os.path.join(out, "sweep", f"b3_L{L}_per_system_stream_s{s}_n0.0.json")
            jobs.append(([sys.executable, script, os.path.join(src, "data", f"stream_s{s}_n0.0.jsonl"), "--out", o,
                          "--layers", str(L), "--steps", str(a.steps), "--threads", "1"], f"sweep L{L} s{s}"))
    run_all(jobs, a.jobs, log)
    table, best = {}, None
    for L in DEPTHS:
        accs = [per_seed(os.path.join(out, "sweep", f"b3_L{L}_per_system_stream_s{s}_n0.0.json"),
                         os.path.join(src, "data", f"keys_s{s}_n0.0.json")) for s in DEV]
        if any(x is None for x in accs):
            raise SystemExit(f"sweep for {L} layers is incomplete; see log.txt")
        table[L] = sum(x["acc"] for x in accs) / len(accs)
    top = max(table.values())
    best = min(L for L in DEPTHS if table[L] >= top - 0.005)
    msg = f"dev accuracy by depth: {table}; chosen depth: {best}"
    print(msg, flush=True)
    log.write(msg + "\n")
    json.dump({"dev_accuracy": table, "chosen_layers": best, "steps": a.steps},
              open(os.path.join(out, "sweep", "choice.json"), "w"))
    # 2. test: chosen depth, seeds 50-52, noise 0 and 0.1, both modes (sequential runs first: longest)
    jobs = []
    for mode in ("sequential", "per_system"):
        for s in TEST:
            for n in NOISE:
                o = os.path.join(out, "test", f"b3_{mode}_stream_s{s}_n{n}.json")
                jobs.append(([sys.executable, script, os.path.join(src, "data", f"stream_s{s}_n{n}.jsonl"), "--out", o,
                              "--layers", str(best), "--steps", str(a.steps), "--mode", mode, "--threads", "1"],
                             f"test {mode} s{s} n{n}"))
    run_all(jobs, a.jobs, log)
    log.close()
    for d, _, files in os.walk(out):                  # resume files are not results
        for f in files:
            if f.endswith(".ckpt.pt"):
                os.remove(os.path.join(d, f))
    zp = shutil.make_archive(os.path.join(work, "b3_results"), "zip", os.path.join(work, "results"))
    print("done ->", zp, flush=True)


if __name__ == "__main__":
    main()
