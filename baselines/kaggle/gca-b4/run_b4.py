"""B4 on Kaggle (GPU T4 x2): smoke test on one system, then the 6 test streams, one per GPU at a time.

    python run_b4.py --src <dataset dir> --work /kaggle/working
Output: <work>/b4_results.zip with b4_stream_s{50,51,52}_n{0.0,0.1}.json and the log.
Test answer keys are not in the dataset; scoring happens on the laptop.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

STREAMS = [f"stream_s{s}_n{n}" for n in ("0.0", "0.1") for s in (50, 51, 52)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    import torch
    ng = max(1, torch.cuda.device_count())
    print("GPUs:", torch.cuda.device_count(), flush=True)
    out = os.path.join(a.work, "results", "b4")
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")
    script = os.path.join(a.src, "lora_llm.py")

    # 1. smoke test: one system, one epoch, 32 questions; stop here if it fails
    t0 = time.time()
    r = subprocess.run([sys.executable, script, os.path.join(a.src, "data", "stream_s50_n0.0.jsonl"),
                        "--out", os.path.join(a.work, "smoke.json"), "--only", "1", "--epochs", "1",
                        "--max-q", "32"], capture_output=True, text=True)
    msg = f"smoke test: exit {r.returncode}, {time.time() - t0:.0f}s\n{r.stdout[-1500:]}"
    if r.returncode != 0:
        msg += "\n" + r.stderr[-3000:]
    print(msg, flush=True)
    log.write(msg + "\n")
    if r.returncode != 0:
        raise SystemExit("smoke test failed; nothing else was run")

    # 2. the 6 streams, one per GPU at a time
    def one(i_st):
        i, st = i_st
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ng))
        t = time.time()
        p = subprocess.run([sys.executable, script, os.path.join(a.src, "data", f"{st}.jsonl"),
                            "--out", os.path.join(out, f"b4_{st}.json")], capture_output=True, text=True, env=env)
        m = f"{st}: exit {p.returncode}, {time.time() - t:.0f}s\n{p.stdout[-600:]}"
        if p.returncode != 0:
            m += "\n" + p.stderr[-2000:]
        return m

    with ThreadPoolExecutor(max_workers=ng) as ex:
        for m in ex.map(one, list(enumerate(STREAMS))):
            print(m, flush=True)
            log.write(m + "\n")
            log.flush()
    log.close()
    for d, _, files in os.walk(out):                  # resume files are not results
        for f in files:
            if f.endswith(".ckpt.pt"):
                os.remove(os.path.join(d, f))
    print("done ->", shutil.make_archive(os.path.join(a.work, "b4_results"), "zip", os.path.join(a.work, "results")),
          flush=True)


if __name__ == "__main__":
    main()
