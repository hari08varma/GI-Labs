"""Run B2 / B2-L (Popper) on Modal, since SWI-Prolog is not installed locally.

    modal run baselines/popper_runner/modal_app.py::main --streams stream_s50_n0.0,stream_s50_n0.1 --mode b2
    modal run baselines/popper_runner/modal_app.py::main --streams stream_s50_n0.0 --mode b2l
    modal run baselines/popper_runner/modal_app.py::main --streams stream_s50_n0.0 --mode b2 --only 1   (smoke test)
    modal run baselines/popper_runner/modal_app.py::fetch --streams stream_s50_n0.0 --mode b2       (merge later)
Use `modal run --detach ...` for long runs: results are stored on a Modal Volume as each job finishes.

b2  : one container per (stream, world): worlds are independent, so they run in parallel.
b2l : one container per stream: worlds run in order, accumulating earlier programs (Forgetgol setting).
Predictions go to results/baselines/popper/<mode>_<stream>.json and are scored locally with
experiments.export_dataset.score, the same rule as every other learner.
"""
from __future__ import annotations

import json
import os
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

image = (
    # official SWI-Prolog image (Popper needs SWI-Prolog >= 9.1.12; Debian's package is older)
    modal.Image.from_registry("swipl:latest", add_python="3.11")
    .apt_install("git", "coreutils", "build-essential", "pkg-config")
    .pip_install("uv")
    .run_commands("git clone --depth 1 https://github.com/logic-and-learning-lab/Popper.git /popper",
                  "cd /popper && uv python install 3.14 && uv sync --python 3.14")
    .add_local_file(os.path.join(HERE, "run_popper.py"), "/app/run_popper.py")
    .add_local_dir(os.path.join(ROOT, "baselines", "data"), "/app/data")
)
# MagicPopper (Hocquette & Cropper 2023, github.com/celinehocquette/magicpopper): an older Popper
# fork that searches for constants ("magic values") itself; README: pyswip master, clingo >= 5.5
magic_image = (
    modal.Image.from_registry("swipl:latest", add_python="3.11")
    .apt_install("git", "build-essential", "pkg-config")
    .run_commands(
        "git clone --depth 1 https://github.com/celinehocquette/magicpopper.git /mp",
        "pip install numpy clingo python-sat git+https://github.com/yuce/pyswip@master#egg=pyswip",
        "cd /mp && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
        "echo '#!/bin/sh' > /usr/local/bin/magicpopper",
        "echo 'exec python \"$(find /mp -maxdepth 2 -name popper.py | head -1)\" \"$@\"' >> /usr/local/bin/magicpopper",
        "chmod +x /usr/local/bin/magicpopper && cat /usr/local/bin/magicpopper && ls /mp",
        # pyswip's embedded Prolog has the 1 GB default stack, which the constant search overflows
        "mkdir -p /root/.config/swi-prolog && echo ':- set_prolog_flag(stack_limit, 8_000_000_000).' > /root/.config/swi-prolog/init.pl")
    .env({"POPPER_MAGIC": "1", "POPPER_CMD": "/usr/local/bin/magicpopper"})
    .add_local_file(os.path.join(HERE, "run_popper.py"), "/app/run_popper.py")
    .add_local_dir(os.path.join(ROOT, "baselines", "data"), "/app/data")
)
app = modal.App("gca-popper", image=image)
# results live on Modal, so a dropped laptop connection loses nothing (run with --detach)
vol = modal.Volume.from_name("gca-popper-results", create_if_missing=True)


def key(mode: str, stream: str, only, learn: bool) -> str:
    return f"/results/{mode}{'' if learn else '_nolearn'}/{stream}__{'all' if only is None else only}.json"


@app.function(cpu=6, memory=16384, timeout=8 * 3600, retries=3, volumes={"/results": vol})
def run(mode: str, stream: str, only, timeout: int = 120, learn: bool = True, reanswer: bool = False) -> str:
    """reanswer: keep the stored programs and only redo the answering (after an answering fix)."""
    return _job(mode, stream, only, timeout, learn, reanswer)


@app.function(cpu=6, memory=16384, timeout=8 * 3600, retries=3, volumes={"/results": vol}, image=magic_image)
def run_magic(mode: str, stream: str, only, timeout: int = 120, learn: bool = True, reanswer: bool = False) -> str:
    return _job(mode, stream, only, timeout, learn, reanswer)


def _job(mode: str, stream: str, only, timeout: int, learn: bool, reanswer: bool) -> str:
    path = key(mode, stream, only, learn)
    vol.reload()
    if os.path.exists(path) and not reanswer:      # resume: never redo a finished job
        return "exists " + path
    out = f"/tmp/{stream}_{only}.json"
    cmd = ["python", "/app/run_popper.py", f"/app/data/{stream}.jsonl", "--out", out, "--timeout", str(timeout)]
    if mode == "b2l":
        cmd.append("--accumulate")
    if mode.startswith("core") or mode.startswith("check"):
        cmd.append("--core")
    if "full" in mode:                             # diagnostic: complete one-argument tables
        cmd += ["--full-unary", f"/app/data/fullunary_{stream}.json"]
    if mode.startswith("check"):
        cmd.append("--check-core")
    if mode.startswith("b2c"):                     # B2+core, two-stage
        cmd.append("--two-stage")
    if only is not None:
        cmd += ["--only", str(only)]
    if not learn:
        cmd.append("--no-learn")
    if reanswer and os.path.exists(path):
        cmd += ["--programs-from", path]
    if only is None:                               # long sequential job: checkpoint every world
        cmd += ["--checkpoint", f"/results/ckpt/{mode}_{stream}.json"]
    with open("/tmp/job.out", "w") as fo, open("/tmp/job.err", "w") as fe:
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, text=True)
        while p.poll() is None:                    # commit the checkpoint while it runs, so a
            try:                                   # preempted container resumes, not restarts
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                vol.commit()
    r = subprocess.CompletedProcess(cmd, p.returncode, open("/tmp/job.out").read(), open("/tmp/job.err").read())
    if r.returncode != 0:
        d = {"error": r.stderr[-4000:], "stdout": r.stdout[-2000:]}
    else:
        d = json.load(open(out))
        d["log"] = r.stdout[-4000:]
    if "error" not in d:                           # failed jobs are not stored, so a rerun retries them
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(d, open(path, "w"))
        vol.commit()
    return ("error " if "error" in d else "done ") + path + " " + d.get("log", d.get("error", ""))[-300:]


@app.function(volumes={"/results": vol}, timeout=600)
def collect(mode: str, stream: str, learn: bool = True) -> list:
    vol.reload()
    d = os.path.dirname(key(mode, stream, None, learn))
    if not os.path.isdir(d):
        return []
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.startswith(stream + "__")]


def jobs_for(names, mode, only, n_systems, timeout, learn, reanswer=False, worlds=""):
    if mode != "b2l":                              # b2 and the time tests: one job per world
        idx = [int(w) for w in worlds.split(",")] if worlds else ([only] if only >= 0 else list(range(n_systems)))
        return [(mode, s, i, timeout, learn, reanswer) for s in names for i in idx]
    return [(mode, s, None, timeout, learn, reanswer) for s in names]


@app.local_entrypoint()
def main(streams: str, mode: str = "b2", only: int = -1, n_systems: int = 24, timeout: int = 120,
         learn: bool = True, reanswer: bool = False, worlds: str = ""):
    """Launch (or resume) the jobs, then merge what is finished."""
    names = [s for s in streams.split(",") if s]
    # spawn, not starmap: each job is submitted on its own and keeps running (and saving to the
    # Volume) even if this local client fails while waiting
    fn = run_magic if mode.startswith("magic") else run
    calls = [fn.spawn(*j) for j in jobs_for(names, mode, only, n_systems, timeout, learn, reanswer, worlds)]
    print("spawned", len(calls), "jobs", flush=True)
    for c in calls:
        try:
            print(str(c.get())[:400], flush=True)
        except Exception as e:                     # noqa: BLE001 - report and keep collecting
            print("job failed:", repr(e)[:300], flush=True)
    _fetch(streams, mode, learn)


@app.local_entrypoint()
def fetch(streams: str, mode: str = "b2", learn: bool = True):
    """Merge the finished results stored on Modal into results/baselines/popper/<mode>_<stream>.json."""
    _fetch(streams, mode, learn)


def _fetch(streams: str, mode: str, learn: bool) -> None:
    os.makedirs(os.path.join(ROOT, "results", "baselines", "popper"), exist_ok=True)
    for s in [s for s in streams.split(",") if s]:
        parts = collect.remote(mode, s, learn)
        merged = {"predictions": {}, "programs": [], "logs": [], "n_jobs": len(parts)}
        for d in parts:
            merged["predictions"].update(d["predictions"])
            merged["programs"] += d["programs"]
            merged["logs"].append(d.get("log", ""))
        path = os.path.join(ROOT, "results", "baselines", "popper", f"{mode}{'' if learn else '_nolearn'}_{s}.json")
        json.dump(merged, open(path, "w"), indent=1)
        print(s, "jobs", len(parts), "predictions", len(merged["predictions"]), "->", path)
