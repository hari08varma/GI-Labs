"""Integrity (plan §11): agent code never imports world internals; budgets hold;
same seed gives byte-identical logs; a 3-system mini stream learns end to end in < 60 s."""
import ast
import json
import os
import time

from gca.agent import GCAAgent
from gca.config import Hparams
from gca.eventlog import EventLog
from experiments.harness import run_stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIRS = ["gca/learn", "gca/memory", "gca/think", "gca/improve"]
AGENT_FILES = ["gca/agent.py", "gca/config.py", "gca/eventlog.py"]


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            yield ("." * node.level) + mod, node
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node


def test_agent_never_imports_world_internals():
    files = [os.path.join(ROOT, f) for f in AGENT_FILES]
    for d in AGENT_DIRS:
        files += [os.path.join(ROOT, d, f) for f in os.listdir(os.path.join(ROOT, d)) if f.endswith(".py")]
    bad = []
    for f in files:
        for mod, _ in _imports(f):
            m = mod.lstrip(".")
            if "world" in m.split(".") and not m.endswith("world.api"):
                bad.append((os.path.relpath(f, ROOT), mod))
    assert not bad, bad


def _run(mini, seed=0):
    sb, systems = mini
    log = EventLog()
    agent = GCAAgent(Hparams(seed=seed), log=log)
    res = run_stream(agent, systems, sb.budget, curve=False)
    return agent, log, res


def test_end_to_end_and_budgets(mini):
    t = time.perf_counter()
    agent, log, res = _run(mini)
    assert time.perf_counter() - t < 60
    for r in res:
        assert r.calls <= r.budget
        assert r.test_acc == 1.0, (r.sid, {k: v.acc for k, v in r.splits.items()})
    assert any(r["ev"] == "lock" for r in log.records)
    rules = agent.rules("S02")
    assert any("repeat" in x for x in rules)


def test_determinism(mini):
    _, a, _ = _run(mini)
    _, b, _ = _run(mini)
    ja = "\n".join(json.dumps(r, sort_keys=True) for r in a.records)
    jb = "\n".join(json.dumps(r, sort_keys=True) for r in b.records)
    assert ja == jb


def test_reversal_and_unification_M3(mini):
    """M3: next/previous unified within <= 6 shared events at eta = 0; reversed phrasing >= 95%."""
    agent, log, res = _run(mini)
    for u in log.of("unify"):
        if u["ctx"] in ("S01", "S02", "S03") and len(u["sigma"]) == 2:
            assert u["support"] <= 6
    assert all(r.splits["S4"].acc >= 0.95 for r in res)


def test_each_rule_locks_once(mini):
    agent, log, _ = _run(mini)
    locks = [r["col"] for r in log.of("lock")]
    assert len(locks) == len(set(locks))
