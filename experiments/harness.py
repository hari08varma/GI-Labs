"""Runs any Learner over a stream and scores it with world-side code (plan §9).

Per system: experience increments (learn() after each, one checker shared by all
increments), then practice questions with feedback, then tests without feedback, then
rest(). Examples-to-criterion X_k = observations + checker calls before S1 (or S4 when a
system has no S1) reaches 95%. At the end every earlier system is re-tested, giving
R[T, i] for backward transfer. Scoring uses gca.world (allowed here: this is the
experimenter, not the agent).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from gca.world.api import UNKNOWN
from gca.world.checker import Checker, score
from gca.world.generator import SystemWorld

SPLITS = ("S1", "S2", "S3", "S4", "S4b")


@dataclass
class SplitScore:
    n: int = 0
    correct: int = 0
    abstain: int = 0
    wrong: int = 0                    # confident wrong answers (hallucinations)

    def add(self, ok: bool, value: str) -> None:
        self.n += 1
        if ok:
            self.correct += 1
        elif value == UNKNOWN:
            self.abstain += 1
        else:
            self.wrong += 1

    @property
    def acc(self) -> float:
        return self.correct / self.n if self.n else float("nan")


@dataclass
class SystemResult:
    sid: str
    block: str
    tags: list
    n: int
    budget: int
    splits: dict = field(default_factory=dict)
    curve: list = field(default_factory=list)       # (labels used, criterion accuracy)
    x_crit: int | None = None
    labels_total: int = 0
    calls: int = 0
    calls_learn: int = 0
    practice_correct: int = 0
    practice_n: int = 0
    test_acc: float = 0.0
    retest_acc: float | None = None
    rules: list = field(default_factory=list)
    wall: float = 0.0
    identifiable: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["splits"] = {k: asdict(v) for k, v in self.splits.items()}
        return d


def answer_all(learner, sw: SystemWorld, qs) -> SplitScore:
    s = SplitScore()
    for q in qs:
        a = learner.answer(q)
        s.add(score(sw, q, a.value), a.value)
    return s


def run_system(learner, sw: SystemWorld, budget: int, *, extra_labels: bool = False,
               curve: bool = True, hook=None) -> SystemResult:
    t0 = time.perf_counter()
    res = SystemResult(sw.sid, sw.block, list(sw.tags), sw.n, budget, identifiable=dict(sw.identifiable))
    learner.start_system(sw.sid)
    checker = Checker(sw, budget)
    crit = "S1" if sw.questions["S1"] else "S4"
    extra = list(sw.extra_labels) if extra_labels else []
    n_inc = max(len(sw.increments), 1)
    per = -(-len(extra) // n_inc) if extra else 0
    seen = 0
    for i, inc in enumerate(sw.increments):
        batch = list(inc) + extra[i * per:(i + 1) * per]
        learner.observe(batch)
        seen += len(batch)
        learner.learn(checker)
        if curve and sw.questions[crit]:
            acc = answer_all(learner, sw, sw.questions[crit]).acc
            used = seen + checker.calls_used()
            res.curve.append((used, round(acc, 4)))
            if res.x_crit is None and acc >= 0.95:
                res.x_crit = used
    res.labels_total = seen
    res.calls_learn = checker.calls_used()
    practice = list(sw.practice)
    if hook is not None:                     # fault injection: corrupt knowledge, add practice
        practice += hook(learner, sw, checker) or []
    for q in practice:
        a = learner.answer(q)
        ok = score(sw, q, a.value)
        res.practice_n += 1
        res.practice_correct += int(ok)
        learner.feedback(q, ok, checker)
    tot = SplitScore()
    for sp in SPLITS:
        s = answer_all(learner, sw, sw.questions[sp])
        res.splits[sp] = s
        tot.n += s.n
        tot.correct += s.correct
    res.test_acc = tot.correct / tot.n if tot.n else float("nan")
    res.calls = checker.calls_used()
    learner.rest()
    if hasattr(learner, "rules"):
        res.rules = learner.rules(sw.sid)
    res.wall = time.perf_counter() - t0
    return res


def retest(learner, sw: SystemWorld) -> float:
    c = n = 0
    for sp in SPLITS:
        s = answer_all(learner, sw, sw.questions[sp])
        c += s.correct
        n += s.n
    return c / n if n else float("nan")


def run_stream(learner, systems: list[SystemWorld], budget_of, *, extra_labels: bool = False,
               curve: bool = True, verbose: bool = False, hook=None) -> list[SystemResult]:
    out = []
    for sw in systems:
        r = run_system(learner, sw, budget_of(sw), extra_labels=extra_labels, curve=curve, hook=hook)
        out.append(r)
        if verbose:
            sp = " ".join(f"{k}={v.acc:.2f}" for k, v in r.splits.items() if v.n)
            print(f"{r.sid} n={r.n} X={r.x_crit} calls={r.calls}/{r.budget} {sp} wall={r.wall:.1f}s", flush=True)
            for rule in r.rules:
                print("   ", rule, flush=True)
    for r, sw in zip(out, systems):
        r.retest_acc = retest(learner, sw)
    return out
