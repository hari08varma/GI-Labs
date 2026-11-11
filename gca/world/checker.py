"""The checker stands in for reality: noiseless, budgeted (plan §2.6, P1).
Also the world-side Episode object used by experiment runners for scoring."""
from __future__ import annotations

from .api import BudgetExceeded, Question
from .generator import SystemWorld


class Checker:
    def __init__(self, sw: SystemWorld, budget: int):
        self._sw = sw
        self._budget = budget
        self._used = 0
        self.log: list[tuple] = []

    def _spend(self) -> None:
        if self._used >= self._budget:
            raise BudgetExceeded(f"{self._sw.sid}: budget {self._budget} exhausted")
        self._used += 1

    def check_fact(self, word: str, args: tuple[str, ...], out: str) -> bool:
        self._spend()
        r = self._sw.true_fact(word, tuple(args), out)
        self.log.append(("fact", word, tuple(args), out, r))
        return r

    def check_answer(self, q: Question, value: str) -> bool:
        self._spend()
        r = self._sw.keys[q.qid] == value
        self.log.append(("answer", q.qid, value, r))
        return r

    def calls_used(self) -> int:
        return self._used

    def remaining(self) -> int:
        return self._budget - self._used


def score(sw: SystemWorld, q: Question, value: str) -> bool:
    """World-side scoring of test answers. Never exposed to learners."""
    return sw.keys[q.qid] == value
