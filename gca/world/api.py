"""Public world API. The ONLY world module a learner may import.

Ground truth (hidden integer values, operator definitions, answer keys) never
crosses this boundary. Learners see tokens, observations, questions, and a
budgeted checker that answers yes/no about claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# Innate truth tokens (truth itself is core knowledge, see plan P2).
TRUE = "#T"
FALSE = "#F"
# Special answers.
UNKNOWN = "unknown"   # abstain: knowledge missing
NONE = "none"         # equation has no solution
MANY = "many"         # equation has more than one solution
VAR = "?"             # the unknown in a solve question


class BudgetExceeded(Exception):
    """Raised when a learner asks the checker more than its budget allows."""


@dataclass(frozen=True)
class Observation:
    system: str               # context token (learner sees the context id, not its content)
    word: str                 # operator / relation word token
    args: tuple[str, ...]     # argument tokens in surface order
    out: str                  # result token (object token, or TRUE/FALSE for relations)


# Expressions in questions: ("obj", token) | ("var",) | ("app", word, expr, ...)
Expr = tuple


@dataclass(frozen=True)
class Question:
    system: str
    qid: str
    split: str                # S1 | S2 | S3 | S4 | S4b
    kind: str                 # "eval": value of expr ; "solve": value of ? such that expr == target
    expr: Expr
    target: str | None = None


@dataclass
class Answer:
    value: str                # token | UNKNOWN | NONE | MANY
    trace: list = field(default_factory=list)   # steps: (word, args, out, source)
    confidence: str = "low"   # state of weakest column used


class CheckerAPI(Protocol):
    def check_fact(self, word: str, args: tuple[str, ...], out: str) -> bool: ...
    def check_answer(self, q: Question, value: str) -> bool: ...
    def calls_used(self) -> int: ...
    def remaining(self) -> int: ...


class Learner(Protocol):
    """Every learner (ours and every baseline) implements this."""

    def start_system(self, system_id: str) -> None: ...
    def observe(self, batch: list[Observation]) -> None: ...
    def learn(self, checker: CheckerAPI) -> None: ...
    def answer(self, q: Question) -> Answer: ...
    def feedback(self, q: Question, correct: bool, checker: CheckerAPI) -> None: ...
    def rest(self) -> None: ...


def leaves(expr: Expr) -> list[str]:
    if expr[0] == "obj":
        return [expr[1]]
    if expr[0] == "var":
        return []
    out: list[str] = []
    for e in expr[2:]:
        out += leaves(e)
    return out


def depth(expr: Expr) -> int:
    if expr[0] != "app":
        return 0
    return 1 + max(depth(e) for e in expr[2:])


def show(expr: Expr) -> str:
    if expr[0] == "obj":
        return expr[1]
    if expr[0] == "var":
        return VAR
    return f"{expr[1]}(" + ", ".join(show(e) for e in expr[2:]) + ")"
