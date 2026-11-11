"""Global workspace (plan §6.3): a blackboard with k slots.

Each slot holds a binding (variable, value, supporting column, confidence). A binding is
released once its consumer has used it, so the slots needed equal the largest number of
values held at once while a question is evaluated. When a search fails with every slot
full, the solver retries with k + 1, up to k_max (dynamic k).
"""
from __future__ import annotations

from dataclasses import dataclass


class WorkspaceFull(Exception):
    pass


@dataclass
class Slot:
    var: int
    value: int
    source: str
    confidence: str


class Workspace:
    def __init__(self, k: int = 7):
        self.k = k
        self.slots: dict[int, Slot] = {}
        self.high_water = 0
        self.broadcasts = 0

    def bind(self, var: int, value: int, source: str, confidence: str) -> None:
        if var not in self.slots and len(self.slots) >= self.k:
            raise WorkspaceFull(self.k)
        self.slots[var] = Slot(var, value, source, confidence)
        self.broadcasts += 1            # every active subgoal re-checks against the new binding
        self.high_water = max(self.high_water, len(self.slots))

    def release(self, var: int) -> None:
        self.slots.pop(var, None)

    def value(self, var: int) -> int | None:
        s = self.slots.get(var)
        return None if s is None else s.value
