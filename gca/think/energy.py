"""Energy (plan §6.4): E(Gamma) = sum_s w_s * violated(s, Gamma) + lambda * L(Gamma).

Weights are fixed hyperparameters, not learned: locked columns and stored events 100,
candidate columns 10, plastic columns 1, lambda = 0.1. "Violated" means the chosen
source's output disagrees with the binding. A configuration with no violation is
"zero-energy" (the L term only orders equally consistent configurations).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    locked: float = 100.0
    candidate: float = 10.0
    event: float = 100.0
    plastic: float = 1.0
    lam: float = 0.1

    def of(self, source_state: str) -> float:
        return {"locked": self.locked, "candidate": self.candidate, "event": self.event,
                "plastic": self.plastic}.get(source_state, self.plastic)


def energy(violations: list[float], bits: float, w: Weights) -> float:
    return sum(violations) + w.lam * bits


CONF_ORDER = ("locked", "candidate", "event", "plastic")


def weakest(states: list[str]) -> str:
    """Confidence of an answer = the lowest state among the sources used."""
    if not states:
        return "low"
    worst = max(CONF_ORDER.index(s) if s in CONF_ORDER else 3 for s in states)
    return {0: "high", 1: "medium", 2: "medium", 3: "low"}[worst]
