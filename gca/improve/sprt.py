"""Wald's sequential probability ratio test for unverified contradictions (plan §7.4).

H1: the column is correct and its error rate is the noise rate, p1 = eta_hat (floor 0.01).
H0: the column is wrong, error rate p0 = 0.3.
A confirmation adds ln((1 - p1) / (1 - p0)); a contradiction adds ln(p1 / p0).
Accept H1 at A = ln((1 - beta) / alpha), H0 at B = ln(beta / (1 - alpha)).
"""
from __future__ import annotations

import math


def thresholds(alpha: float = 0.01, beta: float = 0.01) -> tuple[float, float]:
    return math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))


def step(llr: float, confirmed: bool, eta: float, p0: float = 0.3, floor: float = 0.01) -> float:
    p1 = max(eta, floor)
    if confirmed:
        return llr + math.log((1 - p1) / (1 - p0))
    return llr + math.log(p1 / p0)


def n_to_accept(eta: float, p0: float = 0.3, alpha: float = 0.01, beta: float = 0.01,
                floor: float = 0.01) -> tuple[int, int]:
    """(consecutive confirmations to accept H1, consecutive contradictions to accept H0)."""
    A, B = thresholds(alpha, beta)
    p1 = max(eta, floor)
    up = math.log((1 - p1) / (1 - p0))
    down = math.log(p1 / p0)
    return math.ceil(A / up), math.ceil(B / down)
