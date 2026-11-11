"""Structural repair of a unary frame (Harinath's "days of the week" method).

Overhypothesis, counted from experience: a unary frame is a bijection that forms ONE loop
(every item has exactly one before and one after). When the known facts break that, find
every MINIMAL repair: the fewest unsettled facts to change (plus empty or conflicting cells
to fill) that make the table one loop, respecting every checker verdict. The fact that the
minimal repairs agree on is the suspect, and the learner asks the checker about the
CORRECTED fact once ("does Friday come after Thursday?"): a yes fixes the cell and proves
the old fact was a slip in a single question; a no removes that repair and the search reruns.
"""
from __future__ import annotations

import itertools
from collections import Counter


def is_loop(t: list[int]) -> bool:
    m = len(t)
    if sorted(t) != list(range(m)):
        return False
    c, n = 0, 0
    while True:
        c = t[c]
        n += 1
        if c == 0:
            return n == m


def is_perm(t: list[int]) -> bool:
    return sorted(t) == list(range(len(t)))


def loop_repairs(m: int, known: dict[int, int], weight: dict[int, float], settled: set[int],
                 neg: set[tuple[int, int]], must_fill: list[int], kmax: int = 3, cycle: bool = True,
                 max_fill: int = 6, cap: int = 400) -> tuple[int, list[tuple[float, dict]]]:
    """Minimal repairs. Returns (k, [(cost, {cell: new value})]) for the smallest k with any
    solution, or (-1, []) if none up to kmax. `known` holds single-valued facts; `must_fill`
    are cells with no value or conflicting values. Cost = observations overruled."""
    ok_shape = is_loop if cycle else is_perm
    changeable = sorted(c for c in known if c not in settled)
    for k in range(0, kmax + 1):
        sols = []
        for S in itertools.combinations(changeable, k):
            fixed = {c: v for c, v in known.items() if c not in S}
            if len(set(fixed.values())) != len(fixed):
                continue
            open_cells = list(S) + [c for c in must_fill if c not in fixed]
            free_vals = [v for v in range(m) if v not in set(fixed.values())]
            if len(free_vals) != len(open_cells) or len(open_cells) > max_fill:
                continue
            for vals in itertools.permutations(free_vals):
                asg = dict(zip(open_cells, vals))
                if any(asg[c] == known[c] for c in S):
                    continue                  # not a change: a smaller repair covers it
                if any((c, v) in neg for c, v in asg.items()):
                    continue
                t = [fixed.get(c, asg.get(c, -1)) for c in range(m)]
                if ok_shape(t):
                    sols.append((sum(weight.get(c, 0.0) for c in S), asg))
                    if len(sols) >= cap:
                        break
            if len(sols) >= cap:
                break
        if sols:
            sols.sort(key=lambda s: (s[0], sorted(s[1].items())))
            return k, sols
    return -1, []


def best_question(sols: list[tuple[float, dict]], known: dict[int, int]) -> tuple[int, int] | None:
    """The corrected fact most minimal-cost repairs agree on; overruled facts (the suspects)
    before fills of empty cells."""
    if not sols:
        return None
    best = min(c for c, _ in sols)
    top = [a for c, a in sols if c == best]
    votes = Counter((c, v) for a in top for c, v in a.items())
    return max(votes, key=lambda cv: (votes[cv], cv[0] in known, -cv[0], -cv[1]))
