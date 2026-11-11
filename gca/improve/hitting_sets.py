"""Who is at fault (plan §7.3): consistency-based diagnosis (Reiter 1987).

Components are columns, lexicon entries, name resolutions and stored events. A
diagnosis is a minimal hitting set of all current conflict sets. The sets here are
tiny, so an exhaustive search by increasing size (equivalent to the HS-tree with
Greiner's correction) is exact and cheap.
"""
from __future__ import annotations

import itertools
import math


def minimal_hitting_sets(conflicts: list[frozenset], max_size: int = 3) -> list[frozenset]:
    conflicts = [c for c in conflicts if c]
    if not conflicts:
        return [frozenset()]
    universe = sorted(set().union(*conflicts))
    found: list[frozenset] = []
    for k in range(1, min(max_size, len(universe)) + 1):
        for combo in itertools.combinations(universe, k):
            s = frozenset(combo)
            if any(f <= s for f in found):
                continue
            if all(s & c for c in conflicts):
                found.append(s)
    return found


def rank(diagnoses: list[frozenset], prior: dict) -> list[tuple[frozenset, float]]:
    """Order by prior probability of being faulty: product over members."""
    scored = []
    for d in diagnoses:
        p = math.prod(prior.get(x, 0.05) for x in d)
        scored.append((d, p))
    return sorted(scored, key=lambda t: (-t[1], sorted(map(str, t[0]))))
