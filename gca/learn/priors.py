"""Counted priors: the search skill (plan §5.7). Everything is counts, no weights.

- A Dirichlet-multinomial grammar: P(p | type) = (c_p + alpha * base_p) / (sum c + alpha),
  base uniform over the productions available in the context.
- A schema library: P(schema s) = c_s / (C + beta), P(fresh enumeration) = beta / (C + beta).
- Overhypotheses: system-level outcome counts (Kemp, Perfors & Tenenbaum 2007).

The ablation `no_priors` freezes every count at zero (the base distribution).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from . import hypothesis as H

PRODUCTIONS = {
    "O": ("x", "y", "c", "U", "B", "iter", "iterc", "if"),
    "N": ("pos",),
    "B": ("lt", "R", "eq"),
}


@dataclass
class Schema:
    sid: str
    body: tuple                    # schema expression (holes in parameter slots)
    kinds: dict                    # hole -> frozenset of kinds
    arity: int
    out_type: str
    count: int = 1
    sources: list = field(default_factory=list)    # column ids it came from
    abstraction: bool = False      # proposed in play mode (plan §7.8)
    refuted: int = 0               # recognitions the checker refuted (track record)
    bindings: list = field(default_factory=list)   # (world, frame, column id) bound to it
    versions: list = field(default_factory=list)   # earlier bodies, kept when it was extended
    decisive: dict = field(default_factory=dict)   # check pattern (anchor offsets) -> times it refuted a rival
    rivals: dict = field(default_factory=dict)     # rival concept id -> times it competed and was refuted

    @property
    def trust(self) -> float:
        """How often recognizing this concept turned out right: (ok + 1) / (ok + refuted + 2)."""
        return (self.count + 1) / (self.count + self.refuted + 2)


class Priors:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, frozen: bool = False):
        self.alpha = alpha
        self.beta = beta
        self.frozen = frozen
        self.counts: dict[str, Counter] = {t: Counter() for t in PRODUCTIONS}
        self.schemas: dict[tuple, Schema] = {}
        self.over: Counter = Counter()          # overhypothesis counts
        self.version = 0

    # ---- grammar ---------------------------------------------------------
    def prod_bits(self, typ: str, prod: str, available: tuple[str, ...]) -> float:
        c = self.counts[typ]
        tot = sum(c[p] for p in available)
        base = 1.0 / len(available)
        p = (c[prod] + self.alpha * base) / (tot + self.alpha)
        return -math.log2(p)

    def available(self, typ: str, prims: H.Prims, arity: int) -> tuple[str, ...]:
        have = {"U": bool(prims.unary), "B": bool(prims.binary), "iter": bool(prims.powers),
                "iterc": bool(prims.binary), "pos": bool(prims.chains), "R": bool(prims.rel),
                "y": arity == 2}
        return tuple(p for p in PRODUCTIONS[typ] if have.get(p, True))

    def param_bits(self, kind: str, prims: H.Prims) -> float:
        if kind == "E":
            return math.log2(max(prims.m, 1))
        return math.log2(max(len(prims.kinds(kind)), 1))

    def expr_bits(self, e, prims: H.Prims, arity: int) -> float:
        """L(H) for a fresh (enumerated) expression."""
        t = e[0]
        if t == "table" or t == "map":
            raise ValueError("table/map costs are set by their builders")
        typ = H.OUT_TYPE[t]
        bits = self.prod_bits(typ, t, self.available(typ, prims, arity))
        for _, k in H.PARAM_SLOTS.get(t, ()):
            bits += self.param_bits(k, prims)
        for c in H.children(e):
            bits += self.expr_bits(c, prims, arity)
        return bits

    def fresh_bits(self, arity: int, out_type: str) -> float:
        """Cost of choosing fresh enumeration over the library."""
        C = self._lib_total(arity, out_type)
        return -math.log2(self.beta / (C + self.beta))

    # ---- library ---------------------------------------------------------
    def _count(self, s: Schema) -> float:
        # no_priors: every schema weighs the same; otherwise successes discounted by refutations
        return 1 if self.frozen else max(s.count * s.trust, 0.25)

    def _lib_total(self, arity: int, out_type: str) -> int:
        return sum(self._count(s) for s in self.schemas.values()
                   if s.arity == arity and s.out_type == out_type)

    def schema_bits(self, s: Schema) -> float:
        C = self._lib_total(s.arity, s.out_type)
        return -math.log2(self._count(s) / (C + self.beta))

    def library(self, arity: int, out_type: str) -> list[Schema]:
        return sorted((s for s in self.schemas.values() if s.arity == arity and s.out_type == out_type),
                      key=lambda s: (-s.count, s.sid))

    def add_schema(self, body, arity: int, out_type: str, source: str, abstraction: bool = False) -> Schema:
        sch, kinds = H.to_schema(body)
        key = (sch, arity, out_type)
        s = self.schemas.get(key)
        if s is None:
            s = Schema(f"sch{len(self.schemas)}", sch, kinds, arity, out_type, 1, [source], abstraction)
            self.schemas[key] = s
        else:
            s.count += 1
            s.sources.append(source)
        self.version += 1
        return s

    # ---- counts ----------------------------------------------------------
    def add_body(self, body) -> None:
        """After a system, add the productions used in a locked column's body."""
        if self.frozen or body[0] in ("table", "map"):
            return
        for typ, prod in H.productions(body):
            if typ in self.counts:
                self.counts[typ][prod] += 1
        self.version += 1

    def observe(self, event: str, k: int = 1) -> None:
        """Overhypothesis counts (e.g. 'op:schema', 'op:fresh', 'iso:yes', 'cause:slip')."""
        self.over[event] += k

    def rate(self, event: str, total_events: tuple[str, ...], prior: float = 0.5) -> float:
        """Smoothed rate of `event` among `total_events` (Laplace with a prior mean)."""
        if self.frozen:
            return prior
        n = sum(self.over[e] for e in total_events)
        return (self.over[event] + prior) / (n + 1.0)
