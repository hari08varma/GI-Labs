"""Columns (plan §4.3): equal-size structured knowledge units with keys, ports and
provenance, and a lifecycle. Fork-not-bend: a locked column's body and scope never
change; changes create a new column that forks or supersedes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vsa import VSA

STATES = ("plastic", "candidate", "locked", "forked", "superseded", "merged", "retired")
ACTIVE = ("plastic", "candidate", "locked")
RANK = {"locked": 0, "candidate": 1, "plastic": 2}
S_MAX = 12                       # node budget per column (P3)


@dataclass
class Provenance:
    system: str
    origin: str                  # pool | split | schema:<id> | analogy:<ctx> | table | fork | diagnosis
    created_at: int              # logical time (event-log sequence number)
    sources: tuple = ()          # observation ids behind the data at creation
    parent: str | None = None    # column it forked from / superseded
    diagnosis: str | None = None # id of the diagnosis record that created it


@dataclass
class Stats:
    uses: int = 0
    successes: int = 0
    failures: int = 0
    sprt: float = 0.0            # log-likelihood ratio, H1 (correct) vs H0 (wrong)
    last_used: int = 0
    confirmed: set = field(default_factory=set)    # cells confirmed by the checker (new inputs)


@dataclass
class Column:
    id: str
    kind: str                    # operator | relation | lexicon | property | abstraction | procedure
    ctx: str
    frame: str
    arity: int
    out_type: str                # "O" | "B"
    body: tuple
    sem: np.ndarray              # full-domain table, flattened over the context's grid
    prior_bits: float
    provenance: Provenance
    state: str = "plastic"
    scope: tuple | None = None   # applicability condition (Bool expr over x, y) or None
    calls: tuple = ()            # columns this one links to (refs used)
    tables: dict = field(default_factory=dict)     # for ("table",) / ("map", ...) bodies
    validated: set = field(default_factory=set)    # cells inside its validated scope
    stats: Stats = field(default_factory=Stats)
    key: np.ndarray | None = None
    replaced_by: str | None = None
    note: str = ""

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    @property
    def ref(self) -> str:
        """Name of this column's function when used as a primitive by other columns."""
        return "f:" + self.frame

    def signature(self) -> str:
        return f"{self.arity}->{self.out_type}"

    def applies(self, cell: int, scope_sem: np.ndarray | None) -> bool:
        return self.scope is None or scope_sem is None or bool(scope_sem[cell] == 1)


def column_key(vsa: VSA, kind: str, signature: str, ctx: str | None, frame: str) -> np.ndarray:
    """ROLE_kind (x) HV(kind) + ROLE_sig (x) HV(sig) [+ ROLE_ctx (x) HV(ctx)] + ROLE_frame (x) HV(frame),
    as an integer sum compared by cosine (plan §4.3). Context-generic keys leave out ctx."""
    parts = [vsa.bind(vsa.hv("role:kind"), vsa.hv("kind:" + kind)),
             vsa.bind(vsa.hv("role:sig"), vsa.hv("sig:" + signature)),
             vsa.bind(vsa.hv("role:frame"), vsa.hv("frame:" + frame))]
    if ctx is not None:
        parts.append(vsa.bind(vsa.hv("role:ctx"), vsa.hv("ctx:" + ctx)))
    return vsa.key_sum(parts)
