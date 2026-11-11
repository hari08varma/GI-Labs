"""The three spaces (plan §4.4).

whiteboard : plastic and candidate columns (usable, answers built on them are low-confidence)
notebook   : locked columns (used first, never edited)
mismatch   : refuted hypotheses with their counterexamples, observations flagged as
             noise, superseded columns with the reason. Used as negative constraints in
             search and as evidence in diagnosis.
Columns that leave the active states are kept in `archive` for audit; nothing is deleted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .columns import ACTIVE, RANK, Column


@dataclass
class MismatchRecord:
    kind: str                # refuted | noise | superseded | constraint | rollback
    ctx: str
    frame: str | None
    detail: dict = field(default_factory=dict)
    time: int = 0


class Spaces:
    def __init__(self):
        self.columns: dict[str, Column] = {}
        self.mismatch: list[MismatchRecord] = []
        self._by_frame: dict[tuple[str, str], list[str]] = {}

    # ---- views ---------------------------------------------------------------
    @property
    def whiteboard(self) -> list[Column]:
        return [c for c in self.columns.values() if c.state in ("plastic", "candidate")]

    @property
    def notebook(self) -> list[Column]:
        return [c for c in self.columns.values() if c.state == "locked"]

    @property
    def archive(self) -> list[Column]:
        return [c for c in self.columns.values() if c.state not in ACTIVE]

    def for_frame(self, ctx: str, frame: str, active_only: bool = True, kind: str | None = "operator") -> list[Column]:
        cols = [self.columns[i] for i in self._by_frame.get((ctx, frame), [])]
        if kind is not None:
            cols = [c for c in cols if c.kind == kind]
        if active_only:
            cols = [c for c in cols if c.active]
        return sorted(cols, key=lambda c: (RANK.get(c.state, 9), c.provenance.created_at))

    def in_context(self, ctx: str, active_only: bool = True) -> list[Column]:
        return [c for c in self.columns.values() if c.ctx == ctx and (c.active or not active_only)]

    # ---- writes ----------------------------------------------------------------
    def add(self, col: Column) -> Column:
        assert col.id not in self.columns, col.id
        self.columns[col.id] = col
        self._by_frame.setdefault((col.ctx, col.frame), []).append(col.id)
        return col

    def set_state(self, col: Column, state: str) -> None:
        if col.state == "locked" and state in ("plastic", "candidate"):
            raise ValueError("fork-not-bend: a locked column cannot be unlocked")
        col.state = state

    def note(self, _kind: str, _ctx: str, _frame: str | None, _time: int, **detail) -> MismatchRecord:
        r = MismatchRecord(_kind, _ctx, _frame, detail, _time)
        self.mismatch.append(r)
        return r

    def mismatches(self, ctx: str, frame: str | None = None, kind: str | None = None) -> list[MismatchRecord]:
        return [r for r in self.mismatch if r.ctx == ctx and (frame is None or r.frame == frame)
                and (kind is None or r.kind == kind)]
