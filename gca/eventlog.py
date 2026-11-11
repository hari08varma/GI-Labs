"""JSONL event log with provenance (plan §11 "Logging"). Named eventlog.py rather than
logging.py so it never shadows the standard library module.

Every learning event (observe, unify, sprout, lock, fork, supersede, merge, retire,
diagnose, fix, query, ...) is one record with a sequence number used as logical time.
No wall-clock times are written, so equal seeds give byte-identical logs.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def _clean(v: Any):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(_clean(x) for x in v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    return v


class EventLog:
    def __init__(self, path: str | None = None, keep: bool = True):
        self.records: list[dict] = []
        self.keep = keep
        self.seq = 0
        self._f = open(path, "w", encoding="utf-8") if path else None

    def log(self, _ev: str, **fields) -> int:
        self.seq += 1
        rec = {"seq": self.seq, "ev": _ev, **{k: _clean(v) for k, v in fields.items()}}
        if self.keep:
            self.records.append(rec)
        if self._f:
            self._f.write(json.dumps(rec, sort_keys=True) + "\n")
        return self.seq

    def of(self, ev: str) -> list[dict]:
        return [r for r in self.records if r["ev"] == ev]

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None
