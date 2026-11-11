"""The learner's hypothesis language (plan §5.2) and its evaluator.

Expressions are tuples:
  ("x",) ("y",)                  arguments
  ("c", e)                       constant entity e
  ("U", r, a)                    apply local unary function r
  ("B", r, a, b)                 apply local binary function r
  ("iter", r, k, a)              apply unary r k times starting from a
  ("iterc", r, p, k, a)          apply t -> B_r(p, t) k times starting from a   (curry)
  ("pos", r, o, a)               position of a in the chain o, r(o), r(r(o)), ... (Nat)
  ("lt", k1, k2)                 Nat comparison (Bool)
  ("R", r, a, b)                 local relation (Bool)
  ("eq", a, b)                   equality (Bool)
  ("if", b, a1, a2)              case split
  ("table",)                     the column's own lookup table (fully observed frames)
  ("map", src, phi)              a table pulled back through an analogy (plan §6.6)
  ("H", "?t0")                   subtree hole (abstractions from play mode only)

Types: "O" object, "N" count, "B" boolean. Refs r, anchors o and constants e are
parameters, not nodes. In a schema (a context-free body) every parameter is a hole:
a string "?r0", "?r1", ... for refs and "?e0", ... for entities.

Values are int arrays over the input grid of the context's m entities.
UNDEF = -1 marks an undefined value; tables are stored "extended" with a trailing
UNDEF slot so that indexing with -1 propagates UNDEF without branching.
"""
from __future__ import annotations

import numpy as np

UNDEF = -1

OUT_TYPE = {"x": "O", "y": "O", "c": "O", "U": "O", "B": "O", "iter": "O", "iterc": "O",
            "if": "O", "pos": "N", "lt": "B", "R": "B", "eq": "B", "H": "O"}
CHILD_SLOTS = {"x": (), "y": (), "c": (), "U": (2,), "B": (2, 3), "iter": (2, 3),
               "iterc": (2, 3, 4), "pos": (3,), "lt": (1, 2), "R": (2, 3), "eq": (1, 2),
               "if": (1, 2, 3), "table": (), "map": (), "H": ()}
CHILD_TYPES = {"U": ("O",), "B": ("O", "O"), "iter": ("N", "O"), "iterc": ("O", "N", "O"),
               "pos": ("O",), "lt": ("N", "N"), "R": ("O", "O"), "eq": ("O", "O"),
               "if": ("B", "O", "O")}
# parameter slots: (index, kind). Ref kinds: U unary, P unary with powers, C single
# cycle (chain), B binary, R relation. "E" is an entity.
PARAM_SLOTS = {"c": ((1, "E"),), "U": ((1, "U"),), "B": ((1, "B"),), "iter": ((1, "P"),),
               "iterc": ((1, "B"),), "pos": ((1, "C"), (2, "E")), "R": ((1, "R"),)}


def is_hole(p) -> bool:
    return isinstance(p, str) and p.startswith("?")


def children(e) -> list:
    return [e[i] for i in CHILD_SLOTS[e[0]]]


def size(e) -> int:
    """Node count (the column size budget S_MAX counts these)."""
    if e[0] in ("table", "map", "H"):
        return 1
    return 1 + sum(size(c) for c in children(e))


def depth(e) -> int:
    ch = children(e)
    return 1 + (max(depth(c) for c in ch) if ch else 0)


def params(e) -> list[tuple[str, object]]:
    """All parameters (kind, value) in pre-order."""
    out = [(k, e[i]) for i, k in PARAM_SLOTS.get(e[0], ())]
    for c in children(e):
        out += params(c)
    return out


def refs_used(e) -> set:
    return {v for k, v in params(e) if k != "E"}


def productions(e) -> list[tuple[str, str]]:
    """(type, production) for every node: the counts behind the priors (plan §5.7)."""
    t = e[0]
    out = [(OUT_TYPE.get(t, "O"), t)]
    for c in children(e):
        out += productions(c)
    return out


def replace_params(e, f):
    """Rebuild e with every parameter p of kind k replaced by f(k, p)."""
    t = e[0]
    if t in ("table", "map", "H"):
        return e
    lst = list(e)
    for i, k in PARAM_SLOTS.get(t, ()):
        lst[i] = f(k, e[i])
    for i in CHILD_SLOTS[t]:
        lst[i] = replace_params(e[i], f)
    return tuple(lst)


def show(e, names=str, ent=lambda i: f"e{i}") -> str:
    t = e[0]
    s = lambda x: show(x, names, ent)
    if t in ("x", "y"):
        return t
    if t == "c":
        return ent(e[1])
    if t == "U":
        return f"{names(e[1])}({s(e[2])})"
    if t in ("B", "R"):
        return f"{names(e[1])}({s(e[2])}, {s(e[3])})"
    if t == "iter":
        return f"repeat[{names(e[1])}]({s(e[2])} times, from {s(e[3])})"
    if t == "iterc":
        return f"repeat[t -> {names(e[1])}({s(e[2])}, t)]({s(e[3])} times, from {s(e[4])})"
    if t == "pos":
        return f"pos[{names(e[1])} from {ent(e[2])}]({s(e[3])})"
    if t == "lt":
        return f"({s(e[1])} < {s(e[2])})"
    if t == "eq":
        return f"({s(e[1])} = {s(e[2])})"
    if t == "if":
        return f"(if {s(e[1])} then {s(e[2])} else {s(e[3])})"
    if t == "table":
        return "table"
    if t == "map":
        return f"analogy[{e[1]}]"
    if t == "H":
        return e[1]
    raise ValueError(t)


def _ext1(t: np.ndarray) -> np.ndarray:
    return np.append(t.astype(np.int8), np.int8(UNDEF))


def _ext2(t: np.ndarray) -> np.ndarray:
    m = t.shape[0]
    out = np.full((m + 1, m + 1), UNDEF, dtype=np.int8)
    out[:m, :m] = t
    return out


class Prims:
    """Local primitives of one context, as extended tables over its m entities.

    Unary refs: any total table. Powers (iter) exist for every total unary table.
    Chains (pos) exist for single-cycle bijections, from every anchor.
    Binary refs carry a curried-power cube for iterc."""

    def __init__(self, m: int):
        self.m = m
        self.unary: dict[str, np.ndarray] = {}
        self.binary: dict[str, np.ndarray] = {}
        self.rel: dict[str, np.ndarray] = {}
        self.powers: dict[str, np.ndarray] = {}      # ref -> (m+1, m+1) ext: [k, a] = f^k(a)
        self.chains: dict[str, np.ndarray] = {}      # ref -> (m, m+1) ext: [o, a] = pos
        self.cubes: dict[str, np.ndarray] = {}       # ref -> (m+1, m+1, m+1) ext: [p, k, a]
        self.raw: dict[str, np.ndarray] = {}         # ref -> plain table (for signatures)

    def signature(self) -> tuple:
        return (self.m,) + tuple(sorted((r, t.tobytes()) for r, t in self.raw.items()))

    def kinds(self, kind: str) -> list[str]:
        return sorted({"U": self.unary, "P": self.powers, "C": self.chains,
                       "B": self.binary, "R": self.rel}[kind])

    def add_unary(self, ref: str, t: np.ndarray) -> None:
        m = self.m
        t = np.asarray(t, dtype=np.int64)
        assert t.shape == (m,) and (t >= 0).all() and (t < m).all()
        self.raw[ref] = t
        self.unary[ref] = _ext1(t)
        P = np.full((m + 1, m + 1), UNDEF, dtype=np.int8)
        cur = np.arange(m)
        for k in range(m):
            P[k, :m] = cur
            cur = t[cur]
        self.powers[ref] = P
        orbit, c = [], 0
        for _ in range(m):
            orbit.append(c)
            c = int(t[c])
        if len(set(orbit)) == m and c == 0:          # single cycle through all entities
            C = np.full((m, m + 1), UNDEF, dtype=np.int8)
            for o in range(m):
                c = o
                for k in range(m):
                    C[o, c] = k
                    c = int(t[c])
            self.chains[ref] = C

    def add_binary(self, ref: str, t: np.ndarray) -> None:
        m = self.m
        t = np.asarray(t, dtype=np.int64)
        assert t.shape == (m, m) and (t >= 0).all() and (t < m).all()
        self.raw[ref] = t
        self.binary[ref] = _ext2(t)
        Q = np.full((m + 1, m + 1, m + 1), UNDEF, dtype=np.int8)
        for p in range(m):
            cur = np.arange(m)
            for k in range(m):
                Q[p, k, :m] = cur
                cur = t[p, cur]
        self.cubes[ref] = Q

    def add_rel(self, ref: str, t: np.ndarray) -> None:
        t = np.asarray(t, dtype=np.int64)
        assert t.shape == (self.m, self.m) and set(np.unique(t)) <= {0, 1}
        self.raw[ref] = t
        self.rel[ref] = _ext2(t)


def grid(m: int, arity: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened input grid. Cell index = a*m + b for binary, a for unary."""
    if arity == 1:
        X = np.arange(m)
        return X, X.copy()
    X, Y = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    return X.ravel(), Y.ravel()


def evaluate(e, X: np.ndarray, Y: np.ndarray, P: Prims, tables: dict | None = None) -> np.ndarray:
    """Evaluate over flattened grids X, Y. Returns int64 values, UNDEF where undefined.
    `tables` supplies ("table",) and ("map", ...) bodies: key -> flat table."""
    t = e[0]
    ev = lambda x: evaluate(x, X, Y, P, tables)
    if t == "x":
        return X.astype(np.int64)
    if t == "y":
        return Y.astype(np.int64)
    if t == "c":
        return np.full(X.shape, e[1], dtype=np.int64)
    if t == "U":
        return P.unary[e[1]][ev(e[2])]
    if t == "B":
        return P.binary[e[1]][ev(e[2]), ev(e[3])]
    if t == "R":
        return P.rel[e[1]][ev(e[2]), ev(e[3])]
    if t == "iter":
        k = ev(e[2])
        k = np.where(k >= P.m, UNDEF, k)
        return P.powers[e[1]][k, ev(e[3])]
    if t == "iterc":
        k = ev(e[3])
        k = np.where(k >= P.m, UNDEF, k)
        return P.cubes[e[1]][ev(e[2]), k, ev(e[4])]
    if t == "pos":
        return P.chains[e[1]][e[2]][ev(e[3])]
    if t in ("lt", "eq"):
        a, b = ev(e[1]), ev(e[2])
        r = (a < b) if t == "lt" else (a == b)
        return np.where((a >= 0) & (b >= 0), r.astype(np.int64), UNDEF)
    if t == "if":
        c, a, b = ev(e[1]), ev(e[2]), ev(e[3])
        return np.where(c < 0, UNDEF, np.where(c == 1, a, b))
    if t in ("table", "map"):
        key = e if t == "map" else ("table",)
        tab = tables[key]
        idx = X * P.m + Y if tab.size == P.m * P.m else X
        return tab[idx].astype(np.int64)
    raise ValueError(t)


# ---- schemas: context-free bodies (plan §5.4 "library first", build log "Schemas") ----

def to_schema(body):
    """Replace every parameter with a hole. Equal parameters share one hole, so
    "the same anchor" stays the same. Returns (schema, hole kinds)."""
    ref_h: dict = {}
    ent_h: dict = {}
    kinds: dict[str, set] = {}

    def f(k, p):
        if k == "E":
            h = ent_h.setdefault(p, f"?e{len(ent_h)}")
            kinds.setdefault(h, set()).add("E")
            return h
        h = ref_h.setdefault(p, f"?r{len(ref_h)}")
        kinds.setdefault(h, set()).add(k)
        return h

    sch = replace_params(body, f)
    for h in subtree_holes(sch):
        kinds.setdefault(h, set()).add("T")
    return sch, {h: frozenset(v) for h, v in kinds.items()}


def subtree_options(P: Prims, arity: int) -> list:
    """Fillers for a subtree hole: the arguments and one unary step on them."""
    leaves = [("x",), ("y",)] if arity == 2 else [("x",)]
    return leaves + [("U", r, a) for r in sorted(P.unary) for a in leaves]


def subtree_holes(e) -> list[str]:
    if e[0] == "H":
        return [e[1]]
    out = []
    for c in children(e):
        out += subtree_holes(c)
    return out


def hole_options(kinds: frozenset, P: Prims, arity: int = 2) -> list:
    if "T" in kinds:
        return subtree_options(P, arity)
    if "E" in kinds:
        return list(range(P.m))
    opts = None
    for k in kinds:
        s = set(P.kinds(k))
        opts = s if opts is None else opts & s
    return sorted(opts or [])


def instantiate(schema, fill: dict):
    def sub(e):
        if e[0] == "H":
            return fill[e[1]]
        if e[0] in ("table", "map"):
            return e
        lst = list(e)
        for i, k in PARAM_SLOTS.get(e[0], ()):
            if is_hole(e[i]):
                lst[i] = fill[e[i]]
        for i in CHILD_SLOTS[e[0]]:
            lst[i] = sub(e[i])
        return tuple(lst)
    return sub(schema)
