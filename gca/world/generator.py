"""Hidden generator: builds invented maths systems, experience, questions and
answer keys. PRIVATE to the world; learners see only gca.world.api types."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml

from . import grammar as G
from .api import FALSE, MANY, NONE, TRUE, Observation, Question
from .api import depth as _depth
from .tokens import TokenPool


@dataclass
class WordDef:
    word: str
    spec: G.OpSpec
    frame: str                 # id of the hidden frame (one per operator)
    surface: tuple[int, ...]   # surface position -> canonical position (args..., out)
    foundation: bool = False


@dataclass
class SystemWorld:
    sid: str
    n: int
    block: str
    tags: tuple[str, ...]
    objs: list[str]                              # hidden int -> token
    words: list[WordDef]
    tables: dict[str, np.ndarray]                # word -> surface table (ints; bool as 0/1)
    quasi: dict[str, bool]
    experience: list[Observation] = field(default_factory=list)
    seen: dict[str, set] = field(default_factory=dict)       # word -> set of surface arg tuples (ints)
    increments: list[list[Observation]] = field(default_factory=list)
    questions: dict[str, list[Question]] = field(default_factory=dict)
    practice: list[Question] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)        # qid -> answer value
    extra_labels: list[Observation] = field(default_factory=list)
    identifiable: dict[str, int] = field(default_factory=dict)  # word -> cells added by the check
    copy_of: str | None = None

    # ---- semantics -------------------------------------------------------
    def tok(self, v: int, word: str | None = None) -> str:
        if word is not None and self.boolean(word):
            return TRUE if v else FALSE
        return self.objs[v]

    def val(self, token: str) -> int:
        return self.objs.index(token)

    def boolean(self, word: str) -> bool:
        return self.wdef(word).spec.boolean

    def wdef(self, word: str) -> WordDef:
        for w in self.words:
            if w.word == word:
                return w
        raise KeyError(word)

    def arity(self, word: str) -> int:
        return self.tables[word].ndim

    def apply(self, word: str, args: tuple[int, ...]) -> int:
        return int(self.tables[word][args])

    def eval_int(self, expr) -> int:
        if expr[0] == "obj":
            return self.val(expr[1])
        return self.apply(expr[1], tuple(self.eval_int(e) for e in expr[2:]))

    def true_fact(self, word: str, args: tuple[str, ...], out: str) -> bool:
        try:
            a = tuple(self.val(t) for t in args)
        except ValueError:
            return False
        if len(a) != self.arity(word):
            return False
        return self.tok(self.apply(word, a), word) == out


# ---------------------------------------------------------------------------
def _surface_table(spec: G.OpSpec, n: int, surface: tuple[int, ...]) -> np.ndarray:
    """Turn the canonical table into the table of one phrasing. `surface[i]` is the
    canonical position shown at surface position i (last surface position = out)."""
    t = G.table(spec, n)
    k = spec.arity
    shape = (n,) * k
    out = -np.ones(shape, dtype=np.int64)
    for canon in np.ndindex(*shape):
        tup = canon + (int(t[canon]),)                 # canonical (args..., out)
        surf = tuple(tup[surface[i]] for i in range(k + 1))
        args, o = surf[:-1], surf[-1]
        if out[args] != -1 and out[args] != o:
            raise ValueError("phrasing is not functional")
        out[args] = o
    if (out < 0).any():
        raise ValueError("phrasing is not total")
    return out


def _cells(n: int, k: int) -> list[tuple[int, ...]]:
    return list(np.ndindex(*((n,) * k)))


def _identify(spec: G.OpSpec, n: int, truth: np.ndarray, cells: list, rng) -> tuple[list, int]:
    """Add cells until every hypothesis of the generator's grammar consistent with
    the cells agrees with the truth on the whole domain (plan §2.4)."""
    fam = G.family_tables(spec, n)
    flat = truth.ravel()
    idx = lambda c: int(np.ravel_multi_index(c, truth.shape))
    added = 0
    while True:
        obs = np.array([idx(c) for c in cells], dtype=np.int64)
        consistent = fam[(fam[:, obs] == flat[obs]).all(axis=1)] if len(obs) else fam
        wrong = consistent[(consistent != flat).any(axis=1)]
        if len(wrong) == 0:
            return cells, added
        disagree = (wrong != flat).sum(axis=0)
        disagree[obs] = -1
        best = int(np.argmax(disagree))
        cells = cells + [tuple(int(v) for v in np.unravel_index(best, truth.shape))]
        added += 1


class StreamBuilder:
    def __init__(self, cfg: dict, seed: int, noise: float):
        self.cfg = cfg
        self.d = cfg["defaults"]
        self.seed = seed
        self.noise = noise
        ss = np.random.SeedSequence([seed, int(round(noise * 1000))])
        self.rng = np.random.default_rng(ss)
        # token stream depends only on the seed, so noise levels share tokens
        self.pool = TokenPool(np.random.default_rng([seed, 7]))
        self.built: dict[str, SystemWorld] = {}

    # ---------------------------------------------------------------------
    def build(self) -> list[SystemWorld]:
        out = []
        for scfg in self.cfg["systems"]:
            out.append(self._system(scfg))
        return out

    def _system(self, scfg: dict) -> SystemWorld:
        sid = scfg["id"]
        rng = np.random.default_rng([self.seed, int(sid[1:]), int(round(self.noise * 1000))])
        src = self.built.get(scfg.get("copy_of", ""), None)
        if src is not None:
            n = scfg.get("n", src.n)
            foundation = scfg.get("foundation", any(w.foundation for w in src.words))
            ops = scfg.get("ops", None) or self._ops_of(src)
        else:
            n = scfg["n"]
            foundation = scfg.get("foundation", True)
            ops = scfg.get("ops", [])
        reuse = float(scfg.get("reuse_tokens", 0.0))
        taken: set[str] = set()

        def token() -> str:
            if reuse > 0 and rng.random() < reuse:
                t = self.pool.reuse(taken)
                if t is not None:
                    taken.add(t)
                    return t
            t = self.pool.fresh(taken)
            taken.add(t)
            return t

        objs = [token() for _ in range(n)]
        words: list[WordDef] = []
        if foundation:
            spec = G.OpSpec("succ")
            words.append(WordDef(token(), spec, "F0", (0, 1), True))   # next(a) = b
            words.append(WordDef(token(), spec, "F0", (1, 0), True))   # previous(b) = a
        for i, op in enumerate(ops):
            spec = G.parse(op["def"])
            fid = f"F{i + 1}"
            ident = tuple(range(spec.arity + 1))
            words.append(WordDef(token(), spec, fid, ident))
            if op.get("alt_phrasing"):
                assert spec.arity == 2
                words.append(WordDef(token(), spec, fid, (1, 0, 2)))   # argument-swapped phrasing
        tables = {w.word: _surface_table(w.spec, n, w.surface) for w in words}
        quasi = {}
        for w in words:
            t = tables[w.word]
            quasi[w.word] = (not w.spec.boolean) and (t.ndim == 1 and len(set(t)) == n or
                                                      t.ndim == 2 and G.is_quasigroup(t))
        tags = tuple(scfg.get("tags", []))
        sw = SystemWorld(sid, n, scfg.get("block", ""), tags, objs, words, tables, quasi,
                         copy_of=scfg.get("copy_of"))
        self._experience(sw, rng)
        self._questions(sw, rng)
        self.built[sid] = sw
        return sw

    @staticmethod
    def _ops_of(src: SystemWorld) -> list[dict]:
        ops, seen = [], set()
        for w in src.words:
            if w.foundation or w.frame in seen:
                continue
            seen.add(w.frame)
            alt = sum(1 for v in src.words if v.frame == w.frame) > 1
            p = w.spec.params
            d = w.spec.kind if not p else f"{w.spec.kind}:" + ",".join(str(x) for x in p)
            ops.append({"def": d, "alt_phrasing": alt})
        return ops

    # ---------------------------------------------------------------------
    def _experience(self, sw: SystemWorld, rng) -> None:
        n, d = sw.n, self.d
        obs: list[tuple[str, tuple[int, ...]]] = []
        fwords = [w for w in sw.words if w.foundation]
        if fwords:
            nxt, prv = fwords
            for a in rng.permutation(n):
                a = int(a)
                b = int(sw.tables[nxt.word][a])
                r = rng.random()
                if r < 0.4:
                    obs += [(nxt.word, (a,)), (prv.word, (b,))]
                elif r < 0.7:
                    obs.append((nxt.word, (a,)))
                else:
                    obs.append((prv.word, (b,)))
        frames: dict[str, list[WordDef]] = {}
        for w in sw.words:
            if not w.foundation:
                frames.setdefault(w.frame, []).append(w)
        for fid, ws in frames.items():
            main = ws[0]
            k = main.spec.arity
            cells = _cells(n, k)
            m = max(int(round(d["binary_fraction"] * len(cells))), n + 4) if k == 2 else len(cells)
            m = min(m, len(cells))
            pick = [cells[i] for i in rng.choice(len(cells), m, replace=False)]
            pick, added = _identify(main.spec, n, sw.tables[main.word], pick, rng)
            sw.identifiable[main.word] = added
            obs += [(main.word, c) for c in pick]
            for alt in ws[1:]:
                # half the alt-phrased cells overlap events already seen in the main phrasing
                share = pick[: max(3, len(pick) // 2)]
                swapped = [(c[1], c[0]) for c in share]
                rest = [c for c in cells if c not in set(swapped)]
                extra = [rest[i] for i in rng.choice(len(rest), max(3, len(pick) // 2), replace=False)]
                obs += [(alt.word, c) for c in swapped + extra]
        # noise-free record of what was seen, then the noisy observations
        sw.seen = {w.word: set() for w in sw.words}
        for word, args in obs:
            sw.seen[word].add(args)
        out, is_found = [], []
        fset = {w.word for w in fwords}
        for word, args in obs:
            v = sw.apply(word, args)
            if self.noise > 0 and rng.random() < self.noise:
                if sw.boolean(word):
                    v = 1 - v
                else:
                    v = int((v + rng.integers(1, n)) % n)
            out.append(Observation(sw.sid, word, tuple(sw.objs[a] for a in args), sw.tok(v, word)))
            is_found.append(word in fset)
        order = list(rng.permutation(len(out)))
        if d.get("foundation_first", True):
            # curriculum: the counting sequence is heard first (build log, "World")
            order = [i for i in order if is_found[i]] + [i for i in order if not is_found[i]]
        sw.experience = [out[i] for i in order]
        inc = d["increment"]
        sw.increments = [sw.experience[i:i + inc] for i in range(0, len(sw.experience), inc)]
        # extra labels for baselines that do not query actively (plan §2.6): noise-free
        budget = self.budget(sw)
        pool = [(w.word, c) for w in sw.words for c in _cells(n, sw.arity(w.word))
                if c not in sw.seen[w.word]]
        if pool:
            sel = rng.choice(len(pool), min(budget, len(pool)), replace=False)
            sw.extra_labels = [Observation(sw.sid, pool[i][0], tuple(sw.objs[a] for a in pool[i][1]),
                                           sw.tok(sw.apply(*pool[i]), pool[i][0])) for i in sel]

    def budget(self, sw: SystemWorld) -> int:
        return int(self.d["budget_per_word"]) * len(sw.words)

    # ---------------------------------------------------------------------
    def _questions(self, sw: SystemWorld, rng) -> None:
        Q = int(self.d["questions_per_split"])
        P = int(self.d["practice_questions"])
        qs: dict[str, list[Question]] = {s: [] for s in ("S1", "S2", "S3", "S4", "S4b")}
        used: set = set()
        obj_words = [w.word for w in sw.words if not w.spec.boolean]
        all_words = [w.word for w in sw.words]

        def add(split: str, kind: str, expr, target, key, pool: list):
            sig = (kind, expr, target)
            if sig in used:
                return False
            used.add(sig)
            qid = f"{sw.sid}-{split}-{len(used)}"
            pool.append(Question(sw.sid, qid, split, kind, expr, target))
            sw.keys[qid] = key
            return True

        def leaf():
            return ("obj", sw.objs[int(rng.integers(sw.n))])

        def unseen_step(expr) -> bool:
            if expr[0] != "app":
                return False
            args = tuple(sw.eval_int(e) for e in expr[2:])
            if args not in sw.seen[expr[1]]:
                return True
            return any(unseen_step(e) for e in expr[2:])

        def tree(dep: int, root_words) -> tuple:
            w = root_words[int(rng.integers(len(root_words)))]
            k = sw.arity(w)
            sub = int(rng.integers(k))
            kids = []
            for i in range(k):
                if dep > 1 and (i == sub or rng.random() < 0.3):
                    kids.append(tree(dep - 1, obj_words))
                else:
                    kids.append(leaf())
            return ("app", w, *kids)

        def key_eval(expr) -> str:
            return sw.tok(sw.eval_int(expr), expr[1])

        # S1: single application on an unseen cell (non-foundation words)
        cand = [(w.word, c) for w in sw.words if not w.foundation
                for c in _cells(sw.n, sw.arity(w.word)) if c not in sw.seen[w.word]]
        for i in rng.permutation(len(cand))[: 3 * Q]:
            word, c = cand[i]
            e = ("app", word, *[("obj", sw.objs[a]) for a in c])
            if len(qs["S1"]) < Q:
                add("S1", "eval", e, None, key_eval(e), qs["S1"])
        # S2 / S3: composition and depth, at least one unseen step
        for split, lo, hi in (("S2", 2, 2), ("S3", 3, 5)):
            if not obj_words:
                break
            tries = 0
            while len(qs[split]) < Q and tries < 40 * Q:
                tries += 1
                dep = int(rng.integers(lo, hi + 1))
                e = tree(dep, all_words)
                if _depth(e) != dep and not (split == "S3" and _depth(e) >= 3):
                    continue
                if split == "S2" and len(set(_words_in(e))) < 2 and len(all_words) > 1:
                    continue
                if not unseen_step(e) and any(len(sw.seen[w]) < sw.n ** sw.arity(w) for w in _words_in(e)):
                    continue
                add(split, "eval", e, None, key_eval(e), qs[split])
        # S4: reversal. (a) phrasing reversal: other phrasing of facts seen one way only
        fr: dict[str, list[WordDef]] = {}
        for w in sw.words:
            fr.setdefault(w.frame, []).append(w)
        for ws in fr.values():
            if len(ws) < 2:
                continue
            for w in ws:
                for c in _cells(sw.n, sw.arity(w.word)):
                    if c in sw.seen[w.word]:
                        continue
                    # was the same event seen in another phrasing?
                    if any(self._same_event_seen(sw, w, v, c) for v in ws if v is not w):
                        e = ("app", w.word, *[("obj", sw.objs[a]) for a in c])
                        add("S4", "eval", e, None, key_eval(e), qs["S4"])
        # (b) solve for an input
        def solve_q(expr) -> list[int]:
            """Value of expr for every substitution of the variable."""
            return [sw.eval_int(_subst(expr, ("obj", sw.objs[v]))) for v in range(sw.n)]

        tries = 0
        while len(qs["S4"]) < Q and tries < 40 * Q:
            tries += 1
            qw = [w for w in obj_words if sw.quasi[w]]
            if not qw:
                break
            w = qw[int(rng.integers(len(qw)))]
            k = sw.arity(w)
            pos = int(rng.integers(k))
            nested = rng.random() < 0.3 and len(qw) > 0
            kids = [leaf() for _ in range(k)]
            kids[pos] = ("var",)
            e = ("app", w, *kids)
            if nested:
                w2 = qw[int(rng.integers(len(qw)))]
                k2 = sw.arity(w2)
                kids2 = [leaf() for _ in range(k2)]
                kids2[int(rng.integers(k2))] = e
                e = ("app", w2, *kids2)
            sols = solve_q(e)
            target_v = sols[int(rng.integers(sw.n))]
            hits = [v for v in range(sw.n) if sols[v] == target_v]
            if len(hits) != 1:
                continue
            add("S4", "solve", e, sw.tok(target_v, w if not nested else e[1]), sw.objs[hits[0]], qs["S4"])
        # S4b: honest reversal on non-quasigroup words (answers may be none / many)
        nq = [w for w in all_words if not sw.quasi[w]]
        tries = 0
        while nq and len(qs["S4b"]) < Q and tries < 40 * Q:
            tries += 1
            w = nq[int(rng.integers(len(nq)))]
            k = sw.arity(w)
            kids = [leaf() for _ in range(k)]
            kids[int(rng.integers(k))] = ("var",)
            e = ("app", w, *kids)
            sols = solve_q(e)
            if sw.boolean(w):
                tv = int(rng.integers(2))
            else:
                tv = int(rng.integers(sw.n))
            hits = [v for v in range(sw.n) if sols[v] == tv]
            key = NONE if not hits else (sw.objs[hits[0]] if len(hits) == 1 else MANY)
            add("S4b", "solve", e, sw.tok(tv, w), key, qs["S4b"])
        sw.questions = qs
        # practice questions (with feedback): drawn the same way, disjoint from tests
        # practice set: drawn from the same splits, removed from the tests
        prac: list[Question] = []
        sub: dict[str, list[Question]] = {s: [] for s in qs}
        for split in qs:
            pool = qs[split]
            k = min(len(pool) // 4, max(1, P // 5))
            idx = rng.permutation(len(pool))[:k]
            sub[split] = [pool[i] for i in idx]
        for split, items in sub.items():
            for q in items:
                qs[split].remove(q)
                prac.append(q)
        sw.practice = [prac[i] for i in rng.permutation(len(prac))]

    @staticmethod
    def _same_event_seen(sw: SystemWorld, w: WordDef, v: WordDef, c: tuple[int, ...]) -> bool:
        """Is the event behind surface cell c of phrasing w observed in phrasing v?"""
        k = len(c)
        out = int(sw.tables[w.word][c])
        surf = c + (out,)
        canon = [None] * (k + 1)
        for i, cp in enumerate(w.surface):
            canon[cp] = surf[i]
        vs = tuple(canon[v.surface[i]] for i in range(k + 1))
        return vs[:-1] in sw.seen[v.word]


def _words_in(e) -> list[str]:
    if e[0] != "app":
        return []
    out = [e[1]]
    for x in e[2:]:
        out += _words_in(x)
    return out


def _subst(e, val):
    if e[0] == "var":
        return val
    if e[0] == "obj":
        return e
    return ("app", e[1], *[_subst(x, val) for x in e[2:]])


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_stream(cfg: dict, seed: int, noise: float) -> list[SystemWorld]:
    return StreamBuilder(cfg, seed, noise).build()
