"""Reading: ingest experience and learn the lexicon (plan §5.1).

Each new word starts with its own provisional frame. Cross-situational role
alignment compares a word's events with another frame's events under every role
permutation sigma. A shared event is support; the same input with a different output
is a contradiction. The decision is a likelihood ratio between "w names roles of
frame F under sigma" and "the overlap is chance" (an agreeing cell has chance 1/V):

    ln BF = s * ln((1 - 2 eta) V) + c * ln(2 eta V / (V - 1)) + q * ln V

where s = shared events, c = contradictions, q = checker-confirmed cross-phrasing
predictions (noiseless). A checker refutation rejects sigma. Unify when s + q >= M_UNIFY
and ln BF >= ln(99) + ln(#hypotheses compared).

Design note (build log): the plan's "Beta lower bound on agreement > 0.9" needs about
28 shared events with no contradiction, which contradicts M3's "unified within <= 6
shared events". The likelihood ratio above meets M3 and keeps the same intent.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..memory.stores import TRUTH, Context
from ..world.api import Observation
from .mdl import FrameData


@dataclass
class UnifyCand:
    word: str
    frame: str
    sigma: tuple[int, ...]
    support: int
    contra: int
    confirmed: int
    refuted: bool
    lnbf: float
    lnpos: float = 0.0          # the positive evidence alone (support + checker confirmations)
    post: float = 0.0           # posterior probability (set by the learner from its priors)
    prio: float = 0.0           # value of testing: posterior x (1 + frames the reading unblocks)


def ingest(ctx: Context, batch: list[Observation]) -> list[int]:
    ids = []
    for ob in batch:
        boolean = ob.out in TRUTH
        args = tuple(ctx.ent(t) for t in ob.args)
        out = ctx.value(ob.out, boolean)
        ctx.add_obs(ob.word, args, out, boolean)
        ids.append(len(ctx.obs) - 1)
    return ids


def n_values(ctx: Context, frame: str) -> int:
    return 2 if ctx.frames[frame]["bool"] else max(ctx.m, 2)


def cell_of(ctx: Context, args: tuple[int, ...]) -> int:
    return args[0] * ctx.m + args[1] if len(args) == 2 else args[0]


def canonical_fact(ctx: Context, word: str, args: tuple[int, ...], out: int) -> tuple[str, tuple[int, ...]]:
    e = ctx.lex[word]
    return e.frame, ctx.canon(e, args + (out,))


def frame_data(ctx: Context, frame: str) -> FrameData:
    """Observed (non-noise) events plus checker facts, on canonical cells."""
    info = ctx.frames[frame]
    k = info["arity"]
    cells, outs, w = [], [], []
    for tup, ev in ctx.events.get(frame, {}).items():
        if ev.noise or len(tup) != k + 1:
            continue
        cells.append(cell_of(ctx, tup[:-1]))
        outs.append(tup[-1])
        w.append(len(ev.sources))
    pc, po, nc, no = [], [], [], []
    for word, args, out, ok in ctx.checked:
        if ctx.lex.get(word) is None or ctx.lex[word].frame != frame:
            continue
        _, tup = canonical_fact(ctx, word, args, out)
        if any(v >= ctx.m for v in tup[:-1]):
            continue
        (pc if ok else nc).append(cell_of(ctx, tup[:-1]))
        (po if ok else no).append(tup[-1])
    a = lambda x: np.array(x, dtype=np.int64)
    return FrameData(k, n_values(ctx, frame), a(cells), a(outs), np.array(w, dtype=np.float64),
                     a(pc), a(po), a(nc), a(no))


def _sigmas(arity: int, boolean: bool):
    if boolean:
        for p in itertools.permutations(range(arity)):
            yield tuple(p) + (arity,)
    else:
        yield from itertools.permutations(range(arity + 1))


def _map(sigma: tuple[int, ...], surf: tuple[int, ...]) -> tuple[int, ...]:
    c = [0] * len(surf)
    for i, cp in enumerate(sigma):
        c[cp] = surf[i]
    return tuple(c)


def reading_tag(word: str, frame: str, sigma: tuple) -> str:
    """The purpose tag of a check asked to test one specific reading. Only such checks
    count as its confirmations: checks asked for anything else are selected by other
    criteria and would bias the agreement rate."""
    return f"unify:{word}|{frame}|{','.join(map(str, sigma))}"


def _cell_args(ctx: Context, cell: int, arity: int) -> tuple:
    return (cell // ctx.m, cell % ctx.m) if arity == 2 else (cell,)


def _known(ctx: Context, F: str, tables: dict | None) -> dict[tuple, set]:
    """What frame F says: its column's predictions when it has one (knowledge reads new
    words), otherwise its observed (non-noise) events."""
    k = ctx.frames[F]["arity"]
    tab = (tables or {}).get(F)
    if tab is not None and tab.size == ctx.m ** k:
        return {_cell_args(ctx, c, k): {int(tab[c])} for c in range(tab.size)}
    theirs: dict[tuple, set] = {}
    for t, ev in ctx.events.get(F, {}).items():
        if not ev.noise:
            theirs.setdefault(t[:-1], set()).add(t[-1])
    return theirs


def candidates(ctx: Context, eta: float, tables: dict | None = None) -> list[UnifyCand]:
    """Every (word with its own frame, other frame, sigma) with some overlap. A word whose
    frame already has a locked rule has a validated meaning and is not re-read (two locked
    rules naming one function are merged by the lifecycle instead)."""
    out = []
    own = [w for w, e in ctx.lex.items() if ctx.words_of(e.frame) == [w] and e.provisional
           and e.sigma == tuple(range(len(e.sigma))) and e.frame not in (tables or {})]
    for w in own:
        fw = ctx.lex[w].frame
        info = ctx.frames[fw]
        mine = [t for t, ev in ctx.events.get(fw, {}).items() if not ev.noise]
        for F, finfo in ctx.frames.items():
            if F == fw or finfo["arity"] != info["arity"] or finfo["bool"] != info["bool"]:
                continue
            if not ctx.words_of(F):
                continue
            # "w into F" and "F's word into w's frame" are one hypothesis when both words still
            # have their own frames: consider it once, from the later word
            fws = ctx.words_of(F)
            if len(fws) == 1 and fws[0] in own and fws[0] > w:
                continue
            theirs = _known(ctx, F, tables)
            settled_f = set()
            for word2, a2, o2, ok2 in ctx.checked:
                e2 = ctx.lex.get(word2)
                if ok2 and e2 is not None and e2.frame == F:
                    settled_f.add(ctx.canon(e2, a2 + (o2,))[:-1])
            V = n_values(ctx, F)
            for sg in _sigmas(info["arity"], info["bool"]):
                s = c = 0
                for t in mine:
                    ct = _map(sg, t)
                    o = theirs.get(ct[:-1])
                    if o is None:
                        continue
                    if ct[-1] in o:
                        s += 1
                    else:
                        c += 1
                q, bad = 0, False
                for (word, args, outv, ok), why in zip(ctx.checked, ctx.checked_why):
                    if word != w:
                        continue
                    ct = _map(sg, args + (outv,))
                    o = theirs.get(ct[:-1])
                    if o is None:
                        continue
                    agree = ct[-1] in o
                    if ok and agree:
                        # only predictions asked AS tests of a reading count as confirmations:
                        # facts chosen for other reasons (e.g. query by committee) are selected
                        # where hypotheses disagree, which biases the agreement rate
                        q += why == reading_tag(w, F, sg)
                    elif ok != agree:
                        # a hard refutation needs the other side to be settled too (a checker
                        # fact or a locked rule); against an unverified observation, which may
                        # itself be a slip, the failure is a soft contradiction
                        if ct[:-1] in settled_f or F in (tables or {}):
                            bad = True
                        else:
                            c += 1
                # the reading is symmetric: tests of the other frame's words predicted from
                # this word's observations count too
                mapped = {_map(sg, t) for t in mine}
                for (word, args, outv, ok), why in zip(ctx.checked, ctx.checked_why):
                    if not ok or why != reading_tag(w, F, sg) or word == w or word not in ctx.lex or ctx.lex[word].frame != F:
                        continue
                    if ctx.canon(ctx.lex[word], args + (outv,)) in mapped:
                        q += 1
                if s + q == 0:
                    # no agreement yet is still a testable hypothesis when the two phrasings
                    # are complementary: together they cover (almost) the whole domain. Any
                    # contradictions then enter the likelihood ratio like everywhere else.
                    if bad or not mine or not theirs:
                        continue
                    covered = {a + (o,) for a, os_ in theirs.items() for o in os_} | {_map(sg, t) for t in mine}
                    if len(covered) < 0.9 * max(ctx.m, 1) ** info["arity"]:
                        continue
                e2 = min(max(2 * eta, 1e-3), 0.5)
                pos = s * math.log(max((1 - e2) * V, 1.0001)) + q * math.log(V)
                lnbf = pos + c * math.log(e2 * V / (V - 1))
                out.append(UnifyCand(w, F, sg, s, c, q, bad, -math.inf if bad else lnbf, pos))
    return out


def threshold(cands: list[UnifyCand], ln_threshold: float) -> float:
    """ln(99) plus a correction for the number of readings compared."""
    return ln_threshold + math.log(max(len(cands), 1))


def decide(cands: list[UnifyCand], m_unify: int, ln_threshold: float, use_post: bool = False,
           min_post: float = 0.99) -> list[UnifyCand]:
    """Accepted unifications: at most one per word, best evidence first. With learned priors
    (use_post) a reading is also accepted when its posterior reaches 0.99 on at least one piece
    of evidence: lessons from past worlds lower the evidence needed (learned thresholds)."""
    if not cands:
        return []
    thr = threshold(cands, ln_threshold)
    acc, used = [], set()
    for c in sorted(cands, key=lambda c: (-c.lnbf, c.word, c.frame, c.sigma)):
        if c.word in used or c.refuted:
            continue
        if c.support + c.confirmed >= m_unify and c.lnbf >= thr:
            acc.append(c)
            used.add(c.word)
        elif use_post and c.post >= min_post and c.support + c.confirmed >= 1:
            acc.append(c)
            used.add(c.word)
    return acc


def n_sigmas(arity: int, boolean: bool) -> int:
    return math.factorial(arity) if boolean else math.factorial(arity + 1)


def posterior(c: UnifyCand, prior_pair: float, n_sig: int, p_sigma: float | None = None) -> float:
    """P(this reading | evidence): the pair's prior (an overhypothesis counted from earlier
    readings of the same arity) times P(role order | pair) (counted too; uniform 1/n_sig before
    any evidence), updated by ln BF."""
    ps = (1.0 / max(n_sig, 1)) if p_sigma is None else p_sigma
    p = min(max(prior_pair * ps, 1e-6), 1 - 1e-6)
    if c.lnbf == -math.inf:
        return 0.0
    z = c.lnbf + math.log(p / (1 - p))
    if z < -700:
        return 0.0
    if z > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def pending(cands: list[UnifyCand], m_unify: int, ln_threshold: float, eta: float = 0.05,
            prior=None, min_post: float = 0.05) -> list[UnifyCand]:
    """Promising but undecided, below the threshold: no contradiction, or (under noise)
    few contradictions against positive evidence. Checker confirmations count as positive
    evidence (the checker is reality, P1), so a reading confirmed by the checker but
    contradicted by one observation is investigated, not dropped."""
    done = {c.word for c in decide(cands, m_unify, ln_threshold)}
    thr = threshold(cands, ln_threshold)
    # a three-way sequential test: accept at ln BF >= thr (decide), reject at ln BF <= -thr,
    # investigate in between. Under heavy noise a contradiction is weak evidence, so the
    # likelihood ratio itself keeps a contradicted reading open for testing.
    allow = lambda c: c.lnbf > -thr
    out = [c for c in sorted(cands, key=lambda c: (c.contra > 0, -c.lnbf, c.word, c.frame, c.sigma))
           if c.word not in done and not c.refuted and allow(c)]
    if prior is not None:
        # value of testing: only readings that are still plausible given their prior
        out = [c for c in out if c.prio >= min_post]
        out.sort(key=lambda c: (-c.prio, c.word, c.frame, c.sigma))
    return out


def contradictions(ctx: Context, cand: UnifyCand) -> list[tuple[tuple, tuple]]:
    """Pairs (word's own event, other frame's event) that disagree under sigma."""
    fw = ctx.lex[cand.word].frame
    theirs: dict[tuple, list] = {}
    for t, ev in ctx.events.get(cand.frame, {}).items():
        if not ev.noise:
            theirs.setdefault(t[:-1], []).append(t)
    out = []
    for t, ev in ctx.events.get(fw, {}).items():
        if ev.noise:
            continue
        ct = _map(cand.sigma, t)
        for u in theirs.get(ct[:-1], []):
            if u[-1] != ct[-1]:
                out.append((t, u))
    return out


def prediction(ctx: Context, cand: UnifyCand, rng: np.random.Generator, tables: dict | None = None):
    """A fact of `cand.word` predicted from the other frame (its column, or its events)
    and not yet observed or checked. Returns (word, surface args, surface out) or None."""
    fw = ctx.lex[cand.word].frame
    seen = set(ctx.events.get(fw, {}))
    seen |= {a + (o,) for w, a, o, _ in ctx.checked if w == cand.word}
    opts = []
    for args, outs in sorted(_known(ctx, cand.frame, tables).items()):
        if len(outs) != 1:
            continue
        t = args + (next(iter(outs)),)
        surf = tuple(t[cand.sigma[i]] for i in range(len(t)))
        if surf not in seen and not any(s[:-1] == surf[:-1] for s in seen):
            opts.append(surf)
    if not opts:
        # the other direction: a fact of the frame's own word, predicted from this word's events
        others = [u for u in ctx.words_of(cand.frame)]
        if not others:
            return None
        u = others[0]
        eu = ctx.lex[u]
        known_f = set(ctx.events.get(cand.frame, {}))
        asked = {ctx.canon(eu, a + (o,))[:-1] for w2, a, o, _ in ctx.checked if w2 == u}
        for t in sorted(ctx.events.get(fw, {})):
            if ctx.events[fw][t].noise:
                continue
            ct = _map(cand.sigma, t)
            if ct in known_f or ct[:-1] in asked or any(k[:-1] == ct[:-1] for k in known_f):
                continue
            opts.append(ctx.surface(eu, ct))
        if not opts:
            return None
        surf = opts[int(rng.integers(len(opts)))]
        return u, surf[:-1], surf[-1]
    surf = opts[int(rng.integers(len(opts)))]
    return cand.word, surf[:-1], surf[-1]
