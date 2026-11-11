"""The GCA agent: the loop of plan §1 behind the single Learner protocol.

observe -> read -> learn (with an active query budget) -> answer -> feedback and
diagnosis -> rest. Agent code imports only gca.world.api from the world (enforced by
tests/test_integrity.py).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .config import Hparams
from .eventlog import EventLog
from .improve import sprt
from .improve.concepts import Concepts
from .improve.diagnosis import Diagnoser
from .improve.lifecycle import Lifecycle
from .improve.play import Play
from .improve.value import ValueSignal
from .learn import hypothesis as H
from .learn import reading as R
from .learn.learner import ContextLearner
from .learn.priors import Priors
from .memory.columns import Column, Provenance, column_key
from .memory.spaces import Spaces
from .memory.stores import Context
from .memory.vsa import VSA
from .think.analogy import Analogy
from .think.router import Router
from .think.solver import STEP_KINDS, Solver, Step, is_latin
from .world.api import FALSE, TRUE, UNKNOWN, Answer, CheckerAPI, Observation, Question


NON_DSL = ("table", "map", "lex", "prop")


def eval_scope(scope, X, Y, P) -> np.ndarray:
    """0/1 applicability of a scope condition; ("not", e) nests."""
    if scope[0] == "not":
        return 1 - eval_scope(scope[1], X, Y, P)
    v = H.evaluate(scope, X, Y, P)
    return np.where(v < 0, 0, v).astype(np.int64)


class GCAAgent:
    name = "gca"

    def __init__(self, hp: Hparams | None = None, log: EventLog | None = None):
        self.hp = hp or Hparams()
        f = self.hp.flags
        self.vsa = VSA(self.hp.vsa_dim, self.hp.seed)
        self.priors = Priors(frozen=not f.priors)
        self.spaces = Spaces()
        self.log = log or EventLog()
        self.router = Router(self.vsa, self.hp.vigilance)
        self.solver = Solver(self)
        self.value = ValueSignal(self.hp)
        self.diagnoser = Diagnoser(self)
        self.concepts = Concepts(self)
        self.lifecycle = Lifecycle(self)
        self.play = Play(self)
        self.analogy = Analogy(self) if f.analogy else None
        self.learners: dict[str, ContextLearner] = {}
        self.cur: ContextLearner | None = None
        self.sys_count = 0
        self.noise_found = 0
        self.noise_suspected = 0         # contradictions of locked rules not yet checked
        self.noise_examined = 0
        self.stats: Counter = Counter()
        self.sprt_A, self.sprt_B = sprt.thresholds(self.hp.sprt_alpha, self.hp.sprt_beta)
        self._answers: dict[str, Answer] = {}
        self._n = 0

    # ==== Learner protocol =============================================================
    def start_system(self, system_id: str) -> None:
        cl = ContextLearner(self, system_id, len(self.learners))
        self.learners[system_id] = cl
        self.cur = cl
        self.log.log("start_system", ctx=system_id)

    def observe(self, batch: list[Observation]) -> None:
        for ob in batch:
            cl = self.learners[ob.system]
            ctx = cl.ctx
            if ob.word in ctx.lex and all(t in ctx.index for t in ob.args) and (ob.out in ctx.index or ob.out in (TRUE, FALSE)):
                out = 1 if ob.out == TRUE else 0 if ob.out == FALSE else ctx.index[ob.out]
                cl.observe_check(ob.word, tuple(ctx.index[t] for t in ob.args), out)
            ids = R.ingest(ctx, [ob])
            self.log.log("observe", ctx=ob.system, obs=ids[0], word=ob.word, args=list(ob.args), out=ob.out)

    def learn(self, checker: CheckerAPI) -> None:
        self.cur.learn(checker)

    def answer(self, q: Question) -> Answer:
        cl = self.learners.get(q.system)
        if cl is None:
            return Answer(UNKNOWN, [("unknown context",)], "low")
        a = self.solver.answer(cl, q)
        if self.hp.flags.selfcheck and a.value not in (UNKNOWN, "none", "many"):
            why = self.self_check(cl, a)
            if why:
                self.stats["self_check_abstain"] += 1
                self.log.log("self_check", qid=q.qid, answer=a.value, why=why)
                a = Answer(UNKNOWN, a.trace + [("self-check", why)], "low")
        for s in a.trace:
            col = self.spaces.columns.get(getattr(s, "source", None))
            if col is not None:
                col.stats.uses += 1
                col.stats.last_used = self.sys_count
        self._answers[q.qid] = a
        self.stats["answers"] += 1
        return a

    def feedback(self, q: Question, correct: bool, checker: CheckerAPI) -> None:
        a = self._answers.get(q.qid)
        cl = self.learners.get(q.system)
        if a is None or cl is None:
            return
        self.diagnoser.feedback(cl, q, a, correct, checker)

    def rest(self) -> None:
        cl = self.cur
        if cl is None:
            return
        if self.hp.flags.play:
            self.play.rest(cl)
        self.lifecycle.rest(cl)
        from .improve.concepts import self_test
        self.stats["self_test_warnings"] += self_test(self, cl)
        for col in self.spaces.in_context(cl.cid):
            if col.state == "locked" and col.kind == "operator":
                self.priors.add_body(col.body)
        self.priors.observe("noise:yes" if self.noise_found else "noise:no")
        self._count_readings(cl)
        cl.pools.clear()
        cl.scored.clear()
        self.sys_count += 1
        self.log.log("rest", ctx=cl.cid, locked=len([c for c in self.spaces.in_context(cl.cid) if c.state == "locked"]))

    def _count_readings(self, cl: ContextLearner) -> None:
        """Overhypothesis counts at rest: every pair of phrasings this system tested or read
        as one frame, and whether it ended as a locked reading."""
        ctx = cl.ctx
        pairs = set()
        for why in ctx.checked_why:
            if why.startswith("unify:"):
                w, F, _ = why[6:].split("|")
                pairs.add((w, F))
        for r in self.log.of("unify"):
            if r.get("ctx") == cl.cid:
                pairs.add((r["word"], r["frame"]))
        locked = {(c.body[1], c.frame) for c in self.lexicon_cols(cl) if c.state == "locked"}
        for w, F in sorted(pairs):
            if w not in ctx.lex:
                continue
            k = len(ctx.lex[w].sigma) - 1
            self.priors.observe(f"reading:{k}:" + ("yes" if (w, F) in locked else "no"))
            if (w, F) in locked:
                sg = next(c.body[2] for c in self.lexicon_cols(cl) if c.body[1] == w and c.frame == F)
                self.priors.observe(f"reading-sigma:{k}:" + ",".join(map(str, sg)))

    # ==== helpers used by the learning, thinking and improving modules ==================
    def eta_hat(self) -> float:
        """Running noise estimate: (noise found + prior) / (observations examined + pseudo-count)."""
        return (self.noise_found + self.noise_suspected + self.hp.eps0 * 20) / (self.noise_examined + 20)

    def sprt_step(self, llr: float, ok: bool) -> float:
        return sprt.step(llr, ok, self.eta_hat(), self.hp.sprt_p0, self.hp.sprt_floor)

    @staticmethod
    def token(ctx: Context, frame: str, value: int, out: bool = True) -> str:
        if out and ctx.frames[frame]["bool"]:
            return TRUE if value else FALSE
        return ctx.tokens[value]

    def new_column(self, cl: ContextLearner, F: str, kind: str, body, sem: np.ndarray, prior: float, *,
                   origin: str, state: str, sources: tuple = (), tables: dict | None = None,
                   scope=None, parent: str | None = None, diagnosis: str | None = None) -> Column:
        info = cl.ctx.frames.get(F, {"arity": 0, "bool": False})
        out_type = "B" if info["bool"] else "O"
        self._n += 1
        cid = f"{cl.cid}/{F.split('/')[-1]}#{self._n}"
        calls = []
        for r in sorted(H.refs_used(body)) if body[0] not in NON_DSL else []:
            if isinstance(r, str) and r.startswith("f:"):
                c = cl.active_col(r[2:].rstrip("~"))
                if c is not None:
                    calls.append(c.id)
        col = Column(cid, kind, cl.cid, F, info["arity"], out_type, body, np.asarray(sem, dtype=np.int64),
                     prior, Provenance(cl.cid, origin, self.log.seq, tuple(sources), parent, diagnosis),
                     state=state, scope=scope, calls=tuple(calls), tables=dict(tables or {}))
        col.key = column_key(self.vsa, kind, col.signature(), cl.cid, F).astype(np.int8)
        if scope is not None:
            P = cl.prims(F)
            X, Y = H.grid(cl.ctx.m, info["arity"])
            col.tables["scope"] = eval_scope(scope, X, Y, P)
        self.spaces.add(col)
        self.log.log("sprout", col=cid, ctx=cl.cid, frame=F, kind=kind, state=state, origin=origin,
                     rule=cl.show(body) if body[0] not in ("lex", "prop") else str(body), bits=prior, parent=parent)
        return col

    def replace(self, old: Column, new: Column | None, why: str) -> None:
        old.state = "superseded"
        old.key = None
        old.replaced_by = new.id if new else None
        self.spaces.note("superseded", old.ctx, old.frame, self.log.seq, col=old.id, by=old.replaced_by, why=why)
        self.log.log("supersede", col=old.id, by=old.replaced_by, why=why)
        for c in self.spaces.in_context(old.ctx):
            if old.id in c.calls and c.state != "locked":
                self.retire(c, f"calls superseded {old.id}")

    def retire(self, col: Column, why: str) -> None:
        if not col.active:
            return
        col.state = "retired"
        col.key = None
        self.log.log("retire", col=col.id, why=why)

    def lock(self, cl: ContextLearner, col: Column, d, lc) -> None:
        col.state = "locked"
        col.stats.confirmed = {c for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist()) if int(col.sem[c]) == o}
        # observations the column contradicts are SUSPECTS, not slips: a fact is flagged as
        # noise only once the checker says it is wrong (or, with no budget left, the SPRT)
        m = cl.ctx.m
        mism = [(c, o) for c, o in zip(d.cells.tolist(), d.outs.tolist()) if int(col.sem[c]) != o]
        col.validated = (set(d.cells.tolist()) - {c for c, _ in mism}) | set(d.pos_cells.tolist())
        for c, o in mism:
            cl.suspects[col.id].append((c, o))
            self.noise_suspected += 1
        self.noise_examined += int(d.weights.sum())
        if self.hp.flags.library and col.body[0] not in ("table", "map"):
            s = self.concepts.on_lock(cl, col)
            self.log.log("schema", schema=s.sid, count=s.count, col=col.id)
        self.priors.observe("op:" + col.provenance.origin.split(":")[0])
        if col.body == ("table",) and col.arity == 1:          # overhypothesis counts: one loop?
            from .learn.structure import is_loop
            self.priors.observe("unary:loop:" + ("yes" if is_loop([int(v) for v in col.sem]) else "no"))
        if self.analogy is not None and col.provenance.origin.startswith("analogy:"):
            self.analogy.on_lock(cl, col)
        self.log.log("lock", col=col.id, ctx=cl.cid, frame=col.frame, rule=cl.show(col.body),
                     evidence=lc.as_dict(), m=m)

    def unify(self, cl: ContextLearner, cand) -> None:
        ctx = cl.ctx
        old_frame = ctx.lex[cand.word].frame
        for c in self.spaces.for_frame(cl.cid, old_frame, kind=None):
            self.retire(c, "its word was unified into another frame")
        ctx.unify(cand.word, cand.frame, cand.sigma, cand.support)
        col = self.new_column(cl, cand.frame, "lexicon", ("lex", cand.word, cand.sigma), np.zeros(0), 0.0,
                              origin="reading", state="candidate")
        col.note = cand.word
        self.log.log("unify", ctx=cl.cid, word=cand.word, frame=cand.frame, sigma=cand.sigma,
                     support=cand.support, contra=cand.contra, confirmed=cand.confirmed, lnbf=cand.lnbf)

    def lexicon_cols(self, cl: ContextLearner) -> list[Column]:
        return [c for c in self.spaces.in_context(cl.cid) if c.kind == "lexicon"]

    def cross_evidence(self, cl: ContextLearner, word: str) -> tuple[int, list]:
        """(confirmed cross-phrasing predictions, open predictions) for a unified word."""
        ctx = cl.ctx
        e = ctx.lex[word]
        other = set()
        for tup, ev in ctx.events.get(e.frame, {}).items():
            if not ev.noise and any(ctx.obs[s][0] != word for s in ev.sources):
                other.add(tup)
        conf, asked = 0, set()
        for (w, args, out, ok), why in zip(ctx.checked, ctx.checked_why):
            if w != word:
                continue
            tup = ctx.canon(e, args + (out,))
            asked.add(tup[:-1])
            if ok and tup in other and why == R.reading_tag(word, e.frame, e.sigma):
                conf += 1
        own = {ctx.canon(e, ctx.obs[s][1] + (ctx.obs[s][2],))[:-1] for ev in ctx.events.get(e.frame, {}).values()
               for s in ev.sources if ctx.obs[s][0] == word}
        open_ = [t for t in sorted(other) if t[:-1] not in asked and t[:-1] not in own]
        return conf, open_

    def lexicon_locks(self, cl: ContextLearner, checker) -> None:
        for col in self.lexicon_cols(cl):
            if col.state != "candidate" or not self.hp.flags.locking:
                continue
            w = col.body[1]
            if cl.ctx.lex.get(w) is None or cl.ctx.lex[w].frame != col.frame:
                self.retire(col, "lexicon entry changed")
                continue
            conf, open_ = self.cross_evidence(cl, w)
            k_arity = len(cl.ctx.lex[w].sigma) - 1
            # learned threshold: a kind of reading that has held in past worlds needs fewer checks
            need = 2 if (self.hp.flags.priors and cl.reading_prior(k_arity) >= 0.9) else self.hp.k_lock
            while conf < need and open_ and self.hp.flags.active and checker.remaining() > self.hp.reserve:
                tup = open_.pop(int(cl.rng.integers(len(open_))))
                k = len(tup) - 1
                cell = R.cell_of(cl.ctx, tup[:-1])
                ok = self._ask_word(cl, w, tup, checker, why=R.reading_tag(w, col.frame, col.body[2]))
                if ok is None:
                    break
                conf += int(ok)
                if not ok:
                    # conflict set {this reading, the fact it was predicted from}: test the fact
                    src_ok = cl.verify_event(col.frame, tup, w, checker)
                    if src_ok is False:
                        continue                  # a slip in the source fact; the reading stands
                    self.diagnoser.misread(cl, w, col, checker, cell=cell, k=k)
                    break
            if conf >= need and col.state == "candidate":
                col.state = "locked"
                cl.ctx.lex[w].provisional = False
                self.log.log("lock", col=col.id, ctx=cl.cid, frame=col.frame, rule=f"lexicon {w} {col.body[2]}",
                             evidence={"confirmed": conf})

    def _ask_word(self, cl: ContextLearner, w: str, canon: tuple, checker, why: str = "lexicon-lock") -> bool | None:
        ctx = cl.ctx
        e = ctx.lex[w]
        surf = ctx.surface(e, canon)
        if checker.remaining() <= 0:
            return None
        ok = checker.check_fact(w, tuple(ctx.tokens[i] for i in surf[:-1]), self.token(ctx, e.frame, surf[-1]))
        ctx.record_check(w, tuple(surf[:-1]), surf[-1], bool(ok), why)
        self.stats["queries"] += 1
        self.log.log("query", ctx=cl.cid, word=w, canon=canon, result=bool(ok), why=why)
        return bool(ok)

    def scope_ok(self, cl: ContextLearner, col: Column, cell: int) -> bool:
        if col.scope is None:
            return True
        s = col.tables.get("scope")
        return s is not None and cell < s.size and int(s[cell]) == 1

    def is_latin_frame(self, cl: ContextLearner, F: str) -> bool:
        for c in self.spaces.for_frame(cl.cid, F):
            if c.state in ("locked", "candidate") and c.kind == "operator" and is_latin(c.sem, cl.ctx.m):
                return True
        return False

    def self_check(self, cl: ContextLearner, a: Answer) -> str | None:
        """Check its own work before answering, without the teacher: every step is compared with
        the facts stored for its frame. A step resting on a rule not yet validated that
        contradicts a stored fact, or a kind of step that has often failed before resting on a
        plastic rule, means rethink: abstain rather than answer."""
        kinds = tuple(f"stepfail:{k}" for k in STEP_KINDS)
        seen = sum(self.priors.over[k] for k in kinds)
        for s in a.trace:
            if not isinstance(s, Step) or s.source == "event":
                continue
            col = self.spaces.columns.get(s.source)
            if col is None or col.state == "locked":
                continue
            ev = cl.ctx.majority(s.frame).get(tuple(s.canon[:-1]))
            if ev is not None and ev != s.canon[-1]:
                return f"step {s.word}{s.args} = {s.out} contradicts a stored fact"
            if col.state == "plastic" and seen >= 5 and \
                    self.priors.rate(f"stepfail:{s.kind}", kinds, prior=0.2) >= 0.5:
                return f"{s.kind} step on an unvalidated rule, a kind that often failed before"
        return None

    # ==== reporting ======================================================================
    def rules(self, cid: str) -> list[str]:
        cl = self.learners[cid]
        out = []
        for c in self.spaces.in_context(cid):
            if c.kind == "operator" and c.state in ("locked", "candidate"):
                words = "/".join(cl.ctx.words_of(c.frame)) or c.frame
                out.append(f"[{c.state}] {words}(x, y) = {cl.show(c.body)}" if c.arity == 2
                           else f"[{c.state}] {words}(x) = {cl.show(c.body)}")
        return out
