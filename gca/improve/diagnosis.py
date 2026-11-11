"""Diagnosis: find the first wrong step, find who is at fault, name the cause, fix it
(plan §7.1-§7.5). Like a student analysing a wrong solution (Ohlsson).

Causes and fixes:
  misread        a lexicon entry is wrong        -> split the word, re-read (alternative entry)
  out_of_scope   a locked column is right on its validated inputs but wrong here
                                                 -> fork it under the cheapest separating condition
  wrong_concept  a column is contradicted inside its validated scope (or no cheap
                 condition separates)            -> supersede it, re-induce with the counterexample
  slip           an observation was noise        -> flag it in mismatch, change nothing

Design note (build log): a trace step's claim is local (f(args) = out given its own
inputs), so one checker call cannot test a prefix of the trace and the plan's binary
search over the trace is not sound. Steps are instead checked in order of suspicion
(internal verification first, then prior faultiness), stopping at the first refuted one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..learn import reading as R
from ..learn.active import choose_query
from ..learn.mdl import FrameData
from ..think.solver import STEP_KINDS, Step
from ..world.api import UNKNOWN, BudgetExceeded
from .hitting_sets import minimal_hitting_sets, rank
from .specialize import find_condition


@dataclass
class Fix:
    id: str
    qid: str | None
    component: str
    cause: str
    action: str
    evidence: dict
    new_cols: list = field(default_factory=list)
    held: int = 0
    failed: int = 0
    status: str = "probation"


class Diagnoser:
    def __init__(self, agent):
        self.a = agent
        self.records: list[dict] = []
        self.fixes: list[Fix] = []

    # ---- entry points ----------------------------------------------------------------
    def feedback(self, cl, q, ans, correct: bool, checker) -> None:
        steps = [s for s in ans.trace if isinstance(s, Step)]
        used = {s.source for s in steps}
        for cid in used:
            col = self.a.spaces.columns.get(cid)
            if col is not None:
                if correct:
                    col.stats.successes += 1
                else:
                    col.stats.failures += 1
        for fx in self.fixes:
            if fx.status == "probation" and used & set(fx.new_cols):
                if correct:
                    fx.held += 1
                    if fx.held >= 2:
                        fx.status = "promoted"
                else:
                    fx.failed += 1
                    self._rollback(cl, fx)
        if correct:
            return
        if ans.value == UNKNOWN:
            self.a.log.log("abstain_feedback", qid=q.qid)
            return
        if not self.a.hp.flags.diagnosis:
            frames = {s.frame for s in steps}
            for F in frames:
                for c in self.a.spaces.for_frame(cl.cid, F):
                    if c.state != "locked" and c.kind == "operator":
                        self.a.retire(c, "re-induce from scratch (no diagnosis)")
            cl._induce_all()
            self.a.log.log("reinduce", qid=q.qid, frames=sorted(frames))
            return
        self.retry(cl, q, ans, steps, checker)

    def retry(self, cl, q, ans, steps, checker, max_attempts: int = 3) -> dict:
        """Like a student: find the wrong step, fix it, then solve the SAME problem again and
        check the new answer. The same wrong answer again, or a new answer the checker refutes,
        starts another diagnosis; up to `max_attempts`, within the budget."""
        a = self.a
        tried = {ans.value}
        rec = self.diagnose(cl, q, ans, steps, checker)
        rec["attempts"] = 1
        for attempt in range(2, max_attempts + 1):
            if rec.get("cause") in (None, "unlocated"):
                break
            a2 = a.solver.answer(cl, q)
            st2 = [x for x in a2.trace if isinstance(x, Step)]
            if a2.value == UNKNOWN:
                rec["retry"] = "abstain"
                break
            if a2.value in tried:
                nxt = self.diagnose(cl, q, a2, st2, checker)     # same mistake: look again
            else:
                if checker.remaining() <= 0:
                    rec["retry"] = "no budget"
                    break
                ok = checker.check_answer(q, a2.value)
                a.stats["queries"] += 1
                if ok:
                    rec["retry"] = "solved"
                    a.stats["retry_solved"] += 1
                    for fx in self.fixes:
                        if fx.id == rec.get("fix"):
                            fx.held += 1
                    break
                tried.add(a2.value)
                nxt = self.diagnose(cl, q, a2, st2, checker)
            nxt["retry_of"] = rec["id"]
            rec = nxt
            rec["attempts"] = attempt
        a.stats["retry_attempts"] += rec.get("attempts", 1)
        a.log.log("retry", qid=q.qid, attempts=rec.get("attempts", 1), outcome=rec.get("retry", "-"))
        return rec

    def diagnose(self, cl, q, ans, steps: list[Step], checker) -> dict:
        a, ctx = self.a, cl.ctx
        rec = {"id": f"dx{len(self.records)}", "qid": q.qid, "answer": ans.value, "steps": len(steps),
               "checks": 0, "first_wrong": None, "cause": "unlocated", "component": None}
        self.records.append(rec)
        # 1. internal verification at no cost
        suspect = []
        for s in steps:
            sus = False
            ev = ctx.majority(s.frame).get(s.canon[:-1])
            if s.source != "event" and ev is not None and ev != s.canon[-1]:
                sus = True
            for c in a.spaces.for_frame(cl.cid, s.frame):
                if c.id != s.source and c.kind == "operator" and c.sem.size > s.cell and int(c.sem[s.cell]) != s.canon[-1]:
                    sus = True
            suspect.append(sus)
        # 2. check steps in order of suspicion until one is refuted
        pri = {"plastic": a.hp.prior_plastic, "candidate": a.hp.prior_candidate, "event": 0.2,
               "locked": a.hp.prior_locked}
        # lesson transfer: causes diagnosed before are suspected first next time (counted)
        causes = ("cause:slip", "cause:misread", "cause:wrong_concept", "cause:out_of_scope")
        boost = {c: 1.0 + 2.0 * a.priors.rate(c, causes, prior=0.25) for c in causes}
        # learned expectations: kinds of reasoning step that failed before are suspected first
        kinds = tuple(f"stepfail:{k}" for k in STEP_KINDS)
        kboost = lambda st: 1.0 + 4.0 * a.priors.rate(f"stepfail:{st.kind}", kinds, prior=0.2)
        order = sorted(range(len(steps)),
                       key=lambda i: (-(pri.get(steps[i].state, 0.1) * (5 if suspect[i] else 1) * kboost(steps[i])), i))
        wrong = None
        for i in order:
            ok = self._check_step(cl, steps[i], checker)
            if ok is None:
                break
            rec["checks"] += 1
            if not ok:
                wrong = i
                break
        if wrong is None:
            cand = [i for i in order if suspect[i]]
            if not cand or rec["checks"] == len(steps):
                a.log.log("diagnose", **rec)
                return rec
            wrong = cand[0]
            rec["unverified"] = True
        s = steps[wrong]
        rec["first_wrong"] = {"word": s.word, "args": s.args, "out": s.out, "source": s.source, "kind": s.kind}
        if not rec.get("unverified"):
            a.priors.observe(f"stepfail:{s.kind}")
        # 3. conflict set and minimal hitting sets
        comps, prior = [], {}
        if s.source == "event":
            comps.append(("event", s.frame, s.canon))
            prior[comps[-1]] = 0.5 * boost["cause:slip"]
        else:
            col = a.spaces.columns[s.source]
            comps.append(("col", s.source))
            prior[comps[-1]] = pri.get(col.state, 0.1) * max(boost["cause:wrong_concept"], boost["cause:out_of_scope"])
        if len(ctx.words_of(s.frame)) > 1:
            lex = next((c for c in a.lexicon_cols(cl) if c.body[1] == s.word and c.active), None)
            comps.append(("lex", s.word))
            low = lex is None or lex.state != "locked" or ctx.lex[s.word].support < 10
            prior[comps[-1]] = (a.hp.prior_lexicon_low if low else 0.01) * boost["cause:misread"]
        for t in s.args:
            comps.append(("name", t))
            prior[comps[-1]] = 0.001
        ranked = rank(minimal_hitting_sets([frozenset(comps)]), prior)
        rec["ranked"] = [(sorted(map(str, d)), round(p, 4)) for d, p in ranked[:4]]
        # 4. name the cause of the most probable faulty component, test, fix
        for d, _p in ranked:
            (comp,) = tuple(d)
            if self._resolve(cl, rec, comp, s, checker):
                break
        a.log.log("diagnose", **{k: v for k, v in rec.items()})
        return rec

    # ---- causes ------------------------------------------------------------------------
    def _resolve(self, cl, rec, comp, s: Step, checker) -> bool:
        kind = comp[0]
        if kind == "name":
            return False
        if kind == "event":
            cl.flag_noise(s.frame, (s.cell, s.canon[-1]), "checker refuted the observed fact (slip)")
            self._fix(cl, rec, f"event:{s.frame}:{s.canon}", "slip", "flag observation as noise", {})
            cl.learn_frame(s.frame)
            return True
        if kind == "lex":
            other = [w for w in cl.ctx.words_of(s.frame) if w != s.word]
            if not other:
                return False
            ok = self._check_canon(cl, other[0], s.canon, checker)
            if ok:
                self.split_word(cl, s.word, rec)
                return True
            return False
        col = self.a.spaces.columns[comp[1]]
        self.column_fault(cl, col, s.cell, s.canon[-1], checker, rec)
        return True

    def column_fault(self, cl, col, cell: int, bad_value: int | None, checker, rec: dict) -> None:
        """A column predicted wrongly at `cell` (the claim F(cell) = bad_value is false)."""
        a = self.a
        F = col.frame
        if col.state != "locked":
            self._fix(cl, rec, col.id, "wrong_concept", "re-induce on the whiteboard with the counterexample", {})
            a.retire(col, "refuted by the checker")
            cl.learn_frame(F)
            self._new_cols_since(cl, F, rec)
            return
        d = cl.data(F)
        inside = cell in col.validated or self._contradicted_inside(col, d)
        if not inside:
            bad = self._refuted_cells(cl, col)
            if cell not in bad:
                bad.append(cell)
            good = sorted(col.validated - set(bad))
            pool = cl.pool(col.arity, F)
            cond = find_condition(pool, good, bad)
            if cond is not None:
                self.fork(cl, col, cond, rec, checker)
                return
        self.supersede(cl, col, cell, rec)

    def counterexample(self, cl, col, cell: int, out: int, checker) -> None:
        """An observation the checker confirmed contradicts a locked column (SPRT path)."""
        rec = {"id": f"dx{len(self.records)}", "qid": None, "via": "sprt", "cause": "unlocated", "component": col.id}
        self.records.append(rec)
        self.column_fault(cl, col, cell, int(col.sem[cell]), checker, rec)
        self.a.log.log("diagnose", **rec)

    def record_slip(self, cl, F: str, tup: tuple, via: str) -> None:
        rec = {"id": f"dx{len(self.records)}", "qid": None, "via": via, "cause": "slip",
               "component": f"event:{F}:{tup}"}
        self.records.append(rec)
        self.a.log.log("diagnose", **rec)

    # ---- fixes ---------------------------------------------------------------------------
    def supersede(self, cl, col, cell: int, rec: dict) -> None:
        a, F = self.a, col.frame
        a.concepts.on_fault(cl, col, f"wrong concept in {cl.cid}")
        a.replace(col, None, f"wrong concept: counterexample at cell {cell}")
        a.spaces.note("refuted", cl.cid, F, a.log.seq, col=col.id, cell=cell)
        for ev in cl.ctx.events.get(F, {}).values():       # slips it had explained away are evidence again
            if ev.noise and not self._checker_noise(cl, F, ev.tup):
                ev.noise = False
        self._fix(cl, rec, col.id, "wrong_concept", "supersede and re-induce with the counterexample", {"cell": cell})
        cl.learn_frame(F)
        self._new_cols_since(cl, F, rec)

    def fork(self, cl, col, cond, rec: dict, checker) -> None:
        a, F = self.a, col.frame
        a.concepts.on_fault(cl, col, f"out of scope in {cl.cid}")
        expr, sem, bits = cond
        fork = a.new_column(cl, F, "operator", col.body, col.sem, col.prior_bits + bits, origin="fork",
                            state="locked", scope=expr, parent=col.id, diagnosis=rec["id"])
        fork.validated = set(col.validated)
        col.state = "forked"
        col.key = None
        col.replaced_by = fork.id
        a.log.log("fork", col=col.id, fork=fork.id, condition=cl.show(expr))
        self._fix(cl, rec, col.id, "out_of_scope", f"fork under condition {cl.show(expr)}", {"bits": bits})
        rec["new_cols"] = [fork.id]
        self._induce_complement(cl, F, sem, rec, checker)

    def _induce_complement(self, cl, F: str, cond_sem: np.ndarray, rec: dict, checker) -> None:
        """New plastic column for the region where the condition is false, induced from the
        failing cases and their neighbours, with a few region-restricted queries."""
        a = self.a
        allowed = cond_sem == 0
        for _ in range(3):
            d = self._restrict(cl.data(F), allowed)
            sc = cl._candidates(F, d)
            if sc is None or len(sc.exprs) == 0:
                return
            vs = sc.version_space(a.hp.delta_vs)
            if len(vs) == 0:
                return
            b = sc.best()
            if len(vs) == 1 or checker.remaining() <= 0:
                break
            pick = choose_query(sc, vs, d.known_cells(), d.n_values, cl.rng, allowed=allowed)
            if pick is None:
                break
            if cl.ask(F, pick[0], int(sc.sem[b][pick[0]]), checker, why="complement") is None:
                break
        fork = a.spaces.columns[rec["new_cols"][0]]
        new = a.new_column(cl, F, "operator", sc.exprs[b], sc.sem[b].astype(np.int64), float(sc.prior[b]),
                           origin="fork-complement", state="plastic", scope=("not", fork.scope),
                           parent=fork.provenance.parent, diagnosis=rec["id"])
        rec["new_cols"].append(new.id)
        for fx in self.fixes:
            if fx.id == rec.get("fix"):
                fx.new_cols = list(rec["new_cols"])

    def split_word(self, cl, w: str, rec: dict) -> None:
        a, ctx = self.a, cl.ctx
        e = ctx.lex[w]
        F = e.frame
        a.spaces.note("lexicon-rejected", cl.cid, F, a.log.seq, word=w, frame=F, sigma=list(e.sigma))
        for c in a.lexicon_cols(cl):
            if c.body[1] == w and c.active:
                a.retire(c, "misread")
        ctx.split(w)
        self._fix(cl, rec, f"lex:{w}", "misread", "split the word and re-read it", {"sigma": e.sigma})
        a.log.log("split", ctx=cl.cid, word=w, frame=F, sigma=e.sigma)

    def misread(self, cl, w: str, lexcol, checker, cell=None, k=None) -> None:
        rec = {"id": f"dx{len(self.records)}", "qid": None, "via": "lexicon-lock", "cause": "unlocated",
               "component": f"lex:{w}"}
        self.records.append(rec)
        self.split_word(cl, w, rec)
        self.a.log.log("diagnose", **rec)

    def _rollback(self, cl, fx: Fix) -> None:
        fx.status = "rolled_back"
        for cid in fx.new_cols:
            col = self.a.spaces.columns.get(cid)
            if col is not None and col.state != "locked":
                self.a.retire(col, f"fix {fx.id} failed on a later problem")
        self.a.spaces.note("rollback", cl.cid, None, self.a.log.seq, fix=fx.id)
        self.a.log.log("rollback", fix=fx.id)

    def _fix(self, cl, rec, component: str, cause: str, action: str, evidence: dict) -> Fix:
        rec["cause"], rec["component"], rec["action"] = cause, component, action
        fx = Fix(f"fix{len(self.fixes)}", rec.get("qid"), component, cause, action, evidence)
        rec["fix"] = fx.id
        self.fixes.append(fx)
        self.a.priors.observe("cause:" + cause)
        self.a.log.log("fix", fix=fx.id, component=component, cause=cause, action=action)
        return fx

    def _new_cols_since(self, cl, F: str, rec: dict) -> None:
        col = cl.active_col(F)
        if col is not None:
            rec["new_cols"] = [col.id]
            for fx in self.fixes:
                if fx.id == rec.get("fix"):
                    fx.new_cols = [col.id]

    # ---- checks --------------------------------------------------------------------------
    def _check_step(self, cl, s: Step, checker) -> bool | None:
        ctx = cl.ctx
        if checker.remaining() <= 0:
            return None
        try:
            ok = checker.check_fact(s.word, tuple(s.args), s.out)
        except BudgetExceeded:
            return None
        args = tuple(ctx.index[t] for t in s.args)
        out = 1 if s.out == "#T" else 0 if s.out == "#F" else ctx.index[s.out]
        ctx.record_check(s.word, args, out, bool(ok), "diagnosis")
        self.a.stats["queries"] += 1
        return bool(ok)

    def _check_canon(self, cl, word: str, canon: tuple, checker) -> bool | None:
        return self.a._ask_word(cl, word, canon, checker, why="diagnosis") if checker.remaining() > 0 else None

    def _refuted_cells(self, cl, col) -> list[int]:
        out = []
        ctx = cl.ctx
        for w, args, o, ok in ctx.checked:
            e = ctx.lex.get(w)
            if ok or e is None or e.frame != col.frame:
                continue
            tup = ctx.canon(e, args + (o,))
            c = R.cell_of(ctx, tup[:-1])
            if c < col.sem.size and int(col.sem[c]) == tup[-1]:
                out.append(c)
        return sorted(set(out))

    def _contradicted_inside(self, col, d: FrameData) -> bool:
        """Internal check at no cost: does the column disagree with any fact inside its
        validated scope (checker facts, or observations it accepted when it locked)?"""
        for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist()):
            if c in col.validated and int(col.sem[c]) != o:
                return True
        for c, o in zip(d.cells.tolist(), d.outs.tolist()):
            if c in col.validated and int(col.sem[c]) != o:
                return True
        return False

    def _checker_noise(self, cl, F: str, tup: tuple) -> bool:
        return any(r.detail.get("tup") == tup and str(r.detail.get("why", "")).startswith("checker")
                   for r in self.a.spaces.mismatches(cl.cid, F, "noise"))

    @staticmethod
    def _restrict(d: FrameData, allowed: np.ndarray) -> FrameData:
        keep = allowed[d.cells] if len(d.cells) else np.zeros(0, bool)
        kp = allowed[d.pos_cells] if len(d.pos_cells) else np.zeros(0, bool)
        kn = allowed[d.neg_cells] if len(d.neg_cells) else np.zeros(0, bool)
        return FrameData(d.arity, d.n_values, d.cells[keep], d.outs[keep], d.weights[keep],
                         d.pos_cells[kp], d.pos_outs[kp], d.neg_cells[kn], d.neg_outs[kn])
