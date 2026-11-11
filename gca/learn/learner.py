"""Learning inside one context (one system): read, induce, query, validate, lock.

One column per frame (build log). A frame's column maps canonical args to the
canonical out; every phrasing is answered through the frame. A fully observed unary
frame becomes a table column, and every active (candidate or locked) column becomes a
local primitive that other columns may call (PNN-style lateral links via `calls`).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..memory.columns import Column
from ..memory.stores import Context
from ..world.api import BudgetExceeded
from . import hypothesis as H
from . import reading as R
from .active import choose_query
from .enumerate import Pool
from .induce import pool_part, schema_instances, score, splits
from .linking import linked_part
from .mdl import FrameData, adequate, null_bits, sprout_threshold
from .structure import best_question, is_loop, is_perm, loop_repairs
from .validate import lock_check


class ContextLearner:
    def __init__(self, agent, cid: str, sys_index: int):
        self.a = agent
        self.cid = cid
        self.ctx = Context(cid)
        self.sys_index = sys_index
        self.pools: dict = {}
        self.scored: dict = {}
        self.gain: dict = {}
        self.analog: dict = {}               # frame -> induce.Extra from analogy
        self.m_seen = 0
        self.errors: dict[str, list[int]] = defaultdict(list)
        self.suspects: dict[str, list] = defaultdict(list)    # column id -> contradicting obs
        self.rng = np.random.default_rng([agent.hp.seed, sys_index, 17])
        self.analogy_state: dict = {}
        self.inadequate: dict[str, bool] = {}     # frame -> best fit implausible under the noise model
        self.linked_cache: dict = {}              # linked parts derived from earlier concepts
        self.linked_names: dict[str, str] = {}

    # ---- helpers ----------------------------------------------------------------
    @property
    def hp(self):
        return self.a.hp

    @property
    def flags(self):
        return self.a.hp.flags

    def out_type(self, frame: str) -> str:
        return "B" if self.ctx.frames[frame]["bool"] else "O"

    def active_col(self, frame: str) -> Column | None:
        cols = self.a.spaces.for_frame(self.cid, frame)
        return cols[0] if cols else None

    def data(self, frame: str) -> FrameData:
        d = R.frame_data(self.ctx, frame)
        if not self.flags.mismatch:          # ablation: forget negative evidence
            d.neg_cells = d.neg_cells[:0]
            d.neg_outs = d.neg_outs[:0]
        return d

    def eps(self) -> float:
        return max(self.hp.eps_floor, self.a.eta_hat())

    def ref_name(self, ref: str) -> str:
        if ref.startswith("link:"):
            return "linked[" + self.linked_names.get(ref, ref) + "]"
        base = ref.rstrip("~")
        inv = ref.endswith("~")
        frame = base[2:] if base.startswith("f:") else base
        words = self.ctx.words_of(frame)
        name = words[0] if words else frame.split("/")[-1]
        return f"{name}^-1" if inv else name

    def show(self, e) -> str:
        if e[0] == "not":
            return "not " + self.show(e[1])
        if e[0] == "map":
            src = self.a.spaces.columns.get(e[1])
            if src is not None and src.ctx in self.a.learners:
                return f"analogy to {src.ctx}: {self.a.learners[src.ctx].show(src.body)}"
        return H.show(e, names=self.ref_name, ent=lambda i: self.ctx.tokens[i] if isinstance(i, int)
                      and i < len(self.ctx.tokens) else str(i))

    # ---- primitives and pools ---------------------------------------------------
    def prims(self, exclude_frame: str | None = None) -> H.Prims:
        P = H.Prims(self.ctx.m)
        m = self.ctx.m
        # library first: locked columns are primitives (candidates too when locking is ablated)
        ok_states = ("locked",) if self.flags.locking else ("locked", "candidate")
        for col in self.a.spaces.in_context(self.cid):
            if col.state not in ok_states or col.frame == exclude_frame or col.kind != "operator":
                continue
            if col.scope is not None:
                continue
            if col.sem.size not in (m, m * m) or (col.sem < 0).any():
                continue
            if col.arity == 1:
                P.add_unary(col.ref, col.sem)
                if len(set(col.sem.tolist())) == m:              # bijection: both directions are views
                    inv = np.empty(m, dtype=np.int64)
                    inv[col.sem.astype(np.int64)] = np.arange(m)
                    P.add_unary(col.ref + "~", inv)
            elif col.out_type == "O":
                P.add_binary(col.ref, col.sem.reshape(m, m))
            else:
                P.add_rel(col.ref, col.sem.reshape(m, m))
        return P

    def pool(self, arity: int, exclude_frame: str | None) -> Pool:
        P = self.prims(exclude_frame)
        key = (arity, P.signature(), self.a.priors.version)
        p = self.pools.get(key)
        if p is None:
            hp = self.hp
            p = Pool(P, arity, self.a.priors, s_enum=hp.s_enum, caps={"O": hp.cap_O, "N": hp.cap_N, "B": hp.cap_B},
                     combo_budget=hp.combo_budget, node_budget=hp.node_budget)
            self.a.stats["search_nodes"] += p.nodes
            self.a.stats["pools_built"] += 1
            self.pools[key] = p
        return p

    # ---- the learn step -----------------------------------------------------------
    def learn(self, checker) -> None:
        self._check_domain()
        self._sprt_decisions(checker)
        for _ in range(4):                    # new locks give new primitives and readings: learn again
            locked = self._n_locked()
            if self.flags.lexicon_unify:
                self._unify(checker)
            self._induce_all()
            if self.flags.analogy and self.a.analogy is not None:
                if self.a.analogy.attempt(self, checker):
                    self._induce_all()
            if self.flags.active:
                self._queries(checker)
            if self._n_locked() == locked:
                break
        if self.flags.active:
            self._verify_suspects(checker)

    def _n_locked(self) -> int:
        return sum(1 for c in self.a.spaces.in_context(self.cid) if c.state == "locked")

    def _induce_all(self, rounds: int = 4) -> None:
        for _ in range(rounds):
            changed = False
            for F in self.frames_by_value():
                changed |= self.learn_frame(F)
            if not changed:
                break

    def frames_by_value(self) -> list[str]:
        fs = [F for F in self.ctx.frames if self.ctx.words_of(F)]
        return sorted(fs, key=lambda F: (-self.a.value.value(self, F), F))

    def learn_frame(self, F: str) -> bool:
        cur = self.active_col(F)
        if cur is not None and cur.state == "locked":
            return False
        info = self.ctx.frames[F]
        d = self.data(F)
        if d.n_obs == 0 and len(d.pos_cells) == 0:
            return False
        if info["arity"] == 1 and not info["bool"]:
            tab = self._full_table(d)
            if tab is not None:
                return self._table_column(F, d, tab, cur)
        sc = self._candidates(F, d)
        if sc is None or len(sc.exprs) == 0:
            return False
        b = sc.best()
        total = float(sc.total[b])
        gain = null_bits(d) - total
        self.scored[F], self.gain[F] = sc, gain
        delta = sprout_threshold(self.a.sys_count, self.hp.sprout_start, self.hp.sprout_end, self.hp.sprout_over)
        recognized = self._recognized_map(sc, b) if np.isfinite(total) and gain < delta else False
        if not np.isfinite(total) or (gain < delta and not recognized):
            if cur is not None and not sc.alive.any():
                self.a.retire(cur, "refuted, nothing compresses the data")
                return True
            if cur is not None and cur.state != "locked" and np.isfinite(total)                     and not np.array_equal(cur.sem, sc.sem[b]):
                self.a.retire(cur, "no longer the best hypothesis")      # e.g. a recognition data overturned
                return True
            return False
        vs = sc.version_space(self.hp.delta_vs)
        fit = adequate(float(sc.wrong[b]), d.n_obs, self.a.eta_hat())
        self.inadequate[F] = not fit
        state = "candidate" if len(vs) == 1 and fit else "plastic"
        if cur is not None and np.array_equal(cur.sem, sc.sem[b]):
            if cur.state != state:
                self.a.spaces.set_state(cur, state)
                return True
            return False
        body = sc.exprs[b]
        new = self.a.new_column(self, F, "operator", body, sc.sem[b].astype(np.int64), float(sc.prior[b]),
                                origin=sc.origin[b], state=state, sources=tuple(self._sources(F)))
        if cur is not None:
            self.a.replace(cur, new, "a better hypothesis fits the data")
        return True

    def _candidates(self, F: str, d: FrameData):
        info = self.ctx.frames[F]
        k, ot = info["arity"], self.out_type(F)
        pool = self.pool(k, F)
        parts = [pool_part(pool, ot, self.a.priors)]
        if self.flags.split and ot == "O":
            tol = max(1, int(np.ceil(2 * self.eps() * max(len(d.cells), 1))))
            parts.append(splits(pool, d, self.a.priors, tol=tol))
        if self.flags.library:
            parts.append(schema_instances(pool.P, k, ot, self.a.priors, exclude={"f:" + F, "f:" + F + "~"},
                                          cap=self.hp.schema_cap))
        if F in self.analog:
            parts.append(self.analog[F])
        cur = self.active_col(F)
        if (self.flags.library and k == 2 and ot == "O" and d.n_obs >= 10
                and (cur is None or cur.state != "locked")):
            parts.append(linked_part(self, F, d, pool.P))      # link earlier submodels
        return score(d, self.eps(), parts)

    def _sources(self, F: str) -> list[int]:
        return sorted({s for ev in self.ctx.events.get(F, {}).values() for s in ev.sources})

    # ---- fully observed unary frames: table columns ---------------------------------
    def _full_table(self, d: FrameData) -> np.ndarray | None:
        m = self.ctx.m
        votes: dict[int, dict[int, float]] = defaultdict(dict)
        for c, o, w in zip(d.cells.tolist(), d.outs.tolist(), d.weights.tolist()):
            votes[c][o] = votes[c].get(o, 0) + w
        for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist()):
            votes[c] = {o: 1e9}
        bad = {(c, o) for c, o in zip(d.neg_cells.tolist(), d.neg_outs.tolist())}
        tab = np.full(m, -1, dtype=np.int64)
        for c in range(m):
            v = {o: w for o, w in votes.get(c, {}).items() if (c, o) not in bad}
            if not v:
                return None
            top = max(v.values())
            best = [o for o, w in v.items() if w == top]
            if len(best) > 1:
                return None
            tab[c] = best[0]
        return tab

    def _table_column(self, F: str, d: FrameData, tab: np.ndarray, cur: Column | None) -> bool:
        if cur is not None and np.array_equal(cur.sem, tab):
            return False
        m = self.ctx.m
        bits = m * np.log2(max(m, 2))             # lookup_T costs |T| log2 n (plan §5.2)
        new = self.a.new_column(self, F, "operator", ("table",), tab, float(bits), origin="table",
                                state="candidate", sources=tuple(self._sources(F)),
                                tables={("table",): tab})
        if cur is not None:
            self.a.replace(cur, new, "table changed")
        return True

    # ---- queries --------------------------------------------------------------------
    def ask(self, F: str, cell: int, value: int, checker, why: str) -> bool | None:
        """Check the canonical claim F(cell) = value with the checker, through a word of F."""
        words = self.ctx.words_of(F)
        if not words or checker.remaining() <= 0:
            return None
        w = words[0]
        e = self.ctx.lex[w]
        k = self.ctx.frames[F]["arity"]
        args = (cell // self.ctx.m, cell % self.ctx.m) if k == 2 else (cell,)
        canon = args + (value,)
        surf = self.ctx.surface(e, canon)
        try:
            ok = checker.check_fact(w, tuple(self.ctx.tokens[i] for i in surf[:-1]),
                                    self.a.token(self.ctx, F, surf[-1], out=True))
        except BudgetExceeded:
            return None
        self.ctx.record_check(w, tuple(surf[:-1]), surf[-1], bool(ok), why)
        self.a.log.log("query", ctx=self.cid, frame=F, word=w, cell=cell, value=value, result=bool(ok), why=why)
        self.a.stats["queries"] += 1
        return bool(ok)

    def confirmed_new(self, col: Column, d: FrameData) -> int:
        """Checker-verified correct predictions on inputs not in experience."""
        observed = set(d.cells.tolist())
        n = 0
        for c, o in zip(d.pos_cells.tolist(), d.pos_outs.tolist()):
            if (c not in observed or col.body == ("table",)) and int(col.sem[c]) == o:
                n += 1
        return n

    def _settle_unary(self, F: str, checker) -> None:
        """Settle a unary object frame under the overhypothesis 'unary frames are bijections'
        by checking facts in order of suspicion until it can lock:
          1. conflicts (phrasings disagree on a cell): one fact is a slip;
          2. collisions (two inputs claim one output): one unverified fact is a slip;
          3. one empty cell whose bijection leaves a single candidate: one query decides;
             if the checker refuted every candidate, an unverified known fact is a slip;
          4. unverified single-source facts, until (unsettled cells) x eta_hat <= 0.1.
        A refuted observation is a checker-verified slip; nothing is flagged on assumption."""
        info = self.ctx.frames[F]
        if info["arity"] != 1 or info["bool"] or not self.flags.active:
            return
        col = self.active_col(F)
        if col is not None and col.state == "locked":
            return
        others = [G for G, gi in self.ctx.frames.items() if G != F and gi["arity"] == 1 and not gi["bool"]
                  and self.ctx.words_of(G)]
        if others and self.flags.lexicon_unify:
            return                    # another phrasing is still to be read into this frame
        m = self.ctx.m
        for _ in range(3 * m):
            if checker.remaining() <= self.hp.reserve:
                return
            d = self.data(F)
            neg = set(zip(d.neg_cells.tolist(), d.neg_outs.tolist()))
            pos = dict(zip(d.pos_cells.tolist(), d.pos_outs.tolist()))
            votes: dict[int, dict] = defaultdict(dict)
            for c, o, w in zip(d.cells.tolist(), d.outs.tolist(), d.weights.tolist()):
                if (c, o) not in neg and c not in pos:
                    votes[c][o] = votes[c].get(o, 0) + w
            settled = set(pos) | {c for c, v in votes.items() if len(v) == 1 and sum(v.values()) >= 2}
            known = dict(pos)
            known.update({c: next(iter(v)) for c, v in votes.items() if len(v) == 1})
            act = None
            conflict = [c for c, v in votes.items() if len(v) > 1]
            # 0. structure first (one loop): propose the corrected fact the minimal repairs
            #    agree on, and ask about it once
            empty0 = [c for c in range(m) if c not in known]
            single = {c: v for c, v in known.items() if c not in conflict}
            table = [single.get(c, -1) for c in range(m)]
            cycle = self.a.priors.rate("unary:loop:yes", ("unary:loop:yes", "unary:loop:no"), prior=0.5) >= 0.5
            healthy = not conflict and not empty0 and (is_loop(table) if cycle else is_perm(table))
            if not healthy and len(empty0) + len(conflict) <= 3:
                wts = {c: sum(votes[c].values()) for c in votes}
                k, sols = loop_repairs(m, single, wts, settled, neg, empty0 + conflict, cycle=cycle)
                q = best_question(sols, single)
                if q is not None:
                    act = (q[0], q[1], "loop-repair")
            if act is None and conflict:
                c = conflict[0]
                act = (c, max(sorted(votes[c]), key=lambda o: votes[c][o]), "conflict")
            if act is None:
                by_out: dict[int, list] = defaultdict(list)
                for c, o in known.items():
                    by_out[o].append(c)
                for cs in by_out.values():
                    open_ = [c for c in cs if c not in settled]
                    if len(cs) > 1 and open_:
                        act = (open_[0], known[open_[0]], "collision")
                        break
            empty = [c for c in range(m) if c not in known]
            used = set(known.values())
            free_vals = [v for v in range(m) if v not in used]
            # complete only when the known part is injective and the free values match the
            # empty cells (otherwise the domain may still hold unseen entities)
            if act is None and empty and len(empty) <= 3 and len(used) == len(known) and len(free_vals) == len(empty):
                cand = {c: [v for v in free_vals if (c, v) not in neg] for c in empty}
                c = min(empty, key=lambda c: (len(cand[c]), c))
                if cand[c]:
                    act = (c, cand[c][0], "complete")
                else:
                    # no candidate left: a slip among unverified facts, or an entity not yet
                    # seen. Repair only when a slip is the likelier explanation.
                    unv = [x for x in known if x not in settled]
                    if unv and len(unv) * self.a.eta_hat() >= 0.5:
                        act = (unv[0], known[unv[0]], "conflict-repair")
            if act is None and not empty:
                unsettled = sorted((c for c in known if c not in settled), key=lambda c: (sum(votes[c].values()), c))
                table = [known.get(c, -1) for c in range(m)]
                need_more = not self._table_safe(len(unsettled), table) or len(settled) < min(self.hp.k_lock, m)
                if unsettled and need_more:
                    act = (unsettled[0], known[unsettled[0]], "lock-table")
            if act is None:
                return
            c, v, why = act
            ok = self.ask(F, c, v, checker, why=why)
            if ok is None:
                return
            if not ok and v in votes.get(c, {}):
                self.flag_noise(F, (c, v), f"checker refuted an observed fact ({why})")
            if ok and why in ("conflict", "loop-repair"):
                for o2 in list(votes.get(c, {})):
                    if o2 != v:     # the checker confirmed a different output for this input
                        self.flag_noise(F, (c, o2), f"checker confirmed the corrected fact ({why})")

    def reading_prior(self, arity: int) -> float:
        """Overhypothesis: how often a tested pair of phrasings of this arity turned out to
        name one frame (counted at rest from earlier systems; 0.5 before any evidence)."""
        return self.a.priors.rate(f"reading:{arity}:yes", (f"reading:{arity}:yes", f"reading:{arity}:no"),
                                  prior=0.5)

    def sigma_prior(self, arity: int, boolean: bool, sigma: tuple) -> float:
        """P(role order | the pair is one frame), counted over earlier locked readings."""
        n = R.n_sigmas(arity, boolean)
        if not self.flags.priors:
            return 1.0 / n
        key = f"reading-sigma:{arity}:"
        tot = sum(v for k, v in self.a.priors.over.items() if k.startswith(key))
        return (self.a.priors.over[key + ",".join(map(str, sigma))] + 1.0) / (tot + n)

    def _allowed_unsettled(self) -> int:
        return int(0.1 / max(self.a.eta_hat(), 1e-6))

    def _table_safe(self, unsettled: int, table: list[int]) -> bool:
        """May a unary table lock with `unsettled` unverified facts? If the table is a
        bijection, hidden errors survive only if the wrong values happen to permute among
        themselves: P(wrong and consistent) ~ sum_{k>=2} C(u,k) eta^k (1-eta)^(u-k) k!/(m-1)^k
        (a single error always breaks it). Lock when that is <= 0.001. Otherwise fall back to
        (unsettled x eta_hat) <= 0.1."""
        m, eta = self.ctx.m, self.a.eta_hat()
        if not is_perm(table):
            # under the learned structure (unary frames are bijections), a table that is not
            # one is itself evidence of an unverified slip: it never locks
            if self.a.priors.rate("unary:loop:yes", ("unary:loop:yes", "unary:loop:no"), prior=0.5) >= 0.5:
                return False
            return unsettled <= self._allowed_unsettled()
        p = sum(math.comb(unsettled, k) * eta ** k * (1 - eta) ** (unsettled - k)
                * math.factorial(k) / max(m - 1, 1) ** k for k in range(2, unsettled + 1))
        return p <= 1e-3

    def _queries(self, checker) -> None:
        hp = self.hp
        # 1. unary frames: settle them by checking facts in order of suspicion, then lock
        for F in self.frames_by_value():
            self._settle_unary(F, checker)
            self.learn_frame(F)
            self.try_lock(F, checker)
        # 1b. slip tests: a frame whose best hypothesis would compress if its few mismatching
        #     observations were noise: ask about those observations (slip, or a refutation)
        for F in self.frames_by_value():
            self._slip_tests(F, checker)
        # 2. operator columns: lock checks on collapsed version spaces first (a lock gives
        #    other frames a new primitive), then query by committee by value
        order = self.frames_by_value()
        order.sort(key=lambda F: 0 if (c := self.active_col(F)) is not None and c.state == "candidate" else 1)
        for F in order:
            self._recognize_verify(F, checker)
        for F in order:
            # act on its own: keep asking while a question is worth it (value of information),
            # instead of a fixed number per batch of examples; stop at the reserve
            for _ in range(hp.max_queries_per_learn):
                col = self.active_col(F)
                sc = self.scored.get(F)
                if col is None or col.state == "locked" or col.body == ("table",) or sc is None:
                    break
                if self.inadequate.get(F):
                    break             # the concept is not in reach yet: do not spend the budget
                if checker.remaining() <= hp.reserve:
                    break
                d = self.data(F)
                vs = sc.version_space(hp.delta_vs)
                if len(vs) == 0:
                    break
                # only checker facts are settled: a single-source observation that members of V
                # explain as noise can be disputed, and the checker settles it
                known = set(d.pos_cells.tolist())
                if len(vs) == 1:
                    known |= set(d.cells.tolist())
                pick = choose_query(sc, vs, known, d.n_values, self.rng)
                if pick is None:
                    break
                cell, _h = pick
                if len(vs) > 1 and _h <= hp.voi_min_bits:
                    break             # (near) zero information gain: no answer would change much
                if len(vs) == 1 and self.confirmed_new(col, d) >= hp.k_lock:
                    break             # validated: nothing left to ask before locking
                ok = self.ask(F, cell, int(col.sem[cell]), checker, why="qbc" if len(vs) > 1 else "lock")
                if ok is None:
                    break
                self.learn_frame(F)
                if self.try_lock(F, checker):
                    break
            self.try_lock(F, checker)

    def _slip_tests(self, F: str, checker, max_checks: int = 2) -> None:
        sc = self.scored.get(F)
        col = self.active_col(F)
        if sc is None or len(sc.exprs) == 0 or (col is not None and col.state == "locked"):
            return
        b = sc.best()
        wrong = float(sc.wrong[b])
        if wrong <= 0 or self.inadequate.get(F) or not self.flags.mismatch:
            return
        if wrong > max(1.0, 2.0 * self.a.eta_hat() * self.data(F).n_obs):
            return                    # more mismatches than slips could plausibly explain
        d = self.data(F)
        from .mdl import obs_bits
        hit, miss = obs_bits(self.eps(), d.n_values)
        delta = sprout_threshold(self.a.sys_count, self.hp.sprout_start, self.hp.sprout_end, self.hp.sprout_over)
        gain = self.gain.get(F, -1e9)
        if not np.isfinite(gain):
            return                    # no hypothesis exists that an answer could change
        if gain >= delta and (col is not None and col.state == "candidate"):
            return
        if gain + wrong * miss < delta:
            return
        k = self.ctx.frames[F]["arity"]
        m = self.ctx.m
        pred = sc.sem[b]
        bad = [(c, o) for c, o in zip(d.cells.tolist(), d.outs.tolist()) if int(pred[c]) != o]
        for c, o in bad[:max_checks]:
            if checker.remaining() <= self.hp.reserve:
                return
            args = (c // m, c % m) if k == 2 else (c,)
            self.verify_event(F, args + (o,), "", checker)
        self.learn_frame(F)

    def _settled_cells(self, F: str, d: FrameData) -> set:
        """Cells of a fully observed frame that are as good as verified: checker facts, or
        facts seen in two phrasings that agree (two independent slips agreeing is ~eta^2/m)."""
        settled = set(d.pos_cells.tolist())
        outs: dict[int, set] = defaultdict(set)
        w: dict[int, float] = defaultdict(float)
        for c, o, n in zip(d.cells.tolist(), d.outs.tolist(), d.weights.tolist()):
            outs[c].add(o)
            w[c] += n
        settled |= {c for c in outs if len(outs[c]) == 1 and w[c] >= 2}
        return settled

    def _table_need(self, d: FrameData) -> int:
        """Lock a table only when the expected number of wrong unsettled facts,
        (unsettled cells) x eta_hat, is at most 0.1; and never with fewer than K_LOCK checks."""
        m = self.ctx.m
        return max(min(self.hp.k_lock, m), m - self._allowed_unsettled())

    def recognized(self, col) -> "object | None":
        """The library concept a column instantiates, if its track record allows the fast path."""
        if col is None or not col.provenance.origin.startswith("schema:") or not self.flags.library:
            return None
        sid = col.provenance.origin.split(":", 1)[1]
        s = next((x for x in self.a.priors.schemas.values() if x.sid == sid), None)
        return s if s is not None and s.trust >= 0.6 else None

    def _recognized_map(self, sc, b: int) -> bool:
        """The sprout gate makes the data alone pay for a hypothesis (the noisy-TV guard). A
        trusted concept that is already the clear MAP does not have to wait for that: it forms a
        column at once and is tested by falsifying checks, a stronger guard against noise."""
        if not self.flags.library or not self.flags.locking or not sc.origin[b].startswith("schema:"):
            return False
        sid = sc.origin[b].split(":", 1)[1]
        s = next((x for x in self.a.priors.schemas.values() if x.sid == sid), None)
        if s is None or s.trust < 0.6 or float(sc.wrong[b]) > 0:
            return False
        t = sc.total
        ok = np.isfinite(t)
        w = np.exp2(-(t[ok] - t[b]))
        return float(1.0 / w.sum()) >= 0.9

    def map_posterior(self, F: str, col) -> float:
        """P(the column's function | data, prior) over all surviving candidates (MDL posterior)."""
        sc = self.scored.get(F)
        if sc is None or len(sc.exprs) == 0:
            return 0.0
        t = sc.total
        alive = np.isfinite(t)
        if not alive.any():
            return 0.0
        same = alive & (sc.sem == col.sem.astype(sc.sem.dtype)[None, :]).all(axis=1)
        if not same.any():
            return 0.0
        tb = t[alive].min()
        w = np.exp2(-(t[alive] - tb))
        return float(np.exp2(-(t[same] - tb)).sum() / w.sum())

    def _recognize_verify(self, F: str, checker) -> bool:
        """Recognize, then verify: a column that instantiates a trusted concept and fits the
        data is tested at once with 2-3 falsifying questions, asked where its strongest rivals
        disagree with it. All pass and posterior >= 0.99: lock. A failure sends the recognition to
        mismatch (the concept's trust drops) and learning continues normally."""
        col = self.active_col(F)
        s = self.recognized(col)
        sc = self.scored.get(F)
        if s is None or sc is None or col.state == "locked" or self.inadequate.get(F) or not self.flags.locking:
            return False
        n = 2 if s.trust >= 0.85 else 3
        d = self.data(F)
        known = set(d.cells.tolist()) | set(d.pos_cells.tolist())
        # remembered history first: the checks that were decisive before, and a check that
        # separates it from a rival that fooled it before; then the generic committee choice
        hist = self.a.concepts.history_cells(self, col, s, sc, known)
        for _ in range(n):
            if self.confirmed_new(col, d) >= n or checker.remaining() <= self.hp.reserve:
                break
            hist = [c for c in hist if c not in known]
            if hist:
                cell, why = hist.pop(0), "recognize-history"
            else:
                vs = sc.version_space(2 * self.hp.delta_vs)
                pick = choose_query(sc, vs, known, d.n_values, self.rng)
                if pick is None:
                    break
                cell, why = pick[0], "recognize"
            ok = self.ask(F, cell, int(col.sem[cell]), checker, why=why)
            if ok is None:
                break
            known.add(cell)
            if not ok:
                s.refuted += 1
                self.a.spaces.note("refuted", self.cid, F, self.a.log.seq, col=col.id, concept=s.sid, cell=cell,
                                   why="recognition failed its falsifying check")
                self.learn_frame(F)
                return False
            d = self.data(F)
        self.learn_frame(F)
        return self.try_lock(F, checker)

    def try_lock(self, F: str, checker) -> bool:
        col = self.active_col(F)
        if col is None or not self.flags.locking or col.state == "locked":
            return False
        fast = self.recognized(col)
        if col.state != "candidate" and fast is None:
            return False
        d = self.data(F)
        if col.body == ("table",):
            vs_sem = col.sem[None, :]
            gain = 1e9
        else:
            sc = self.scored.get(F)
            if sc is None:
                return False
            vs_sem = sc.sem[sc.version_space(self.hp.delta_vs)]
            gain = self.gain.get(F, -1e9)
        delta = sprout_threshold(self.a.sys_count, self.hp.sprout_start, self.hp.sprout_end, self.hp.sprout_over)
        conflicts = len([r for r in self.a.spaces.mismatches(self.cid, F, "conflict") if r.detail.get("col") == col.id
                         and not r.detail.get("resolved")])
        lc = lock_check(col, vs_sem, gain, delta, conflicts, self.a.sprt_A, checker.remaining())
        if col.body == ("table",):
            settled = self._settled_cells(F, d) & set(range(self.ctx.m))
            safe = self._table_safe(self.ctx.m - len(settled), [int(v) for v in col.sem])
            need = min(self.hp.k_lock, self.ctx.m) if safe else self.ctx.m + 1
            lc.confirmed = len(settled)
        else:
            need = self.hp.k_lock
            lc.confirmed = self.confirmed_new(col, d)
            if fast is not None:
                # recognize-then-verify: posterior collapse instead of version-space collapse, and
                # the passed falsifying checks stand in for the compression gain (the checker is a
                # stronger guard against noise than compression)
                need = 2 if fast.trust >= 0.85 else 3
                if not lc.collapsed:
                    lc.collapsed = self.map_posterior(F, col) >= 0.99 and not self.inadequate.get(F)
                if lc.confirmed >= need:
                    lc.gain_ok = True
        if not lc.ok(need):
            return False
        self.a.lock(self, col, d, lc)
        return True

    # ---- lexicon ----------------------------------------------------------------------
    def _unify(self, checker) -> None:
        hp = self.hp
        rejected = {(r.detail["word"], r.detail["frame"], tuple(r.detail["sigma"]))
                    for r in self.a.spaces.mismatches(self.cid, None, "lexicon-rejected")}
        for _ in range(6):
            tables = self._column_tables()
            cands = [c for c in R.candidates(self.ctx, self.a.eta_hat(), tables)
                     if (c.word, c.frame, c.sigma) not in rejected]
            for c in cands:
                info = self.ctx.frames[c.frame]
                c.post = R.posterior(c, self.reading_prior(info["arity"]), R.n_sigmas(info["arity"], info["bool"]),
                                     self.sigma_prior(info["arity"], info["bool"], c.sigma))
                own = self.ctx.lex[c.word].frame
                unblock = max(self.a.value.blocking(self, c.frame), self.a.value.blocking(self, own))
                c.prio = min(1.0, c.post * (1 + unblock))
            acc = R.decide(cands, hp.m_unify, hp.unify_ln_threshold, use_post=self.flags.priors)
            if acc:
                for c in acc:
                    # an earlier unification in this pass may have removed the target frame
                    if c.frame in self.ctx.frames and self.ctx.words_of(c.frame) and                             self.ctx.words_of(self.ctx.lex[c.word].frame) == [c.word]:
                        self.a.unify(self, c)
                continue
            pend = R.pending(cands, hp.m_unify, hp.unify_ln_threshold, self.a.eta_hat(), prior=True)
            if not pend or not self.flags.active or checker.remaining() <= hp.reserve:
                break
            # resolve a contradiction only when the answer can change the decision: the
            # positive evidence alone must already reach the threshold. Otherwise gather
            # positive evidence first (a prediction test), which can refute the reading.
            if pend[0].contra > 0 and pend[0].lnpos >= R.threshold(cands, hp.unify_ln_threshold):
                if not self._resolve_contradiction(pend[0], checker):
                    rejected.add((pend[0].word, pend[0].frame, pend[0].sigma))
                continue
            pr = R.prediction(self.ctx, pend[0], self.rng, tables)
            if pr is None:
                break
            w, args, out = pr
            try:
                ok = checker.check_fact(w, tuple(self.ctx.tokens[i] for i in args),
                                        self.a.token(self.ctx, self.ctx.lex[w].frame, out, out=True))
            except BudgetExceeded:
                break
            self.ctx.record_check(w, args, out, bool(ok), R.reading_tag(pend[0].word, pend[0].frame, pend[0].sigma))
            self.a.stats["queries"] += 1
            self.a.log.log("query", ctx=self.cid, word=w, args=args, out=out, result=bool(ok), why="unify")
            if not ok:     # the prediction came from one fact of the other side: was that fact a slip?
                c = pend[0]
                # verify the source only when a slip is the likelier explanation of the failure:
                # P(slip | fail) = eta p / (eta p + (1 - p)(1 - 1/V)), p = P(reading)
                V = R.n_values(self.ctx, c.frame)
                eta = self.a.eta_hat()
                p_slip = eta * c.post / (eta * c.post + (1 - c.post) * (1 - 1 / V) + 1e-12)
                if p_slip < 0.5:
                    continue
                if w == c.word:
                    self.verify_event(c.frame, R._map(c.sigma, args + (out,)), w, checker)
                else:
                    ct = self.ctx.canon(self.ctx.lex[w], args + (out,))
                    src = tuple(ct[c.sigma[i]] for i in range(len(ct)))
                    self.verify_event(self.ctx.lex[c.word].frame, src, w, checker)
        self.a.lexicon_locks(self, checker)

    def _column_tables(self) -> dict:
        """Full-domain tables of frames with a candidate or locked column."""
        out = {}
        for F in self.ctx.frames:
            c = self.active_col(F)
            if c is not None and c.state == "locked" and c.scope is None:      # validated knowledge only
                out[F] = c.sem
        return out

    def _resolve_contradiction(self, cand, checker) -> bool:
        """Slip or real? Check both facts of a contradicting pair. A refuted fact is noise
        (the reading survives); two true facts refute the reading."""
        pairs = R.contradictions(self.ctx, cand)
        if not pairs:
            return True
        mine, theirs = pairs[0]
        fw = self.ctx.lex[cand.word].frame
        ok1 = self.verify_event(fw, mine, "", checker)
        if ok1 is False:
            return True
        ok2 = self.verify_event(cand.frame, theirs, "", checker)
        if ok2 is False:
            return True
        if ok1 and ok2:
            self.a.spaces.note("lexicon-rejected", self.cid, cand.frame, self.a.log.seq, word=cand.word,
                               frame=cand.frame, sigma=list(cand.sigma))
            return False
        return True

    def verify_event(self, F: str, tup: tuple, not_word: str, checker) -> bool | None:
        """Check a stored event of frame F through a word (other than `not_word`) that
        observed it. A refuted event is flagged as noise (a slip)."""
        ev = self.ctx.events.get(F, {}).get(tuple(tup))
        if ev is None or checker.remaining() <= 0:
            return None
        for w2, a2, o2, ok2 in self.ctx.checked:          # already settled by the checker
            e2 = self.ctx.lex.get(w2)
            if ok2 and e2 is not None and e2.frame == F and self.ctx.canon(e2, a2 + (o2,)) == tuple(tup):
                return True
        words = [self.ctx.obs[s][0] for s in ev.sources if self.ctx.obs[s][0] != not_word]
        if not words or self.ctx.lex.get(words[0]) is None:
            return None
        ok = self.a._ask_word(self, words[0], tuple(tup), checker, why="verify-source")
        if ok is False:
            k = self.ctx.frames[F]["arity"]
            self.flag_noise(F, (R.cell_of(self.ctx, tuple(tup[:k])), tup[-1]), "checker refuted a source fact")
            self.a.diagnoser.record_slip(self, F, tuple(tup), "checker")
        return ok

    # ---- noise, SPRT and domain changes -------------------------------------------
    def flag_noise(self, F: str, cell_out: tuple[int, int], why: str) -> None:
        c, o = cell_out
        k = self.ctx.frames[F]["arity"]
        args = (c // self.ctx.m, c % self.ctx.m) if k == 2 else (c,)
        ev = self.ctx.events.get(F, {}).get(args + (o,))
        if ev is None or ev.noise or not self.flags.mismatch:
            return
        ev.noise = True
        self.a.noise_found += len(ev.sources)
        self.a.spaces.note("noise", self.cid, F, self.a.log.seq, tup=args + (o,), why=why)
        self.a.log.log("noise", ctx=self.cid, frame=F, tup=args + (o,), why=why)

    def observe_check(self, word: str, args: tuple[int, ...], out: int) -> None:
        """Prequential prediction of a new observation, before it is ingested."""
        e = self.ctx.lex.get(word)
        if e is None or any(a >= self.ctx.m for a in args) or (out >= self.ctx.m and not self.ctx.frames[e.frame]["bool"]):
            return
        F = e.frame
        tup = self.ctx.canon(e, args + (out,))
        col = self.active_col(F)
        k = len(tup) - 1
        if col is None or col.sem.size != self.ctx.m ** k:
            self.errors[F].append(1)
            return
        cell = R.cell_of(self.ctx, tup[:-1])
        ok = int(col.sem[cell]) == tup[-1]
        self.errors[F].append(0 if ok else 1)
        if col.state == "locked":
            self.a.noise_examined += 1
            col.stats.sprt = self.a.sprt_step(col.stats.sprt, ok)
            if not ok:
                self.suspects[col.id].append((cell, tup[-1]))
                self.a.noise_suspected += 1

    def _sprt_decisions(self, checker) -> None:
        for cid, sus in list(self.suspects.items()):
            col = self.a.spaces.columns[cid]
            if not sus or col.state != "locked":
                self.suspects[cid] = []
                continue
            if col.stats.sprt <= self.a.sprt_B:
                # H0 on a locked column never supersedes alone: one checker verification
                cell, out = sus[-1]
                ok = self.ask(col.frame, cell, out, checker, why="sprt-H0") if checker.remaining() > 0 else None
                if ok is None:
                    self.a.spaces.note("conflict", self.cid, col.frame, self.a.log.seq, col=col.id, cell=cell, out=out)
                    continue
                if ok:
                    self.a.diagnoser.counterexample(self, col, cell, out, checker)
                else:
                    self.flag_noise(col.frame, (cell, out), "checker refuted an observation (SPRT H0 test)")
                    self.a.diagnoser.record_slip(self, col.frame, (cell, out), "checker")
                    col.stats.sprt = 0.0
                self.suspects[cid] = [s for s in sus if s != (cell, out)]
                self.a.noise_suspected = max(0, self.a.noise_suspected - 1)
            elif col.stats.sprt >= self.a.sprt_A and checker.remaining() <= 0:
                # plan §7.4: the SPRT stands in for the checker only when the budget is gone
                for cell, out in sus:
                    self.flag_noise(col.frame, (cell, out), "SPRT accepted the column, no budget left (slip)")
                    self.a.diagnoser.record_slip(self, col.frame, (cell, out), "sprt")
                col.stats.sprt = 0.0
                self.suspects[cid] = []

    def _verify_suspects(self, checker, max_checks: int = 2) -> None:
        """Spare budget tests locked rules against the observations that contradict them:
        a refuted observation is a slip; a confirmed one is a counterexample."""
        done = 0
        for cid, sus in list(self.suspects.items()):
            col = self.a.spaces.columns[cid]
            while sus and col.state == "locked" and done < max_checks and checker.remaining() > self.hp.reserve:
                cell, out = sus.pop(0)
                ok = self.ask(col.frame, cell, out, checker, why="suspect")
                done += 1
                if ok is None:
                    sus.insert(0, (cell, out))
                    return
                self.a.noise_suspected = max(0, self.a.noise_suspected - 1)
                if ok:
                    self.a.diagnoser.counterexample(self, col, cell, out, checker)
                    break
                self.flag_noise(col.frame, (cell, out), "checker refuted an observation a locked rule contradicts")
                self.a.diagnoser.record_slip(self, col.frame, (cell, out), "checker")

    def _check_domain(self) -> None:
        m = self.ctx.m
        if m == self.m_seen:
            return
        if self.m_seen:
            self.a.log.log("domain", ctx=self.cid, old=self.m_seen, new=m)
            self.pools.clear()
            self.scored.clear()
            self.analog.clear()
            for col in self.a.spaces.in_context(self.cid):
                if col.kind == "operator" and col.sem.size not in (m, m * m):
                    if col.state == "locked":
                        self.a.replace(col, None, "the domain grew")
                    else:
                        self.a.retire(col, "the domain grew")
        self.m_seen = m
