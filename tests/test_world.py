"""World: generator, grammar, identifiability, questions and the checker (plan §2, M1)."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gca.world import grammar as G
from gca.world.api import MANY, NONE, BudgetExceeded, Question, depth
from gca.world.checker import Checker, score
from gca.world.generator import StreamBuilder, _subst


def test_all_systems_identifiable(stream0):
    """Plan §2.4: every program of the generator's grammar consistent with the observed
    cells agrees with the truth on the whole domain."""
    _, systems = stream0
    for sw in systems:
        seen_canon = {}
        for w in sw.words:
            if w.foundation:
                continue
            fam = G.family_tables(w.spec, sw.n)
            truth = sw.tables[w.word].ravel()
            obs = [int(np.ravel_multi_index(c, sw.tables[w.word].shape)) for c in sw.seen[w.word]]
            if w.frame in seen_canon:
                continue
            seen_canon[w.frame] = True
            consistent = fam[(fam[:, obs] == truth[obs]).all(axis=1)]
            assert (consistent == truth).all(), (sw.sid, w.word)


def test_foundation_complete_and_first(stream0):
    _, systems = stream0
    for sw in systems:
        f = [w for w in sw.words if w.foundation]
        if not f:
            continue
        nxt, prv = f
        covered = {a for (a,) in sw.seen[nxt.word]} | {int(sw.tables[prv.word][b]) for (b,) in sw.seen[prv.word]}
        assert covered == set(range(sw.n)), sw.sid           # every link of the chain is heard
        fw = {nxt.word, prv.word}
        kinds = [ob.word in fw for ob in sw.experience]
        k = sum(kinds)
        assert all(kinds[:k]) and not any(kinds[k:]), sw.sid  # the counting sequence comes first


def test_question_keys(stream0):
    _, systems = stream0
    for sw in systems[:9]:
        for split, qs in sw.questions.items():
            for q in qs:
                key = sw.keys[q.qid]
                if q.kind == "eval":
                    v = sw.eval_int(q.expr)
                    tok = sw.tok(v, q.expr[1])
                    assert key == tok
                else:
                    sols = [v for v in range(sw.n) if sw.tok(sw.eval_int(_subst(q.expr, ("obj", sw.objs[v]))),
                                                            q.expr[1]) == q.target]
                    expect = NONE if not sols else (sw.objs[sols[0]] if len(sols) == 1 else MANY)
                    assert key == expect
                if split == "S3":
                    assert depth(q.expr) >= 3
                if split == "S2":
                    assert depth(q.expr) == 2


def test_s4b_has_none_and_many(stream0):
    _, systems = stream0
    keys = [sw.keys[q.qid] for sw in systems for q in sw.questions["S4b"]]
    assert NONE in keys and MANY in keys


def test_checker_budget_and_truth(stream0):
    _, systems = stream0
    sw = systems[1]
    ch = Checker(sw, 3)
    w = sw.words[-1].word
    a, b = sw.objs[1], sw.objs[2]
    true_out = sw.tok(sw.apply(w, (1, 2)), w)
    assert ch.check_fact(w, (a, b), true_out) is True
    wrong = sw.objs[(sw.objs.index(true_out) + 1) % sw.n]
    assert ch.check_fact(w, (a, b), wrong) is False
    q = sw.questions["S1"][0]
    assert ch.check_answer(q, sw.keys[q.qid]) is True
    assert ch.remaining() == 0
    with pytest.raises(BudgetExceeded):
        ch.check_fact(w, (a, b), true_out)


def test_determinism_of_world(cfg):
    a = StreamBuilder(cfg, 3, 0.05).build()
    b = StreamBuilder(cfg, 3, 0.05).build()
    for x, y in zip(a, b):
        assert x.objs == y.objs and x.experience == y.experience
        assert [q.expr for q in x.questions["S1"]] == [q.expr for q in y.questions["S1"]]


def test_noise_only_in_experience(cfg):
    sb = StreamBuilder(cfg, 0, 0.2)
    sw = sb.build()[1]
    wrong = sum(1 for ob in sw.experience if not sw.true_fact(ob.word, ob.args, ob.out))
    assert 0 < wrong < len(sw.experience)
    assert all(sw.true_fact(ob.word, ob.args, ob.out) for ob in sw.extra_labels)


def test_reused_tokens_across_systems(stream0):
    _, systems = stream0
    early = set().union(*[set(s.objs) | {w.word for w in s.words} for s in systems[:9]])
    reused = [t for s in systems[9:14] for t in s.objs + [w.word for w in s.words] if t in early]
    assert reused, "transfer systems must reuse some tokens with new meanings"


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 13), st.integers(0, 12), st.integers(0, 12), st.integers(0, 12))
def test_lin_quasigroup_iff_units(n, a, b, c):
    t = G.table(G.OpSpec("poly", (0, 0, 0, a % n, b % n, c % n)), n)
    from math import gcd
    assert G.is_quasigroup(t) == (gcd(a % n, n) == 1 and gcd(b % n, n) == 1)


def test_score_is_world_side(stream0):
    _, systems = stream0
    sw = systems[1]
    q: Question = sw.questions["S1"][0]
    assert score(sw, q, sw.keys[q.qid])
