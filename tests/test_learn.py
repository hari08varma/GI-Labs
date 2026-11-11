"""Learner DSL, enumeration, MDL, priors, active learning (plan §5, M4)."""
import math

import numpy as np

from gca.learn import hypothesis as H
from gca.learn.active import choose_query
from gca.learn.enumerate import Pool, row_hash
from gca.learn.induce import pool_part, schema_instances, score, splits
from gca.learn.mdl import FrameData, adequate, data_bits, null_bits, obs_bits, sprout_threshold
from gca.learn.priors import Priors


def prims(m, binary=False):
    P = H.Prims(m)
    P.add_unary("next", (np.arange(m) + 1) % m)
    P.add_unary("next~", (np.arange(m) - 1) % m)
    if binary:
        X, Y = H.grid(m, 2)
        P.add_binary("add", ((X + Y) % m).reshape(m, m))
        P.add_binary("mul", ((X * Y) % m).reshape(m, m))
    return P


def test_evaluate_matches_arithmetic():
    m = 9
    P = prims(m, binary=True)
    X, Y = H.grid(m, 2)
    cases = {
        ("iter", "next", ("pos", "next", 0, ("y",)), ("x",)): (X + Y) % m,
        ("iter", "next~", ("pos", "next", 0, ("y",)), ("x",)): (X - Y) % m,
        ("iterc", "add", ("x",), ("pos", "next", 0, ("y",)), ("c", 0)): (X * Y) % m,
        ("lt", ("pos", "next", 0, ("x",)), ("pos", "next", 0, ("y",))): (X < Y).astype(int),
        ("if", ("lt", ("pos", "next", 0, ("x",)), ("pos", "next", 0, ("y",))), ("y",), ("x",)): np.maximum(X, Y),
        ("B", "add", ("x",), ("U", "next", ("y",))): (X + Y + 1) % m,
    }
    for e, want in cases.items():
        assert np.array_equal(H.evaluate(e, X, Y, P), want), H.show(e)


def test_size_and_schema_roundtrip():
    e = ("iter", "next", ("pos", "next", 3, ("y",)), ("x",))
    assert H.size(e) == 4
    sch, kinds = H.to_schema(e)
    assert sch == ("iter", "?r0", ("pos", "?r0", "?e0", ("y",)), ("x",))
    assert kinds["?r0"] == frozenset({"P", "C"}) and kinds["?e0"] == frozenset({"E"})
    assert H.instantiate(sch, {"?r0": "next", "?e0": 3}) == e


def test_row_hash_exact():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 13, size=(500, 169)).astype(np.int8)
    B = np.concatenate([A[250:], A[:250]])
    ha, hb = row_hash(A), row_hash(B)
    assert np.array_equal(np.concatenate([ha[250:], ha[:250]]), hb)
    assert len(set(ha.tolist())) == 500


def test_pool_observational_equivalence_and_targets():
    m = 7
    pool = Pool(prims(m), 2, Priors(), s_enum=6)
    sem, cost, exprs, _ = pool.flat("O")
    assert len(set(row_hash(sem).tolist())) == len(sem)      # no two candidates compute the same function
    X, Y = H.grid(m, 2)
    for f in ((X + Y) % m, (X - Y) % m, (2 * X + 1) % m):
        assert (sem == f).all(axis=1).any()


def test_split_reaches_large_affine_operator():
    m = 11
    P = prims(m, binary=True)
    pool = Pool(P, 2, Priors(), s_enum=6)
    X, Y = H.grid(m, 2)
    f = (2 * X + 3 * Y + 1) % m
    cells = np.random.default_rng(1).choice(m * m, 20, replace=False)
    d = FrameData(2, m, cells, f[cells], np.ones(20))
    sc = score(d, 0.05, [pool_part(pool, "O", Priors()), splits(pool, d, Priors(), tol=0)])
    assert np.array_equal(sc.sem[sc.best()], f)


def test_data_bits_formula():
    sem = np.array([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=np.int8)
    d = FrameData(1, 4, np.array([0, 3]), np.array([0, 3]), np.array([1.0, 2.0]))
    db, wrong = data_bits(sem, d, 0.05)
    hit, miss = obs_bits(0.05, 4)
    assert math.isclose(db[0], 3 * hit) and math.isclose(db[1], hit + 2 * miss)
    assert list(wrong) == [0, 2]
    assert math.isclose(null_bits(d), 3 * 2.0)


def test_sprout_threshold_schedule():
    assert sprout_threshold(0) == 16 and sprout_threshold(5) == 8 and sprout_threshold(9) == 8
    assert 8 < sprout_threshold(2) < 16


def test_noise_cannot_sprout():
    """Learnable-error guard: random labels do not compress, so nothing sprouts."""
    m = 9
    pool = Pool(prims(m), 2, Priors(), s_enum=6)
    rng = np.random.default_rng(2)
    cells = rng.choice(m * m, 40, replace=False)
    d = FrameData(2, m, cells, rng.integers(0, m, 40), np.ones(40))
    sc = score(d, 0.05, [pool_part(pool, "O", Priors())])
    gain = null_bits(d) - sc.total[sc.best()]
    assert gain < sprout_threshold(5)


def test_adequacy_test():
    assert adequate(0, 40, 0.01)
    assert adequate(2, 40, 0.05)
    assert not adequate(15, 40, 0.05)


def test_priors_learn_productions_and_schemas():
    pr = Priors()
    body = ("iter", "next", ("pos", "next", 0, ("y",)), ("x",))
    P = prims(9)
    before = pr.expr_bits(body, P, 2)
    pr.add_body(body)
    assert pr.expr_bits(body, P, 2) < before
    s = pr.add_schema(body, 2, "O", "c1")
    inst = schema_instances(P, 2, "O", pr)
    assert len(inst.exprs) == 2 * 9 and all(o == "schema:" + s.sid for o in inst.origin)
    frozen = Priors(frozen=True)
    frozen.add_body(body)
    assert frozen.expr_bits(body, P, 2) == Priors().expr_bits(body, P, 2)


def test_query_by_committee_picks_disagreement():
    m = 5
    sem = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 0], [0, 1, 2, 4, 4]], dtype=np.int8)
    d = FrameData(1, m, np.array([0, 1]), np.array([0, 1]), np.ones(2))
    sc = score(d, 0.05, [type("E", (), {"sem": sem, "prior": np.array([1.0, 1.0, 1.0]),
                                         "exprs": ["a", "b", "c"], "origin": ["p"] * 3})()])
    cell, h = choose_query(sc, sc.version_space(10), {0, 1}, m, np.random.default_rng(0))
    assert cell in (3, 4) and h > 0


def test_loop_repair_days_of_the_week():
    """Harinath's example: Monday has two days before it (Sunday, Thursday) and Friday none.
    Only changing the Thursday fact to Friday makes one loop, so that is the one question."""
    from gca.learn.structure import best_question, loop_repairs
    known = {0: 1, 1: 2, 2: 3, 3: 0, 4: 5, 5: 6, 6: 0}      # Mon=0 ... Sun=6; 3 -> 0 is wrong
    k, sols = loop_repairs(7, known, {c: 1.0 for c in known}, set(), set(), [])
    assert k == 1 and [a for _, a in sols] == [{3: 4}]
    assert best_question(sols, known) == (3, 4)
    # a checker refusal removes that repair; with the true fact confirmed there is none left
    k2, sols2 = loop_repairs(7, known, {c: 1.0 for c in known}, {3}, {(3, 4)}, [])
    assert all((3, 4) not in a.items() for _, a in sols2)
