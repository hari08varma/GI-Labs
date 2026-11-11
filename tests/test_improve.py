"""Improve-side building blocks: SPRT, hitting sets, specialization, play, value."""
import numpy as np

from gca.improve import sprt
from gca.improve.hitting_sets import minimal_hitting_sets, rank
from gca.improve.play import _assoc, anti_unify
from gca.improve.specialize import find_condition
from gca.improve.value import ValueSignal
from gca.config import Hparams
from gca.learn import hypothesis as H
from gca.learn.enumerate import Pool
from gca.learn.priors import Priors


def test_sprt_thresholds():
    """Plan §7.4: at eta = 0.05 about 16 confirmations accept H1, about 3 contradictions H0."""
    up, down = sprt.n_to_accept(0.05)
    assert up == 16 and down == 3
    A, B = sprt.thresholds()
    assert abs(A - 4.595) < 0.01 and abs(B + 4.595) < 0.01


def test_minimal_hitting_sets():
    hs = minimal_hitting_sets([frozenset("ab"), frozenset("bc")])
    assert frozenset("b") in hs and frozenset("ac") in hs and len(hs) == 2
    ranked = rank(hs, {"a": 0.3, "b": 0.001, "c": 0.3})
    assert ranked[0][0] == frozenset("ac")


def test_specialization_finds_cheap_condition():
    m = 7
    P = H.Prims(m)
    P.add_unary("next", (np.arange(m) + 1) % m)
    pool = Pool(P, 2, Priors(), s_enum=5)
    X, Y = H.grid(m, 2)
    good = [int(c) for c in np.flatnonzero(X != 3)][:20]
    bad = [int(c) for c in np.flatnonzero(X == 3)][:3]
    expr, sem, bits = find_condition(pool, good, bad)
    assert all(sem[c] == 1 for c in good) and all(sem[c] == 0 for c in bad)


def test_anti_unification_generalises_parameters_and_subtrees():
    a = ("iter", "?r0", ("pos", "?r0", "?e0", ("y",)), ("x",))
    b = ("iter", "?r0", ("pos", "?r0", "?e0", ("y",)), ("U", "?r0", ("x",)))
    g = anti_unify(a, b)
    assert g[0] == "iter" and g[3][0] == "H"
    c = ("iter", "?r1", ("pos", "?r0", "?e0", ("y",)), ("x",))
    g2 = anti_unify(a, c)
    assert g2[1] != g2[2][1]                  # the iterated ref became its own hole


def test_associativity_check():
    m = 5
    x = np.arange(m)
    add = (x[:, None] + x[None, :]) % m
    sub = (x[:, None] - x[None, :]) % m
    assert _assoc(add) and not _assoc(sub)


def test_learning_progress_signal():
    v = ValueSignal(Hparams())
    assert v.progress([1] * 5 + [0] * 5) == 1.0
    assert v.progress([1] * 10) == 0.0            # noise: no progress, no attraction
