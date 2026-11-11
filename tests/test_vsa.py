"""VSA layer (plan §4.1): fixed random MAP-B codes."""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from gca.memory.columns import column_key
from gca.memory.vsa import VSA


def test_bind_self_inverse_and_dissimilar():
    v = VSA(seed=1)
    a, b = v.hv("a"), v.hv("b")
    ab = v.bind(a, b)
    assert np.array_equal(v.bind(a, ab), b)
    assert abs(VSA.sim(ab, a)) < 0.05 and abs(VSA.sim(ab, b)) < 0.05


@settings(max_examples=20, deadline=None)
@given(st.text(min_size=1, max_size=6), st.text(min_size=1, max_size=6))
def test_bind_commutes_and_inverts(x, y):
    v = VSA(dim=2000, seed=2)
    a, b = v.hv(x), v.hv(y)
    assert np.array_equal(v.bind(a, b), v.bind(b, a))
    assert np.array_equal(v.bind(v.bind(a, b), b), a)


def test_vsa_capacity():
    """Role-filler pairs bundled in one vector are all recovered by cleanup (plan §4.1)."""
    v = VSA(seed=0)
    items = [f"i{k}" for k in range(2000)]
    rng = np.random.default_rng(0)
    for n_pairs in (3, 7, 15, 31, 63):
        fillers = [items[i] for i in rng.choice(len(items), n_pairs, replace=False)]
        binds = {f"r{k}": f for k, f in enumerate(fillers)}
        F = v.frame_vector(f"cap{n_pairs}", binds)
        for r, f in binds.items():
            got, _ = v.cleanup(v.unbind_role(F, r), items)
            assert got == f, (n_pairs, r)


def test_per_frame_tie_breaking_decorrelates():
    """Shared tie vectors make unrelated 4-item bundles correlate; per-frame ties do not."""
    v = VSA(seed=3)
    sims = []
    for k in range(20):
        A = v.frame_vector(f"fa{k}", {f"r{j}": f"a{k}_{j}" for j in range(4)})
        B = v.frame_vector(f"fb{k}", {f"r{j}": f"b{k}_{j}" for j in range(4)})
        sims.append(VSA.sim(A, B))
    assert abs(float(np.mean(sims))) < 0.03


def test_key_cosine_counts_matching_fields():
    v = VSA(seed=4)
    q = column_key(v, "operator", "2->O", "S01", "F1")
    for m, key in [(4, column_key(v, "operator", "2->O", "S01", "F1")),
                   (3, column_key(v, "operator", "2->O", "S01", "F2")),
                   (2, column_key(v, "operator", "2->O", "S02", "F2")),
                   (1, column_key(v, "operator", "1->O", "S02", "F2"))]:
        assert abs(VSA.sim(q, key) - m / 4) < 0.04, m


def test_resonator_factors_a_product():
    v = VSA(dim=4000, seed=5)
    books = [[f"x{i}" for i in range(6)], [f"y{i}" for i in range(6)]]
    s = v.bind(v.hv("x3"), v.hv("y4"))
    names, _ = v.resonator(s, books)
    assert names == ["x3", "y4"]
