"""Concept index (plan §4.2): each fact stored once, linked from every participant."""
from gca.learn import reading as R
from gca.memory.stores import Context
from gca.world.api import Observation


def _ctx_next_prev():
    ctx = Context("S")
    obs = [Observation("S", "nxt", ("a",), "b"), Observation("S", "nxt", ("b",), "c"),
           Observation("S", "prv", ("b",), "a"), Observation("S", "prv", ("c",), "b"),
           Observation("S", "prv", ("d",), "c")]
    R.ingest(ctx, obs)
    return ctx


def test_each_fact_stored_once_after_unify():
    ctx = _ctx_next_prev()
    F = ctx.lex["nxt"].frame
    assert len(ctx.events[F]) == 2
    ctx.unify("prv", F, (1, 0), support=2)
    ev = ctx.events[F]
    assert set(ev) == {(0, 1), (1, 2), (2, 3)}           # a->b, b->c, c->d: three facts, once each
    assert len(ev[(0, 1)].sources) == 2                   # seen in both phrasings
    assert all(f == F for f, _ in ctx.by_entity[1])       # indexed from every participant
    assert {t for _, t in ctx.by_entity[1]} == {(0, 1), (1, 2)}


def test_split_restores_own_frame():
    ctx = _ctx_next_prev()
    F = ctx.lex["nxt"].frame
    ctx.unify("prv", F, (1, 0), support=2)
    ctx.split("prv")
    G = ctx.lex["prv"].frame
    assert G != F
    assert set(ctx.events[F]) == {(0, 1), (1, 2)}
    assert set(ctx.events[G]) == {(1, 0), (2, 1), (3, 2)}


def test_names_resolved_per_context():
    """Reused tokens name different entities in different contexts (crowding/binding)."""
    a, b = Context("S1"), Context("S2")
    R.ingest(a, [Observation("S1", "w", ("dax",), "wug")])
    R.ingest(b, [Observation("S2", "w", ("wug",), "dax")])
    assert a.index["dax"] == 0 and b.index["dax"] == 1
    assert a.ents[0].id != b.ents[1].id


def test_frame_data_uses_checker_facts():
    ctx = _ctx_next_prev()
    F = ctx.lex["nxt"].frame
    ctx.record_check("nxt", (2,), 3, True, "test")
    ctx.record_check("nxt", (3,), 0, False, "test")
    d = R.frame_data(ctx, F)
    assert list(d.pos_cells) == [2] and list(d.pos_outs) == [3]
    assert list(d.neg_cells) == [3] and list(d.neg_outs) == [0]
