"""Shared concepts: join, extend-first (proofs preserved), propagation."""
from gca.improve.concepts import concrete_share, match
from gca.improve.play import anti_unify
from gca.learn import hypothesis as H

ADD = ("iter", "u:a", ("pos", "u:a", 3, ("y",)), ("x",))
SUB = ("iter", "u:a~", ("pos", "u:a", 3, ("y",)), ("x",))


def test_match_exact_instances():
    sch, _ = H.to_schema(ADD)
    f = match(sch, ADD)
    assert f is not None and H.instantiate(sch, f) == ADD
    assert match(sch, SUB) is None          # different refs in the two slots: not this form


def test_extension_keeps_every_proven_rule():
    """Generalizing add to cover sub keeps add an exact instance (nothing it proved breaks)."""
    s_add, _ = H.to_schema(ADD)
    g, _ = H.to_schema(anti_unify(s_add, H.to_schema(SUB)[0]))
    assert match(g, ADD) is not None and match(g, SUB) is not None
    assert H.size(g) == H.size(ADD) and concrete_share(g) == 1.0


def test_concepts_in_a_stream(mini):
    from gca.agent import GCAAgent
    from gca.config import Hparams
    from experiments.harness import run_stream
    sb, systems = mini
    agent = GCAAgent(Hparams())
    run_stream(agent, systems, sb.budget, curve=False)
    joined = [r for r in agent.log.of("concept") if r["how"] in ("join", "extend")]
    assert joined, "S03's addition-like rules must reuse S02's concept"
    s = next(iter(agent.priors.schemas.values()))
    assert all(agent.spaces.columns[c].state == "locked" for _, _, c in s.bindings)


def test_history_and_retry_in_a_stream(mini):
    from gca.agent import GCAAgent
    from gca.config import Hparams
    from gca.think.solver import STEP_KINDS
    from experiments.harness import run_stream
    sb, systems = mini
    agent = GCAAgent(Hparams())
    res = run_stream(agent, systems, sb.budget, curve=False)
    assert all(r.test_acc == 1.0 for r in res)
    # every trace step carries a kind
    q = systems[1].questions["S2"][0]
    a = agent.answer(q)
    assert all(getattr(s, "kind", None) in STEP_KINDS for s in a.trace if hasattr(s, "kind"))
