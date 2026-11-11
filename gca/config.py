"""Hyperparameters and ablation flags. Defaults follow the plan; configs/hparams.yaml
can override any field. Hyperparameters are fixed before results exist and are never
tuned on test seeds (plan §10, fairness rules)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

import yaml


@dataclass(frozen=True)
class Flags:
    """Each flag switches one mechanism; ablations (plan §9.3) turn one off."""
    library: bool = True          # reuse of earlier locked columns (schemas)
    priors: bool = True           # counted priors and overhypotheses
    mismatch: bool = True         # persistent negative constraints
    diagnosis: bool = True        # diagnose errors (off: re-induce from scratch)
    locking: bool = True          # lock validated columns (off: overwrite in place)
    play: bool = True             # abstraction and conjectures at rest
    analogy: bool = True          # structure transfer across systems
    lexicon_unify: bool = True    # phrasings unified into shared frames
    split: bool = True            # top-down split through invertible primitives
    active: bool = True           # spend the checker budget on chosen queries
    extra_labels: bool = False    # B1: take the budget as extra random labels instead
    # check its own answer against stored facts before giving it. OFF: an A/B on dev seeds showed
    # it only withheld correct answers (the stored facts are the same evidence the rules came from)
    selfcheck: bool = False


ABLATIONS = {
    "full": {},
    "no_library": {"library": False},
    "no_priors": {"priors": False},
    "no_mismatch": {"mismatch": False},
    "no_diagnosis": {"diagnosis": False},
    "no_locking": {"locking": False},
    "no_play": {"play": False},
    "no_analogy": {"analogy": False},
    "no_lexicon_unify": {"lexicon_unify": False},
    "with_selfcheck": {"selfcheck": True},
}

# B1 enumerator (plan §10): the same innate core, search and MDL, with no library, no priors,
# no diagnosis and no memory across systems (no analogy, no play). It reads phrasings within a
# system like we do. It never queries: its checker budget arrives as extra random labels, so
# nothing can be checker-validated and candidate rules act as primitives (locking off).
B1_FLAGS = Flags(library=False, priors=False, mismatch=True, diagnosis=False, locking=False,
                 play=False, analogy=False, lexicon_unify=True, split=True, active=False,
                 extra_labels=True)


@dataclass(frozen=True)
class Hparams:
    seed: int = 0
    vsa_dim: int = 10_000
    # search (plan §5.4)
    s_enum: int = 7
    cap_O: int = 3000
    cap_N: int = 1000
    cap_B: int = 1500
    combo_budget: int = 60_000
    node_budget: int = 1_000_000
    schema_cap: int = 4000
    # scoring and version space (plan §5.3, §5.5)
    delta_vs: float = 10.0
    eps0: float = 0.05
    eps_floor: float = 0.005
    sprout_start: float = 16.0
    sprout_end: float = 8.0
    sprout_over: int = 5
    # validation (plan §5.6, P7)
    k_lock: int = 3
    # reading (plan §5.1)
    m_unify: int = 3
    unify_ln_threshold: float = 4.595      # ln(99)
    # thinking (plan §6)
    vigilance: float = 0.8
    workspace_k: int = 7
    workspace_kmax: int = 12
    w_locked: float = 100.0
    w_candidate: float = 10.0
    w_event: float = 100.0
    w_plastic: float = 1.0
    lam: float = 0.1
    # improving (plan §7)
    sprt_p0: float = 0.3
    sprt_floor: float = 0.01
    sprt_alpha: float = 0.01
    sprt_beta: float = 0.01
    queries_per_learn: int = 2
    max_queries_per_learn: int = 8     # value-of-information loop: hard cap per frame per step
    voi_min_bits: float = 0.1          # ask only while the question's entropy exceeds this
    reserve: int = 4
    lp_window: int = 5
    eps_explore: float = 0.05
    goal_off: float = 0.3
    retire_superseded_after: int = 5
    retire_plastic_after: int = 2
    column_budget: int = 2000
    play_min_saving: float = 8.0
    analogy_top: int = 3
    analogy_max_maps: int = 64
    analogy_node_limit: int = 200_000
    prior_plastic: float = 0.30
    prior_candidate: float = 0.10
    prior_locked: float = 0.001
    prior_lexicon_low: float = 0.20
    flags: Flags = field(default_factory=Flags)

    def with_flags(self, **kw) -> "Hparams":
        return replace(self, flags=replace(self.flags, **kw))

    def to_dict(self) -> dict:
        return asdict(self)


def load_hparams(path: str | None = None, **overrides) -> Hparams:
    hp = Hparams()
    if path:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        flags = d.pop("flags", {}) or {}
        names = {f.name for f in fields(Hparams)}
        hp = replace(hp, **{k: v for k, v in d.items() if k in names})
        hp = hp.with_flags(**flags)
    flags = overrides.pop("flags", None)
    hp = replace(hp, **overrides)
    if flags:
        hp = replace(hp, flags=flags)
    return hp
