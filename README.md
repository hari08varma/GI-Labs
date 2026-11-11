# GCA: Growing Compositional Architecture

*(working name)*

A research prototype of a different way for an AI system to learn. It does not use next-token
prediction, a pretrained transformer, learned weights or gradient descent.

> The system receives experience, develops reusable knowledge, reasons with that knowledge,
> identifies its own mistakes, and improves its future learning.

Knowledge is held as small, readable, structured units (rules, frames and procedures, each with
conditions and provenance). Learning changes that structure: writing records, comparing,
counting, diagnosing, adding conditions, forking, locking, merging and retiring. Nothing is
locked until it has been validated against an external checker.

The test domain is **invented mathematical systems**. A hidden generator creates systems nobody
has seen, with new symbols, objects, operators and relations, so no pretrained model can have
memorised them. The learner sees only examples and checker feedback.

---

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [The test domain](#the-test-domain)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the results](#reproducing-the-results)
- [Baselines](#baselines)
- [Repository layout](#repository-layout)
- [Testing and integrity rules](#testing-and-integrity-rules)
- [Limitations and open problems](#limitations-and-open-problems)
- [References](#references)

---

## Results

All numbers come from the held-out **test seeds 50, 51 and 52**. Each stream has 24 systems.
Accuracy is computed per system over all its test questions, averaged over the systems of a
stream, then given as mean ± 95% CI (t-distribution) over seeds. Every learner gets the **same
data and the same checker budget**. Development used seeds 0–12; seeds 50–52 were run once, at
the end.

### Main comparison

| learner | noise 0 | noise 0.1 | confidently wrong (noise 0 / 0.1) |
|---|---|---|---|
| **GCA (this repo)** | **0.988 ± 0.027** | **0.819 ± 0.040** | **0.0% / 0.9%** |
| B1: brute-force rule enumerator (same innate core, no memory across systems) | 0.764 ± 0.070 | 0.276 ± 0.129 | 1.3% / 3.3% |
| B2+core: Popper ILP + our innate core as background knowledge, two-stage | 0.692 ± 0.029 | 0.428 ± 0.018 | 11.4% / 15.5% |
| B2: Popper ILP, fresh for each system | 0.640 ± 0.045 | 0.421 ± 0.037 | 12.5% / 16.2% |
| B2-L: Popper ILP, keeps earlier programs (Forgetgol setting) | 0.529 ± 0.007 | 0.392 ± 0.017 | 10.6% / 15.8% |
| B3: transformer from scratch, sequential (2 layers, chosen on dev seeds) | 0.217 | 0.202 | 77.9% / 79.6% |
| B3: transformer from scratch, one model per system | 0.206 | 0.190 | 79.1% / 80.6% |
| B4: Qwen2.5-0.5B + LoRA, sequential | 0.178 ± 0.001 | 0.163 ± 0.010 | 78.6% / 81.1% |

"Confidently wrong" means a definite answer that was wrong. Abstaining (`unknown`) is not
counted as wrong.

**Forgetting.** When every system is asked again at the end of the stream, GCA's backward transfer
is about zero (BWT −0.001 ± 0.004 at noise 0). B3 falls to 0.034 / 0.027 and B4 to 0.034 / 0.035.
B3 and B4 have no way to abstain, so part of their confident-wrong rate reflects that.

### GCA under noise

| noise | accuracy | abstains | confidently wrong | locked rules that are wrong |
|---|---|---|---|---|
| 0.00 | 0.988 ± 0.027 | 1.3% | 0.0% | 0 / 204 |
| 0.05 | 0.899 ± 0.087 | 10.5% | 0.5% | 0 / 193 |
| 0.10 | 0.819 ± 0.040 | 17.9% | 0.9% | 0 / 171 |
| 0.20 | 0.632 ± 0.113 | 36.1% | 0.7% | 0 / 115 |

Across all main runs, **0 of 683 locked rules were wrong**, and none were wrong in any ablation
run either. As noise rises, GCA abstains more; it does not guess more.

### Per-split accuracy (GCA, noise 0)

| S1 in-distribution | S2 composition | S3 depth | S4 reversal | S4b honest reversal (`none`/`many`) |
|---|---|---|---|---|
| 0.986 | 0.989 | 0.983 | 1.000 | 0.967 |

### Ablations (one switched-off mechanism each; noise 0, full GCA = 0.988)

| ablation | accuracy | ablation | accuracy |
|---|---|---|---|
| no_locking | 0.818 | no_library | 0.909 |
| no_lexicon_unify | 0.855 | no_priors | 0.983 |
| no_mismatch | 0.893 | no_play | 0.988 |
| no_analogy | 0.906 | no_diagnosis | 0.990 |

At noise 0.1 (full = 0.819): no_mismatch 0.311, no_locking 0.680, no_diagnosis 0.817. Diagnosis
and play add nothing measurable on clean streams, because there are no errors there to fix.
Diagnosis shows its effect under fault injection.

### Self-diagnosis (fault injection, seeds 50–52)

A known fault is injected into the learner's knowledge, and the run checks whether diagnosis
names the right component **and** the right cause:

| fault | correct diagnosis |
|---|---|
| misread (role order swapped) | 66 / 69 |
| out_of_scope (concept used outside its region) | 62 / 62 |
| slip (burst of contradicting noisy observations) | 68 / 69 |
| wrong_concept (concept body replaced) | 66 / 66 |

### Does keeping earlier knowledge help?

Groups: cumulative block F (S21–S24, rules only buildable from earlier concepts); transfer copies
(S10–S14, the same systems as S02/S05/S06/S04/S08 under new and reused symbols); the originals.

| noise 0 | cumulative S21–S24 | copies S10–S14 | originals | all 24 |
|---|---|---|---|---|
| B2 Popper, fresh for each system | 0.574 | 0.651 | 0.645 | 0.640 |
| B2-L Popper, keeps earlier programs | 0.464 | 0.586 | 0.628 | 0.529 |
| GCA without its library | 0.488 | 1.000 | 0.974 | 0.909 |
| **GCA** | **0.960** | **1.000** | **0.974** | **0.988** |

| noise 0.1 | cumulative | copies | originals | all |
|---|---|---|---|---|
| B2 | 0.396 | 0.434 | 0.447 | 0.421 |
| B2-L | 0.342 | 0.419 | 0.444 | 0.392 |
| GCA | 0.575 | 0.920 | 0.868 | 0.819 |

- Keeping its earlier programs made Popper **worse** in every group, even though it does reuse
  them (81 of 190 clauses on the copies call an earlier system's program). A likely cause, **not
  tested**: the library adds many body predicates, so the same 120 s budget searches a much
  larger space.
- GCA's library is what solves block F (0.960 with it, 0.488 without). On the copies GCA reaches
  1.000 with or without its library, so there the library only saves examples, and only a
  little (see [Limitations](#limitations-and-open-problems)).
- Per-rule cost to lock at noise 0 is 10.2 examples + checker calls with the library and 13.7
  without it (26% less).

Result files: `results/final3/` (GCA, B1, ablations, faults, tables) and `results/baselines/`
(B2, B2+core, B2-L, B3, B4).

---

## How it works

```
experience ──► read ──► learn (active queries, shared checker budget) ──► answer ──► feedback
                 ▲                                                             │
                 └──────────── diagnose mistake → fix → re-test ◄──────────────┘
                                      rest / play (no task running)
```

| component | prototype implementation | code |
|---|---|---|
| **Micro-columns** | Equal-size knowledge units (fixed node budget `S_MAX`) with keys, ports and provenance. Lifecycle: plastic → candidate → validated-locked, plus forked, merged, retired and superseded. New knowledge first tries to **extend** a related column; the extension is accepted only if everything that column already proved still holds. | `gca/memory/columns.py`, `gca/improve/lifecycle.py`, `gca/learn/linking.py` |
| **Concept index** | Separate stores for identities, names, kinds, frames (situation types with roles), lexicon (relation words → roles) and events. Each fact is stored **once**, as a frame with roles, and linked from every participant, so inverse relations are discovered rather than memorised (reversal). | `gca/memory/stores.py`, `gca/improve/concepts.py` |
| **Binding (VSA)** | MAP bipolar hypervectors (~10,000 dims) with bind/unbind/bundle/permute, cleanup through the concept index, and resonator networks. Fixed random codes; nothing about them is learned. | `gca/memory/vsa.py` |
| **Resonance router** | ART-style and non-parametric: VSA keys plus a vigilance threshold pick which columns run. | `gca/think/router.py` |
| **Global workspace** | A k-slot blackboard where the chosen columns compete, combine and broadcast. | `gca/think/workspace.py` |
| **Three spaces** | Whiteboard (provisional), notebook (validated), mismatch (shown wrong, kept as negative examples). | `gca/memory/spaces.py` |
| **Energy-based reasoner** | Finds the most consistent configuration (violated constraints + description length) instead of predicting a next token. | `gca/think/energy.py`, `gca/think/solver.py` |
| **Factorized decoder** | Turns a settled configuration into an answer, including `none` / `many` / `unknown`. | `gca/think/decoder.py` |
| **Rule learning** | Enumeration over a small innate core (equality, constants, position, iteration) scored by MDL, with priors counted from earlier systems, analogy, and active queries. | `gca/learn/` |
| **Diagnosis** | Finds the first wrong step and names its cause: *misread* → fix the reading; *out of scope* → add a condition; *wrong concept* → supersede the column; *slip* → learn nothing. Noise is flagged as a slip only after the checker refutes it (an SPRT stands in only when the budget is spent). | `gca/improve/diagnosis.py`, `gca/improve/sprt.py`, `gca/improve/hitting_sets.py` |
| **Value signal** | Goal relevance × learning progress; decides what is worth pursuing. | `gca/improve/value.py` |
| **Play** | With no task running, invents primitives and keeps only those that pass a usefulness / compression test. | `gca/improve/play.py` |

Perception encoders, a neural decoder, physical tests and natural language are out of scope for
this prototype.

---

## The test domain

`gca/world/` holds the **hidden world**: the generator (`generator.py`), the private operator
grammar (`grammar.py`) and the checker (`checker.py`). Agent code may import only
`gca/world/api.py`, and a test enforces this.

A stream is 24 systems in sequence (`configs/stream_default.yaml`), in blocks:

| block | systems | purpose |
|---|---|---|
| A foundations | S01–S04 | successor chains, addition, order |
| B building on it | S05–S09 | multiplication, subtraction (alternative phrasing), affine maps |
| C transfer | S10–S14 | isomorphic copies of earlier systems under new and reused tokens |
| D harder | S15–S18 | structure from tables only (no successor statements) |
| E new structures | S19–S20 | late-life learning speed |
| F cumulative | S21–S24 | rules too large to find from scratch, buildable by linking earlier concepts |

Every system is **identifiable** from its examples. Some operators are invertible, so reversal
can be tested. Their properties (identity, inverse, commutativity, associativity, …) are hidden.

**Test splits:**

| split | what it tests | example |
|---|---|---|
| S1 in-distribution | one operator, forward, on an unseen cell | `zorp(dax, wug) = ?` |
| S2 composition | learned operators combined in ways never seen | `zorp(dax, kiv(wug)) = ?` |
| S3 depth | expressions deeper than any in experience (depth 3–6) | `zorp(zorp(dax,wug), kiv(kiv(fep))) = ?` |
| S4 reversal | solve for an input; reversed phrasing of facts seen one way | `zorp(dax, ?) = fep` |
| S4b honest reversal | non-invertible cases: the answer is `none` or `many` | `mul(?, z0) = dax` |
| S5 transfer and binding | isomorphic systems under new tokens; reused tokens with new meanings | examples-to-criterion vs the original |

Default settings: 35% of each binary table observed, 6 checker calls per word (the same for every
learner), 20 practice questions with feedback, 60 test questions per split, label noise
η ∈ {0, 0.05, 0.1, 0.2}.

---

## Installation

Tested on Python 3.14 (the code uses 3.10+ syntax). The core runs on **CPU only** and uses only the
standard library, `numpy` and `pyyaml`.

```bash
git clone https://github.com/hari08varma/GI-Labs.git
cd GI-Labs
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
# test suite (47 tests, about 30 s)
python -m pytest -q

# GCA on one 24-system stream: seed 50, noise 0.1 (a few minutes on a laptop CPU)
python -m experiments.run_stream --seed 50 --noise 0.1 --learner gca --out results/my_run.json

# the same stream with one mechanism switched off
python -m experiments.run_stream --seed 50 --noise 0.1 --learner gca --ablation no_library --out results/my_nolib.json
```

Ablation names are listed in `gca/config.py` (`ABLATIONS`). The rules GCA learns are readable
and are stored in the run JSON under `systems[*].rules`.

---

## Reproducing the results

Everything is seeded. The same seed gives byte-identical logs, and a test checks this.

**GCA, B1 and ablations** (CPU):

```bash
for s in 50 51 52; do for n in 0.0 0.05 0.1 0.2; do
  python -m experiments.run_stream --seed $s --noise $n --learner gca --out results/final3/main/gca_s${s}_n${n}.json
  python -m experiments.run_stream --seed $s --noise $n --learner b1  --out results/final3/main/b1_s${s}_n${n}.json
done; done

for a in no_library no_priors no_mismatch no_diagnosis no_locking no_play no_analogy no_lexicon_unify; do
  for s in 50 51 52; do
    python -m experiments.run_stream --seed $s --noise 0.0 --learner gca --ablation $a --out results/final3/ablations/${a}_s${s}_n0.0.json
  done
done
for a in no_mismatch no_diagnosis no_locking; do for s in 50 51 52; do     # also at noise 0.1
  python -m experiments.run_stream --seed $s --noise 0.1 --learner gca --ablation $a --out results/final3/ablations/${a}_s${s}_n0.1.json
done; done

python -m experiments.report results/final3/main      > results/final3/main_table.md
python -m experiments.report results/final3/ablations > results/final3/ablation_table.md
python -m experiments.continual results/final3/main results/final3/ablations > results/final3/continual.md
```

**Fault injection:**

```bash
for t in misread out_of_scope wrong_concept slip; do for s in 50 51 52; do
  python -m experiments.run_faults --seed $s --types $t --out results/final3/faults/${t}_s${s}.json
done; done
```

**Data for the external baselines.** Streams and answer keys are not stored in the repo. They are
regenerated exactly from the generator (byte-identical; checked):

```bash
python -m experiments.export_dataset --seeds 50 51 52 --noise 0.0 0.1 --out baselines/data
python -m experiments.export_dataset --seeds 0 1 2    --noise 0.0     --out baselines/data   # B3 dev seeds
```

Each `stream_s*_n*.jsonl` holds exactly what every learner sees: experience, the checker budget
as extra noise-free labels, practice questions with answers, and test questions. Answer keys go
to a separate `keys_*.json` that only the scorer reads.

**Scoring the external baselines:**

```bash
python -m experiments.score_baselines --modes b2 b2c b2l                                 # Popper variants
python -m experiments.score_baselines --preds results/baselines/b4 --modes b4            # right after each system
python -m experiments.score_baselines --preds results/baselines/b4 --modes b4 --part after_stream   # forgetting
python -m experiments.b2l_compare                                                        # B2 vs B2-L vs GCA
```

---

## Baselines

All baselines get the same data and the same checker budget (spent as extra noise-free labels,
because they do not query actively).

| id | baseline | where it runs | code |
|---|---|---|---|
| B1 | brute-force rule enumerator: the same innate core, search and MDL, but no library, priors, diagnosis or memory across systems | CPU | `gca/config.py`, `--learner b1` |
| B2 | [Popper](https://github.com/logic-and-learning-lab/Popper) ILP (learning from failures), fresh for each system, 120 s per word | SWI-Prolog ≥ 9.1.12 | `baselines/popper_runner/run_popper.py` |
| B2+core | Popper + our innate core (`iter`, `pos`, `iterc`, `lt`) as background knowledge, learned in two stages | as B2 | `run_popper.py --core --two-stage` |
| B2-L | Popper that keeps earlier programs (lifted into background knowledge), Forgetgol setting; **no human-written bias per system** | as B2 | `run_popper.py --accumulate` |
| B3 | transformer from scratch (d 128, 2 layers picked from {2,3,4} on dev seeds, 3000 steps, AdamW, weight decay 1.0) | Kaggle GPU | `baselines/transformer_scratch.py` |
| B4 | Qwen2.5-0.5B + LoRA (r 16, α 32, q/k/v/o, 8 epochs per system), one adapter trained through the stream | Kaggle GPU | `baselines/lora_llm.py` |

**Popper setup.** Popper is pinned at commit `897b087`. It needs SWI-Prolog ≥ 9.1.12, `uv` and
`clingo`. Ways to run it:

- **WSL / Linux:** `bash baselines/popper_runner/run_wsl.sh <run_popper args>`. On GCC 14, build
  `janus-swi` with `CFLAGS=-Wno-error=incompatible-pointer-types`.
- **Kaggle CPU notebooks:** `baselines/kaggle/gca-b2l/` (`setup.sh`, `run_b2l.sh`) and
  `b2l_stream_*.ipynb`, one notebook per stream.
- **Modal:** `baselines/popper_runner/modal_app.py`.

The fixed background knowledge for every world is written by one rule (no per-system human
bias). Answering uses standard ILP B + H semantics, with tabling and a 2 s limit per question; a
time-out scores `unknown`.

**Kaggle packages** (`baselines/kaggle/gca-b3/`, `gca-b4/`, `gca-b2l/`): each holds a driver
script and its notebook. Before uploading as a Kaggle dataset, copy the matching streams from
`baselines/data/` into `<package>/data/`. **Test answer keys never go to Kaggle**; scoring runs
locally.

**Not yet run:** a frontier model through its API with in-context examples (B5).

---

## Repository layout

```
gca/
  agent.py            the GCA agent loop behind a single Learner protocol
  config.py           hyperparameters, ablation switches, B1 definition
  eventlog.py         deterministic event log
  world/              HIDDEN world: generator, grammar, checker; agents may import only api.py
  memory/             columns, concept stores, three spaces, VSA
  learn/              reading, enumeration, MDL, priors, induction, linking, validation, active queries
  think/              router, workspace, energy solver, analogy, decoder
  improve/            diagnosis, lifecycle, shared concepts, SPRT, value signal, play
configs/
  stream_default.yaml the 24-system stream
experiments/          harness, run_stream, run_faults, report, continual, curves, export_dataset,
                      score_baselines, b2l_compare, analysis probes
baselines/
  transformer_scratch.py   B3
  lora_llm.py              B4
  popper_runner/           B2 / B2+core / B2-L (run_popper.py, Modal and WSL launchers)
  kaggle/                  Kaggle packages and notebooks for B2-L, B3, B4
tests/                47 tests: VSA, stores, world, learning, concepts, improvement, integrity
results/
  final3/             final GCA / B1 / ablation / fault-injection runs on test seeds 50–52, tables
  baselines/          B2, B2+core, B2-L (popper/), B3 (b3/), B4 (b4/) predictions and logs
  final, final2, main, main_v1, main_v2, ablations, faults, dev/   earlier runs, kept for the record
```

---

## Testing and integrity rules

```bash
python -m pytest -q
```

`tests/test_integrity.py` enforces the rules that keep the evaluation honest:

- **Agent code never imports world internals.** Only `gca/world/api.py` is allowed. The test
  parses the imports of every agent file.
- **Budgets hold:** no learner gets more checker calls than its budget.
- **Determinism:** the same seed gives byte-identical event logs.
- **End to end:** a 3-system mini stream learns in under 60 s.

Design rules followed throughout:

- No learned weights and no gradient descent in the agent. Hypervectors are fixed random codes.
- Nothing is locked until it is validated against the checker. Locked evidence is frozen and old
  versions are archived, so later learning cannot cause forgetting.
- A new column sprouts only on **learnable, compressible** error, which guards against the
  "noisy TV" trap.

---

## Limitations and open problems

These are reported honestly; none of them is hidden in the tables above.

- **Transfer speed (milestone M7) is not met.** Transfer copies need 0.52–1.02× the original's
  examples; the target is ≤ 0.30×. The library's measured benefit is the cumulative block F and a
  26% lower cost per locked rule, not fast relearning of copies.
- **Noise costs accuracy.** At η = 0.2, GCA reaches 0.632 and abstains on 36% of questions. It
  stays calibrated (0.7% confidently wrong), but it does not recover the missing knowledge.
- **Diagnosis and play** have no measurable effect on clean streams; diagnosis is validated by
  fault injection only.
- **A self-check before answering** (`Flags.selfcheck`, ablation `with_selfcheck`) is off by
  default: in A/B tests on dev seeds it withheld only correct answers, because its second route
  reads the same evidence as the first.
- **B2-L run failures (kept, not rerun):** a Prolog tabling-dependency error on 4 words (S04 in
  three streams, S13 in one), 2 hung words (S16, s52 n0) and 7 unanswered questions (s51 n0).
  Upper bound on their effect: 1.3 points on the noise-0 mean, 0 at noise 0.1.
- **B3 / B4 cannot abstain**, which inflates their confidently-wrong rate. A confidence threshold
  chosen on dev seeds and a data-size curve (35 / 60 / 90% of each table) are planned, not yet run.
- **B5** (a frontier model with in-context examples) is not yet run.
- The scope is a CPU prototype on invented maths. Perception, language and physical tasks are
  later phases. Nothing here shows how the approach scales beyond this domain.
- Known code debt: `gca/world/generator.py` has dead lines in `_questions` / `_experience`.

---

## References

- Cropper & Morel, *Learning programs by learning from failures* (Popper). http://andrewcropper.com/pubs/popper.pdf
- Cropper et al., *Inductive logic programming at 30*. https://arxiv.org/abs/2008.07912
- Cropper, *Forgetting to learn logic programs* (Forgetgol). https://arxiv.org/abs/2002.11002
- Wang & Sun, *The reversal curse as a binding problem*. https://arxiv.org/abs/2504.01928
- Kleyko et al., *A survey on hyperdimensional computing / vector symbolic architectures*. https://arxiv.org/abs/2111.06077
- Ellis et al., *DreamCoder*. https://arxiv.org/abs/2006.08381
- Lightman et al., *Let's Verify Step by Step*. https://arxiv.org/abs/2305.20050
- Shinn et al., *Reflexion*. https://arxiv.org/abs/2303.11366
- Ohlsson, *learning from performance errors*. https://edutechwiki.unige.ch/en/Learning_theory_of_performance_errors
- Doumas, Hummel & Sandhofer, *DORA: a theory of the discovery and predication of relational concepts*. https://labs.psychology.illinois.edu/~jehummel/pubs/DORA_PR_2008.pdf
- Goyal et al., *Recurrent Independent Mechanisms* and *Coordination among neural modules through a shared global workspace*.
- Lopez-Paz & Ranzato, *Gradient Episodic Memory* (ACC / BWT / FWT definitions). https://arxiv.org/abs/1706.08840
- Power et al., *Grokking*. https://arxiv.org/abs/2201.02177

## License

No license has been chosen yet. Until one is added, all rights are reserved by the author.
