"""B2: Popper (Cropper & Morel 2021) on the same streams (plan §10).

Setup, fixed once for all systems (no human writes bias or BK per system):
- Words are renamed by a fixed rule to slot names r1, r2, ... in order of first appearance,
  so one generator writes every bias file. Each word becomes a target in turn; the other
  words' facts of the same system are its background knowledge.
- Examples: experience + the same budget as extra labels (Popper does not query actively).
  Functional facts give negatives for every other output (plan §10). --noisy for noise > 0.
- B2-L (--accumulate): programs learned in earlier systems are lifted over slot names and
  added to later systems' BK as lib_k predicates (the Forgetgol setting).
- Answers come from SWI-Prolog: forward questions by query, solve questions by findall
  (none / unique / many), abstain ("unknown") when the program gives nothing.

Needs: SWI-Prolog on PATH, `pip install clingo`, and Popper
(`pip install git+https://github.com/logic-and-learning-lab/Popper@main`).
Usage:
    python -m experiments.export_dataset --seeds 10 --noise 0.0 --out baselines/data
    python baselines/popper_runner/run_popper.py baselines/data/stream_s10_n0.0.jsonl --out results/popper_s10_n0.0.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor


def slots(rec: dict) -> dict:
    """Fixed renaming: words -> r1, r2, ... in order of first appearance."""
    order = []
    for o in rec["experience"] + rec["extra_labels"]:
        if o["word"] not in order:
            order.append(o["word"])
    return {w: f"r{i + 1}" for i, w in enumerate(order)}


def atom(tok: str) -> str:
    return {"#T": "true", "#F": "false"}.get(tok, "o_" + re.sub(r"\W", "_", tok))


def facts(rec: dict, names: dict) -> dict:
    """slot -> list of (args..., out) with tokens as atoms."""
    out: dict[str, list] = {}
    for o in rec["experience"] + rec["extra_labels"]:
        out.setdefault(names[o["word"]], []).append(tuple(atom(t) for t in o["args"]) + (atom(o["out"]),))
    return out


def objects(rec: dict) -> list[str]:
    toks = set()
    for o in rec["experience"] + rec["extra_labels"]:
        toks |= set(o["args"])
        if o["out"] not in ("#T", "#F"):
            toks.add(o["out"])
    return sorted(atom(t) for t in toks)


def write_task(d: str, target: str, fs: dict, objs: list[str], lib: list[str], noisy: bool) -> None:
    os.makedirs(d, exist_ok=True)
    ar = len(fs[target][0])
    boolean = any(f[-1] in ("true", "false") for f in fs[target])
    with open(os.path.join(d, "exs.pl"), "w") as f:
        seen = set()
        for t in fs[target]:
            f.write(f"pos({target}({','.join(t)})).\n")
            seen.add(t)
        outs = ["true", "false"] if boolean else objs
        for t in fs[target]:
            for o in outs:
                n = t[:-1] + (o,)
                if o != t[-1] and n not in seen:
                    f.write(f"neg({target}({','.join(n)})).\n")
    with open(os.path.join(d, "bk.pl"), "w") as f:
        f.write(":- discontiguous obj/1.\n:- discontiguous eq/2.\n")
        f.writelines(x + "\n" for x in lib_decls(lib))
        # earlier systems' programs can call each other in a loop (r1 = flip of r2, r2 = flip of
        # r1): tabled, such a loop simply has no answers instead of hanging Popper's tests
        f.writelines(f":- table {n}/{a}.\n" for n, a in sorted(lib_heads(lib)))
        for s, ts in fs.items():
            if s != target:
                for t in ts:
                    f.write(f"{s}({','.join(t)}).\n")
        # the innate core in Prolog, written by one fixed rule for every world: equality and one
        # constant predicate per object (our learner has equality and constants too)
        for i, o in enumerate(objs):
            f.write(f"obj({o}).\neq({o},{o}).\n" + ("" if MAGIC else f"c{i}({o}).\n"))
        f.writelines(x + "\n" for x in lib)
        if CORE:
            f.writelines(x + "\n" for x in core_bk(fs, target, len(objs))[0])
    with open(os.path.join(d, "bias.pl"), "w") as f:
        f.write(f"head_pred({target},{ar}).\n")
        if CORE:
            f.writelines(x + "\n" for x in core_bk(fs, target, len(objs))[1])
        for s, ts in fs.items():
            if s != target:
                f.write(f"body_pred({s},{len(ts[0])}).\n")
        f.write("body_pred(eq,2).\n")
        if MAGIC:                             # MagicPopper finds constants itself: no c_i, none named
            f.write("magic_value_all.\nmax_magic(2).\nmax_vars(6).\nmax_body(5).\n")
        else:
            for i in range(len(objs)):
                f.write(f"body_pred(c{i},1).\n")
        for name, n in sorted(lib_heads(lib)):
            if not is_lib_const(name):         # old systems' constants are not building blocks here
                f.write(f"body_pred({name},{n}).\n")
        f.write("enable_recursion.\n")         # max_vars / max_body / noisy are command-line flags


def core_bk(fs: dict, exclude: str | None, n_objs: int, pre: str = "") -> tuple[list[str], list[str]]:
    """Our innate core in Prolog (gca/learn/hypothesis.py), written by one fixed rule for every
    world, over the facts of each OTHER word (`pre` + slot names the fact predicate):
      iter_u(K, X, Z)      apply unary word u K times starting from X        (iter)
      pos_u(A, X, K)       position of X in the chain A, u(A), u(u(A)), ...  (pos, shortest K)
      iterc_b(P, K, X, Z)  apply t -> b(P, t) K times starting from X        (iterc)
      lt(K1, K2)           comparison of counts                              (lt)
    Counts are integers 0..n-1 (n = number of objects). Returns (clauses, body_pred lines)."""
    # Popper calls BK with any bindings, so a count slot holding an object must fail, not raise
    cl = [f"cnt(K) :- integer(K), !, K >= 0, K =< {n_objs - 1}.",
          f"cnt(K) :- var(K), between(0,{n_objs - 1},K).",
          "cntok(K) :- var(K), !.",
          "cntok(K) :- integer(K).",
          "lt(K1,K2) :- cnt(K1), cnt(K2), K1 < K2."]
    bias = ["body_pred(lt,2)."]
    for s, ts in fs.items():
        if s == exclude or any(t[-1] in ("true", "false") for t in ts):   # relations: no chains
            continue
        f = pre + s
        if len(ts[0]) == 2:                    # unary word: arg, out
            cl += [f"iter_{s}(K,X,Z) :- cntok(K), obj(X), iter_{s}_(X,0,K,Z).",
                   f"iter_{s}_(X,K,K,X).",
                   f"iter_{s}_(X,I,K,Z) :- I < {n_objs - 1}, {f}(X,Y), I1 is I+1, iter_{s}_(Y,I1,K,Z).",
                   f"pos_{s}(A,X,K) :- obj(A), obj(X), once(iter_{s}_(A,0,K0,X)), K = K0."]
            bias += [f"body_pred(iter_{s},3).", f"body_pred(pos_{s},3)."]
        elif len(ts[0]) == 3:                  # binary word: arg1, arg2, out
            cl += [f"iterc_{s}(P,K,X,Z) :- cntok(K), obj(P), obj(X), iterc_{s}_(P,X,0,K,Z).",
                   f"iterc_{s}_(_,X,K,K,X).",
                   f"iterc_{s}_(P,X,I,K,Z) :- I < {n_objs - 1}, {f}(P,X,Y), I1 is I+1, iterc_{s}_(P,Y,I1,K,Z)."]
            bias += [f"body_pred(iterc_{s},4)."]
    return cl, bias


CORE = os.environ.get("POPPER_CORE") == "1"        # set by --core


def lib_heads(lib: list[str]) -> set:
    """(name, arity) of every lib_ predicate defined by a clause head in lib."""
    out = set()
    for x in lib:
        m = re.match(r"(lib_\w+)\(([^)]*)\)", x)
        if m:
            out.add((m.group(1), len(m.group(2).split(","))))
    return out


def lib_decls(lib: list[str]) -> list[str]:
    """Declare every lib_ predicate that is called, so a call to a slot never learned fails
    instead of raising an existence error."""
    called = set()
    for x in lib:
        for m in re.finditer(r"(lib_\w+)\(([^()]*)\)", x):
            called.add((m.group(1), len(m.group(2).split(","))))
    return [f":- dynamic {n}/{a}." for n, a in sorted(called - lib_heads(lib))]


POPPER_CMD = os.environ.get("POPPER_CMD", "uv run --project /popper /popper/popper.py").split()
MAGIC = os.environ.get("POPPER_MAGIC") == "1"      # MagicPopper (Hocquette & Cropper 2023)


def popper(d: str, noisy: bool, timeout: int) -> tuple[list[str], str]:
    """Run Popper's command-line tool on one task directory; return (learned clauses, raw log tail)."""
    cmd = POPPER_CMD + [d, "--timeout", str(timeout)]
    if not MAGIC:                             # MagicPopper reads these from the bias file
        cmd += ["--max-vars", "6", "--max-body", "5"]
    if noisy and not MAGIC:
        cmd.append("--noisy")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        return [], "timeout"
    raw = r.stdout[-1500:] + " --stderr-- " + r.stderr[-1500:]
    out, inside = [], False
    for ln in (r.stdout + "\n" + r.stderr).splitlines():      # Popper logs its solution to stderr
        t = re.sub(r"^\d+(\.\d+)?s\s+", "", ln.strip())          # drop the timestamp prefix
        if "SOLUTION" in t:
            inside = True
            continue
        if inside and t.startswith("*"):
            break
        if inside and (":-" in t or re.match(r"^\w+\(.*\)\.$", t)):
            out.append(t)
    return out, raw


def to_query(expr, names: dict, var: list[int]):
    """Expression -> (prolog goal list, result variable)."""
    if expr[0] == "obj":
        return [], atom(expr[1])
    if expr[0] == "var":
        return [], "Q"
    goals, args = [], []
    for e in expr[2:]:
        g, v = to_query(e, names, var)
        goals += g
        args.append(v)
    var[0] += 1
    out = f"V{var[0]}"
    goals.append(f"{names.get(expr[1], 'missing_' + str(var[0]))}({','.join(args + [out])})")
    return goals, out


def run_swipl(prog: str, timeout: int) -> str:
    """Run a Prolog program's main/0; return its output (what was printed before a time-out)."""
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as f:
        f.write(":- use_module(library(time)).\n" + prog)
        path = f.name
    try:
        return subprocess.run(["swipl", "-q", "-g", "main", "-t", "halt", path], capture_output=True, text=True,
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired as e:
        return e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    finally:
        os.unlink(path)


def complete_unary(units: dict, programs: list[str], objs: list[str]) -> dict:
    """Two-stage B2+core, stage 1 -> 2: fill each one-argument word's table with ITS OWN learned
    program (B + H over the observed facts), so pos/iter in stage 2 run on Popper's own successor.
    An object gets a value only where the program gives exactly one; observed facts always stay."""
    lines = program_lines(units, programs, objs, None)
    qs = [f"q :- obj(X), findall(Z, {s}(X,Z), L), sort(L, [Z1]), writeq(val({s}, X, Z1)), nl, fail."
          for s in units]
    out = run_swipl("\n".join(lines + qs + ["q.", "main :- q."]) + "\n", timeout=300)
    full = {s: set(ts) for s, ts in units.items()}
    for m in re.finditer(r"val\((\w+),\s*(\w+),\s*(\w+)\)", out):
        s, x, z = m.groups()
        if not any(t[0] == x for t in full[s]):
            full[s].add((x, z))
    return {s: sorted(v) for s, v in full.items()}


def program_lines(fs: dict, programs: list[str], objs: list[str], lib: list[str] | None) -> list[str]:
    """Each word's program is evaluated with the background knowledge it was learned with (standard
    ILP: B + H): its body calls the OTHER words' facts (f_*), a recursive call goes to its own
    program (h_*). Answering a word = its observed facts or its program. Tabling makes recursion
    over the finite set of objects terminate."""
    ar = {s: len(ts[0]) for s, ts in fs.items()}
    tabled = sorted({(f"h_{s}", a) for s, a in ar.items()} | {(s, a) for s, a in ar.items()}
                    | lib_heads(list(lib or [])))
    lines = [":- discontiguous obj/1.", ":- discontiguous eq/2."]
    lines += [f":- table {n}/{a}." for n, a in tabled]
    lines += [f":- dynamic f_{s}/{a}." for s, a in ar.items()]
    lines += lib_decls(list(lib or []))
    lines += [f"f_{s}({','.join(t)})." for s, ts in fs.items() for t in ts]
    for i, o in enumerate(objs):                     # the same background knowledge as learning
        lines += [f"obj({o}).", f"eq({o},{o}).", f"c{i}({o})."]
    lines += list(lib or [])
    if CORE:                                  # the same core predicates, over the facts (f_*)
        lines += core_bk(fs, None, len(objs), pre="f_")[0]
    for c in programs:
        m = re.match(r"(r\d+)\(", c)
        if not m:
            continue
        head = m.group(1)
        lines.append(re.sub(r"\b(r\d+)\(", lambda x: f"h_{head}(" if x.group(1) == head
                            else (f"f_{x.group(1)}(" if x.group(1) in ar else x.group(0)), c))
    for s, a in ar.items():
        v = ",".join(f"A{i}" for i in range(a))
        lines += [f"{s}({v}) :- f_{s}({v}).", f"{s}({v}) :- h_{s}({v})."]
    defined = {re.match(r"(r\d+)\(", c).group(1) for c in programs if re.match(r"(r\d+)\(", c)}
    lines += [f"h_{s}({','.join('_' for _ in range(a))}) :- fail." for s, a in ar.items() if s not in defined]
    return lines


def answer_all(rec: dict, names: dict, programs: list[str], fs: dict, timeout: int = 20,
               lib: list[str] | None = None) -> dict:
    lines = program_lines(fs, programs, objects(rec), lib)
    qs = []
    for q in rec["questions"]:
        goals, out = to_query(q["expr"], names, [0])
        if q["kind"] == "eval":
            qs.append((q["qid"], f"findall({out}, ({','.join(goals)}), L)"))
        else:
            tgt = atom(q["target"])
            # the unknown ranges over the system's objects, so the answer set is complete
            qs.append((q["qid"], f"findall(Q, (obj(Q), {','.join(goals)}, {out} == {tgt}), L)"))
    # a time-out or an error prints no answer line, so it scores as "unknown" (not as "none")
    body = "\n".join(f"q('{qid}') :- catch(call_with_time_limit(2, ({g})), _, fail), sort(L, S), "
                     f"writeq(ans('{qid}', S)), nl, flush_output." for qid, g in qs)
    prog = "\n".join(lines) + "\n" + body + "\nmain :- forall(clause(q(X), _), (q(X) -> true ; true)).\n"
    stdout = run_swipl(prog, timeout=3 * len(qs) + 60)    # keeps the answers printed before a time-out
    inv = {atom(t): t for o in rec["experience"] + rec["extra_labels"] for t in o["args"] + [o["out"]]}
    preds = {}
    kinds = {q["qid"]: q["kind"] for q in rec["questions"]}
    for m in re.finditer(r"ans\('([^']+)',\s*\[([^\]]*)\]\)", stdout):
        qid, vals = m.group(1), [v.strip() for v in m.group(2).split(",") if v.strip()]
        if kinds[qid] == "eval":
            preds[qid] = inv.get(vals[0], "unknown") if len(vals) == 1 else "unknown"
        else:
            preds[qid] = "none" if not vals else (inv.get(vals[0], "unknown") if len(vals) == 1 else "many")
    return preds


def lift(clauses: list[str], names: dict, k: int, objs: list[str]) -> list[str]:
    """B2-L: rename a system's learned program over slot names into lib_<k>_<slot> predicates.
    Its constants c_i (system k's objects) become lib_<k>_c<i> with system k's objects as facts,
    so the program keeps exactly the meaning it was learned with (in a later system those objects
    do not occur, so such a clause simply does not fire)."""
    slot = set(names.values())                 # head and body: a body call must stay in system k
    out = []
    for c in clauses:
        c = re.sub(r"\b(r\d+)\(", lambda m: f"lib_{k}_{m.group(1)}(" if m.group(1) in slot else m.group(0), c)
        out.append(re.sub(r"\bc(\d+)\(", lambda m: f"lib_{k}_c{m.group(1)}(", c))
    used = sorted({int(i) for c in out for i in re.findall(rf"\blib_{k}_c(\d+)\(", c)})
    return out + [f"lib_{k}_c{i}({objs[i]})." for i in used if i < len(objs)]


def is_lib_const(name: str) -> bool:
    return re.fullmatch(r"lib_\d+_c\d+", name) is not None


def check_core(rec: dict, fs: dict, objs: list[str]) -> list[str]:
    """Is the core rule  t(X,Y,Z) :- pos_u(A,Y,K), iter_u(K,X,Z)  (x + y as 'count y steps from x')
    expressible here? For each binary word t, unary word u and anchor A: positives and negatives
    covered. Uses the same bk.pl Popper gets."""
    out = []
    for t, ts in fs.items():
        if len(ts[0]) != 3:
            continue
        with tempfile.TemporaryDirectory() as d:
            write_task(d, t, fs, objs, [], False)
            us = [s for s, r in fs.items() if s != t and len(r[0]) == 2]
            goal = (f"chk :- member(U, [{','.join(us)}]), member(A, [{','.join(objs)}]), "
                    f"atom_concat(pos_, U, P), atom_concat(iter_, U, I), "
                    f"aggregate_all(count, (pos({t}(X,Y,Z)), call(P,A,Y,K), call(I,K,X,Z)), Pc), "
                    f"aggregate_all(count, pos({t}(_,_,_)), Pn), "
                    f"aggregate_all(count, (neg({t}(X2,Y2,Z2)), call(P,A,Y2,K2), call(I,K2,X2,Z2)), Nc), "
                    f"format('~w ~w ~w pos ~w/~w neg ~w~n', [{t}, U, A, Pc, Pn, Nc]), fail.\nchk.")
            prog = os.path.join(d, "chk.pl")
            with open(prog, "w") as fh:
                fh.write(f":- consult('{d}/bk.pl').\n:- consult('{d}/exs.pl').\n{goal}\n")
            r = subprocess.run(["swipl", "-q", "-g", "chk", "-t", "halt", prog], capture_output=True, text=True,
                               timeout=300)
            rows = [ln for ln in r.stdout.splitlines() if " pos " in ln]
            best = sorted(rows, key=lambda ln: (-int(ln.split("pos ")[1].split("/")[0]), int(ln.split("neg ")[1])))
            out += [rec["system"] + " " + ln for ln in best[:3]] + ([r.stderr[-300:]] if not rows else [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--accumulate", action="store_true", help="B2-L: add earlier programs to BK")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--only", type=int, default=None, help="run one system (index) only; B2 in parallel")
    ap.add_argument("--no-learn", action="store_true", help="answer from the facts alone (pipeline check)")
    ap.add_argument("--programs-from", default=None, help="reuse programs learned in an earlier run (re-answer)")
    ap.add_argument("--core", action="store_true", help="add our innate core (iter/pos/iterc/lt) to the BK")
    ap.add_argument("--full-unary", default=None,
                    help="DIAGNOSTIC only: complete one-argument tables from the hidden generator")
    ap.add_argument("--check-core", action="store_true", help="only check that the core rule fits")
    ap.add_argument("--checkpoint", default=None, help="save/resume state after every system (B2-L)")
    ap.add_argument("--max-par", type=int, default=None,
                    help="at most this many Popper searches at once (= cores), so each gets a full core")
    ap.add_argument("--two-stage", action="store_true",
                    help="B2+core: learn one-argument words first, fill their chains with their own "
                         "programs, then learn the rest with the core over those chains (implies --core)")
    a = ap.parse_args()
    global CORE
    CORE = CORE or a.core or a.two_stage
    full = json.load(open(a.full_unary, encoding="utf-8")) if a.full_unary else {}
    old = {}
    if a.programs_from:
        for e in json.load(open(a.programs_from, encoding="utf-8"))["programs"]:
            old[(e["system"], e["target"])] = e
    noisy = "_n0.0" not in a.stream
    lib: list[str] = []
    preds_all, log = {}, []
    done = -1
    if a.checkpoint and os.path.exists(a.checkpoint):  # resume after a preemption
        ck = json.load(open(a.checkpoint, encoding="utf-8"))
        lib, preds_all, log, done = ck["lib"], ck["predictions"], ck["programs"], ck["done"]
        print("resuming after system index", done, flush=True)
    for k, line in enumerate(open(a.stream, encoding="utf-8")):
        if (a.only is not None and k != a.only) or k <= done:
            continue
        rec = json.loads(line)
        names = slots(rec)
        fs = facts(rec, names)
        for w, rows in full.get(rec["system"], {}).items():      # diagnostic: complete chains
            if w in names:
                fs[names[w]] = sorted(set(fs.get(names[w], [])) | {tuple(atom(t) for t in r) for r in rows})
        objs = objects(rec)
        if a.check_core:
            print("\n".join(check_core(rec, fs, objects(rec))), flush=True)
            continue
        programs, t0 = [], time.time()
        with tempfile.TemporaryDirectory() as tmp:
            # the words of one system are independent learning tasks (each sees only the other
            # words' facts and earlier systems' lib), so they run in parallel, one process each
            def learn_one(target: str, fsx: dict) -> dict:
                if (rec["system"], target) in old:
                    return old[(rec["system"], target)]
                d = os.path.join(tmp, target)
                write_task(d, target, fsx, objs, lib if a.accumulate else [], noisy)
                prog, raw = ([], "") if a.no_learn else popper(d, noisy, a.timeout)
                return {"system": rec["system"], "target": target, "program": prog, "raw": raw}

            def learn_all(targets: list[str], fsx: dict) -> None:
                nonlocal programs
                n_par = len(targets) if a.max_par is None else min(len(targets), a.max_par)
                with ThreadPoolExecutor(max_workers=max(1, n_par)) as ex:
                    for e in ex.map(lambda t: learn_one(t, fsx), targets):   # map keeps the order
                        log.append(e)
                        programs += e["program"]

            if a.two_stage:
                # stage 1: one-argument words, from the other one-argument words only (so successor
                # cannot lean on addition); their own programs then fill the chains for stage 2
                units = {s: ts for s, ts in fs.items()
                         if len(ts[0]) == 2 and not any(t[-1] in ("true", "false") for t in ts)}
                learn_all(list(units), units)
                if units:
                    filled = complete_unary(units, programs, objs)
                    print(rec["system"], "chains filled:",
                          {s: f"{len(units[s])}->{len(filled[s])}" for s in units}, flush=True)
                    fs = {**fs, **filled}
                learn_all([s for s in fs if s not in units], fs)
            else:
                learn_all(list(fs), fs)
        try:                                  # an answering failure must not lose the learned programs
            preds_all.update(answer_all(rec, names, programs, fs, lib=lib if a.accumulate else None))
        except Exception as e:                # noqa: BLE001
            print(rec["system"], "answering failed:", repr(e)[:300], flush=True)
        if a.accumulate:
            lib += lift(programs, names, k, objs)
        print(rec["system"], f"{time.time() - t0:.0f}s", len(programs), "clauses", flush=True)
        if a.checkpoint:                      # one world finished: save the state to resume from
            os.makedirs(os.path.dirname(a.checkpoint) or ".", exist_ok=True)
            tmp_ck = a.checkpoint + ".tmp"
            json.dump({"done": k, "lib": lib, "predictions": preds_all, "programs": log}, open(tmp_ck, "w"))
            os.replace(tmp_ck, a.checkpoint)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"predictions": preds_all, "programs": log}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
