"""B3: a transformer trained from scratch (plan §10). Runs on CPU (laptop) or GPU.

Same data as every learner: a system's experience + its checker budget as extra noise-free
labels + its practice questions with answers (we got feedback on those, so they get them as
labels). Each fact or question is a token sequence ending in "=", and the model predicts the
answer token at that position (objects, #T/#F, none, many). Two settings:
  --mode per_system   a fresh model per system
  --mode sequential   one model trained through the stream in order (tests forgetting;
                      earlier systems are re-scored at the end)
Training budget is fixed before results exist (grokking needs many steps and weight decay,
Power et al. 2022): --steps 3000, AdamW lr 1e-3, weight decay 1.0, batch 64, d = 128.
Depth (plan: 2-4 layers) is chosen on dev seeds 0-2 only, never on test seeds.
Progress is saved after every system (<out>.ckpt.pt) and a rerun resumes from it.

    python baselines/transformer_scratch.py baselines/data/stream_s50_n0.0.jsonl --out results/baselines/b3/... --layers 3
Score:  python -m experiments.score_baselines --preds results/baselines/b3 --modes b3_L3_per_system
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn

SPECIAL = ["<pad>", "(", ")", ",", "=", "?", "none", "many", "#T", "#F"]


def seq(expr) -> list[str]:
    if expr[0] == "obj":
        return [expr[1]]
    if expr[0] == "var":
        return ["?"]
    out = [expr[1], "("]
    for i, e in enumerate(expr[2:]):
        if i:
            out.append(",")
        out += seq(e)
    return out + [")"]


def fact_seq(o) -> list[str]:
    return [o["word"], "("] + sum(([a, ","] for a in o["args"]), [])[:-1] + [")", "="]


def q_seq(q) -> list[str]:
    s = seq(q["expr"]) + ["="]
    if q["kind"] == "solve":
        s = s[:-1] + ["=", q["target"], "?"]
    return s


class Model(nn.Module):
    def __init__(self, vocab: int, maxlen: int, d: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(maxlen, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        h = self.enc(h, src_key_padding_mask=(x == 0))
        last = (x != 0).sum(1) - 1
        return self.out(h[torch.arange(len(x)), last])


def build_examples(rec, vocab):
    ex = []
    for o in rec["experience"] + rec["extra_labels"]:
        ex.append((fact_seq(o), o["out"]))
    for q in rec["practice"]:
        ex.append((q_seq(q), q["key"]))
    for s, y in ex:
        for t in s + [y]:
            vocab.setdefault(t, len(vocab))
    for q in rec["questions"]:
        for t in q_seq(q):
            vocab.setdefault(t, len(vocab))
    return ex


def encode(s, vocab, maxlen):
    assert len(s) <= maxlen, f"sequence of {len(s)} tokens exceeds maxlen {maxlen}"   # never truncate
    ids = [vocab.get(t, 0) for t in s]
    return ids + [0] * (maxlen - len(ids))


def trim(x):
    """Cut a batch to its longest sequence: the padding is masked anyway, so the result is the same,
    only faster (most facts are 6-8 tokens, the longest question about 100)."""
    return x[:, : int((x != 0).sum(1).max())]


def train(model, ex, vocab, steps, dev, seed, maxlen):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    X = torch.tensor([encode(s, vocab, maxlen) for s, _ in ex], device=dev)
    Y = torch.tensor([vocab[y] for _, y in ex], device=dev)
    model.train()
    for _ in range(steps):
        idx = torch.tensor([rng.randrange(len(ex)) for _ in range(min(64, len(ex)))], device=dev)
        loss = nn.functional.cross_entropy(model(trim(X[idx])), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def predict(model, rec, vocab, dev, maxlen):
    model.eval()
    inv = {i: t for t, i in vocab.items()}
    X = torch.tensor([encode(q_seq(q), vocab, maxlen) for q in rec["questions"]], device=dev)
    out = model(trim(X)).argmax(-1).tolist()
    return {q["qid"]: inv[i] for q, i in zip(rec["questions"], out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="per_system", choices=["per_system", "sequential"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=int, default=None, help="one system index (timing check)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = [json.loads(line) for line in open(a.stream, encoding="utf-8")]
    if a.only is not None:
        recs = [recs[a.only]]
    # longest sequence in this stream (+1 spare): nothing is ever cut off
    maxlen = 1 + max(len(s) for r in recs for s in
                     [fact_seq(o) for o in r["experience"] + r["extra_labels"]] +
                     [q_seq(q) for q in r["practice"] + r["questions"]])
    ck_path = a.out + ".ckpt.pt"
    ck = torch.load(ck_path, weights_only=False) if os.path.exists(ck_path) else None
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.manual_seed(a.seed)
    if a.mode == "per_system":
        preds = ck["preds"] if ck else {}
        done = ck["done"] if ck else -1
        for k, rec in enumerate(recs):
            if k <= done:
                continue
            t0 = time.time()
            torch.manual_seed(a.seed * 1000 + k)        # each system's model is reproducible on its own
            vocab = {t: i for i, t in enumerate(SPECIAL)}
            ex = build_examples(rec, vocab)
            model = Model(len(vocab), maxlen, layers=a.layers).to(dev)
            train(model, ex, vocab, a.steps, dev, a.seed, maxlen)
            preds.update(predict(model, rec, vocab, dev, maxlen))
            torch.save({"done": k, "preds": preds}, ck_path)
            print(rec["system"], f"{time.time() - t0:.0f}s", flush=True)
        result = {"predictions": preds}
    else:
        vocab = {t: i for i, t in enumerate(SPECIAL)}
        exs = [build_examples(rec, vocab) for rec in recs]       # vocabulary fixed up front
        model = Model(len(vocab), maxlen, layers=a.layers).to(dev)
        after_own, done = {}, -1
        if ck:
            model.load_state_dict(ck["model"])
            after_own, done = ck["after_own"], ck["done"]
        for k, (rec, ex) in enumerate(zip(recs, exs)):
            if k <= done:
                continue
            t0 = time.time()
            train(model, ex, vocab, a.steps, dev, a.seed + k, maxlen)
            after_own.update(predict(model, rec, vocab, dev, maxlen))
            torch.save({"done": k, "after_own": after_own, "model": model.state_dict()}, ck_path)
            print(rec["system"], f"{time.time() - t0:.0f}s", flush=True)
        final = {}
        for rec in recs:
            final.update(predict(model, rec, vocab, dev, maxlen))
        result = {"predictions": after_own, "after_own_system": after_own, "after_stream": final}
    json.dump(result, open(a.out, "w"))


if __name__ == "__main__":
    main()
