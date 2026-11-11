"""B4: a small pretrained LLM with a LoRA fine-tune (plan §10). Kaggle GPU.

Model: Qwen/Qwen2.5-0.5B (fixed before results). Same data as every learner: experience +
the checker budget as extra noise-free labels + practice questions with answers, written as
plain lines such as "zorp(dax, wug) = fep" or "zorp(dax, ?) = fep ; ? = wug". The LoRA adapter
is trained SEQUENTIALLY through the stream (one adapter, systems in order) to measure
forgetting; every system's questions are answered right after its own training and again at
the end. Answers are greedy generations; the first word is the answer, and anything that is not
a token seen in this system's data, #T/#F, none or many counts as "unknown".
Hyperparameters fixed before results: LoRA r 16, alpha 32, dropout 0.05 on q/k/v/o; AdamW lr 2e-4;
8 epochs per system; batch 16.

Progress is saved after every system (<out>.ckpt.pt: LoRA weights + answers) and a rerun resumes.
    python lora_llm.py stream_s50_n0.0.jsonl --out b4_stream_s50_n0.0.json
    python lora_llm.py stream_s50_n0.0.jsonl --out smoke.json --only 0 --epochs 1 --max-q 32   (smoke test)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"


def show(expr) -> str:
    if expr[0] == "obj":
        return expr[1]
    if expr[0] == "var":
        return "?"
    return f"{expr[1]}(" + ", ".join(show(e) for e in expr[2:]) + ")"


def prompt(q) -> str:
    if q["kind"] == "eval":
        return f"{show(q['expr'])} ="
    return f"{show(q['expr'])} = {q['target']} ; ? ="


def lines(rec) -> list[str]:
    out = [f"{o['word']}({', '.join(o['args'])}) = {o['out']}" for o in rec["experience"] + rec["extra_labels"]]
    out += [f"{prompt(q)} {q['key']}" for q in rec["practice"]]
    return out


def valid_answers(rec) -> set[str]:
    """Every token this system's data showed, plus the special answers."""
    v = {t for o in rec["experience"] + rec["extra_labels"] for t in o["args"] + [o["out"]]}
    v |= {q["key"] for q in rec["practice"]}
    return v | {"#T", "#F", "none", "many"}


def train(model, tok, texts, epochs, lr, dev, seed):
    rng = random.Random(seed)
    texts = list(texts)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    tok.padding_side = "right"                       # training: plain right padding, masked in the loss
    for _ in range(epochs):
        rng.shuffle(texts)
        for i in range(0, len(texts), 16):
            batch = tok([f"[{t}]" for t in texts[i:i + 16]], return_tensors="pt", padding=True).to(dev)
            out = model(**batch, labels=batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100))
            opt.zero_grad()
            out.loss.backward()
            opt.step()


@torch.no_grad()
def answer(model, tok, rec, dev, max_q=None, bs=64):
    """Greedy answers, batched; left padding so every prompt ends at the same position."""
    model.eval()
    tok.padding_side = "left"
    valid = valid_answers(rec)
    qs = rec["questions"][:max_q] if max_q else rec["questions"]
    preds = {}
    for i in range(0, len(qs), bs):
        chunk = qs[i:i + bs]
        enc = tok([f"[{prompt(q)}" for q in chunk], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        for q, text in zip(chunk, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            words = text.replace("]", " ").split()
            preds[q["qid"]] = words[0] if words and words[0] in valid else "unknown"
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=int, default=None, help="one system index (smoke test)")
    ap.add_argument("--max-q", type=int, default=None, help="answer only the first N questions (smoke test)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    recs = [json.loads(x) for x in open(a.stream, encoding="utf-8")]
    if a.only is not None:
        recs = [recs[a.only]]
    ck_path = a.out + ".ckpt.pt"
    after_own, done = {}, -1
    if os.path.exists(ck_path):                      # resume after an interruption
        ck = torch.load(ck_path, weights_only=False, map_location=dev)
        model.load_state_dict(ck["lora"], strict=False)
        after_own, done = ck["after_own"], ck["done"]
        print("resuming after system index", done, flush=True)
    for k, rec in enumerate(recs):
        if k <= done:
            continue
        t0 = time.time()
        train(model, tok, lines(rec), a.epochs, a.lr, dev, a.seed + k)
        t1 = time.time()
        after_own.update(answer(model, tok, rec, dev, a.max_q))
        lora = {n: p.detach().cpu() for n, p in model.state_dict().items() if "lora_" in n}
        torch.save({"done": k, "after_own": after_own, "lora": lora}, ck_path)
        print(rec["system"], f"train {t1 - t0:.0f}s answer {time.time() - t1:.0f}s", flush=True)
    t0 = time.time()
    final = {}
    for rec in recs:                                 # every system re-asked at the end (forgetting)
        final.update(answer(model, tok, rec, dev, a.max_q))
    print(f"final re-ask {time.time() - t0:.0f}s", flush=True)
    json.dump({"predictions": after_own, "after_own_system": after_own, "after_stream": final}, open(a.out, "w"))


if __name__ == "__main__":
    main()
