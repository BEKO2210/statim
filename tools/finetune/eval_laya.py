#!/usr/bin/env python3
"""Accuracy of a Laya checkpoint directory on the public suites, on GPU, with an optional
token budget override (max_len / head_max_len). Same prompts as bench/eval_accuracy.py.

    .venv-train/bin/python tools/finetune/eval_laya.py models/laya-multilingual --suites banking77 --head-max-len 192 512
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))
from eval_accuracy import metrics, suites  # noqa: E402


def probs_for(agent, states, questions, qid, keys, max_len, head_max_len, batch=16):
    out = []
    for i in range(0, len(states), batch):
        res = agent.predict_batch(states[i:i + batch], questions, max_len=max_len, head_max_len=head_max_len)
        for r in res:
            p = r["answers"][qid]["probabilities"]
            out.append([p[k] for k in keys])
    return out


def typed_decisions(agent, max_len, head_max_len, batch=16):
    """Hard-label accuracy on LocalLLaMA/typed-decisions test (as in Laya's fine-tune notebook)."""
    import numpy as np
    from datasets import load_dataset
    rows = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="test"))
    corr = []
    for r in rows:
        state, qs, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        ans = agent.predict_batch([state], qs, max_len=max_len, head_max_len=head_max_len)[0]["answers"]
        for qid, qd in qs.items():
            p, g = ans[qid], gold[qid]
            if qd["type"] == "choice":
                corr.append(p["choice"] == str(g["label"]))
            elif qd["type"] == "noul":
                corr.append(("true" if p["noul"] >= 0.5 else "false") == str(g["label"]).lower())
            else:
                pl = [p["probabilities"].get(str(i), 0.0) for i in range(len(qd.get("criteria", [])))]
                corr.append(int(np.argmax(pl)) == int(g.get("label", round(g.get("score", 0.0)))))
    return {"accuracy": round(float(np.mean(corr)), 4), "n": len(corr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--suites", nargs="+", default=["ag_news", "emotion", "banking77", "typed"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, nargs="+", default=[None])
    ap.add_argument("--out", default=None, help="append JSON lines here")
    a = ap.parse_args()
    import laya
    agent = laya.load(os.path.abspath(a.model), device="cuda")
    data = suites(a.n)
    for hml in a.head_max_len:
        for name in a.suites:
            if name == "typed":
                t0 = time.time()
                row = {"model": a.model, "suite": "typed_decisions", "head_max_len": hml or agent.cfg.get("head_max_len"),
                       **typed_decisions(agent, a.max_len, hml), "seconds": round(time.time() - t0, 1)}
                print(json.dumps(row), flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                continue
            states, questions, qid, keys, gold = data[name]
            ml = a.max_len or (max(agent.cfg.get("max_len", 512), (hml or 0) + 128) if hml else None)
            t0 = time.time()
            probs = probs_for(agent, states, questions, qid, keys, ml, hml)
            row = {"model": a.model, "suite": name, "max_len": ml or agent.cfg.get("max_len"),
                   "head_max_len": hml or agent.cfg.get("head_max_len"), **metrics(probs, gold),
                   "seconds": round(time.time() - t0, 1)}
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
