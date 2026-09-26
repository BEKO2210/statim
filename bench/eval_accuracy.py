#!/usr/bin/env python3
"""Accuracy/calibration on the public suites Laya compares against Jev (same construction as
laya research/scripts/bench_apps.py: first N test rows, identical prompts).

    python bench/eval_accuracy.py --url http://127.0.0.1:8412 --ensemble 1 4 --n 400
    python bench/eval_accuracy.py --laya models/laya-multilingual --n 100   # official package, for cross-checking
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request


def suites(n):
    from datasets import load_dataset
    out = {}
    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    keys = list(crit)
    out["ag_news"] = ([{"article": r["text"]} for r in list(d)[:n]],
                      {"topic": {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": crit}},
                      "topic", keys, [int(r["label"]) for r in list(d)[:n]])
    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    out["emotion"] = ([{"text": r["text"]} for r in list(d)[:n]],
                      {"emotion": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?",
                                   "criteria": {k: None for k in names}}},
                      "emotion", names, [int(r["label"]) for r in list(d)[:n]])
    d = load_dataset("mteb/banking77", split="test")
    labels = sorted(set(d["label_text"]))
    keys77 = [x.replace("_", " ") for x in labels]
    rows = list(d)[:n]
    out["banking77"] = ([{"message": r["text"]} for r in rows],
                        {"intent": {"type": "choice", "instructions": "Which banking intent does `message` express?",
                                    "criteria": {k: None for k in keys77}}},
                        "intent", keys77, [keys77.index(r["label_text"].replace("_", " ")) for r in rows])
    return out


def metrics(probs, gold):
    n = len(gold)
    acc = sum(max(range(len(p)), key=p.__getitem__) == g for p, g in zip(probs, gold)) / n
    nll = -sum(math.log(max(p[g], 1e-12)) for p, g in zip(probs, gold)) / n
    brier = sum(sum((pi - (1.0 if i == g else 0.0)) ** 2 for i, pi in enumerate(p)) for p, g in zip(probs, gold)) / n
    bins = [[0, 0.0, 0.0] for _ in range(15)]
    for p, g in zip(probs, gold):
        c = max(p)
        b = min(14, int(c * 15))
        bins[b][0] += 1
        bins[b][1] += c
        bins[b][2] += float(max(range(len(p)), key=p.__getitem__) == g)
    ece = sum(abs(s_c - s_a) for cnt, s_c, s_a in bins if cnt) / n
    return {"accuracy": round(acc, 4), "ece": round(ece, 4), "nll": round(nll, 4), "brier": round(brier, 4)}


def run_statim(url, states, questions, qid, keys, ensemble, batch=16, api_key=None, model=None):
    probs, t0 = [], time.time()
    run_statim.logits = []
    for i in range(0, len(states), batch):
        body = json.dumps({"states": states[i:i + batch], "questions": questions, "ensemble": ensemble,
                           "return_logits": True, **({"model": model} if model else {})}).encode()
        req = urllib.request.Request(url + "/v1/systemone/batch", data=body, headers={"Content-Type": "application/json"})
        if api_key:
            req.add_header("Authorization", "Bearer " + api_key)
        res = json.load(urllib.request.urlopen(req, timeout=600))
        for r in res["results"]:
            pr = r["answers"][qid]["probabilities"]
            probs.append([pr[k] for k in keys])
            run_statim.logits.append(r["answers"][qid].get("logits"))
    return probs, time.time() - t0


def run_laya(path, states, questions, qid, keys):
    import laya
    agent = run_laya.agent = getattr(run_laya, "agent", None) or laya.load(os.path.abspath(path), device="cpu")
    probs, t0 = [], time.time()
    for s in states:
        pr = agent.system_one(s, questions)["answers"][qid]["probabilities"]
        probs.append([pr[k] for k in keys])
    return probs, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None, help="model name on the Statim server")
    ap.add_argument("--laya", default=None, help="checkpoint dir: evaluate the official Python package instead")
    ap.add_argument("--ensemble", type=int, nargs="+", default=[1])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--suites", nargs="+", default=["ag_news", "emotion", "banking77"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="write per-case probabilities + gold (jsonl) for offline analysis")
    a = ap.parse_args()
    data = suites(a.n)
    report = {}
    for name in a.suites:
        states, questions, qid, keys, gold = data[name]
        if a.laya:
            probs, secs = run_laya(a.laya, states, questions, qid, keys)
            report[name] = {"laya": dict(metrics(probs, gold), seconds=round(secs, 1))}
            print(name, report[name], flush=True)
            continue
        report[name] = {}
        for e in a.ensemble:
            probs, secs = run_statim(a.url, states, questions, qid, keys, e, api_key=os.environ.get("STATIM_API_KEY"), model=a.model)
            report[name]["statim_ensemble_%d" % e] = dict(metrics(probs, gold), seconds=round(secs, 1))
            if a.dump:
                with open(a.dump, "a") as f:
                    f.write(json.dumps({"suite": name, "model": a.model, "ensemble": e, "keys": keys, "gold": gold, "probs": probs,
                                        "logits": run_statim.logits}) + "\n")
            print(name, e, report[name]["statim_ensemble_%d" % e], flush=True)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
