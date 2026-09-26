#!/usr/bin/env python3
"""Offline check of contextual calibration (Zhao et al., 2021, "Calibrate Before Use") on dumped logits.

For every suite the question is scored once on content-free states (same JSON shape, empty or
"N/A" strings); calibrated logits are z - mean(z_null). No labels are used.

    python bench/analyze_calibration.py --url http://127.0.0.1:8412 --dump bench/results/logits.jsonl
"""
import argparse
import json
import math
import urllib.request

import eval_accuracy as ev


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def null_logits(url, model, state_key, questions, qid):
    states = [{state_key: ""}, {state_key: "N/A"}, {state_key: "[MASK]"}]
    body = json.dumps({"states": states, "questions": questions, "return_logits": True, "model": model}).encode()
    res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/systemone/batch", data=body), timeout=600))
    zs = [r["answers"][qid]["logits"] for r in res["results"]]
    # average in probability space, as in the paper, then back to log space
    ps = [softmax(z) for z in zs]
    mean = [sum(p[i] for p in ps) / len(ps) for i in range(len(ps[0]))]
    return [math.log(max(v, 1e-12)) for v in mean]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    data = ev.suites(a.n)
    key_of = {"ag_news": "article", "emotion": "text", "banking77": "message"}
    rows = [json.loads(l) for l in open(a.dump)]
    for r in rows:
        states, questions, qid, keys, gold = data[r["suite"]]
        zn = null_logits(a.url, r["model"], key_of[r["suite"]], questions, qid)
        raw = [softmax(z) for z in r["logits"]]
        out = {"suite": r["suite"], "model": r["model"], "raw": ev.metrics(raw, gold)}
        for alpha in (0.5, 1.0):
            cal = [softmax([zi - alpha * ni for zi, ni in zip(z, zn)]) for z in r["logits"]]
            out["cal_%.1f" % alpha] = ev.metrics(cal, gold)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
