#!/usr/bin/env python3
"""Offline check: fuse the English (ModernBERT-large) and multilingual (mmBERT) checkpoints by
averaging their log-probabilities. Laya's Router always answers with a single checkpoint.

    python bench/analyze_consensus.py --dump bench/results/logits.jsonl
"""
import argparse
import json
import math

import eval_accuracy as ev


def log_softmax(z):
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return [v - lse for v in z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--url", default=None, help="Statim server: also evaluate contextual calibration")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.dump)]
    data = ev.suites(400) if a.url else None
    by = {(r["suite"], r["model"]): r for r in rows}
    for suite in ("emotion", "ag_news", "banking77"):
        if (suite, "english") not in by or (suite, "multilingual") not in by:
            continue
        en, ml = by[(suite, "english")], by[(suite, "multilingual")]
        gold = en["gold"]
        out = {"suite": suite}
        variants = [("", en["logits"], ml["logits"])]
        if a.url:
            import analyze_calibration as ac
            _, questions, qid, _, _ = data[suite]
            key = {"ag_news": "article", "emotion": "text", "banking77": "message"}[suite]
            ne = ac.null_logits(a.url, "english", key, questions, qid)
            nm = ac.null_logits(a.url, "multilingual", key, questions, qid)
            variants.append(("cal_", [[x - y for x, y in zip(z, ne)] for z in en["logits"]],
                             [[x - y for x, y in zip(z, nm)] for z in ml["logits"]]))
        for prefix, zen, zml in variants:
            for name, w in (("english", 1.0), ("multilingual", 0.0), ("consensus_50_50", 0.5)):
                probs = []
                for ze, zm in zip(zen, zml):
                    le, lm = log_softmax(ze), log_softmax(zm)
                    probs.append([math.exp(v) for v in log_softmax([w * x + (1 - w) * y for x, y in zip(le, lm)])])
                out[prefix + name] = ev.metrics(probs, gold)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
