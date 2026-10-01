#!/usr/bin/env python3
"""Model selection on held-out *validation* data only (never the test splits reported in README):

    banking77   500 rows of Banking77 train held out by the trainers (same seed, never trained on)
    massive     MASSIVE validation split, 12 languages x --per-lang rows (seeded)
    sentiment   tyqiangz multilingual-sentiments valid.csv, 12 languages x --per-lang rows (seeded)
    emotion     dair-ai/emotion validation split (seeded sample)
    ag_news     AG News train rows (labels never used in training; texts only in distillation)

Prompts and option order match the evaluation harnesses; head_max_len 512.

    .venv-train/bin/python tools/finetune/eval_dev.py models/laya-multilingual-banking77 models/laya-multilingual-multitask
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench"))
from prediction_items import write_prediction_rows  # noqa: E402

SEED = 20260926
EVAL_LANGS = ["de", "en", "fr", "es", "it", "tr", "pl", "ru", "ja", "zh-CN", "ar", "hi"]


def suites(per_lang, n):
    # imported here, not at module level: argument checks must work without numpy and the training stack
    from train_multitask import MASSIVE_URL, SENT_LANGS, SENT_OPTIONS, SENT_URL, load  # noqa: PLC0415
    from datasets import load_dataset
    out = {}
    bank = list(load_dataset("mteb/banking77", split="train"))
    random.Random(SEED).shuffle(bank)  # identical to the trainers' first shuffle -> same 500 dev rows
    keys = sorted({r["label_text"].replace("_", " ") for r in bank})
    rows = bank[:500]
    out["banking77"] = ([{"message": r["text"]} for r in rows],
                        {"q": {"type": "choice", "instructions": "Which banking intent does `message` express?",
                               "criteria": {k: None for k in keys}}},
                        keys, [keys.index(r["label_text"].replace("_", " ")) for r in rows])
    rng = random.Random(SEED + 1)
    st, gold = [], []
    for lang in EVAL_LANGS:
        rows = load("json", MASSIVE_URL % ("validation", lang))
        rng.shuffle(rows)
        for r in rows[:per_lang]:
            st.append({"utterance": r["text"]})
            gold.append(r["label_text"].replace("_", " "))
    keys = sorted({r["label_text"].replace("_", " ") for r in load("json", MASSIVE_URL % ("train", "en"))})
    out["massive"] = (st, {"q": {"type": "choice", "instructions": "Which intent does `utterance` express?",
                                 "criteria": {k: None for k in keys}}}, keys, [keys.index(g) for g in gold])
    st, gold = [], []
    for lang in SENT_LANGS:
        rows = [r for r in load("csv", SENT_URL % (lang, "valid")) if r["text"] and r["label"] in SENT_OPTIONS]
        rng.shuffle(rows)
        for r in rows[:per_lang]:
            st.append({"text": r["text"]})
            gold.append(SENT_OPTIONS.index(r["label"]))
    out["sentiment"] = (st, {"q": {"type": "choice", "instructions": "What is the sentiment of `text`?",
                                   "criteria": {k: None for k in SENT_OPTIONS}}}, SENT_OPTIONS, gold)
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    rows = list(load_dataset("dair-ai/emotion", "split", split="validation"))
    rng.shuffle(rows)
    out["emotion"] = ([{"text": r["text"]} for r in rows[:n]],
                      {"q": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?",
                             "criteria": {k: None for k in names}}}, names, [int(r["label"]) for r in rows[:n]])
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    rows = list(load_dataset("fancyzhx/ag_news", split="train"))
    rng.shuffle(rows)
    out["ag_news"] = ([{"article": r["text"]} for r in rows[:n]],
                      {"q": {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": crit}},
                      list(crit), [int(r["label"]) for r in rows[:n]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--per-lang", type=int, default=50)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--predictions", default=None,
                    help="JSONL, one per-item gold/prediction record for paired comparisons")
    a = ap.parse_args()
    if a.predictions and len(a.models) != 1:
        ap.error("--predictions requires exactly one model")
    if a.predictions:
        os.makedirs(os.path.dirname(os.path.abspath(a.predictions)) or ".", exist_ok=True)
    import laya
    data = suites(a.per_lang, a.n)
    for m in a.models:
        cfg = json.load(open(os.path.join(m, "rl_agent_config.json")))
        calib = cfg.get("training_banking77", {}).get("calib", 500)
        if calib < 500:  # the model trained on part of these 500 Banking77 rows
            raise SystemExit(f"{m}: trained with --calib {calib} < 500, overlaps the Banking77 dev rows")
        agent = laya.load(os.path.abspath(m), device="cuda")
        row, t0 = {"model": m, "n": {name: len(values[0]) for name, values in data.items()}}, time.time()
        predictions = open(a.predictions, "w", encoding="utf-8") if a.predictions else None
        for name, (states, questions, keys, gold) in data.items():
            probs = []
            for i in range(0, len(states), 32):
                for r, g in zip(agent.predict_batch(states[i:i + 32], questions, max_len=1024, head_max_len=512),
                                gold[i:i + 32]):
                    p = r["answers"]["q"]["probabilities"]
                    probs.append([p[key] for key in keys])
            row[name] = round(sum(max(range(len(p)), key=p.__getitem__) == g
                                  for p, g in zip(probs, gold)) / len(states), 4)
            if predictions:
                lang = "mixed" if name in ("massive", "sentiment") else "en"
                write_prediction_rows(predictions, name, lang, states, questions, gold, probs)
        row["mean"] = round(sum(row[k] for k in data) / len(data), 4)
        row["seconds"] = round(time.time() - t0)
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if predictions:
            predictions.close()
        del agent


if __name__ == "__main__":
    main()
