#!/usr/bin/env python3
"""Build a broad, licence-clean training mixture from tasksource/tasksource-jev-typed-decisions
(2.5M typed decisions in Statim's own choice / score / noul schema, 667 sources).

Rules:
- Only rows marked license_use == "commercial".
- Sources that feed our evaluations are excluded entirely: AG News, every emotion source
  (so DAIR Emotion stays an unseen concept), Banking77, MASSIVE, tyqiangz multilingual sentiment,
  tweet_eval and tweet_sentiment_multilingual (tweet_eval texts are used for distillation;
  the latter is the source of tyqiangz multilingual sentiment), typed-decisions.
- At most --per-source rows per source (Laurer et al. 2023 cap diversity rather than volume).
- Any row whose state text occurs in one of our test splits is dropped (exact match after
  whitespace normalisation): Banking77 test, MASSIVE test (51 languages), tyqiangz test (12),
  AG News test, DAIR Emotion test, typed-decisions test.

Output: gzipped JSONL, one item per line:
  {"state": str, "q": {"type", "instructions", "criteria"}, "target": [...], "src": str}

    .venv-train/bin/python tools/finetune/build_mixture.py --out data/mixture-v1.jsonl.gz --per-source 120
"""
import argparse
import collections
import glob
import gzip
import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_multitask import MASSIVE_URL, SENT_LANGS, SENT_URL, load  # noqa: E402

REPO = "tasksource/tasksource-jev-typed-decisions"
EXCLUDE = re.compile(r"(^|/)(ag_news|emotion|go_emotions|text_emotion|banking77|massive|multilingual-sentiments|"
                     r"tweet_eval|tweet_sentiment_multilingual|universal-joy|emo2019|emobank-[a-z]+)(/|$)|LocalLLaMA|typed-decisions/(all|default)$", re.I)
SEED = 20260927


def norm(t):
    return " ".join(str(t).split()).lower()


def test_texts():
    from datasets import load_dataset
    out = set()
    out |= {norm(r["text"]) for r in load_dataset("mteb/banking77", split="test")}
    langs = [x["path"].split("/")[-1].replace(".json.gz", "") for x in json.load(urllib.request.urlopen(
        "https://huggingface.co/api/datasets/mteb/amazon_massive_intent/tree/main/test"))]
    for lang in langs:
        out |= {norm(r["text"]) for r in load("json", MASSIVE_URL % ("test", lang))}
    for lang in SENT_LANGS:
        out |= {norm(r["text"]) for r in load("csv", SENT_URL % (lang, "test")) if r["text"]}
    out |= {norm(r["text"]) for r in load_dataset("fancyzhx/ag_news", split="test")}
    out |= {norm(r["text"]) for r in load_dataset("dair-ai/emotion", "split", split="test")}
    for r in load_dataset("LocalLLaMA/typed-decisions", "all", split="test"):
        out.add(norm(r["state"]))
    return out


def to_item(r):
    opts = r["options"] or []
    opts = json.loads(opts) if isinstance(opts, str) else list(opts)
    target = json.loads(r["target"]) if isinstance(r["target"], str) else list(r["target"])
    kind = r["kind"]
    if kind == "noul":
        p = float(target[0])
        return {"state": r["state"], "q": {"type": "noul", "instructions": r["question"]},
                "target": [1.0 - p, p], "src": r["source"]}
    if len(opts) != len(target) or len(opts) < 2 or len(set(opts)) != len(opts):
        return None
    s = sum(target)
    if s <= 0:
        return None
    target = [t / s for t in target]
    crit = {str(o): None for o in opts} if kind == "choice" else [str(o) for o in opts]
    return {"state": r["state"], "q": {"type": kind, "instructions": r["question"], "criteria": crit},
            "target": target, "src": r["source"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=120)
    ap.add_argument("--max-state-chars", type=int, default=6000)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, repo_type="dataset", allow_patterns=["data/*.parquet"])
    rng = random.Random(SEED)
    banned = test_texts()
    print(f"test texts to avoid: {len(banned)}", flush=True)
    by_src = collections.defaultdict(list)
    stats = collections.Counter()
    for f in sorted(glob.glob(os.path.join(path, "data", "train-*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            stats["rows"] += 1
            if r["license_use"] != "commercial":
                stats["not_commercial"] += 1
                continue
            if EXCLUDE.search(r["source"]):
                stats["excluded_source"] += 1
                continue
            if len(r["state"]) > a.max_state_chars:
                stats["too_long"] += 1
                continue
            by_src[r["source"]].append(r)
    items, per_src = [], {}
    for src, rows in sorted(by_src.items()):
        rng.shuffle(rows)
        kept = 0
        for r in rows:
            if kept >= a.per_source:
                break
            if norm(r["state"]) in banned:
                stats["test_duplicate"] += 1
                continue
            it = to_item(r)
            if it is None:
                stats["malformed"] += 1
                continue
            items.append(it)
            kept += 1
        per_src[src] = kept
    rng.shuffle(items)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    kinds = collections.Counter(it["q"]["type"] for it in items)
    manifest = {"repo": REPO, "items": len(items), "sources": len(per_src), "kinds": kinds, "stats": stats,
                "per_source_cap": a.per_source, "per_source": per_src, "seed": SEED}
    with open(a.out.replace(".jsonl.gz", ".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"items {len(items)} from {len(per_src)} sources | kinds {dict(kinds)} | {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
