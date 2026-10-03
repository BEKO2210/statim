#!/usr/bin/env python3
"""Build a broad, licence-clean training mixture from tasksource/tasksource-jev-typed-decisions
(2.5M typed decisions in Statim's own choice / score / noul schema, 667 sources).

Rules:
- Only rows marked license_use == "commercial" AND whose every listed licence is permissive
  (Apache-2.0, MIT, BSD, CC0, CC-BY, ODC-By, AFL-3.0). ShareAlike (CC-BY-SA), copyleft (GPL,
  AGPL, MPL, ODbL), "Custom", "other" and corpus-specific terms are excluded: Statim's weights are
  licensed PolyForm Noncommercial with paid commercial licences, which ShareAlike or copyleft data
  could contradict.
- Sources that feed our evaluations are excluded entirely: AG News, every emotion source
  (so DAIR Emotion stays an unseen concept), Banking77, MASSIVE, tyqiangz multilingual sentiment,
  tweet_eval and tweet_sentiment_multilingual (tweet_eval texts are used for distillation;
  the latter is the source of tyqiangz multilingual sentiment), typed-decisions.
- With --audit (tools/finetune/licence_audit.json): a source is used only if it matches a
  keep_families prefix of the audit and is not listed in its exclude map. The audit checked the
  origin of each source's text and labels upstream, because the recorded licence tags are
  best-effort and sometimes wrong (e.g. Stack Overflow tagged Apache-2.0, wikiHow tagged MIT).
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_multitask import MASSIVE_URL, SENT_LANGS, SENT_URL, load  # noqa: E402

REPO = "tasksource/tasksource-jev-typed-decisions"
EXCLUDE = re.compile(r"(^|/)(ag_news|emotion|go_emotions|text_emotion|banking77|massive|multilingual-sentiments|"
                     r"tweet_eval|tweet_sentiment_multilingual|universal-joy|emo2019|emobank-[a-z]+)(/|$)|LocalLLaMA|typed-decisions/(all|default)$", re.I)
SEED = 20260927
PERMISSIVE = {"apache-2.0", "apache license 2.0 (dpi)", "mit", "mit license (dpi)", "bsd",
              "bsd 2-clause license (dpi)", "cc0-1.0", "cc0 1.0 (dpi)", "cc-by-4.0", "cc by 4.0 (dpi)",
              "cc-by-3.0", "odc-by", "afl-3.0"}
POLICY_PATH = Path(__file__).resolve().parent / "sources" / "policy.json"
CURRENT_V5_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "mixture-v5.manifest.json"


def recorded_revision():
    if CURRENT_V5_MANIFEST.exists():
        return json.loads(CURRENT_V5_MANIFEST.read_text(encoding="utf-8")).get("revision")
    return None


def audited(src, audit, policy):
    """Fail closed on the policy exclusions before applying the v5 audit allowlist."""
    denied = [x["id"] for x in policy["exclusions"]
              if x["registry"] == "v5" and x["scope"] == "source"]
    if any(src == prefix or src.startswith(prefix + "/") for prefix in denied):
        return False
    return audit is None or (src not in audit["exclude"] and
                             any(src.startswith(p) for p in audit["keep_families"]))


def permissive(license_field):
    parts = [p.strip().lower() for p in (license_field or "").split(",") if p.strip()]
    return bool(parts) and all(p in PERMISSIVE for p in parts)


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
    ap.add_argument("--audit", default=None, help="licence audit JSON (whitelist of sources)")
    ap.add_argument("--revision", default=recorded_revision(),
                    help="pinned tasksource dataset revision (required unless the current v5 manifest records it)")
    a = ap.parse_args()
    if not a.revision:
        ap.error("--revision is required: the current v5 manifest records no dataset revision")
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, repo_type="dataset", revision=a.revision,
                             allow_patterns=["data/*.parquet"])
    rng = random.Random(SEED)
    audit = json.load(open(a.audit)) if a.audit else None
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

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
            if not permissive(r["license"]):
                stats["not_permissive"] += 1
                continue
            if not audited(r["source"], audit, policy):
                stats["audit_excluded"] += 1
                continue
            if EXCLUDE.search(r["source"]):
                stats["excluded_source"] += 1
                continue
            if len(r["state"]) > a.max_state_chars:
                stats["too_long"] += 1
                continue
            by_src[r["source"]].append(r)
    items, per_src, licenses = [], {}, {}
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
        licenses[src] = rows[0]["license"] if rows else ""
    rng.shuffle(items)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    kinds = collections.Counter(it["q"]["type"] for it in items)
    manifest = {"repo": REPO, "items": len(items), "sources": len(per_src), "kinds": kinds, "stats": stats,
                "per_source_cap": a.per_source, "audit": a.audit and os.path.basename(a.audit),
                "revision": a.revision, "per_source": per_src, "licenses": licenses, "seed": SEED}
    with open(a.out.replace(".jsonl.gz", ".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"items {len(items)} from {len(per_src)} sources | kinds {dict(kinds)} | {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
