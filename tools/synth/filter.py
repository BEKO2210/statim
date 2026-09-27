#!/usr/bin/env python3
"""Drop exact and near-duplicate items, and anything that overlaps an eval split.

Near-duplicates: normalised text, MinHash candidates, then character 5-gram
Jaccard >= 0.8. No third-party MinHash library.

Eval ban list:
- test_texts() from tools/finetune/build_mixture.py (held-out test splits)
- full test text of the zero-shot suites (not the 150-row samples):
  go_emotions, multi-hatecheck, SIB-200 (FLORES sentences), HWU64,
  IndoNLI, FarsTail, Belebele (FLORES passages), SemRel.

A failure to load any suite aborts the filter. Silent skip would let eval
text into training.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

from common import (
    MinHash, NEAR_DUP, char_ngrams, item_fingerprint, jaccard, norm, sha256_text,
)

ROOT = Path(__file__).resolve().parents[2]
BAN_CACHE = Path("/tmp/statim-synth-ban-v1.txt.gz")
ANCHOR = 24
ANCHOR_STEP = 48

HATE_LANGS = [
    ("en", "eng"), ("de", "deu"), ("fr", "fra"), ("es", "spa"), ("it", "ita"),
    ("nl", "nld"), ("pl", "pol"), ("pt", "por"), ("zh", "cmn"), ("ar", "ara"), ("hi", "hin"),
]
SIB_LANGS = [("en", "eng_Latn"), ("de", "deu_Latn"), ("ar", "arb_Arab"), ("hi", "hin_Deva")]
BELEBELE_LANGS = SIB_LANGS
SEMREL_LANGS = [("en", "eng"), ("ar", "arb"), ("hi", "hin")]


def _add(bag, value):
    if value is None:
        return
    text = norm(value)
    if text:
        bag.add(text)


def _dataset():
    from datasets import load_dataset
    return load_dataset


def suite_strings():
    """Every eval string the zero-shot suites score, full splits."""
    load = _dataset()
    bag = set()
    print("ban: go_emotions", flush=True)
    ds = load("google-research-datasets/go_emotions", "simplified", split="test")
    for text in ds["text"]:
        _add(bag, text)

    print("ban: multi_hatecheck", flush=True)
    for _code, file_code in HATE_LANGS:
        url = "https://huggingface.co/datasets/mteb/multi-hatecheck/resolve/main/test/%s.jsonl.gz" % file_code
        ds = load("json", data_files={"test": url}, split="test")
        for text in ds["text"]:
            _add(bag, text)

    print("ban: sib200/flores", flush=True)
    for _code, config in SIB_LANGS:
        ds = load("Davlan/sib200", config, split="test")
        for text in ds["text"]:
            _add(bag, text)

    print("ban: hwu64", flush=True)
    ds = load("DeepPavlov/hwu64", split="test")
    for text in ds["utterance"]:
        _add(bag, text)

    print("ban: indonli", flush=True)
    url = "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/test_expert.jsonl"
    ds = load("json", data_files={"test": url}, split="test")
    for row in ds:
        _add(bag, row["premise"])
        _add(bag, row["hypothesis"])
        _add(bag, str(row["premise"]) + "\n" + str(row["hypothesis"]))

    print("ban: farstail", flush=True)
    url = "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv"
    ds = load("csv", data_files={"test": url}, split="test", delimiter="\t")
    for row in ds:
        _add(bag, row["premise"])
        _add(bag, row["hypothesis"])
        _add(bag, str(row["premise"]) + "\n" + str(row["hypothesis"]))

    print("ban: belebele", flush=True)
    for _code, config in BELEBELE_LANGS:
        ds = load("facebook/belebele", config, split="test")
        for row in ds:
            _add(bag, row["flores_passage"])
            _add(bag, row["question"])
            _add(bag, str(row["flores_passage"]) + "\n" + str(row["question"]))
            for i in range(1, 5):
                _add(bag, row["mc_answer%d" % i])

    print("ban: semrel", flush=True)
    for _code, config in SEMREL_LANGS:
        ds = load("SemRel/SemRel2024", config, split="test")
        for row in ds:
            _add(bag, row["sentence1"])
            _add(bag, row["sentence2"])
            _add(bag, str(row["sentence1"]) + "\n" + str(row["sentence2"]))
    print("ban: zero-shot strings %d" % len(bag), flush=True)
    return bag


def test_split_strings():
    finetune = ROOT / "tools" / "finetune"
    sys.path.insert(0, str(finetune))
    from build_mixture import test_texts  # noqa: E402
    print("ban: test_texts()", flush=True)
    texts = test_texts()
    print("ban: test_texts %d" % len(texts), flush=True)
    return set(texts)


def load_ban_strings():
    if BAN_CACHE.exists() and BAN_CACHE.stat().st_size > 0:
        try:
            with gzip.open(BAN_CACHE, "rt", encoding="utf-8") as handle:
                lines = [line.rstrip("\n") for line in handle if line.strip()]
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError):
            lines = []
        if lines:
            print("ban: cache %s (%d)" % (BAN_CACHE, len(lines)), flush=True)
            return set(lines)
    strings = set()
    strings |= test_split_strings()
    strings |= suite_strings()
    BAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(BAN_CACHE, "wt", encoding="utf-8") as handle:
        for text in sorted(strings):
            handle.write(text + "\n")
    print("ban: wrote %s (%d)" % (BAN_CACHE, len(strings)), flush=True)
    return strings


class BanIndex:
    def __init__(self, strings):
        self.exact = set(strings)
        self.long_text = []
        self.anchors = collections.defaultdict(list)
        self.mh = MinHash(seed=20260929)
        self.sig_buckets = collections.defaultdict(list)
        self.sigs = []
        for text in self.exact:
            if len(text) < 80:
                continue
            idx = len(self.long_text)
            self.long_text.append(text)
            for start in range(0, len(text) - ANCHOR + 1, ANCHOR_STEP):
                self.anchors[text[start:start + ANCHOR]].append(idx)
            sig = self.mh.signature(char_ngrams(text))
            self.sigs.append(sig)
            for key in self.mh.band_keys(sig):
                self.sig_buckets[key].append(idx)
        print("ban: exact %d long %d anchors %d" % (
            len(self.exact), len(self.long_text), len(self.anchors)), flush=True)

    def overlap(self, state):
        text = norm(state)
        if not text:
            return "empty"
        if text in self.exact:
            return "exact"
        if len(text) >= ANCHOR:
            hits = set()
            upper = len(text) - ANCHOR + 1
            for i in range(0, upper):
                found = self.anchors.get(text[i:i + ANCHOR])
                if found:
                    hits.update(found)
                if len(hits) > 32:
                    break
            for idx in hits:
                other = self.long_text[idx]
                if other in text or text in other:
                    return "contains"
        if len(text) < 40:
            return None
        grams = char_ngrams(text)
        sig = self.mh.signature(grams)
        cands = set()
        for key in self.mh.band_keys(sig):
            cands.update(self.sig_buckets.get(key, ()))
        for idx in cands:
            if jaccard(grams, char_ngrams(self.long_text[idx])) >= NEAR_DUP:
                return "near"
        return None


def dedup(items):
    """Keep the lowest id. Exact normalised fingerprint, then 5-gram Jaccard."""
    mh = MinHash(seed=20260928)
    kept = []
    kept_grams = []
    exact = {}
    buckets = collections.defaultdict(list)
    dropped = {"exact": 0, "near": 0}
    for item in sorted(items, key=lambda row: row["_id"]):
        text = norm(item_fingerprint(item))
        digest = sha256_text(text)
        if digest in exact:
            dropped["exact"] += 1
            continue
        grams = char_ngrams(text)
        sig = mh.signature(grams)
        cands = set()
        for key in mh.band_keys(sig):
            cands.update(buckets.get(key, ()))
        if any(jaccard(grams, kept_grams[j]) >= NEAR_DUP for j in cands):
            dropped["near"] += 1
            continue
        idx = len(kept)
        kept.append(item)
        kept_grams.append(grams)
        exact[digest] = idx
        for key in mh.band_keys(sig):
            buckets[key].append(idx)
    return kept, dropped


def public_item(item):
    return {"state": item["state"], "q": item["q"], "target": item["target"], "src": item["src"]}


def apply_filter(items, ban):
    survivors = []
    dropped = collections.Counter()
    for item in items:
        reason = ban.overlap(item["state"])
        if reason:
            dropped["eval_" + reason] += 1
            continue
        survivors.append(item)
    kept, dup_dropped = dedup(survivors)
    dropped["dup_exact"] = dup_dropped["exact"]
    dropped["dup_near"] = dup_dropped["near"]
    return kept, dropped


def run_checks():
    same = "a" * 30
    assert jaccard(char_ngrams(same), char_ngrams(same)) == 1.0
    assert jaccard(char_ngrams("aaaaa bbbbb"), char_ngrams("zzzzz yyyyy")) < 0.2
    base = ("the facilities desk moved the tuesday maintenance window to thursday "
            "because the loading dock is closed for a delivery that morning. ") * 3
    edited = base.replace("thursday", "friday")
    assert jaccard(char_ngrams(base), char_ngrams(edited)) >= NEAR_DUP
    ban = BanIndex({norm("short ticket"), norm(base)})
    assert ban.overlap("short ticket") == "exact"
    assert ban.overlap("totally different sentence about nothing") is None
    assert ban.overlap(edited) in {"near", "contains", "exact"}
    wrapped = "preamble. " + base + " trailing note from the clerk."
    assert ban.overlap(wrapped) == "contains"
    items = []
    for i, state in enumerate((base, edited, "an unrelated payroll question about a missing bonus line")):
        items.append({
            "_id": "id%d" % i,
            "state": state,
            "q": {"type": "noul", "instructions": "Does the desk need to act today?",
                  "criteria": {"false": "no action today", "true": "act today"}},
            "target": [1.0, 0.0],
            "src": "synth-v1/business/en",
        })
    items.append(dict(items[0], _id="id0b"))
    kept, dropped = apply_filter(items, BanIndex(set()))
    assert len(kept) == 2, (len(kept), dropped)
    assert dropped["dup_exact"] + dropped["dup_near"] == 2
    print("filter checks ok")


def main():
    ap = argparse.ArgumentParser(description="Dedupe synthetic items and drop eval overlaps")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        raise SystemExit("filter runs inside generate.py; use --check for local checks")
    run_checks()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
