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
import math
import re
import sys
import unicodedata
from pathlib import Path

from common import (
    MinHash, NEAR_DUP, char_ngrams, item_fingerprint, jaccard, norm, sha256_text,
)

# Same prefix cut as tools/synth/report.py. A repeated opening is capped inside
# one task and language; a prefix that occurs once is kept.
PREFIX_CAP = 0.03
CLASS_CAP = 0.45
SEMANTIC_COSINE = 0.92
E5_MODEL = "intfloat/multilingual-e5-small"

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


def task_lang(item):
    parts = str(item.get("src", "")).split("/")
    task = parts[1] if len(parts) > 1 else "?"
    lang = parts[2] if len(parts) > 2 else "?"
    return task, lang


def _tokens(text, lang):
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    if lang in {"ja", "zh"}:
        return [ch for ch in text if unicodedata.category(ch)[0] in {"L", "N"}]
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def first_prefix(text, lang):
    """First three tokens. Japanese and Chinese use the first three letters."""
    seq = _tokens(text, lang)
    if not seq:
        return ""
    if lang in {"ja", "zh"}:
        return "".join(seq[:3])
    return " ".join(seq[:3])


def _field_text(item, field):
    if field == "state":
        return item.get("state") or ""
    return ((item.get("q") or {}).get("instructions")) or ""


def item_class(item):
    """Gold class inside one task. Score uses the level position, not the phrase."""
    target = item.get("target") or []
    if 1.0 not in target:
        return "?"
    index = target.index(1.0)
    q = item.get("q") or {}
    kind = q.get("type")
    criteria = q.get("criteria")
    if kind == "noul":
        return "true" if index == 1 else "false"
    if kind == "score":
        return "level:%d/%d" % (index + 1, len(target))
    if isinstance(criteria, dict):
        keys = list(criteria)
        return keys[index] if index < len(keys) else str(index)
    return "pos:%d/%d" % (index + 1, len(target))


def _rank(item, seed, salt):
    blob = "%s\n%s\n%s" % (seed, salt, item.get("_id") or item.get("state") or "")
    return sha256_text(blob)


def _cap_groups(items, group_fn, key_fn, max_share, seed, salt, trim_only_key=False):
    """Keep at most floor(max_share * n) items of one key, n counted before the cap.

    The later items in a seeded order are the ones dropped. A single class is
    left intact: two labels cannot both sit at or under 0.45. A prefix that is
    the only opening in the cell is still trimmed, because the cap is there to
    remove a repeated opening.
    """
    groups = collections.defaultdict(list)
    for item in items:
        groups[group_fn(item)].append(item)
    kept = []
    dropped = {}
    for cell, rows in groups.items():
        counts = collections.defaultdict(list)
        for row in rows:
            counts[key_fn(row)].append(row)
        if len(counts) == 1 and not trim_only_key:
            kept.extend(rows)
            continue
        limit = max(1, int(math.floor(max_share * len(rows) + 1e-9)))
        cell_dropped = 0
        for members in counts.values():
            if len(members) <= limit:
                kept.extend(members)
                continue
            members.sort(key=lambda row: _rank(row, seed, salt + str(cell)))
            kept.extend(members[:limit])
            cell_dropped += len(members) - limit
        if cell_dropped:
            dropped[cell] = cell_dropped
    return kept, {"by_cell": dropped, "total": sum(dropped.values())}


def cap_prefixes(items, max_share, seed, field="state"):
    """Cap one first-three-token prefix inside each task/language cell."""
    def group_fn(item):
        task, lang = task_lang(item)
        return "%s/%s" % (task, lang)

    def key_fn(item):
        _task, lang = task_lang(item)
        return first_prefix(_field_text(item, field), lang)

    return _cap_groups(items, group_fn, key_fn, max_share, seed, "prefix:" + field, trim_only_key=True)


def cap_classes(items, max_share, seed):
    """No gold class may exceed max_share of that task's remaining items."""
    if not max_share or max_share >= 1:
        return list(items), {"by_cell": {}, "total": 0}

    def group_fn(item):
        return task_lang(item)[0]

    return _cap_groups(items, group_fn, item_class, max_share, seed, "class", trim_only_key=False)


def _l2(vectors):
    out = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            out.append([value / norm for value in vector])
        else:
            out.append(list(vector))
    return out


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def embed_states(texts, batch_size=32):
    """Mean-pooled multilingual-e5-small on CPU. Same pooling as report._e5_mean_pooled."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer
    print("embed: %d states with %s" % (len(texts), E5_MODEL), flush=True)
    tok = AutoTokenizer.from_pretrained(E5_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(E5_MODEL, local_files_only=True).eval()
    prefixed = ["query: " + (text or "") for text in texts]
    pooled = []
    with torch.no_grad():
        for start in range(0, len(prefixed), batch_size):
            enc = tok(prefixed[start:start + batch_size], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled.extend(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).tolist())
    return _l2(pooled)


def semantic_dedup(items, vectors, seed, threshold=SEMANTIC_COSINE):
    """Within one task and language, drop the later item of a pair at cosine >= threshold."""
    groups = collections.defaultdict(list)
    for index, item in enumerate(items):
        task, lang = task_lang(item)
        groups["%s/%s" % (task, lang)].append(index)
    drop = set()
    by_cell = {}
    for cell, indices in groups.items():
        order = sorted(indices, key=lambda index: _rank(items[index], seed, "sem:" + cell))
        kept_idx = []
        cell_drop = 0
        for index in order:
            if any(_dot(vectors[index], vectors[other]) >= threshold for other in kept_idx):
                drop.add(index)
                cell_drop += 1
                continue
            kept_idx.append(index)
        if cell_drop:
            by_cell[cell] = cell_drop
    kept = [item for index, item in enumerate(items) if index not in drop]
    return kept, {"by_cell": by_cell, "total": len(drop)}


def prefix_tops(items, field):
    by_lang = collections.defaultdict(list)
    for item in items:
        by_lang[task_lang(item)[1]].append(item)
    out = {}
    for lang, rows in sorted(by_lang.items()):
        counts = collections.Counter(first_prefix(_field_text(row, field), lang) for row in rows)
        counts.pop("", None)
        if not counts:
            continue
        prefix, count = counts.most_common(1)[0]
        out[lang] = {"prefix": prefix, "count": count, "share": count / float(len(rows)), "n": len(rows)}
    return out


def semantic_nn_shares(items, vectors):
    """Fraction of items whose nearest neighbour in the same language is at least 0.92."""
    by_lang = collections.defaultdict(list)
    for index, item in enumerate(items):
        by_lang[task_lang(item)[1]].append(vectors[index])
    out = {}
    for lang, vecs in sorted(by_lang.items()):
        n = len(vecs)
        if n < 2:
            out[lang] = None
            continue
        high = 0
        for i in range(n):
            best = max(_dot(vecs[i], vecs[j]) for j in range(n) if j != i)
            if best >= SEMANTIC_COSINE:
                high += 1
        out[lang] = high / float(n)
    return out


def apply_filter(items, ban, seed=20260927, max_class_share=CLASS_CAP, max_prefix_share=PREFIX_CAP,
                 semantic=False, embed_fn=None):
    survivors = []
    flat = collections.Counter()
    for item in items:
        reason = ban.overlap(item["state"])
        if reason:
            flat["eval_" + reason] += 1
            continue
        survivors.append(item)
    kept, dup_dropped = dedup(survivors)
    flat["dup_exact"] = dup_dropped["exact"]
    flat["dup_near"] = dup_dropped["near"]
    before = list(kept)
    vectors = None
    if semantic and before:
        if embed_fn is None:
            embed_fn = embed_states
        vectors = embed_fn([item.get("state") or "" for item in before])
        if len(vectors) != len(before):
            raise SystemExit("embedder returned %d vectors for %d items" % (len(vectors), len(before)))
    id_to_vec = {}
    if vectors is not None:
        for item, vector in zip(before, vectors):
            id_to_vec[id(item)] = vector
    kept, state_prefix = cap_prefixes(kept, max_prefix_share, seed, field="state")
    # Instructions come from six fixed paraphrases per task and language (about 16 % each by
    # design), so a 3 % prefix cap on them only threw good items away (pilot 3: 89 of 298).
    instr_prefix = {"total": 0, "by_cell": {}}
    sem_info = {"by_cell": {}, "total": 0}
    if vectors is not None and kept:
        aligned = [id_to_vec[id(item)] for item in kept]
        nn_before = semantic_nn_shares(before, vectors)
        kept, sem_info = semantic_dedup(kept, aligned, seed)
        aligned_after = [id_to_vec[id(item)] for item in kept]
        nn_after = semantic_nn_shares(kept, aligned_after) if kept else {}
    else:
        nn_before = {}
        nn_after = {}
    kept, class_info = cap_classes(kept, max_class_share, seed)
    if vectors is not None and kept:
        nn_after = semantic_nn_shares(kept, [id_to_vec[id(item)] for item in kept])
    dropped = dict(flat)
    dropped.update({
        "prefix": state_prefix["total"],
        "prefix_dropped_by_cell": state_prefix["by_cell"],
        "instruction_prefix": instr_prefix["total"],
        "instruction_prefix_dropped_by_cell": instr_prefix["by_cell"],
        "semantic": sem_info["total"],
        "semantic_dropped_by_cell": sem_info["by_cell"],
        "class_cap": class_info["total"],
        "class_cap_dropped_by_task": class_info["by_cell"],
        "prefix_state_top_before": prefix_tops(before, "state"),
        "prefix_state_top_after": prefix_tops(kept, "state"),
        "prefix_instruction_top_before": prefix_tops(before, "instructions"),
        "prefix_instruction_top_after": prefix_tops(kept, "instructions"),
        "semantic_nn_before": nn_before,
        "semantic_nn_after": nn_after,
    })
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
    items[2]["q"] = dict(items[2]["q"], instructions="Which payroll desk should review the missing bonus line?")
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
