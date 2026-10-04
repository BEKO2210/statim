"""Fail-closed held-out/S1Bench contamination guard for grounded synthesis."""
from __future__ import annotations

import collections
import hashlib
import sys
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORD_SHINGLE = 8
CHAR_SHINGLE = 20
CONTAIN_CHARS = 40
INDEX_CHARS = 4000


def norm(text):
    return " ".join(str(text or "").split()).lower()


def _units(text):
    words = text.split()
    return (words, WORD_SHINGLE, " ") if len(words) >= WORD_SHINGLE else (
        list(text), CHAR_SHINGLE, "")


def anchors(text):
    text = norm(text)
    units, size, separator = _units(text)
    return {separator.join(units[i:i + size]) for i in range(len(units) - size + 1)}


def _ordered_anchors(text):
    text = norm(text)
    units, size, separator = _units(text)
    return [separator.join(units[i:i + size]) for i in range(len(units) - size + 1)]


def grams(text, n=WORD_SHINGLE):
    """Word shingles used for the independent seed/generated-copy rejection."""
    words = norm(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _strings(value):
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _eval_definition():
    if str(ROOT) not in sys.path:  # run as a script from tools/synth, the repo root is not on the path
        sys.path.insert(0, str(ROOT))
    from tools.finetune.mixture_v6 import eval_texts
    categories = eval_texts._eval_categories().fingerprint()
    suite_fp = hashlib.sha256(json.dumps({
        "suites": eval_texts.SUITES, "categories": categories,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return eval_texts.SUITES, categories, suite_fp


def _load_eval_cache(path):
    if not path.is_file():
        raise FileNotFoundError(
            "held-out cache not found: %s (build it outside this pilot first)" % path)
    payload = pickle.loads(path.read_bytes())
    suites, categories, suite_fp = _eval_definition()
    if not isinstance(payload, dict) or payload.get("suites") != suites:
        raise ValueError("held-out cache suite list/fingerprint mismatch: %s" % path)
    if payload.get("categories") != categories:
        raise ValueError("held-out cache category fingerprint mismatch: %s" % path)
    held = payload.get("texts")
    if not isinstance(held, set):
        raise ValueError("invalid held-out cache texts: %s" % path)
    if not held:
        raise ValueError("held-out cache is empty: %s" % path)
    return held, suite_fp


def _s1bench_texts(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("S1Bench export directory not found: %s" % directory)
    index_path = directory / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError("S1Bench index not found: %s" % index_path)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unreadable S1Bench index %s: %s" % (index_path, exc)) from exc
    subsets = index.get("subsets") if isinstance(index, dict) else None
    if not isinstance(subsets, dict) or not subsets:
        raise ValueError("S1Bench index has no subsets: %s" % index_path)
    texts = []
    for subset in subsets:
        path = directory / (subset + ".json")
        if not path.is_file():
            raise FileNotFoundError("S1Bench subset not found: %s" % path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("unreadable S1Bench subset %s: %s" % (path, exc)) from exc
        texts.extend(_strings(payload))
    if not texts:
        raise ValueError("S1Bench export contains no text: %s" % directory)
    fingerprint = hashlib.sha256(index_path.read_bytes()).hexdigest()
    return texts, fingerprint, len(subsets)


def local_heldout_texts(eval_cache=None, s1bench_dir=None):
    cache = Path(eval_cache) if eval_cache else ROOT / "data" / "eval-texts.pkl"
    s1dir = Path(s1bench_dir) if s1bench_dir else ROOT / "data" / "s1bench"
    held, suite_fp = _load_eval_cache(cache)
    s1texts, s1_fp, subsets = _s1bench_texts(s1dir)
    return held | set(s1texts), {
        "suite_fingerprint": suite_fp,
        "s1bench_fingerprint": s1_fp,
        "s1bench_subsets": subsets,
        "eval_texts": len(held),
        "s1bench_texts": len(s1texts),
    }


class LeakageGuard:
    """Exact, shingle, and 40-character containment predicates from eval_categories."""

    deferred = False

    def __init__(self, banned_texts, metadata=None):
        texts = {norm(text) for text in banned_texts if norm(text)}
        if not texts:
            raise ValueError("leakage guard refuses an empty held-out set")
        self.texts = len(texts)
        self.metadata = dict(metadata or {})
        self.exact = texts
        self.anchor_index = set()
        self.first = collections.defaultdict(list)
        self.inner = collections.defaultdict(list)
        for text in sorted(texts):
            self.anchor_index.update(anchors(text))
            if len(text) < CONTAIN_CHARS:
                continue
            ordered = _ordered_anchors(text)
            if ordered:
                self.first[ordered[0]].append(text)
            for anchor in anchors(text[:INDEX_CHARS]):
                self.inner[anchor].append(text)

    @classmethod
    def from_local(cls, eval_cache=None, s1bench_dir=None):
        texts, metadata = local_heldout_texts(eval_cache, s1bench_dir)
        return cls(texts, metadata)

    def overlap(self, value):
        for raw in _strings(value):
            text = norm(raw)
            if not text:
                continue
            if text in self.exact or anchors(text) & self.anchor_index:
                return True
            if len(text) < CONTAIN_CHARS:
                continue
            ordered = _ordered_anchors(text)
            for anchor in ordered:
                if any(pool_text in text for pool_text in self.first.get(anchor, ())):
                    return True
            if ordered and any(text in pool_text for pool_text in self.inner.get(ordered[0], ())):
                return True
        return False


class DeferredGuard:
    """Colab-only marker; output remains unusable until ``filter_dir`` succeeds."""
    texts = None
    metadata = {}
    deferred = True

    def overlap(self, value):
        return False


def generated_text(row):
    """The generated part of a training row: its state.

    The question and criteria are the fixed task template, shared on purpose with the
    evaluation suites; checking them would reject every row.
    """
    state = row.get("state")
    if not isinstance(state, str) or not state.strip():
        raise SystemExit("row %s has no state text" % row.get("id"))
    return state


def _read_gzip(path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_gzip(path, rows):
    import gzip
    part = path.with_name(path.name + ".part")
    with gzip.open(part, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    part.replace(path)


def filter_dir(src, dst, eval_cache=None, s1bench_dir=None):
    """Filter deferred output, joining provenance by stable item id and rewriting samples."""
    from grounded import write_samples
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit("refusing to write into existing %s" % dst)
    guard = LeakageGuard.from_local(eval_cache, s1bench_dir)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dst.mkdir(parents=True)
    removed, survivors = {}, {}
    for items_path in sorted(src.glob("*.jsonl.gz")):
        if items_path.name.endswith(".provenance.jsonl.gz"):
            continue
        task = items_path.name[:-len(".jsonl.gz")]
        items = _read_gzip(items_path)
        prov_rows = _read_gzip(src / (task + ".provenance.jsonl.gz"))
        prov = {}
        for row in prov_rows:
            ident = row.get("id")
            if not ident or ident in prov:
                raise SystemExit("%s: missing or duplicate provenance id" % task)
            prov[ident] = row
        item_ids = [row.get("id") for row in items]
        if any(not ident for ident in item_ids) or len(set(item_ids)) != len(item_ids):
            raise SystemExit("%s: missing or duplicate training-row id" % task)
        if set(item_ids) != set(prov):
            raise SystemExit("%s: item/provenance id sets differ" % task)
        kept = [row for row in items if not guard.overlap(generated_text(row))]
        kept_prov = [prov[row["id"]] for row in kept]
        removed[task] = len(items) - len(kept)
        _write_gzip(dst / (task + ".jsonl.gz"), kept)
        _write_gzip(dst / (task + ".provenance.jsonl.gz"), kept_prov)
        survivors[task] = [{"id": row["id"], "item": row,
                            "provenance": prov[row["id"]]} for row in kept]
    write_samples(dst / "samples.md", survivors, 30, manifest.get("seed", 20261003))
    manifest["leakage"] = dict(manifest.get("leakage", {}), checked=True,
        checked_by="leakage.py --filter", heldout_texts_indexed=guard.texts,
        removed=removed, source_dir=str(src), **guard.metadata)
    (dst / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return removed


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Filter deferred synthesis output against held-out text")
    ap.add_argument("--filter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-cache", type=Path, default=None)
    ap.add_argument("--s1bench-dir", type=Path, default=None)
    args = ap.parse_args()
    print(json.dumps(filter_dir(args.filter, args.out, args.eval_cache, args.s1bench_dir)))
