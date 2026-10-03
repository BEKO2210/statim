"""Exact eight-word overlap guard for held-out and S1Bench text."""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = 8
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def tokens(text):
    return _WORD.findall(" ".join(str(text or "").casefold().split()))


def grams(text, n=N):
    words = tokens(text)
    return {hashlib.blake2b("\x1f".join(words[i:i + n]).encode(), digest_size=8).digest()
            for i in range(len(words) - n + 1)}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _s1bench_texts(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Export layouts have changed; recursively indexing strings fails safe.
        yield from _strings(payload)


def local_heldout_texts(eval_cache=None, s1bench_dir=None):
    cache = Path(eval_cache) if eval_cache else ROOT / "data" / "eval-texts.pkl"
    if not cache.is_file():
        raise FileNotFoundError("held-out cache not found: %s (build it outside this pilot first)" % cache)
    payload = pickle.loads(cache.read_bytes())
    held = payload.get("texts") if isinstance(payload, dict) else payload
    if not isinstance(held, set):
        raise ValueError("invalid held-out cache: %s" % cache)
    for text in held:
        yield text
    candidates = []
    if s1bench_dir:
        candidates.append(Path(s1bench_dir))
    candidates.extend((ROOT / "data" / "s1bench", ROOT.parent / "lev" / "data" / "s1bench"))
    used = set()
    for directory in candidates:
        resolved = directory.resolve()
        if resolved in used:
            continue
        used.add(resolved)
        yield from _s1bench_texts(resolved)


class LeakageGuard:
    def __init__(self, banned_texts):
        self.index = set()
        self.texts = 0
        for text in banned_texts:
            self.index.update(grams(text))
            self.texts += 1

    @classmethod
    def from_local(cls, eval_cache=None, s1bench_dir=None):
        return cls(local_heldout_texts(eval_cache, s1bench_dir))

    def overlap(self, value):
        for text in _strings(value):
            hit = grams(text) & self.index
            if hit:
                return True
        return False


class DeferredGuard:
    """For a machine without the held-out texts (e.g. Colab): nothing is rejected here, the manifest
    records ``checked: false``, and `leakage.py --filter` must run on a machine that has them before
    any item is used."""
    texts = None
    deferred = True

    def overlap(self, value):
        return False


def filter_dir(src, dst, eval_cache=None, s1bench_dir=None):
    """Copy a grounded-synthesis output dir from ``src`` to ``dst`` without leaking items."""
    import gzip
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit("refusing to write into existing %s" % dst)
    guard = LeakageGuard.from_local(eval_cache, s1bench_dir)
    dst.mkdir(parents=True)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    removed = {}
    for items_path in sorted(src.glob("*.jsonl.gz")):
        if items_path.name.endswith(".provenance.jsonl.gz"):
            continue
        task = items_path.name[:-len(".jsonl.gz")]
        prov_path = src / (task + ".provenance.jsonl.gz")
        with gzip.open(items_path, "rt", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        with gzip.open(prov_path, "rt", encoding="utf-8") as f:
            prov = [json.loads(line) for line in f if line.strip()]
        if len(items) != len(prov):
            raise SystemExit("%s: items and provenance differ in length" % task)
        keep = [i for i, item in enumerate(items) if not guard.overlap(item)]
        removed[task] = len(items) - len(keep)
        for name, rows in ((task + ".jsonl.gz", items), (task + ".provenance.jsonl.gz", prov)):
            with gzip.open(dst / name, "wt", encoding="utf-8") as f:
                for i in keep:
                    f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    manifest["leakage"] = dict(manifest.get("leakage", {}), checked=True, checked_by="leakage.py --filter",
                               heldout_texts_indexed=guard.texts, removed=removed, source_dir=str(src))
    (dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    return removed


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Filter a deferred-check synthesis dir against held-out texts")
    ap.add_argument("--filter", type=Path, required=True, help="input dir (written with --defer-leakage-check)")
    ap.add_argument("--out", type=Path, required=True, help="new output dir (must not exist)")
    ap.add_argument("--eval-cache", type=Path, default=None)
    ap.add_argument("--s1bench-dir", type=Path, default=None)
    a = ap.parse_args()
    print(json.dumps(filter_dir(a.filter, a.out, a.eval_cache, a.s1bench_dir)))
