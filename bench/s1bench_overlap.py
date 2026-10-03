#!/usr/bin/env python3
"""Exact-overlap check between S1Bench items and a training mixture (protocol:
docs/reproductions/s1bench-protocol.md).

    python3 bench/s1bench_overlap.py --tasks <lev>/data/s1bench --mixture data/mixture-v8.jsonl.gz --out overlap.json

Every string of at least 30 characters in an item's state (lower-cased, whitespace collapsed) is
looked up among the same normalisation of every string in the mixture's states. Strings are kept
as 8-byte BLAKE2 digests, so the check fits in memory. Standard library only.
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

MIN_LEN = 30
WS = re.compile(r"\s+")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def key(text):
    norm = WS.sub(" ", text.lower()).strip()
    return hashlib.blake2b(norm.encode(), digest_size=8).digest() if len(norm) >= MIN_LEN else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--mixture", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    seen, rows = set(), 0
    with gzip.open(a.mixture, "rt", encoding="utf-8") as f:
        for line in f:
            rows += 1
            state = json.loads(line).get("state")
            if isinstance(state, str) and state[:1] in "[{":
                try:
                    state = json.loads(state)
                except ValueError:
                    pass
            for s in strings(state):
                k = key(s)
                if k:
                    seen.add(k)
    index = json.loads((a.tasks / "index.json").read_text())
    report = {"mixture": str(a.mixture), "mixture_rows": rows, "mixture_strings": len(seen),
              "min_len": MIN_LEN, "subsets": {}}
    for name in index["subsets"]:
        items = json.loads((a.tasks / f"{name}.json").read_text())["items"]
        hits = [i for i, it in enumerate(items) if any(key(s) in seen for s in strings(it["state"]) if key(s))]
        report["subsets"][name] = {"n": len(items), "overlap": len(hits), "items": hits}
        print(f"{name:22s} {len(hits):4d} / {len(items)}")
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
