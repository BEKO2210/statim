#!/usr/bin/env python3
"""Paired comparison of per-item predictions from `bench/eval_categories.py --predictions`.

Every run must cover the same sample (same suites, languages, --n, --seed and --skip), so item i of
a cell is the same text in every file. For each other run against the base run it reports:

- changed: items whose predicted option differs from the base;
- b: the base was right and the other run wrong; c: the base was wrong and the other run right;
- exact McNemar p (two-sided) on b against c, the paired test for a change in accuracy.

    python3 bench/compare_predictions.py build-ort/predictions-statim-f32.jsonl \\
        build-ort/predictions-statim-q8_0.jsonl build-ort/predictions-ort-f32.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from paired_stats import mcnemar_exact
except ModuleNotFoundError:  # imported as bench.compare_predictions
    from .paired_stats import mcnemar_exact


def load(path: Path) -> dict[tuple[str, str, int], tuple[int, int, str | None]]:
    items = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            items[(row["suite"], row["lang"], int(row["i"]))] = (int(row["gold"]), int(row["pred"]), row.get("item"))
    if not items:
        raise SystemExit(f"{path}: no predictions (did the evaluation fail before its first cell?)")
    return items


def compare(base: dict, other: dict, suites: list[str] | None = None) -> dict:
    keys = sorted(k for k in base if suites is None or k[0] in suites)
    if not keys:
        raise SystemExit(f"no items for suites {suites}")
    missing = [k for k in keys if k not in other]
    if missing:
        raise SystemExit(f"{len(missing)} base items missing from the other run, e.g. {missing[0]}")
    changed = b = c = 0
    for key in keys:
        (gold, p0, item0), (gold1, p1, item1) = base[key], other[key]
        if gold != gold1 or (item0 and item1 and item0 != item1):
            raise SystemExit(f"item {key} differs between the runs: they did not use the same sample")
        changed += p0 != p1
        b += p0 == gold and p1 != gold
        c += p0 != gold and p1 == gold
    base_acc = sum(base[k][1] == base[k][0] for k in keys) / len(keys)
    other_acc = sum(other[k][1] == other[k][0] for k in keys) / len(keys)
    return {"items": len(keys), "changed": changed, "base_right_other_wrong": b, "base_wrong_other_right": c,
            "base_accuracy": base_acc, "other_accuracy": other_acc, "mcnemar_exact_p": mcnemar_exact(b, c)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("others", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a Markdown table")
    a = ap.parse_args()
    base = load(a.base)
    suites = sorted({k[0] for k in base})
    results = []
    for path in a.others:
        other = load(path)
        for scope in [None, *([[s] for s in suites] if len(suites) > 1 else [])]:
            results.append({"base": a.base.name, "other": path.name, "suites": scope or suites,
                            **compare(base, other, scope)})
    if a.json:
        json.dump(results, sys.stdout, indent=2)
        print()
        return 0
    print("| Run against " + a.base.name + " | Suites | Items | Changed | Base right, run wrong | "
          "Base wrong, run right | Accuracy (base → run) | Exact McNemar p |")
    print("|---|---|---:|---:|---:|---:|---|---:|")
    for r in results:
        print(f"| {r['other']} | {', '.join(r['suites'])} | {r['items']} | {r['changed']} | "
              f"{r['base_right_other_wrong']} | {r['base_wrong_other_right']} | "
              f"{r['base_accuracy']:.4f} → {r['other_accuracy']:.4f} | {r['mcnemar_exact_p']:.3g} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
