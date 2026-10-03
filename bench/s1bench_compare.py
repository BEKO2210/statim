#!/usr/bin/env python3
"""Summarise two S1Bench runs from bench/s1bench_run.py: per-subset accuracy, paired exact
McNemar with Holm correction, and the macros fixed in docs/reproductions/s1bench-protocol.md.

    python3 bench/s1bench_compare.py bench/results/s1bench/s1-statim070-f32.json.gz \
        bench/results/s1bench/s1-laya-f32.json.gz --overlap bench/results/s1bench/overlap-v8.json

Lev and Jev columns are levbench numbers quoted from Lev's docs/FINDINGS.md (not re-measured).
Standard library only.
"""
import argparse
import gzip
import json
from math import comb

# Abhinavexists/lev docs/FINDINGS.md (levbench, all 13 subsets), as on the Lev model card
LEV = {"vitaminc-dev": .668, "massive-en-US": .857, "massive-de-DE": .823, "boolq": .827, "squad2": .813,
       "paws": .776, "multinli": .890, "civil_comments": .760, "aegis2": .800, "helpsteer2": .386,
       "summeval-relevance": .358, "summeval-consistency": .271, "pubmedqa": .732}
JEV = {"vitaminc-dev": .801, "massive-en-US": .874, "massive-de-DE": .871, "boolq": .893, "squad2": .836,
       "paws": .900, "multinli": .836, "civil_comments": .803, "aegis2": .804, "helpsteer2": .341,
       "summeval-relevance": .358, "summeval-consistency": .812, "pubmedqa": .764}
UNSEEN = ["vitaminc-dev", "boolq", "squad2", "paws", "civil_comments", "helpsteer2",
          "summeval-relevance", "summeval-consistency", "pubmedqa"]
BOARD = ["vitaminc-dev", "massive-en-US", "boolq", "helpsteer2", "aegis2", "paws"]


def load(path):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)["subsets"]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--overlap")
    x = ap.parse_args()
    a, b = load(x.a), load(x.b)
    skip = {s: set(v["items"]) for s, v in json.load(open(x.overlap))["subsets"].items()} if x.overlap else {}
    rows = []
    for s in a:
        ra, rb = a[s]["records"], b[s]["records"]
        gain = sum(1 for p, q in zip(ra, rb) if p["pred"] == p["truth"] and q["pred"] != q["truth"])
        loss = sum(1 for p, q in zip(ra, rb) if p["pred"] != p["truth"] and q["pred"] == q["truth"])
        rows.append([s, a[s]["accuracy"], b[s]["accuracy"], gain, loss, mcnemar(gain, loss), a[s]["ece"], b[s]["ece"]])
    order, prev = sorted(range(len(rows)), key=lambda i: rows[i][5]), 0.0
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (len(rows) - k) * rows[i][5]))
        rows[i].append(prev)
    print("| Subset | A | B | A right, B wrong | A wrong, B right | Holm p | ECE A | ECE B | Lev | Jev |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, aa, bb, g, l, _, ea, eb, ph in rows:
        print(f"| {s} | {aa:.3f} | {bb:.3f} | {g} | {l} | {ph:.3g} | {ea:.3f} | {eb:.3f} | {LEV[s]:.3f} | {JEV[s]:.3f} |")
    A = {s: a[s]["accuracy"] for s in a}
    B = {s: b[s]["accuracy"] for s in b}
    clean = {s: (lambda r: sum(p["pred"] == p["truth"] for p in r) / len(r))(
        [p for i, p in enumerate(a[s]["records"]) if i not in skip.get(s, set())]) for s in a}
    mac = lambda d, ks: sum(d[k] for k in ks) / len(ks)
    print("\n| Macro | A | A without overlap items | B | Lev | Jev |\n|---|---:|---:|---:|---:|---:|")
    for label, ks in [("all 13 subsets", list(a)), ("9 subsets from sources A never trained on", UNSEEN),
                      ("6 subsets of the board snapshot", BOARD)]:
        print(f"| {label} | {mac(A, ks):.3f} | {mac(clean, ks):.3f} | {mac(B, ks):.3f} | {mac(LEV, ks):.3f} | {mac(JEV, ks):.3f} |")
    print(f"\nmean ECE: A {sum(r[6] for r in rows) / len(rows):.3f}, B {sum(r[7] for r in rows) / len(rows):.3f}")


if __name__ == "__main__":
    main()
