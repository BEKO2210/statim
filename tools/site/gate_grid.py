#!/usr/bin/env python3
"""Render the no-harm gate result of a release as the site's suite grid.

    python3 tools/site/gate_grid.py models/laya-multilingual models/laya-multilingual-big1 site/index.html

Reads both models' eval.json (written by tools/finetune/gate.py eval), classifies every held-out
suite with the gate's rule (a gain is more than two combined binomial standard errors; a regression
must stay significant after Holm-Bonferroni over all suites, a nominal drop counts as noise) and
replaces the block between <!-- gate-grid:start --> and <!-- gate-grid:end --> in the page.
"""
import html
import json
import math
import os
import re
import sys

NAMES = {
    "amazon_massive_intent": "MASSIVE intents", "belebele": "Belebele reading", "farstail": "FarsTail NLI",
    "go_emotions": "GoEmotions", "hwu64": "HWU64 intents", "indonli": "IndoNLI",
    "multi_hatecheck": "HateCheck", "multilingual_sentiments": "Sentiment", "semrel": "SemRel similarity",
    "sib200": "SIB-200 topics", "test/ag_news": "AG News", "test/banking77": "Banking77",
    "test/emotion": "DAIR Emotion", "test/typed_decisions": "typed-decisions",
}
CATEGORIES = {"complaint": "Complaint", "emotion": "Emotion", "fact_check": "Fact-check", "formality": "Formality",
              "intent": "Intent", "nli": "NLI", "pii": "PII", "reading": "Reading", "safety": "Safety",
              "sentiment": "Sentiment", "similarity": "Similarity", "stance": "Stance", "topic": "Topic",
              "urgency": "Urgency"}
TRAINED = lambda k: k.startswith("amazon_massive_intent/") or k in ("test/banking77", "test/typed_decisions")


def label(k):
    if k.startswith("categories:"):
        cat, lang = k[len("categories:"):].split("/", 1)
        return CATEGORIES.get(cat, cat), lang
    if k.startswith("test/"):
        return NAMES[k], ""
    suite, lang = k.split("/", 1)
    return NAMES.get(suite, suite), lang


def main(champ, chall, page):
    a = json.load(open(os.path.join(champ, "eval.json")))["heldout"]
    b = json.load(open(os.path.join(chall, "eval.json")))["heldout"]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "finetune"))
    from gate import holm_regressions  # the gate's own rule
    keys = [k for k in sorted(set(a) & set(b), key=lambda k: (not k.startswith("test/"), k))
            if not (k.startswith("categories:") and a[k].get("pool") != b[k].get("pool"))]
    tests = []
    for k in keys:
        x, y = a[k], b[k]
        se = math.sqrt(x["acc"] * (1 - x["acc"]) / x["n"] + y["acc"] * (1 - y["acc"]) / y["n"])
        tests.append((k, x["acc"], y["acc"], y["acc"] - x["acc"], se))
    losses = {t[0] for t in holm_regressions(tests)}
    cells = {"trained": [], "categories": [], "held": []}
    counts = {"gain": 0, "noise": 0, "loss": 0}
    for k, xa, ya, d, se in tests:
        x, y = a[k], b[k]
        kind = "gain" if d > 2 * se else "loss" if k in losses else "noise"
        counts[kind] += 1
        name, lang = label(k)
        note = " (nominal drop, not significant after Holm-Bonferroni)" if kind == "noise" and d < -2 * se else ""
        text = f"{name}{' ' + lang if lang else ''}: {x['acc']:.3f} to {y['acc']:.3f}{note}"
        group = "categories" if k.startswith("categories:") else "trained" if TRAINED(k) else "held"
        cells[group].append(
            f'<li class="cell {kind}" title="{html.escape(text)}"><span class="sr">{html.escape(text)}</span>'
            f'<b>{"+" if d >= 0 else "−"}{abs(d) * 100:.0f}</b></li>')
    block = (
        f'<div class="gate-group"><h4>Trained tasks <span>{len(cells["trained"])} suites</span></h4>'
        f'<ul class="cells" role="list">{"".join(cells["trained"])}</ul></div>'
        + (f'<div class="gate-group"><h4>Decision categories <span>{len(cells["categories"])} suites</span></h4>'
           f'<ul class="cells" role="list">{"".join(cells["categories"])}</ul></div>' if cells["categories"] else "")
        + f'<div class="gate-group"><h4>Never trained on <span>{len(cells["held"])} suites</span></h4>'
        f'<ul class="cells" role="list">{"".join(cells["held"])}</ul></div>'
        f'<p class="gate-sum" data-gain="{counts["gain"]}" data-noise="{counts["noise"]}" data-loss="{counts["loss"]}">'
        f'<span class="item"><span class="k gain"></span>{counts["gain"]} significant gains</span>'
        f'<span class="item"><span class="k noise"></span>{counts["noise"]} within noise</span>'
        f'<span class="item"><span class="k loss"></span>{counts["loss"]} regressions</span></p>')
    src = open(page).read()
    new, n = re.subn(r"(<!-- gate-grid:start -->).*?(<!-- gate-grid:end -->)", lambda m: m.group(1) + block + m.group(2), src, flags=re.S)
    if n != 1:
        raise SystemExit("gate-grid markers not found exactly once in " + page)
    open(page, "w").write(new)
    print(json.dumps(counts), {k: len(v) for k, v in cells.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
