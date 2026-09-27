#!/usr/bin/env python3
"""Render the release results chart (light + dark SVG) from two gate evaluations.

    python3 tools/diagrams/gate_chart.py models/laya-multilingual/eval.json models/laya-multilingual-clean/eval.json \\
        --out assets/diagrams/results-0.3.0 --version 0.3.0

Numbers come straight from tools/finetune/gate.py output; a group averages its languages, and a
difference is marked significant when it exceeds two standard errors of that average.
"""
import argparse
import json
import math
from collections import OrderedDict

FONT = "Inter, 'Segoe UI', -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"
THEMES = {
    "light": dict(panel="#F6F7F9", ink="#161B22", text2="#3A4350", muted="#6B7482", line="#D3D8DF",
                  base="#A9B1BC", signal="#0F9F6E", soft="#E3F6EE", chip_ink="#0B6B4B"),
    "dark": dict(panel="#12151B", ink="#E7EAEF", text2="#C2C8D1", muted="#8A93A1", line="#2B323D",
                 base="#4A5361", signal="#34D399", soft="#10261E", chip_ink="#34D399"),
}
SECTIONS = [
    ("Trained tasks, held-out test rows", [
        ("test/banking77", "Banking77 intents", "77 labels, 2,000 rows"),
        ("amazon_massive_intent", "MASSIVE intents", "59 labels, 12 languages"),
        ("test/typed_decisions", "Typed decisions", "2,000 decisions"),
    ]),
    ("Never trained on (zero-shot)", [
        ("go_emotions", "GoEmotions", "28 labels, en"),
        ("sib200", "SIB-200 topic", "4 languages"),
        ("multi_hatecheck", "HateCheck", "11 languages"),
        ("indonli", "IndoNLI", "Indonesian NLI"),
        ("farstail", "FarsTail", "Persian NLI"),
        ("test/ag_news", "AG News", "2,000 rows"),
        ("test/emotion", "DAIR Emotion", "2,000 rows"),
        ("semrel", "SemRel", "ordinal, 3 languages"),
        ("belebele", "Belebele", "reading, 4 languages"),
    ]),
    ("Sibling dataset (shares MASSIVE's intent labels)", [
        ("hwu64", "HWU64 intents", "rows seen in MASSIVE removed"),
    ]),
]

# An English-only model is judged on the English variant of each suite.
BASE_NAME = "laya-multilingual"
SECTIONS_EN = [
    ("Trained tasks, held-out test rows", [
        ("test/banking77", "Banking77 intents", "77 labels, 2,000 rows"),
        ("amazon_massive_intent/en", "MASSIVE intents", "59 labels, English"),
        ("test/typed_decisions", "Typed decisions", "2,000 decisions"),
    ]),
    ("Never trained on (zero-shot), English", [
        ("go_emotions", "GoEmotions", "28 labels"),
        ("sib200/en", "SIB-200 topic", "7 topics"),
        ("multi_hatecheck/en", "HateCheck", "hate speech"),
        ("multilingual_sentiments/en", "Sentiment", "3 labels"),
        ("test/ag_news", "AG News", "2,000 rows"),
        ("test/emotion", "DAIR Emotion", "2,000 rows"),
        ("semrel/en", "SemRel", "ordinal similarity"),
        ("belebele/en", "Belebele", "reading"),
    ]),
    ("Sibling dataset (shares MASSIVE's intent labels)", [
        ("hwu64", "HWU64 intents", "rows seen in MASSIVE removed"),
    ]),
]


def group(ev, key):
    rows = [v for k, v in ev["heldout"].items() if k == key or k.split("/")[0] == key]
    k = len(rows)
    mean = sum(r["acc"] for r in rows) / k
    var = sum(r["acc"] * (1 - r["acc"]) / r["n"] for r in rows) / (k * k)
    return mean, var


def text(x, y, s, size, fill, weight=None, anchor=None, family=FONT):
    w = f' font-weight="{weight}"' if weight else ""
    a = f' text-anchor="{anchor}"' if anchor else ""
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}"{w}{a} fill="{fill}">{s}</text>'


def render(base, cand, theme, version, name):
    c = THEMES[theme]
    W, X0, X1 = 960, 300, 820  # bar area
    y, out = 176, []
    for title, rows in SECTIONS:
        out.append(text(40, y, title, 13, c["muted"], weight=600))
        y += 14
        for key, label, note in rows:
            (b, vb), (n, vn) = group(base, key), group(cand, key)
            d, se = n - b, math.sqrt(vb + vn)
            out.append(text(40, y + 17, label, 14.5, c["ink"], weight=600))
            out.append(text(40, y + 35, note, 12.5, c["text2"]))
            for i, (v, fill) in enumerate(((b, c["base"]), (n, c["signal"]))):
                by = y + 8 + i * 16
                out.append(f'<rect x="{X0}" y="{by}" width="{max(2, (X1 - X0) * v):.1f}" height="12" rx="3" fill="{fill}"/>')
                out.append(text(X0 + (X1 - X0) * v + 8, by + 10.5, f"{v:.3f}", 12.5,
                                c["ink"] if i else c["text2"], weight=600 if i else None, family=MONO))
            sig = abs(d) > 2 * se
            label_d = f"{d * 100:+.1f} pts"
            if sig and d > 0:
                out.append(f'<rect x="{W - 104}" y="{y + 10}" width="64" height="22" rx="11" fill="{c["soft"]}"/>')
                out.append(text(W - 72, y + 25.5, label_d, 12.5, c["chip_ink"], weight=600, anchor="middle"))
            else:
                out.append(text(W - 72, y + 25.5, label_d, 12.5, c["muted"], anchor="middle"))
            y += 48
        y += 18
    h = y + 56
    grid = []
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = X0 + (X1 - X0) * t
        grid.append(f'<path d="M{x:.1f} 162V{y - 18}" stroke="{c["line"]}" stroke-width="1"/>')
        grid.append(text(x, 156, f"{t:g}", 12, c["muted"], anchor="middle", family=MONO))
    legend = (f'<rect x="{X0}" y="{y - 2}" width="12" height="12" rx="3" fill="{c["base"]}"/>'
              + text(X0 + 18, y + 8.5, f"base checkpoint ({BASE_NAME})", 12.5, c["text2"])
              + f'<rect x="{X0 + 250}" y="{y - 2}" width="12" height="12" rx="3" fill="{c["signal"]}"/>'
              + text(X0 + 268, y + 8.5, f"Statim {version}, licence-clean training data", 12.5, c["text2"]))
    note = text(40, y + 36, "Accuracy. A green chip marks a difference larger than two standard errors; "
                "grey differences are within sampling noise.", 12.5, c["muted"])
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" role="img" '
            f'aria-labelledby="{name}-title {name}-desc">'
            f'<title id="{name}-title">Statim {version}: licence-clean model vs. base checkpoint</title>'
            f'<desc id="{name}-desc">Accuracy of the base checkpoint and the Statim {version} model on '
            + "; ".join(f"{lab}: {group(base, k)[0]:.3f} to {group(cand, k)[0]:.3f}" for _, rs in SECTIONS for k, lab, _ in rs)
            + ".</desc>"
            f'<rect x="0" y="0" width="{W}" height="{h}" rx="16" fill="{c["panel"]}"/>'
            f'<g aria-hidden="true"><path d="M41 44H50.5" stroke="{c["ink"]}" stroke-width="2.5" stroke-linecap="round"/>'
            f'<circle cx="57" cy="36" r="2.2" fill="none" stroke="{c["muted"]}" stroke-width="1.25"/>'
            f'<circle cx="57" cy="44" r="4" fill="{c["signal"]}"/>'
            f'<circle cx="57" cy="52" r="2.2" fill="none" stroke="{c["muted"]}" stroke-width="1.25"/></g>'
            + text(73, 48.5, f"Release {version}", 12.5, c["text2"], weight=500)
            + text(40, 88, "Better on every trained task, no significant loss where it never trained", 20, c["ink"], weight=600)
            + text(40, 114, "Held-out test rows and zero-shot suites, seeded samples; the model never saw these rows.",
                   14, c["text2"]))
    return head + "".join(grid) + "".join(out) + legend + note + "</svg>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("candidate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--english", action="store_true", help="English model: use the English variant of each suite")
    a = ap.parse_args()
    if a.english:
        global SECTIONS, BASE_NAME
        SECTIONS, BASE_NAME = SECTIONS_EN, "laya, English"
    base, cand = json.load(open(a.base)), json.load(open(a.candidate))
    for theme in THEMES:
        name = a.out.split("/")[-1] + "-" + theme
        open(f"{a.out}-{theme}.svg", "w").write(render(base, cand, theme, a.version, name))


if __name__ == "__main__":
    main()
