#!/usr/bin/env python3
"""Content audit of the mixture v6 adapters: does every source ask a decision a reader can answer?

    python tools/finetune/mixture_v6/audit.py --rows 200 --out data/mixture-v6-audit.json

Loads a sample of every enabled source (the same loaders as build.py), runs its adapter and checks
the items. A flag fails the audit (exit 1) unless ALLOW lists it for that source with a reason:

  serialized-option  a choice option that is a serialized structure ("[{'start': 0, ...}]",
                     "['hello', 'non strategic']") or longer than 160 characters
  numeric-option     every option of a choice question is a number ("0", "1", "2", "0.18"):
                     a ClassLabel id or a score, not something to choose between
  constant-yes-no    one answer in >= 90 % of the yes/no items of one task that reach training
                     (registry.drop_constant_yes_no already removes a task at >= 97 %; the
                     report lists what it removed, counted on the adapter output before the drop)
  constant-choice    one gold option in >= 90 % of a source's items for one question
  constant-score     one level in >= 90 % of a source's score items for one question
  option-mismatch    a sentiment or NLI question whose options are not sentiment / NLI labels
  opaque-option      an option that is an id, not a name: "event4", "LABEL_2", a single Latin letter

A source that fails to load or yields no items is reported as a warning, not a failure: CI
downloads from the Hub and a flaky source must not block an unrelated change.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.finetune.mixture_v6.registry import ADAPTERS, ENTRIES, drop_constant_yes_no, source_key
else:
    from .registry import ADAPTERS, ENTRIES, drop_constant_yes_no, source_key

MIN_ITEMS = 20
YES_NO_SHARE = 0.90
CHOICE_SHARE = 0.90
MAX_OPTION_CHARS = 160
SENTIMENT_OPTIONS = {"positive", "negative", "neutral", "mixed", "very positive", "very negative", "no impact",
                     "liked", "not liked", "did not say"}
NLI_OPTIONS = {"entailment", "neutral", "contradiction"}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
SERIALIZED = re.compile(r"^\s*[\[{]|'\s*:\s*|\"\s*:\s*")
OPAQUE = re.compile(r"(?i)^(?:[a-z]|(?:label|class|event|cat|category|topic|intent|tag)[ _-]?\d+)$")

# source key -> {flag: reason}. Every entry needs a reason a reviewer can check.
BUILD_BALANCES = ("build.py draws the per-source cap round-robin over (question type, gold) buckets, so the "
                  "mixture gets every minority item")
ALLOW = {
    "mteb/toxic_conversations_50k::default": {
        "constant-choice": "about 8 % of the rows are toxic, the true rate of the source; " + BUILD_BALANCES,
        "constant-yes-no": "about 8 % of the rows are toxic, the true rate of the source; " + BUILD_BALANCES},
    "OpenAssistant/oasst2::default": {
        "constant-choice": "about 5 % of the messages have a harmful crowd vote >= 0.5; " + BUILD_BALANCES,
        "constant-yes-no": "about 5 % of the messages have a harmful crowd vote >= 0.5; " + BUILD_BALANCES},
    "gretelai/synthetic_pii_finance_multilingual::default": {
        "constant-yes-no": "about 6 % of the documents have no PII span, the only negatives among the PII "
                           "sources; " + BUILD_BALANCES},
}


def _gold(item):
    return list(item["q"]["criteria"])[item["target"].index(1.0)]


def audit_items(items):
    """Flags for one source's items: {flag: [details]}."""
    flags = collections.defaultdict(list)
    yes_no = collections.defaultdict(collections.Counter)
    choice = collections.defaultdict(collections.Counter)
    score = collections.defaultdict(collections.Counter)
    for item in items:
        q, task = item["q"], item.get("_task")
        if q["type"] == "noul":
            yes_no[task][item["target"][1] == 1.0] += 1
            continue
        if q["type"] == "score":
            score[(task, tuple(q["criteria"]))][item["target"].index(1.0)] += 1
            continue
        if q["type"] != "choice":
            continue
        options = list(q["criteria"])
        for option in options:
            if SERIALIZED.search(option) or len(option) > MAX_OPTION_CHARS:
                flags["serialized-option"].append(option[:80])
                break
        opaque = [o for o in options if OPAQUE.fullmatch(o.strip())]
        if opaque:
            flags["opaque-option"].append(", ".join(opaque[:6]))
        if all(NUMBER.fullmatch(o.strip()) for o in options):
            flags["numeric-option"].append(", ".join(options[:6]))
        allowed = SENTIMENT_OPTIONS if task == "sentiment" else NLI_OPTIONS if task == "nli" else None
        if allowed and not set(o.lower() for o in options) <= allowed:
            flags["option-mismatch"].append("%s: %s" % (task, ", ".join(sorted(set(options) - allowed)[:5])))
        choice[(task, tuple(sorted(options)))][_gold(item)] += 1
    for task, counts in yes_no.items():
        n = sum(counts.values())
        if n >= MIN_ITEMS and max(counts.values()) >= YES_NO_SHARE * n:
            answer = "yes" if counts[True] >= counts[False] else "no"
            flags["constant-yes-no"].append("%s: %s in %d of %d" % (task, answer, max(counts.values()), n))
    for (task, _options), counts in choice.items():
        n = sum(counts.values())
        top, count = counts.most_common(1)[0]
        if n >= MIN_ITEMS and count >= CHOICE_SHARE * n:
            flags["constant-choice"].append("%s: %r in %d of %d" % (task, top, count, n))
    for (task, _levels), counts in score.items():
        n = sum(counts.values())
        top, count = counts.most_common(1)[0]
        if n >= MIN_ITEMS and count >= CHOICE_SHARE * n:
            flags["constant-score"].append("%s: level %d in %d of %d" % (task, top, count, n))
    return {flag: sorted(set(details))[:5] for flag, details in flags.items()}


def audit_source(entry, rows_per_source, seed=20260927):
    from tools.finetune.mixture_v6.build import valid_item
    from tools.finetune.mixture_v6.loaders import load_rows
    key = source_key(entry)
    try:
        rows, _warnings = load_rows(entry, rows_per_source)
        raw = [item for item in ADAPTERS[key](entry, list(rows), seed) if item and valid_item(item)]
        items = drop_constant_yes_no(raw)  # what adapt() hands to training
    except Exception as exc:  # reported as a warning
        return {"key": key, "error": "%s: %s" % (type(exc).__name__, (str(exc).splitlines() or [""])[0][:200])}
    dropped = collections.Counter(item.get("_task") for item in raw if item["q"]["type"] == "noul") - \
        collections.Counter(item.get("_task") for item in items if item["q"]["type"] == "noul")
    result = {"key": key, "rows": len(rows), "items": len(items), "flags": audit_items(items),
              "dropped_yes_no": dict(dropped)}  # constant yes/no tasks removed before training
    allowed = ALLOW.get(key, {})
    result["failing"] = sorted(f for f in result["flags"] if f not in allowed)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=400, help="rows loaded per source")
    ap.add_argument("--only", nargs="+", default=None, metavar="ID", help="registry ids or source keys")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSON report")
    a = ap.parse_args(argv)
    entries = [e for e in ENTRIES if not a.only or e["id"] in a.only or source_key(e) in a.only]
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda e: audit_source(e, a.rows), entries))
    failing = [r for r in results if r.get("failing")]
    warnings = [r for r in results if "error" in r or not r.get("items")]
    for r in results:
        if "error" in r:
            status = "WARN load: " + r["error"]
        elif not r["items"]:
            status = "WARN no items from %d rows" % r["rows"]
        elif r["failing"]:
            status = "FAIL " + "; ".join("%s (%s)" % (f, " | ".join(r["flags"][f])) for f in r["failing"])
        elif r["flags"]:
            status = "ok (allowed: %s)" % ", ".join(sorted(r["flags"]))
        else:
            status = "ok"
        if r.get("dropped_yes_no"):
            status += " [dropped constant yes/no: %s]" % ", ".join("%s %d" % kv for kv in sorted(r["dropped_yes_no"].items()))
        print("%-100s %s" % (r["key"][:100], status), flush=True)
    print("\n%d sources: %d failing, %d warnings" % (len(results), len(failing), len(warnings)))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
