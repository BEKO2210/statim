#!/usr/bin/env python3
"""Gold-label accuracy/calibration for Statim on public multilingual test splits.

Reuses metrics() and run_statim() from eval_accuracy.py (not copied).

    python bench/eval_multilingual.py --url http://127.0.0.1:8090 --model multilingual --n 150 \\
        --out bench/results/multilingual_multilingual.jsonl

Suites, test split only (train files are never requested). Rows are a seeded
stratified sample (--seed, default 20260926), not a prefix of the file.
A language whose sample contains fewer than 2 classes aborts the run.

  amazon_massive_intent    mteb/amazon_massive_intent, Apache-2.0.
                           One config per language. Columns: id, text, label, label_text.
                           The id column is shared across languages (parallel corpus);
                           every language is scored on the same ids. Within each intent,
                           ids are sorted and shuffled with --seed, then drawn
                           round-robin across intents until n (an exhausted intent is
                           skipped). Choice "Which intent does `utterance` express?".
                           Options = sorted test-split intent names, '_' -> ' '.
                           The packaged test split has 59 of MASSIVE's 60 intents
                           (cooking_query is train-only). head_max_len 512.
  multilingual_sentiments  HF dataset tyqiangz/multilingual-sentiments (card license
                           Apache-2.0) ships only a loading script, no parquet.
                           datasets 5 cannot execute that script. The script's test
                           files are the per-language CSVs in
                           github.com/tyqiangz/multilingual-sentiment-datasets.
                           That repo's license is Apache-2.0 (GitHub API
                           GET /repos/tyqiangz/multilingual-sentiment-datasets,
                           license.spdx_id, checked 2026-09-26). Only
                           data/<lang>/test.csv is downloaded.
                           ClassLabel names on the HF card and in dataset_infos.json,
                           and the CSV `label` strings: ["positive", "neutral",
                           "negative"] (indices 0, 1, 2). Choice "What is the
                           sentiment of `text`?" with options negative / neutral /
                           positive. Sample: n/3 rows per class (when n is not
                           divisible by 3, the first classes in that option order
                           get one extra row).
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_accuracy import metrics, run_statim  # noqa: E402

MASSIVE_LANGS = ["de", "en", "fr", "es", "it", "tr", "pl", "ru", "ja", "zh-CN", "ar", "hi"]
# (report code, github directory). All twelve configs that have a test split.
SENTIMENT_LANGS = [
    ("ar", "arabic"), ("zh", "chinese"), ("en", "english"), ("fr", "french"),
    ("de", "german"), ("hi", "hindi"), ("id", "indonesian"), ("it", "italian"),
    ("ja", "japanese"), ("ms", "malay"), ("pt", "portuguese"), ("es", "spanish"),
]
# Dataset ClassLabel order. The choice list below is the order Statim sees.
SENTIMENT_CLASS_NAMES = ["positive", "neutral", "negative"]
SENTIMENT_OPTIONS = ["negative", "neutral", "positive"]
SUITES = ["amazon_massive_intent", "multilingual_sentiments"]
MODELS = ["english", "multilingual", "consensus", "banking77", "multitask"]
LANG_GROUPS = {
    "de": {"de", "german"}, "german": {"de", "german"},
    "en": {"en", "english"}, "english": {"en", "english"},
    "es": {"es", "spanish"}, "spanish": {"es", "spanish"},
    "fr": {"fr", "french"}, "french": {"fr", "french"},
    "it": {"it", "italian"}, "italian": {"it", "italian"},
    "ja": {"ja", "japanese"}, "japanese": {"ja", "japanese"},
    "zh": {"zh", "zh-cn", "chinese"}, "zh-cn": {"zh", "zh-cn", "chinese"},
    "chinese": {"zh", "zh-cn", "chinese"},
    "ar": {"ar", "arabic"}, "arabic": {"ar", "arabic"},
    "hi": {"hi", "hindi"}, "hindi": {"hi", "hindi"},
    "id": {"id", "indonesian"}, "indonesian": {"id", "indonesian"},
    "ms": {"ms", "malay"}, "malay": {"ms", "malay"},
    "pt": {"pt", "portuguese"}, "portuguese": {"pt", "portuguese"},
    "tr": {"tr"}, "pl": {"pl"}, "ru": {"ru"},
}
DISPLAY_ORDER = ["de", "en", "fr", "es", "it", "tr", "pl", "ru", "ja", "zh-CN", "zh", "ar", "hi", "id", "ms", "pt"]

_CACHE = {}


def _dataset():
    from datasets import load_dataset
    return load_dataset


def lang_wanted(selected, *names):
    if not selected:
        return True
    want = set()
    for raw in selected:
        key = raw.strip().lower()
        want |= LANG_GROUPS.get(key, {key})
    have = set()
    for name in names:
        key = name.lower()
        have |= LANG_GROUPS.get(key, {key})
        if name == "zh-CN":
            have |= LANG_GROUPS["zh-cn"]
    return bool(want & have)


def massive_test(lang):
    if ("massive", lang) not in _CACHE:
        url = "https://huggingface.co/datasets/mteb/amazon_massive_intent/resolve/main/test/%s.json.gz" % lang
        _CACHE[("massive", lang)] = _dataset()("json", data_files={"test": url}, split="test")
    return _CACHE[("massive", lang)]


def sentiment_test(config):
    if ("sentiment", config) not in _CACHE:
        url = "https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data/%s/test.csv" % config
        _CACHE[("sentiment", config)] = _dataset()("csv", data_files={"test": url}, split="test")
    return _CACHE[("sentiment", config)]


def _require_classes(suite, lang, gold, minimum=2):
    n_cls = len(set(gold))
    if n_cls < minimum:
        raise SystemExit("%s %s: stratified sample covers %d class(es), need at least %d" % (
            suite, lang, n_cls, minimum))


def _massive_by_id(lang):
    ds = massive_test(lang)
    ids = [str(i) for i in ds["id"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("amazon_massive_intent %s: duplicate id" % lang)
    if any(a != b for a, b in zip(ds["label"], ds["label_text"])):
        raise SystemExit("amazon_massive_intent %s: label != label_text" % lang)
    return {i: (lab, text) for i, lab, text in zip(ids, ds["label_text"], ds["text"])}


def stratified_massive_ids(rows, n, seed):
    """Round-robin over intents. Ids are sorted before the seeded shuffle, so the
    draw does not depend on file order or on which language supplied `rows`."""
    by_intent = {}
    for example_id, (lab, _text) in rows.items():
        by_intent.setdefault(lab, []).append(example_id)
    intents = sorted(by_intent)
    rng = random.Random(seed)
    for intent in intents:
        by_intent[intent].sort()
        rng.shuffle(by_intent[intent])
    picked = []
    while len(picked) < n:
        grew = False
        for intent in intents:
            bucket = by_intent[intent]
            if not bucket:
                continue
            picked.append(bucket.pop())
            grew = True
            if len(picked) >= n:
                break
        if not grew:
            break
    return picked


def massive_task(langs, n, seed):
    """One shared id list for every language. Option list = sorted test intents."""
    maps = {lang: _massive_by_id(lang) for lang in langs}
    ref_lang = langs[0]
    ref_labels = {i: lab for i, (lab, _text) in maps[ref_lang].items()}
    for lang, rows in maps.items():
        labels = {i: lab for i, (lab, _text) in rows.items()}
        if labels != ref_labels:
            raise SystemExit("amazon_massive_intent: id/label map of %s differs from %s" % (lang, ref_lang))
    example_ids = stratified_massive_ids(maps[ref_lang], n, seed)
    if len(example_ids) < n:
        raise SystemExit("amazon_massive_intent: test split has only %d rows, asked for %d" % (
            len(example_ids), n))
    keys = [name.replace("_", " ") for name in sorted(set(ref_labels.values()))]
    questions = {"intent": {"type": "choice",
                            "instructions": "Which intent does `utterance` express?",
                            "criteria": {k: None for k in keys}}}
    tasks = []
    for lang in langs:
        rows = maps[lang]
        states = [{"utterance": rows[i][1] if rows[i][1] is not None else ""} for i in example_ids]
        gold = [keys.index(rows[i][0].replace("_", " ")) for i in example_ids]
        _require_classes("amazon_massive_intent", lang, gold)
        tasks.append((lang, states, questions, "intent", keys, gold, example_ids))
    return tasks


def sentiment_quotas(n):
    """n/3 per class. A remainder goes to the first classes in option order."""
    base, rem = divmod(n, len(SENTIMENT_OPTIONS))
    return {name: base + (1 if i < rem else 0) for i, name in enumerate(SENTIMENT_OPTIONS)}


def stratified_sentiment_indices(labels, n, seed):
    quota = sentiment_quotas(n)
    buckets = {name: [] for name in SENTIMENT_OPTIONS}
    for idx, lab in enumerate(labels):
        buckets[lab].append(idx)
    rng = random.Random(seed)
    chosen = []
    for name in SENTIMENT_OPTIONS:
        need = quota[name]
        have = buckets[name]
        if len(have) < need:
            raise SystemExit("multilingual_sentiments: class %s has %d test rows, need %d" % (
                name, len(have), need))
        rng.shuffle(have)
        chosen.extend(have[:need])
    chosen.sort()
    return chosen


def sentiment_task(langs, n, seed):
    opt_index = {name: i for i, name in enumerate(SENTIMENT_OPTIONS)}
    questions = {"sentiment": {"type": "choice",
                               "instructions": "What is the sentiment of `text`?",
                               "criteria": {k: None for k in SENTIMENT_OPTIONS}}}
    tasks = []
    for code, config in SENTIMENT_LANGS:
        if code not in langs:
            continue
        ds = sentiment_test(config)
        labels = [str(x) for x in ds["label"]]
        if labels and not set(labels) <= set(SENTIMENT_OPTIONS):
            raise SystemExit("multilingual_sentiments %s: unexpected labels %s (expected %s)"
                             % (config, sorted(set(labels)), SENTIMENT_OPTIONS))
        missing = set(SENTIMENT_CLASS_NAMES) - set(labels)
        if missing:
            raise SystemExit("multilingual_sentiments %s: test split missing classes %s" % (config, sorted(missing)))
        texts = list(ds["text"])
        try:
            indices = stratified_sentiment_indices(labels, n, seed)
        except SystemExit as exc:
            raise SystemExit("multilingual_sentiments %s: %s" % (code, exc)) from exc
        states = [{"text": texts[i] if texts[i] is not None else ""} for i in indices]
        gold = [opt_index[labels[i]] for i in indices]
        _require_classes("multilingual_sentiments", code, gold)
        tasks.append((code, states, questions, "sentiment", SENTIMENT_OPTIONS, gold, None))
    return tasks


def macro(rows):
    n = len(rows)
    out = {k: round(sum(r[k] for r in rows) / n, 4) for k in ("accuracy", "ece", "nll", "brier")}
    out["seconds"] = round(sum(r["seconds"] for r in rows) / n, 1)
    return out


def print_table(records):
    suites_present = [s for s in SUITES if any(r["suite"] == s and r["lang"] != "macro" for r in records)]
    langs = []
    seen = set()
    for lang in DISPLAY_ORDER:
        if any(r["lang"] == lang for r in records):
            langs.append(lang)
            seen.add(lang)
    for r in records:
        if r["lang"] not in seen and r["lang"] != "macro":
            langs.append(r["lang"])
            seen.add(r["lang"])
    by = {(r["suite"], r["lang"]): r for r in records}
    header = "Sprache".ljust(10) + "".join(s.rjust(28) for s in suites_present)
    print("\n" + header)
    print("-" * len(header))

    def cell(suite, lang):
        row = by.get((suite, lang))
        return "—".rjust(28) if row is None else ("%0.4f" % row["accuracy"]).rjust(28)

    for lang in langs:
        print(lang.ljust(10) + "".join(cell(s, lang) for s in suites_present))
    print("Makro".ljust(10) + "".join(cell(s, "macro") for s in suites_present))
    print("(Zellen = accuracy, Makro = ungewichtet über Sprachen)")
    for s in suites_present:
        m = by.get((s, "macro"))
        if m:
            print("Makro %-24s  acc %0.4f  ece %0.4f  nll %0.4f  brier %0.4f  sec %s  (%d Sprachen)" % (
                s, m["accuracy"], m["ece"], m["nll"], m["brier"], m["seconds"], m["n_langs"]))


def main():
    ap = argparse.ArgumentParser(description="Multilingual Statim benchmark (test splits only)")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    ap.add_argument("--langs", nargs="+", default=None,
                    help="language codes or dataset names (default: every language of each suite)")
    ap.add_argument("--n", type=int, default=200, help="stratified sample size per language")
    ap.add_argument("--seed", type=int, default=20260926, help="shuffle seed for the stratified sample")
    ap.add_argument("--out", default=None, help="JSONL path, one record per suite/language plus a macro line")
    a = ap.parse_args()
    if a.langs:
        unknown = [t for t in a.langs if t.strip().lower() not in LANG_GROUPS and t.strip() not in ("zh-CN",)]
        if unknown:
            raise SystemExit("unknown language(s): %s" % ", ".join(unknown))
    if a.n < 1:
        raise SystemExit("--n must be >= 1")

    jobs = []
    if "amazon_massive_intent" in a.suites:
        langs = [lang for lang in MASSIVE_LANGS if lang_wanted(a.langs, lang)]
        if langs:
            for lang, states, questions, qid, keys, gold, example_ids in massive_task(langs, a.n, a.seed):
                jobs.append(("amazon_massive_intent", lang, states, questions, qid, keys, gold, 512, example_ids))
    if "multilingual_sentiments" in a.suites:
        langs = [code for code, _cfg in SENTIMENT_LANGS if lang_wanted(a.langs, code)]
        if langs:
            for lang, states, questions, qid, keys, gold, example_ids in sentiment_task(langs, a.n, a.seed):
                jobs.append(("multilingual_sentiments", lang, states, questions, qid, keys, gold, None, example_ids))
    if not jobs:
        raise SystemExit("nothing to run: no suite/language matched")

    out = None
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        out = open(a.out, "w", encoding="utf-8")
    records = []
    try:
        by_suite = {}
        for suite, lang, states, questions, qid, keys, gold, head_max_len, example_ids in jobs:
            print("run %s %s n=%d options=%d classes=%d" % (
                suite, lang, len(gold), len(keys), len(set(gold))), flush=True)
            probs, secs = run_statim(a.url, states, questions, qid, keys, 1,
                                     api_key=os.environ.get("STATIM_API_KEY"), model=a.model,
                                     head_max_len=head_max_len)
            row = {"suite": suite, "lang": lang, "model": a.model, "n": len(gold), "n_options": len(keys),
                   "n_gold_labels": len(set(gold)), "seed": a.seed, "head_max_len": head_max_len}
            if example_ids is not None:
                row["example_ids"] = example_ids
            if suite == "multilingual_sentiments":
                counts = Counter(SENTIMENT_OPTIONS[g] for g in gold)
                row["class_names"] = SENTIMENT_CLASS_NAMES
                row["options"] = SENTIMENT_OPTIONS
                row["gold_counts"] = {k: counts[k] for k in SENTIMENT_OPTIONS}
            row.update(metrics(probs, gold))
            row["seconds"] = round(secs, 1)
            records.append(row)
            by_suite.setdefault(suite, []).append(row)
            print(suite, lang, {k: row[k] for k in ("accuracy", "ece", "nll", "brier", "seconds")}, flush=True)
            if out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
        for suite, rows in by_suite.items():
            m = macro(rows)
            rec = {"suite": suite, "lang": "macro", "model": a.model, "n": sum(r["n"] for r in rows),
                   "n_langs": len(rows), **m}
            records.append(rec)
            if out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        if out:
            out.close()
    print_table(records)


if __name__ == "__main__":
    main()
