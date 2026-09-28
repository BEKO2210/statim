#!/usr/bin/env python3
"""Held-out accuracy per decision category, built from the mixture v6 sources.

    python bench/eval_categories.py --url http://127.0.0.1:8090 --model multilingual \\
        --out bench/results/categories_multilingual.jsonl

One suite per decision category. Every suite reads the test split, or another split the mixture
never loads, of sources listed in tools/finetune/sources/v6-keep.json. The split the mixture
trains on is never read: HELD_OUT names the exact split for each source and
test_eval_categories.py checks that it differs from the registry's training split.

Items come from the mixture v6 adapters (tools/finetune/mixture_v6/registry.py), run in a
canonical mode: the first instruction paraphrase in the item's language (the same fallback to
English as training), no option shuffle, and the options of a choice question sorted
alphabetically. The question wording therefore matches training, the order is fixed, and a
re-run sends the same requests. Integer ClassLabel columns are decoded to their names first, so
no option is a bare id such as "0", "1", "2". Three sources get a documented row
repair (PREP) where the adapter would otherwise emit a wrong gold label; sources that cannot be
repaired that way are listed in EXCLUDED with the reason.

Sample: per suite and language, a seeded stratified draw of --n items (default 150). Buckets are
(source, gold option); items in a bucket are sorted by text, shuffled with --seed, then drawn
round-robin over the sorted buckets. A language with fewer than --n pooled items, or where one
gold class holds more than 70 % of the draw, is skipped and listed in the output. Every pooled
item (not only the drawn ones) is part of the banned set in tools/finetune/mixture_v6/eval_texts.py,
so another --seed or --n can never pick a text training has seen.

Scoring: accuracy (argmax = gold), balanced accuracy (mean recall over gold options), ECE, NLL and
Brier as in bench/eval_accuracy.metrics. A noul question scores [1 - p(true), p(true)]; a score
question uses its level probabilities. One request carries one question, so items are grouped
by question.

--exclude-mixture PATH drops every pooled item that shares a text with the built mixture the model
was trained on (exact normalised match, or containment of 40+ characters in either direction), and
the mixture's hash becomes part of the pool fingerprint in every record. gate.py compares a
category cell only between two models with the same fingerprint.

--adapter NAME sends "adapter": NAME with every request (a LoRA adapter loaded with
`statim serve --adapter [model:]NAME=FILE`), checks that each answer was computed with it
(routing.adapter), and records it in every record ("adapter", and "model" as "<model>:<NAME>"), so a
base run and an adapter run of the same server stay apart. tools/finetune/lora_experiment.py compares
the two with gate.adapter_decision.

Cells in GATE_EXCLUDED are reported with "gate": false and a reason, and gate.py keeps them out of
the "categories" family: zero-shot cells (no training source in that language, or labels the
training never saw) and suites whose items the registry in use would bias.

The pool is cached in data/category-suites.jsonl.gz, keyed by a fingerprint of HELD_OUT, the pool
size, the adapter and loader source and v6-keep.json. A pool with a source that failed to load is
never cached. Pass --rebuild to fetch the splits again. --list prints the draw
without a server.

Not covered: sarcasm/humour (no v6 source has a split the mixture does not read) and the
'mixed' sentiment class (only in the train split of poem_sentiment). See EXCLUDED.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gzip
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

CACHE = ROOT / "data" / "category-suites.jsonl.gz"
POOL_ROWS = 3000        # rows read per source split, spread evenly over the split
DEFAULT_N = 150
DEFAULT_SEED = 20260927
# Request-side cap on the state. The server rejects a state above 50,000 characters and keeps only the
# first max_len tokens (1,024 for the released models); 40,000 characters is far beyond what that
# window can hold, so the cut never changes what the model sees. Pool and banned set keep the full text.
STATE_CHARS = 40_000
MAX_CLASS_SHARE = 0.7     # a sample where one gold class exceeds this is skipped (a constant answer scores it)
PARQUET_REV = "refs/convert/parquet"
MODELS = ["english", "multilingual", "consensus", "banking77", "multitask"]

CLAIMBUSTER_CROWD = {
    # Same Zenodo record as the training loader, which reads groundtruth.csv only.
    "url": "https://zenodo.org/api/records/3609356/files/crowdsourced.csv/content",
    "sha256": "11026f911d96476daed6c9a83ba0fbf3555598c04ef9b7fa567eeadee8693843",  # md5 matches Zenodo
    "filename": "crowdsourced.csv",
    "cache": "claimbuster",
}


@dataclass(frozen=True)
class Source:
    """One held-out split of one registry entry.

    entry / entry_config: the registry entry (adapter, languages, label fields).
    repo / configs / split: parquet files under refs/convert/parquet (repo defaults to entry),
        or `loader` for a file outside the Hub. Rows of several configs go through one adapter
        call, as in training, so a label vocabulary spans all of them.
    kinds: question types kept. tasks: adapter tasks kept (None = all).
    lang: item language for a config that is not the registry entry's own.
    prep: row transform in PREP (label repair for a known adapter defect, see its docstring).
    adapter: a registry adapter used instead of the entry's own (by function name).
    label_names: ((column, names), ...) for integer labels documented on the dataset card only.
    limit: rows read (spread over the split; split evenly over configs).
    template_fields: text_fields that hold a fixed prompt shared by train and test (CUAD's 41
        clause questions). They stay in the state but are not added to the banned set on their
        own: that would drop every training row that asks the same question."""
    entry: str
    entry_config: str
    split: str
    repo: str = ""
    configs: tuple = ("default",)
    kinds: tuple = ("choice",)
    tasks: tuple | None = None
    lang: str = ""
    loader: str = ""
    prep: str = ""
    adapter: str = ""
    label_names: tuple = ()
    limit: int = 0
    template_fields: tuple = ()

    @property
    def key(self):
        return "%s::%s::%s/%s" % (self.entry, self.entry_config, "+".join(self.configs), self.split)


def _brighter(cfg, lang):
    # Multi-label (a third to a half of the rows carry several emotions): single-label rows only,
    # so the gold is not just the alphabetically first emotion.
    return Source("brighter-dataset/BRIGHTER-emotion-categories", "hin", "test", configs=(cfg,), lang=lang,
                  prep="single_emotion")


# Suite -> held-out sources. The split is never the one the mixture reads for that entry
# (test_eval_categories.py checks this against v6-keep.json). Sources whose v6 adapter yields a
# constant or wrong gold label on these rows are left out and listed in EXCLUDED.
HELD_OUT = {
    "sentiment": [
        Source("google-research-datasets/poem_sentiment", "default", "test"),
        Source("google-research-datasets/poem_sentiment", "default", "validation"),
        Source("Kenshiii/synthetic-product-reviews", "default", "test", tasks=("sentiment",)),
        # Card: "label: 0 表示负面，1 表示正面" (0 negative, 1 positive); stored as a plain int.
        Source("YiMeng-SYSU/chinese-logic-sentiment-dataset", "default", "test",
               label_names=(("label", ("negative", "positive")),)),
    ],
    "emotion": [
        Source("JusteLeo/French-emotion", "default", "test"),
        # BRIGHTER: the mixture reads the hin and mar train splits only.
        *[_brighter(cfg, lang) for cfg, lang in (("hin", "hi"), ("deu", "de"), ("eng", "en"), ("esp", "es"),
                                                 ("rus", "ru"), ("chn", "zh"), ("ptbr", "pt"))],
    ],
    "complaint": [
        Source("cngchis/Support-Ticket-Router-12K-Cleaned", "default", "test"),
        Source("liri-uzh/cfpb-complaints-mini", "default", "test"),
    ],
    "nli": [
        Source("nyu-mll/multi_nli", "default", "validation_matched"),
        Source("boun-tabi/nli_tr", "multinli_tr", "validation_matched", configs=("multinli_tr",)),
        Source("takehika/wanli-ja-nli", "ja_only", "test", configs=("ja_only",)),
        Source("GoktugD/turkish-nli-constructed-1.5m", "default", "test"),
    ],
    "safety": [
        Source("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", "test", kinds=("noul",)),
        Source("mteb/toxic_conversations_50k", "default", "test", kinds=("noul",), prep="toxic_label"),
        Source("3nesdeniz/agentic-prompt-injection-5k", "default", "test", kinds=("noul",)),
        Source("3nesdeniz/turkish-conversation-prompt-injection", "default", "test", kinds=("noul",)),
    ],
    "reading": [
        Source("theatticusproject/cuad-qa", "default", "test", kinds=("noul",), template_fields=("question",)),
    ],
    "similarity": [
        Source("stjiris/IRIS_sts", "default", "test", kinds=("noul",)),
    ],
    "topic": [
        Source("NortheasternUniversity/big_patent", "a..y (one per CPC section)", "test",
               configs=tuple("abcdefghy"), lang="en", prep="cpc_section", limit=1800),
    ],
    "intent": [
        Source("GoktugD/turkish-intent-classification-1m", "default", "test"),
        Source("clips/VaccinChatNL", "default", "test"),
        Source("masakhane/InjongoIntent", "eng + 16 African configs", "test", configs=("eng",), lang="en"),
    ],
    "stance": [
        Source("pacoreyes/StanceSentences", "default", "test"),
    ],
    "formality": [
        Source("GoktugD/turkish-formality-rewrite-500k", "default", "test", kinds=("noul",)),
        Source("NagaYu/deference-keigo-corpus", "default", "test", kinds=("noul",)),
    ],
    "urgency": [
        Source("IDinsight/urgency_detection_maternal_health_synthetic", "default", "validation"),
    ],
    "fact_check": [
        # ClaimBuster: the mixture reads groundtruth.csv; crowdsourced.csv is never loaded.
        Source("zenodo:3609356 (ClaimBuster)", "crowdsourced/groundtruth", "crowdsourced.csv", loader="claimbuster"),
    ],
    "pii": [
        Source("gretelai/synthetic_pii_finance_multilingual", "default", "test", kinds=("noul",),
               prep="pii_spans", adapter="pii_adapter"),
        Source("Wismut/nym-pii-multilingual-data", "default", "test", kinds=("noul",),
               prep="pii_spans", adapter="pii_adapter"),
    ],
}
SUITES = list(HELD_OUT)

# Held-out splits that exist but are not used, and why. Kept here so the choice is reviewable.
EXCLUDED = {
    "sarcasm": "every v6 sarcasm/humour source ships only a train split (or one file) and the mixture reads all of it",
    "mteb/toxic_conversations_50k (as is)": "safety_adapter does not treat 'not toxic' as negative: every row becomes "
                                           "true; used with prep=toxic_label",
    "allenai/prosocial-dialog test": "safety_adapter maps every safety_label (incl. __casual__) to true",
    "nvidia/Aegis-AI-Content-Safety-Dataset-1.0 test": "safety_adapter yields true for every row",
    "tasksource/esci test": "similarity_adapter maps the E/S/C/I labels to score 0 for every row",
    "napsternxg/wands test": "similarity_adapter maps Exact/Partial/Irrelevant to score 0 for every row (as ClassLabel "
                             "ints it reads 1 as 'same meaning' and 2 as 'moderately related')",
    "nyu-mll/quality validation": "the adapter reads 'article' but the loader writes 'context': the passage is dropped",
    "theatticusproject/maud test": "reading_adapter has no options for MAUD rows: every item is a constant false",
    "WorkInTheDark/FairytaleQA test": "every row has an answer: the answerability item is a constant true",
    "joelniklaus/covid19_emergency_event test": "fewer than 150 items per language; labels are event1..event8",
    "joelniklaus/german_argument_mining test": "labels are argument roles, not stance toward a target",
    "nvidia/Nemotron-PII, gretelai/gretel-pii-masking-en-v1 test": "every row contains PII (constant true)",
    "RichardSakaguchiMS/brazilian-customer-service-conversations test": "95 rows, below --n for pt",
    "leonvanbokhorst/synthetic-complaints-v2 test": "classification_adapter mixes topic, style and sentiment "
                                                   "values in one option list",
    "sentiment 'mixed'": "only poem_sentiment has a mixed class, and only in its train split",
    "hblim/customer-complaints, tasksource/it-support-tickets test": "v6 trains these on ClassLabel ids "
                                                                     "('0', '1', ...); the suite would be zero-shot",
}

# Cells that are reported but kept out of the gate's "categories" family, and why. A value is a
# reason, or a callable that returns one (or None) for the registry in use.
def _reading_windows(reg):
    if not hasattr(reg, "WINDOW"):
        return ("this v6 registry cuts answerable contexts around the answer and unanswerable ones from "
                "the start (1,500 chars): the length gives the label away")
    return None


def _topic_labels(reg):
    if not hasattr(reg, "CPC_SECTIONS"):
        return "this v6 registry trains big_patent on the section letters a..y; the suite asks for section titles"
    return None


def _urgency_labels(reg):
    if not hasattr(reg, "_special_field_labels"):
        return "this v6 registry asks which of 35 warning signs applies, not whether the message is urgent"
    return None


GATE_EXCLUDED = {
    ("reading", "*"): _reading_windows,
    ("topic", "*"): _topic_labels,
    ("urgency", "*"): _urgency_labels,
}


def gate_note(suite, lang):
    """Why a cell stays out of the gate family, or None."""
    for key in ((suite, lang), (suite, "*")):
        rule = GATE_EXCLUDED.get(key)
        if rule is None:
            continue
        reason = rule(_registry()) if callable(rule) else rule
        if reason:
            return reason
    return None

# --------------------------------------------------------------------------- registry glue

def _registry():
    from tools.finetune.mixture_v6 import registry
    return registry


def entry_for(src):
    reg = _registry()
    for entry in reg.ENTRIES:
        if entry["id"] == src.entry and entry.get("config", "default") == src.entry_config:
            return entry
    raise KeyError("no enabled registry entry %s / %s" % (src.entry, src.entry_config))


class _FirstParaphrase:
    """Stand-in RNG for templates.instruction: item language (random() < 0.70) and bank[0]."""

    def random(self):
        return 0.0

    def choice(self, seq):
        return seq[0]


@contextlib.contextmanager
def canonical():
    """Run the registry adapters with the first paraphrase, no shuffle, and the task on each item."""
    reg = _registry()
    saved = {name: getattr(reg, name) for name in ("instruction", "shuffle_choice", "_render", "_base")}
    last_task = [None]

    def instruction(kind, item_lang, _rng, category=None, **fmt):
        return saved["instruction"](kind, item_lang, _FirstParaphrase(), category, **fmt)

    def shuffle_choice(options, gold, _rng):
        return list(options), [1.0 if i == gold else 0.0 for i in range(len(options))]

    def _render(kind, lang, rng, task, fmt=None):
        last_task[0] = task
        return saved["_render"](kind, lang, rng, task, fmt)

    def _base(*args, **kwargs):
        item = saved["_base"](*args, **kwargs)
        item["_task"] = last_task[0]
        return item

    reg.instruction, reg.shuffle_choice, reg._render, reg._base = instruction, shuffle_choice, _render, _base
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(reg, name, value)


def sort_options(item):
    """Choice options in casefolded alphabetical order; the one-hot target follows."""
    q = item["q"]
    if q["type"] != "choice":
        return item
    keys = list(q["criteria"])
    order = sorted(range(len(keys)), key=lambda i: (keys[i].casefold(), keys[i]))
    q["criteria"] = {keys[i]: q["criteria"][keys[i]] for i in order}
    item["target"] = [item["target"][i] for i in order]
    return item


def gold_index(item):
    return item["target"].index(1.0)


def option_names(item):
    q = item["q"]
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def question_id(q):
    return hashlib.sha256(json.dumps(q, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- loading

def _norm(text):
    return " ".join(str(text).split()).lower()


def parquet_files(repo, config, split):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=PARQUET_REV)
    prefix = "%s/%s/" % (config, split)
    out = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not out:
        raise RuntimeError("%s: no parquet files for %s/%s" % (repo, config, split))
    return out


def spread(rows, limit):
    if len(rows) <= limit:
        return rows
    step = len(rows) / float(limit)
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(limit)]


def _class_labels(path):
    """{column: [names]} for ClassLabel columns, from the datasets metadata in the parquet schema."""
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata or {}
    try:
        features = json.loads(meta[b"huggingface"])["info"]["features"]
    except (KeyError, ValueError, TypeError):
        return {}
    out = {}
    for col, feat in features.items():
        if isinstance(feat, dict) and feat.get("_type") == "ClassLabel" and feat.get("names"):
            out[col] = list(feat["names"])
    return out


def decode_class_labels(rows, names):
    """Replace ClassLabel ints by their names. The v6 adapters would otherwise offer "0", "1", "2" as options."""
    for row in rows:
        for col, labels in names.items():
            value = row.get(col)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(labels):
                row[col] = labels[value]
    return rows


def _load_parquet(src, limit):
    """Rows of every config in src.configs, `limit` per config, spread evenly over each split."""
    from huggingface_hub import hf_hub_download
    from tools.finetune.mixture_v6.loaders import sample_parquet
    import pyarrow.parquet as pq
    repo = src.repo or src.entry
    rows = []
    per_config = max(1, limit // len(src.configs))
    for config in src.configs:
        paths = [hf_hub_download(repo, f, repo_type="dataset", revision=PARQUET_REV)
                 for f in parquet_files(repo, config, src.split)]
        sizes = [pq.ParquetFile(p).metadata.num_rows or 0 for p in paths]
        total = sum(sizes)
        for path, size in zip(paths, sizes):
            want = max(1, round(per_config * size / total)) if total else 0
            part = decode_class_labels(sample_parquet(path, want), _class_labels(path))
            for row in part:
                row.setdefault("_v6_config", config)
            rows.extend(part)
    return rows


def _load_claimbuster(_src, limit):
    from tools.finetune.mixture_v6.loaders import _csv_rows, pinned_path
    return spread(_csv_rows(pinned_path(CLAIMBUSTER_CROWD), 10 ** 9), limit)


LOADERS = {"claimbuster": _load_claimbuster}

CPC_SECTIONS = {
    "a": "human necessities", "b": "performing operations; transporting", "c": "chemistry; metallurgy",
    "d": "textiles; paper", "e": "fixed constructions",
    "f": "mechanical engineering; lighting; heating; weapons; blasting", "g": "physics", "h": "electricity",
    "y": "general tagging of new technological developments",
}


def _prep_toxic_label(rows):
    """safety_adapter counts 'not toxic' as a violation; give it the negative word it knows."""
    for row in rows:
        if str(row.get("label_text", "")).strip().lower() == "not toxic":
            row["label_text"] = "safe"
    return rows


def _prep_cpc_section(rows):
    """big_patent labels are the config letters; the adapter reads _v6_config. Use the CPC section title,
    unless the registry already maps the letters itself (it does since the v6 adapter fixes)."""
    if hasattr(_registry(), "CPC_SECTIONS"):
        return rows
    for row in rows:
        row["_v6_config"] = CPC_SECTIONS[row["_v6_config"]]
    return rows


def _prep_pii_spans(rows):
    """Span-annotated rows in the shape pii_adapter reads (tokenized_text + ner); the entry's own
    classification adapter takes the whole span list as one label."""
    out = []
    for row in rows:
        text = row.get("generated_text") or row.get("text") or ""
        spans = row.get("pii_spans") if "pii_spans" in row else row.get("entities")
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except ValueError:
                continue
        if not isinstance(text, str) or not text.strip() or not isinstance(spans, list):
            continue
        kinds = [str(s.get("label")) for s in spans if isinstance(s, dict) and s.get("label")]
        new = {"tokenized_text": text.split(), "ner": [[0, 0, k] for k in kinds]}
        if row.get("language"):
            new["language"] = row["language"]
        out.append(new)
    return out


def _prep_single_emotion(rows):
    """BRIGHTER rows with exactly one emotion (the `emotions` list, or the 0/1 columns)."""
    names = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
    out = []
    for row in rows:
        flags = [n for n in names if str(row.get(n, "")).strip() in ("1", "1.0", "True", "true")]
        if len(flags) == 1:
            row["emotions"] = flags
            out.append(row)
    return out


PREP = {"toxic_label": _prep_toxic_label, "cpc_section": _prep_cpc_section, "pii_spans": _prep_pii_spans,
        "single_emotion": _prep_single_emotion}


def load_source_rows(src, limit=None):
    limit = limit or src.limit or POOL_ROWS
    rows = (LOADERS[src.loader] if src.loader else _load_parquet)(src, limit)
    for col, names in src.label_names:
        decode_class_labels(rows, {col: list(names)})
    if src.lang:
        for row in rows:
            row["_v6_lang"] = src.lang
    if src.prep:
        rows = PREP[src.prep](rows)
    return rows


def items_from_rows(src, rows, seed=DEFAULT_SEED):
    """Adapter output in canonical form, filtered to src.kinds / src.tasks, one item per question+state."""
    from tools.finetune.mixture_v6.build import valid_item
    from tools.finetune.mixture_v6.registry import adapt
    reg = _registry()
    entry = entry_for(src)
    with canonical():
        if src.adapter:
            raw = list(getattr(reg, src.adapter)(entry, list(rows), seed))
        else:
            raw = list(adapt(entry, rows, seed))
    templates = {str(row.get(f)).strip() for row in rows for f in src.template_fields if row.get(f)}
    out, seen = [], set()
    for item in raw:
        if not item or not valid_item(item) or item["q"]["type"] not in src.kinds:
            continue
        if src.tasks is not None and item.get("_task") not in src.tasks:
            continue
        item = sort_options(item)
        key = (_norm(item["state"]), question_id(item["q"]))
        if key in seen:
            continue
        seen.add(key)
        item["source"] = src.key
        item["_texts"] = [t for t in item.get("_texts") or [] if isinstance(t, str) and t not in templates]
        out.append(item)
    return out


def fingerprint():
    spec = {s: [asdict(src) for src in srcs] for s, srcs in HELD_OUT.items()}
    code = ROOT / "tools" / "finetune" / "mixture_v6"
    adapters = {name: hashlib.sha256((code / name).read_bytes()).hexdigest()
                for name in ("registry.py", "templates.py", "loaders.py")}
    adapters["v6-keep.json"] = hashlib.sha256(
        (ROOT / "tools" / "finetune" / "sources" / "v6-keep.json").read_bytes()).hexdigest()
    import inspect
    here = [canonical, sort_options, _class_labels, decode_class_labels, _load_parquet, load_source_rows,
            items_from_rows, *LOADERS.values(), *PREP.values()]
    adapters["eval_categories"] = hashlib.sha256("".join(inspect.getsource(f) for f in here).encode()).hexdigest()
    raw = json.dumps({"spec": spec, "pool_rows": POOL_ROWS, "cpc": CPC_SECTIONS, "adapters": adapters, "version": 1},
                     sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_pool(suites=None, log=print):
    pool, failures = [], {}
    for suite, srcs in HELD_OUT.items():
        if suites and suite not in suites:
            continue
        for src in srcs:
            try:
                items = items_from_rows(src, load_source_rows(src))
            except Exception as exc:  # a missing source must not hide the others
                failures[src.key] = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0] if str(exc) else "")
                log("pool %-10s %s FAILED %s" % (suite, src.key, failures[src.key]))
                continue
            for item in items:
                item["suite"] = suite
            pool.extend(items)
            log("pool %-10s %s: %d items" % (suite, src.key, len(items)))
    return pool, failures


def load_pool(cache=CACHE, rebuild=False, log=print):
    """Full pool (every suite). Cached; the header line carries the fingerprint (spec, pool size and
    the adapter source), so a change to any of them rebuilds the pool."""
    cache = Path(cache)
    fp = fingerprint()
    if cache.exists() and not rebuild:
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            if header.get("fingerprint") == fp:
                return [json.loads(line) for line in fh], header.get("failures", {})
    pool, failures = build_pool(log=log)
    if failures:
        return pool, failures  # never cache an incomplete pool: the next run retries the sources
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"fingerprint": fp, "failures": failures, "items": len(pool)}) + "\n")
        for item in pool:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(cache)
    return pool, failures


def suite_texts(pool):
    """Normalised texts for the eval banned set: every state and every source text of every pooled item."""
    out = set()
    for item in pool:
        out.add(_norm(item["state"]))
        for text in item.get("_texts") or []:
            out.add(_norm(text))
    out.discard("")
    return out


# --------------------------------------------------------------------------- mixture exclusion

CONTAIN_CHARS = 40   # shorter texts are compared exactly only
WORD_SHINGLE = 8     # words per anchor for space-separated text
CHAR_SHINGLE = 20    # characters per anchor for text without spaces (Chinese, Japanese)
INDEX_CHARS = 4000   # of a pool text, the part indexed for "training text inside a pool text"


def _units(text):
    """Word units for space-separated text, characters otherwise (CJK)."""
    words = text.split()
    return (words, WORD_SHINGLE, " ") if len(words) >= WORD_SHINGLE else (list(text), CHAR_SHINGLE, "")


def _anchors(text):
    units, k, sep = _units(text)
    return [sep.join(units[i:i + k]) for i in range(len(units) - k + 1)]


def mixture_states(path):
    """Normalised states of a built mixture (jsonl.gz, one item per line)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            state = json.loads(line).get("state")
            if not isinstance(state, str):
                state = json.dumps(state, ensure_ascii=False, sort_keys=True)
            yield _norm(state)


def contaminated(pool, states):
    """Indices of pool items that share a text with the mixture: an exact normalised match of the
    state or any source text, or containment of a text of at least CONTAIN_CHARS characters in either
    direction (a pool text inside a training state, or a training state inside a pool text).
    Containment is found through anchors (the first WORD_SHINGLE words or CHAR_SHINGLE characters)
    and then checked on the full strings."""
    texts = []  # (item index, normalised text)
    for i, item in enumerate(pool):
        for text in [item["state"], *(item.get("_texts") or [])]:
            t = _norm(text)
            if t:
                texts.append((i, t))
    by_item = collections.defaultdict(list)
    for i, t in texts:
        by_item[i].append(t)
    exact = collections.defaultdict(set)
    first = collections.defaultdict(list)   # anchor -> pool texts that start with it
    inner = collections.defaultdict(set)    # every anchor of the start of a pool text -> item indices
    for i, t in texts:
        exact[t].add(i)
        if len(t) >= CONTAIN_CHARS:
            anchors = _anchors(t)
            if anchors:
                first[anchors[0]].append((i, t))
            for a in _anchors(t[:INDEX_CHARS]):
                inner[a].add(i)
    hit = set()
    for state in states:
        hit |= exact.get(state, set())
        if len(state) < CONTAIN_CHARS:
            continue
        anchors = _anchors(state)
        for a in anchors:  # a pool text inside this training state
            for i, t in first.get(a, ()):
                if i not in hit and t in state:
                    hit.add(i)
        if anchors:        # this training state inside a pool text
            for i in inner.get(anchors[0], ()):
                if i not in hit and any(state in t for t in by_item[i]):
                    hit.add(i)
    return hit


def exclude_mixture(pool, path, log=print):
    """Pool without the items whose text occurs in the mixture at `path`, and a fingerprint of that file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    hit = contaminated(pool, mixture_states(path))
    by_suite = collections.Counter(pool[i]["suite"] for i in hit)
    log("mixture %s: %d pooled items share a text with it %s" % (path, len(hit), dict(sorted(by_suite.items()))))
    return [item for i, item in enumerate(pool) if i not in hit], digest.hexdigest()[:16]


# --------------------------------------------------------------------------- sampling

def stratified(items, n, seed):
    """Round-robin over sorted (source, gold option) buckets; each bucket sorted, then shuffled with seed."""
    buckets = collections.defaultdict(list)
    for item in items:
        buckets[(item["source"], option_names(item)[gold_index(item)])].append(item)
    rng = random.Random(seed)
    keys = sorted(buckets)
    for key in keys:
        buckets[key].sort(key=lambda it: (_norm(it["state"]), question_id(it["q"])))
        rng.shuffle(buckets[key])
    picked = []
    while len(picked) < n:
        grew = False
        for key in keys:
            if buckets[key]:
                picked.append(buckets[key].pop())
                grew = True
                if len(picked) == n:
                    break
        if not grew:
            break
    return picked


def make_tasks(pool, suites, langs, n, seed):
    """[(suite, lang, items)] plus [(suite, lang, reason)] for skipped languages."""
    by = collections.defaultdict(list)
    for item in pool:
        if item["suite"] in suites:
            by[(item["suite"], item["lang"])].append(item)
    tasks, skipped = [], []
    for suite in suites:
        for lang in sorted(l for s, l in by if s == suite):
            if langs and lang not in langs:
                continue
            items = by[(suite, lang)]
            if len(items) < n:
                skipped.append((suite, lang, "only %d pooled items, need %d" % (len(items), n)))
                continue
            picked = stratified(items, n, seed)
            counts = collections.Counter(option_names(it)[gold_index(it)] for it in picked)
            if len(counts) < 2 or max(counts.values()) > MAX_CLASS_SHARE * len(picked):
                skipped.append((suite, lang, "one gold class holds %d of %d sampled items" % (
                    max(counts.values()), len(picked))))
                continue
            tasks.append((suite, lang, picked))
    return tasks, skipped


# --------------------------------------------------------------------------- HTTP

def probabilities(answer, item):
    q = item["q"]
    if q["type"] == "noul":
        p = float(answer["noul"])
        return [1.0 - p, p]
    pr = answer["probabilities"]
    return [float(pr[k]) for k in option_names(item)]


def request_body(states, questions, model=None, head_max_len=None, adapter=None):
    """Body of one /v1/systemone/batch request. `adapter` names a LoRA adapter the server loaded
    (statim serve --adapter); without it the base weights answer and the body has no "adapter" key."""
    body = {"states": states, "questions": questions, "ensemble": 1}
    if model:
        body["model"] = model
    if head_max_len:
        body["head_max_len"] = head_max_len
    if adapter:
        body["adapter"] = adapter
    return body


def run_items(url, items, model=None, api_key=None, batch=16, head_max_len=None, adapter=None):
    """Probabilities per item, in input order. Items are grouped by question (one request = one question).
    With a named adapter every answer must report it in routing.adapter: a server without LoRA support
    ignores the unknown field and would silently answer with the base weights."""
    groups = collections.OrderedDict()
    for i, item in enumerate(items):
        groups.setdefault(question_id(item["q"]), []).append(i)
    probs = [None] * len(items)
    for idxs in groups.values():
        q = items[idxs[0]]["q"]
        questions = {"q": q}
        for start in range(0, len(idxs), batch):
            chunk = idxs[start:start + batch]
            body = request_body([items[i]["state"][:STATE_CHARS] for i in chunk], questions, model=model,
                                head_max_len=head_max_len, adapter=adapter)
            req = urllib.request.Request(url + "/v1/systemone/batch", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            if api_key:
                req.add_header("Authorization", "Bearer " + api_key)
            res = json.load(urllib.request.urlopen(req, timeout=600))
            for i, r in zip(chunk, res["results"]):
                routed = (r.get("routing") or {}).get("adapter")
                if adapter and adapter not in ("auto", "none") and routed != adapter:
                    raise SystemExit("the server did not apply adapter %r (routing %r): it needs a statim build with "
                                     "LoRA support started with --adapter [model:]%s=FILE" % (adapter, r.get("routing"), adapter))
                probs[i] = probabilities(r["answers"]["q"], items[i])
    return probs


def score(items, probs):
    from eval_accuracy import metrics
    # metrics() needs one width per call; group by option count and pool the sums.
    n = len(items)
    total = {"accuracy": 0.0, "ece": 0.0, "nll": 0.0, "brier": 0.0}
    by_width = collections.defaultdict(list)
    for item, p in zip(items, probs):
        by_width[len(p)].append((p, gold_index(item)))
    ece_probs, ece_gold = [], []
    for rows in by_width.values():
        m = metrics([p for p, _ in rows], [g for _, g in rows])
        for k in ("accuracy", "nll", "brier"):
            total[k] += m[k] * len(rows)
        ece_probs += [p for p, _ in rows]
        ece_gold += [g for _, g in rows]
    out = {k: round(v / n, 4) for k, v in total.items() if k != "ece"}
    # ECE only needs max-prob and correctness, which is width independent.
    out["ece"] = metrics(ece_probs, ece_gold)["ece"]
    # Balanced accuracy: mean recall over gold options, so a skewed sample cannot reward a constant answer.
    hits = collections.defaultdict(list)
    for item, p in zip(items, probs):
        g = gold_index(item)
        hits[option_names(item)[g]].append(max(range(len(p)), key=p.__getitem__) == g)
    out["balanced_accuracy"] = round(sum(sum(v) / len(v) for v in hits.values()) / len(hits), 4)
    return out


# --------------------------------------------------------------------------- CLI

def print_table(records):
    rows = [r for r in records if r["lang"] != "macro"]
    langs = sorted({r["lang"] for r in rows})
    suites = [s for s in SUITES if any(r["suite"] == s for r in rows)]
    by = {(r["suite"], r["lang"]): r for r in records}
    header = "Kategorie".ljust(12) + "".join(l.rjust(7) for l in langs) + "Makro".rjust(8)
    print("\n" + header)
    print("-" * len(header))
    for s in suites:
        cells = "".join(("%.3f" % by[(s, l)]["accuracy"]).rjust(7) if (s, l) in by else "—".rjust(7) for l in langs)
        m = by.get((s, "macro"))
        print(s.ljust(12) + cells + (("%.3f" % m["accuracy"]).rjust(8) if m else ""))
    print("(Zellen = accuracy, Makro = ungewichtet über Sprachen)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Held-out Statim benchmark per decision category")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="multilingual", choices=MODELS,
                    help="model name sent with each request (a single-model server answers with its model)")
    ap.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    ap.add_argument("--langs", nargs="+", default=None, help="ISO codes (default: every language with enough items)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="stratified sample size per suite and language")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--head-max-len", type=int, default=512)
    ap.add_argument("--out", default=None, help="JSONL, one record per suite/language plus one macro line per suite")
    ap.add_argument("--rebuild", action="store_true", help="fetch the held-out splits again")
    ap.add_argument("--exclude-mixture", default=None, metavar="PATH",
                    help="built mixture (jsonl.gz) the model was trained on: drop pooled items that share a text with it")
    ap.add_argument("--list", action="store_true", help="print the sampled suites and exit (no server needed)")
    ap.add_argument("--adapter", default=None, metavar="NAME",
                    help='LoRA adapter sent as "adapter" with every request (statim serve --adapter [model:]NAME=FILE); '
                         "recorded in each record; default: the base weights")
    a = ap.parse_args(argv)
    if a.n < 1:
        raise SystemExit("--n must be >= 1")

    pool, failures = load_pool(rebuild=a.rebuild, log=lambda m: print(m, flush=True))
    pool_fp = fingerprint()
    if a.exclude_mixture:
        pool, mixture_fp = exclude_mixture(pool, a.exclude_mixture, log=lambda m: print(m, flush=True))
        pool_fp = "%s+%s" % (pool_fp, mixture_fp)
    for key, why in failures.items():
        print("source unavailable: %s (%s)" % (key, why), flush=True)
    tasks, skipped = make_tasks(pool, a.suites, set(a.langs) if a.langs else None, a.n, a.seed)
    for suite, lang, why in skipped:
        print("skip %s %s: %s" % (suite, lang, why), flush=True)
    if not tasks:
        raise SystemExit("nothing to run: no suite/language has enough held-out items")
    if a.list:
        for suite, lang, items in tasks:
            gold = collections.Counter(option_names(it)[gold_index(it)] for it in items)
            print(suite, lang, len(items), dict(sorted(gold.items())))
        return 0

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    records, by_suite = [], collections.defaultdict(list)
    label = "%s:%s" % (a.model, a.adapter) if a.adapter else a.model  # base and adapter runs stay apart
    extra = {"adapter": a.adapter} if a.adapter else {}
    try:
        for suite, lang, items in tasks:
            t0 = time.time()
            probs = run_items(a.url, items, model=a.model, api_key=os.environ.get("STATIM_API_KEY"),
                              head_max_len=a.head_max_len, adapter=a.adapter)
            gold = collections.Counter(option_names(it)[gold_index(it)] for it in items)
            row = {"family": "categories", "suite": suite, "lang": lang, "model": label, **extra, "n": len(items), "seed": a.seed,
                   "pool": pool_fp,
                   "sources": dict(collections.Counter(it["source"] for it in items)),
                   "question_types": dict(collections.Counter(it["q"]["type"] for it in items)),
                   "gold_counts": dict(sorted(gold.items())), **score(items, probs),
                   "seconds": round(time.time() - t0, 1)}
            note = gate_note(suite, lang)
            row["gate"] = note is None
            if note:
                row["gate_note"] = note
            records.append(row)
            by_suite[suite].append(row)
            print(suite, lang, {k: row[k] for k in ("accuracy", "balanced_accuracy", "ece", "nll", "brier", "seconds")},
                  "" if row["gate"] else "(not in the gate family: %s)" % row["gate_note"], flush=True)
            if out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
        for suite, rows in by_suite.items():
            m = {k: round(sum(r[k] for r in rows) / len(rows), 4)
                 for k in ("accuracy", "balanced_accuracy", "ece", "nll", "brier")}
            rec = {"family": "categories", "suite": suite, "lang": "macro", "model": label, **extra, "n": sum(r["n"] for r in rows),
                   "n_langs": len(rows), **m}
            records.append(rec)
            if out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if out:
            for suite, lang, why in skipped:
                out.write(json.dumps({"family": "categories", "suite": suite, "lang": lang, "skipped": why},
                                     ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()
    print_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
