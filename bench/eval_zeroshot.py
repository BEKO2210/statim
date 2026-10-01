#!/usr/bin/env python3
"""Zero-shot generalisation of the base Statim checkpoints on public datasets
that are not in our training mixture.

Reuses metrics() and run_statim() from eval_accuracy.py (not copied).

    python bench/eval_zeroshot.py --url http://127.0.0.1:8090 --model multilingual --n 150 \\
        --out bench/results/zeroshot_multilingual.jsonl

Held-out rule. A dataset qualifies only when it is absent from our training
data. Training data is Banking77, MASSIVE, LocalLLaMA/typed-decisions, and
every "per_source" key of data/mixture-v1.manifest.json and
data/mixture-v2.manifest.json (tasksource names). main() reloads both
manifests and aborts if a suite needle matches a key. Obvious derivatives
were rejected the same way, not loaded:

  CLINC150          manifest key clinc_oos/plus (v1 and v2)
  xSID              translation of SNIPS; manifest key snips_built_in_intents
  mtop              multilingual/mtop (v1)
  XNLI              tasksource source multilingual/xnli; MNLI is glue/mnli
                    and gen_debiased_nli/mnli_* (v1)
  OCNLI             multilingual/clue/ocnli
  PAWS / PAWS-X     paws/labeled_* and multilingual/paws-x/*
  SciTail           scitail/snli_format
  SICK              sick/label and sick/relatedness (the relatedness score too)
  STS Benchmark     glue/stsb and sts-companion
  SST-5             same Stanford Sentiment Treebank sentences as glue/sst2
  Amazon star ratings  amazon_polarity and multilingual/amazon_reviews_multi
  Yelp stars        yelp_review_full
  IndoNLI translate_train  MNLI translated to Indonesian; not loaded
  Belebele assembled train (RACE, SciQ, MultiRC, ReClor, ...)  several of
                    those sources are in the mixture; only the Belebele
                    test configs are loaded

Upstream scan, 2026-09-27. Unique `source` values of
tasksource/tasksource-jev-typed-decisions (local snapshot
071f0cf2201f06b2dea3e55323aa3e264284e139, 2,530,000 rows, 667 sources)
were listed from the parquet `source` column. Seven suites below match
no source. go_emotions does: the upstream name is go_emotions/simplified.
Both mixture manifests omit it because tools/finetune/build_mixture.py
drops every emotion source. The public Laya card names DAIR Emotion as
held out and does not say whether the base checkpoints saw
go_emotions/simplified. It is included because it passes the manifest
rule the task defines; the upstream hit is recorded here so the number
is not read as a pure holdout of the unfiltered tasksource corpus.

Rows are a seeded stratified sample (--seed, default 20260927): each
class is shuffled, then drawn round-robin, so a rare class is not
dropped while a frequent one fills the sample. The draw stops early
only when the eligible split itself has fewer than --n rows, which
aborts. Train splits are never requested. A sample with fewer than
two gold classes aborts. Questions with more than 10 options send
head_max_len 512.

Suites (evaluation only; ShareAlike and unstated licences are not used
for training by this script):

  go_emotions       google-research-datasets/go_emotions, config
                    simplified, split test. HF card licence apache-2.0
                    (README YAML, checked 2026-09-27). Rows whose
                    `labels` list has exactly one id; the other test
                    rows are multi-label and dropped. 28 ClassLabel
                    names, short glosses as descriptions.
                    Choice "Which emotion is most strongly expressed
                    in `text`?". head_max_len 512. Language en.
  multi_hatecheck   mteb/multi-hatecheck test files
                    test/<lang>.jsonl.gz (the card's per-config paths
                    under multi-hatecheck/test/ do not exist on the
                    Hub). HF card licence cc-by-4.0. Official repo
                    rewire-online/multilingual-hatecheck licence
                    CC-BY-4.0 (GitHub API license.spdx_id, 2026-09-27).
                    Labels hateful / non-hateful. The dataset is a test
                    suite: there is no train split. Choice "Is `text`
                    hateful or non-hateful?". Languages en de fr es it
                    nl pl pt zh ar hi (files eng deu fra spa ita nld
                    pol por cmn ara hin).
  sib200            Davlan/sib200 test split only (204 rows, 7 topics).
                    HF card YAML licence cc-by-sa-4.0; the card text
                    also says "CC 4.0 Commercial". Code repo
                    dadelani/sib-200 is Apache-2.0 (GitHub API,
                    2026-09-27). Sentences are FLORES-200. Evaluation
                    only. Choice "What is the topic of `text`?".
                    Languages en de ar hi (configs eng_Latn, deu_Latn,
                    arb_Arab, hin_Deva). Not MasakhaNews and not
                    DBPedia-14.
  hwu64             DeepPavlov/hwu64 split test (1,076 rows, 64
                    intents). Intent names from the `intents` config.
                    The HF card states no licence. Upstream corpus
                    xliuhw/NLU-Evaluation-Data is CC-BY-4.0 (GitHub API
                    and the repo README, 2026-09-27). Not SNIPS, CLINC,
                    MASSIVE, Banking77 or mtop, but MASSIVE descends from
                    it: test rows occurring verbatim in MASSIVE are
                    dropped, and the suite counts as sibling-dataset
                    transfer (shared intent taxonomy), not zero-shot.
                    Choice "Which intent
                    does `utterance` express?". head_max_len 512.
                    Language en.
  indonli           Human-annotated Indonesian NLI, expert test only:
                    https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/test_expert.jsonl
                    (2,984 rows, labels e/n/c). datasets 5 cannot run
                    the HF loading script. translate_train (MNLI
                    translated) is not downloaded. GitHub licence for
                    ir-nlp-csui/indonli is null. README: premises from
                    Indonesian Wikipedia (CC-BY-SA 3.0 and GFDL), UD
                    PUD/GSD (CC-BY-SA 3.0 and CC-BY-SA 4.0) and IndoSum
                    (Apache-2.0). HF card afaji/indonli says
                    cc-by-sa-4.0. Evaluation only. Choice "What is the
                    relation of `hypothesis` to `premise`?". Language id.
  farstail          Persian NLI test file only, TSV with letter labels
                    e/n/c (1,564 rows):
                    https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv
                    Repo dml-qom/FarsTail licence Apache-2.0 (GitHub
                    API, 2026-09-27). The mteb mirror's card says
                    licence unknown and stores unlabelled integers, so
                    it is not used. Same choice wording as indonli.
                    Language fa.
  belebele          facebook/belebele test configs only (900 parallel
                    items). HF card licence cc-by-sa-4.0. Official
                    README (facebookresearch/belebele): the dataset is
                    under LICENSE_CC-BY-SA4.0; GitHub's detected
                    licence is NOASSERTION. The assembled training set
                    is a different licence and is not loaded.
                    link+question_number is shared across languages and
                    the correct option number agrees, so one stratified
                    draw is scored in every language. Each item has its
                    own four answers, so each request is one state
                    (run_statim, batch 1) with options "1".."4" and the
                    answer text as the description. Choice "Which
                    option answers `question` given `passage`?".
                    Languages en de ar hi (eng_Latn, deu_Latn,
                    arb_Arab, hin_Deva).
  semrel            SemRel/SemRel2024 split test (not train, not dev).
                    Relatedness in [0, 1] is binned into five ordinal
                    levels at 0.2: [0, 0.2), [0.2, 0.4), [0.4, 0.6),
                    [0.6, 0.8), [0.8, 1]. Score "How related in meaning
                    are `sentence1` and `sentence2`?". Official repo
                    semantic-textual-relatedness/Semantic_Relatedness_SemEval2024
                    has no licence file (GitHub API license null,
                    2026-09-27). The HF card has no licence field.
                    Evaluation only; the publishers do not state a
                    SPDX licence on either source. Not STS-Benchmark
                    and not SICK. Languages en ar hi (configs eng, arb,
                    hin).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_accuracy import metrics, run_statim  # noqa: E402
from prediction_items import write_prediction_rows  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFESTS = [
    os.path.join(ROOT, "data", "mixture-v1.manifest.json"),
    os.path.join(ROOT, "data", "mixture-v2.manifest.json"),
]
SUITES = [
    "go_emotions", "multi_hatecheck", "sib200", "hwu64",
    "indonli", "farstail", "belebele", "semrel",
]
MODELS = ["english", "multilingual"]
# Needle that must not occur in any mixture per_source key. go_emotions is
# in the upstream tasksource corpus and absent from both manifests; the
# check still refuses it if a future mixture adds it.
SUITE_NEEDLES = {
    "go_emotions": ["go_emotions", "goemotions"],
    "multi_hatecheck": ["hatecheck", "multi-hatecheck", "multilingual-hatecheck"],
    "sib200": ["sib200", "sib-200", "sib_200"],
    "hwu64": ["hwu64", "hwu_64", "nlu-evaluation-data"],
    "indonli": ["indonli", "indo_nli", "indo-nli"],
    "farstail": ["farstail", "fars_tail", "fars-tail"],
    "belebele": ["belebele"],
    "semrel": ["semrel", "semrel2024"],
}
LANG_GROUPS = {
    "en": {"en", "eng", "english"}, "eng": {"en", "eng", "english"}, "english": {"en", "eng", "english"},
    "de": {"de", "deu", "german"}, "deu": {"de", "deu", "german"}, "german": {"de", "deu", "german"},
    "fr": {"fr", "fra", "french"}, "fra": {"fr", "fra", "french"}, "french": {"fr", "fra", "french"},
    "es": {"es", "spa", "spanish"}, "spa": {"es", "spa", "spanish"}, "spanish": {"es", "spa", "spanish"},
    "it": {"it", "ita", "italian"}, "ita": {"it", "ita", "italian"}, "italian": {"it", "ita", "italian"},
    "nl": {"nl", "nld", "dutch"}, "nld": {"nl", "nld", "dutch"}, "dutch": {"nl", "nld", "dutch"},
    "pl": {"pl", "pol", "polish"}, "pol": {"pl", "pol", "polish"}, "polish": {"pl", "pol", "polish"},
    "pt": {"pt", "por", "portuguese"}, "por": {"pt", "por", "portuguese"}, "portuguese": {"pt", "por", "portuguese"},
    "zh": {"zh", "cmn", "zho", "chinese"}, "cmn": {"zh", "cmn", "zho", "chinese"},
    "zho": {"zh", "cmn", "zho", "chinese"}, "chinese": {"zh", "cmn", "zho", "chinese"},
    "ar": {"ar", "ara", "arb", "arabic"}, "ara": {"ar", "ara", "arb", "arabic"},
    "arb": {"ar", "ara", "arb", "arabic"}, "arabic": {"ar", "ara", "arb", "arabic"},
    "hi": {"hi", "hin", "hindi"}, "hin": {"hi", "hin", "hindi"}, "hindi": {"hi", "hin", "hindi"},
    "id": {"id", "ind", "indonesian"}, "ind": {"id", "ind", "indonesian"}, "indonesian": {"id", "ind", "indonesian"},
    "fa": {"fa", "fas", "persian"}, "fas": {"fa", "fas", "persian"}, "persian": {"fa", "fas", "persian"},
}
DISPLAY_ORDER = ["en", "de", "fr", "es", "it", "nl", "pl", "pt", "zh", "ar", "hi", "id", "fa"]

GO_GLOSSES = {
    "admiration": "respect or warm approval",
    "amusement": "finding something funny",
    "anger": "strong displeasure or rage",
    "annoyance": "mild irritation",
    "approval": "agreeing or accepting",
    "caring": "kind concern for someone",
    "confusion": "not understanding",
    "curiosity": "wanting to know more",
    "desire": "wanting something",
    "disappointment": "sad that hopes were not met",
    "disapproval": "rejecting or condemning",
    "disgust": "strong revulsion",
    "embarrassment": "feeling self-conscious or ashamed",
    "excitement": "eager enthusiasm",
    "fear": "being afraid",
    "gratitude": "being thankful",
    "grief": "deep sorrow after a loss",
    "joy": "happiness",
    "love": "affection",
    "nervousness": "anxious unease",
    "optimism": "expecting a good outcome",
    "pride": "satisfaction in something worthy",
    "realization": "suddenly understanding",
    "relief": "ease after a worry ends",
    "remorse": "regret for a wrong",
    "sadness": "unhappiness or sorrow",
    "surprise": "being startled by the unexpected",
    "neutral": "no clear emotion",
}
HATE_LANGS = [
    ("en", "eng"), ("de", "deu"), ("fr", "fra"), ("es", "spa"), ("it", "ita"),
    ("nl", "nld"), ("pl", "pol"), ("pt", "por"), ("zh", "cmn"), ("ar", "ara"), ("hi", "hin"),
]
HATE_OPTIONS = ["non-hateful", "hateful"]
SIB_LANGS = [("en", "eng_Latn"), ("de", "deu_Latn"), ("ar", "arb_Arab"), ("hi", "hin_Deva")]
SIB_OPTIONS = {
    "entertainment": "films, music, art and celebrity culture",
    "geography": "places, the earth and the environment",
    "health": "medicine, the body and wellbeing",
    "politics": "government, elections and public affairs",
    "science/technology": "science, research and technology",
    "sports": "sport and athletic competition",
    "travel": "travel, tourism and transport",
}
BELEBELE_LANGS = [("en", "eng_Latn"), ("de", "deu_Latn"), ("ar", "arb_Arab"), ("hi", "hin_Deva")]
BELEBELE_KEYS = ["1", "2", "3", "4"]
NLI_OPTIONS = ["entailment", "neutral", "contradiction"]
NLI_LETTERS = {"e": "entailment", "n": "neutral", "c": "contradiction"}
SEMREL_LANGS = [("en", "eng"), ("ar", "arb"), ("hi", "hin")]
SEMREL_LEVELS = [
    "unrelated: the sentences do not share a meaning",
    "weakly related: only a slight topical overlap",
    "moderately related: a shared topic, not the same claim",
    "related: closely connected meanings",
    "highly related: nearly the same meaning",
]
SEMREL_KEYS = ["0", "1", "2", "3", "4"]

_CACHE = {}


def _dataset():
    from datasets import load_dataset
    return load_dataset


def manifest_sources():
    keys = []
    for path in MANIFESTS:
        if not os.path.isfile(path):
            raise SystemExit("missing training manifest %s" % path)
        data = json.load(open(path, encoding="utf-8"))
        per = data.get("per_source")
        if not isinstance(per, dict) or not per:
            raise SystemExit("%s has no per_source map" % path)
        keys.extend(per)
    return keys


def assert_held_out(suites):
    keys = [k.lower() for k in manifest_sources()]
    for suite in suites:
        for needle in SUITE_NEEDLES[suite]:
            hit = [k for k in keys if needle.lower() in k]
            if hit:
                raise SystemExit(
                    "%s is not held out: %r matches training source %s" % (suite, needle, hit[:8]))
    return len(keys)


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
    return bool(want & have)


def stratified_indices(labels, n, seed, what):
    """Round-robin over classes. Each class is sorted by row index, then
    shuffled with `seed`, so the draw does not depend on dict order."""
    buckets = {}
    for idx, lab in enumerate(labels):
        buckets.setdefault(lab, []).append(idx)
    classes = sorted(buckets, key=lambda lab: str(lab))
    rng = random.Random(seed)
    for lab in classes:
        buckets[lab].sort()
        rng.shuffle(buckets[lab])
    picked = []
    while len(picked) < n:
        grew = False
        for lab in classes:
            bucket = buckets[lab]
            if not bucket:
                continue
            picked.append(bucket.pop())
            grew = True
            if len(picked) >= n:
                break
        if not grew:
            break
    if len(picked) < n:
        raise SystemExit("%s: eligible split has %d rows, asked for %d" % (what, len(picked), n))
    picked.sort()
    return picked


def _require_classes(suite, lang, gold):
    n_cls = len(set(gold))
    if n_cls < 2:
        raise SystemExit("%s %s: sample covers %d class(es), need at least 2" % (suite, lang, n_cls))


def _text(value):
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _choice(qid, instruction, criteria):
    return {qid: {"type": "choice", "instructions": instruction, "criteria": criteria}}


def _head(n_options):
    return 512 if n_options > 10 else None


def _jsonl(url):
    if url not in _CACHE:
        _CACHE[url] = _dataset()("json", data_files={"test": url}, split="test")
    return _CACHE[url]


def go_emotions_task(n, seed):
    ds = _dataset()("google-research-datasets/go_emotions", "simplified", split="test")
    names = list(ds.features["labels"].feature.names)
    if names != list(GO_GLOSSES):
        raise SystemExit("go_emotions: label names differ from the card list: %s" % names)
    rows = [(i, int(r["labels"][0])) for i, r in enumerate(ds) if len(r["labels"]) == 1]
    if len(rows) < n:
        raise SystemExit("go_emotions: %d single-label test rows, asked for %d" % (len(rows), n))
    labels = [lab for _, lab in rows]
    chosen = stratified_indices(labels, n, seed, "go_emotions")
    texts = list(ds["text"])
    states = [{"text": _text(texts[rows[j][0]])} for j in chosen]
    gold = [labels[j] for j in chosen]
    _require_classes("go_emotions", "en", gold)
    questions = _choice(
        "emotion", "Which emotion is most strongly expressed in `text`?",
        {name: GO_GLOSSES[name] for name in names})
    return [("en", states, questions, "emotion", names, gold, _head(len(names)), "test")]


def hate_task(langs, n, seed):
    questions = _choice(
        "hate", "Is `text` hateful or non-hateful?",
        {"non-hateful": "does not attack or demean a protected group",
         "hateful": "attacks or demeans a protected group"})
    out = []
    for code, file_code in HATE_LANGS:
        if code not in langs:
            continue
        url = "https://huggingface.co/datasets/mteb/multi-hatecheck/resolve/main/test/%s.jsonl.gz" % file_code
        ds = _jsonl(url)
        if set(ds["is_hateful"]) - set(HATE_OPTIONS):
            raise SystemExit("multi_hatecheck %s: unexpected labels %s" % (code, sorted(set(ds["is_hateful"]))))
        labels = [HATE_OPTIONS.index(x) for x in ds["is_hateful"]]
        chosen = stratified_indices(labels, n, seed, "multi_hatecheck %s" % code)
        texts = list(ds["text"])
        states = [{"text": _text(texts[i])} for i in chosen]
        gold = [labels[i] for i in chosen]
        _require_classes("multi_hatecheck", code, gold)
        out.append((code, states, questions, "hate", HATE_OPTIONS, gold, None, "test"))
    return out


def sib_task(langs, n, seed):
    questions = _choice("topic", "What is the topic of `text`?", dict(SIB_OPTIONS))
    keys = list(SIB_OPTIONS)
    out = []
    for code, config in SIB_LANGS:
        if code not in langs:
            continue
        ds = _dataset()("Davlan/sib200", config, split="test")
        if set(ds["category"]) != set(keys):
            raise SystemExit("sib200 %s: categories %s" % (code, sorted(set(ds["category"]))))
        labels = [keys.index(x) for x in ds["category"]]
        chosen = stratified_indices(labels, n, seed, "sib200 %s" % code)
        texts = list(ds["text"])
        states = [{"text": _text(texts[i])} for i in chosen]
        gold = [labels[i] for i in chosen]
        _require_classes("sib200", code, gold)
        out.append((code, states, questions, "topic", keys, gold, None, "test"))
    return out


def hwu_task(n, seed):
    intents = _dataset()("DeepPavlov/hwu64", "intents", split="intents")
    by_id = {}
    for row in intents:
        name = " ".join(str(row["name"]).replace("_", " ").split())
        if not name:
            raise SystemExit("hwu64: empty intent name for id %s" % row["id"])
        by_id[int(row["id"])] = name
    if len(by_id) != 64 or len(set(by_id.values())) != 64:
        raise SystemExit("hwu64: expected 64 distinct intent names, got %d" % len(set(by_id.values())))
    keys = [by_id[i] for i in range(64)]
    ds = _dataset()("DeepPavlov/hwu64", split="test")
    if set(ds["label"]) - set(by_id):
        raise SystemExit("hwu64: test label outside 0..63")
    # MASSIVE (a Statim training set) localises SLURP, which grew out of HWU64: a third of HWU64
    # test utterances occur verbatim in MASSIVE. Drop those rows; the remaining ones still share
    # MASSIVE's intent taxonomy, so this suite measures transfer to a sibling dataset rather than
    # a fully unseen label set (reported as such).
    massive = set()
    for split in ("train", "validation", "test"):
        url = "https://huggingface.co/datasets/mteb/amazon_massive_intent/resolve/main/%s/en.json.gz" % split
        massive |= {" ".join(r["text"].lower().split()) for r in _dataset()("json", data_files={"x": url}, split="x")}
    texts_all = list(ds["utterance"])
    keep = [i for i, t in enumerate(texts_all) if " ".join(str(t).lower().split()) not in massive]
    print("hwu64: dropped %d of %d test rows that occur in MASSIVE" % (len(texts_all) - len(keep), len(texts_all)),
          flush=True)
    labels_all = [int(x) for x in ds["label"]]
    labels = [labels_all[i] for i in keep]
    texts = [texts_all[i] for i in keep]
    chosen = stratified_indices(labels, n, seed, "hwu64")
    states = [{"utterance": _text(texts[i])} for i in chosen]
    gold = [labels[i] for i in chosen]
    _require_classes("hwu64", "en", gold)
    questions = _choice("intent", "Which intent does `utterance` express?", {k: None for k in keys})
    return [("en", states, questions, "intent", keys, gold, _head(len(keys)), "test")]


def _nli_questions():
    return _choice(
        "relation", "What is the relation of `hypothesis` to `premise`?",
        {"entailment": "the hypothesis follows from the premise",
         "neutral": "the hypothesis might be true or false given the premise",
         "contradiction": "the hypothesis conflicts with the premise"})


def indonli_task(n, seed):
    url = "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/test_expert.jsonl"
    ds = _jsonl(url)
    for row in ds:
        if row.get("annotator_type") != "expert" or row.get("data_split") != "test":
            raise SystemExit("indonli: row is not expert test (%s)" % {k: row.get(k) for k in ("annotator_type", "data_split")})
    if set(ds["label"]) - set(NLI_LETTERS):
        raise SystemExit("indonli: unexpected labels %s" % sorted(set(ds["label"])))
    labels = [NLI_OPTIONS.index(NLI_LETTERS[x]) for x in ds["label"]]
    chosen = stratified_indices(labels, n, seed, "indonli")
    premises, hyps = list(ds["premise"]), list(ds["hypothesis"])
    states = [{"premise": _text(premises[i]), "hypothesis": _text(hyps[i])} for i in chosen]
    gold = [labels[i] for i in chosen]
    _require_classes("indonli", "id", gold)
    return [("id", states, _nli_questions(), "relation", NLI_OPTIONS, gold, None, "test_expert")]


def farstail_task(n, seed):
    url = "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv"
    if "Test-word.csv" not in url:
        raise SystemExit("farstail: refusing a URL that is not the test file")
    if url not in _CACHE:
        _CACHE[url] = _dataset()("csv", data_files={"test": url}, split="test", delimiter="\t")
    ds = _CACHE[url]
    if set(ds["label"]) - set(NLI_LETTERS):
        raise SystemExit("farstail: unexpected labels %s" % sorted(set(ds["label"])))
    labels = [NLI_OPTIONS.index(NLI_LETTERS[x]) for x in ds["label"]]
    chosen = stratified_indices(labels, n, seed, "farstail")
    premises, hyps = list(ds["premise"]), list(ds["hypothesis"])
    states = [{"premise": _text(premises[i]), "hypothesis": _text(hyps[i])} for i in chosen]
    gold = [labels[i] for i in chosen]
    _require_classes("farstail", "fa", gold)
    return [("fa", states, _nli_questions(), "relation", NLI_OPTIONS, gold, None, "test")]


def _semrel_bin(value):
    x = float(value)
    if x < 0.0 or x > 1.0:
        raise SystemExit("semrel: relatedness %r is outside [0, 1]" % value)
    if x >= 1.0:
        return 4
    return min(4, int(x / 0.2))


def semrel_task(langs, n, seed):
    questions = {"relatedness": {
        "type": "score",
        "instructions": "How related in meaning are `sentence1` and `sentence2`?",
        "criteria": list(SEMREL_LEVELS)}}
    out = []
    for code, config in SEMREL_LANGS:
        if code not in langs:
            continue
        ds = _dataset()("SemRel/SemRel2024", config, split="test")
        labels = [_semrel_bin(x) for x in ds["label"]]
        chosen = stratified_indices(labels, n, seed, "semrel %s" % code)
        s1, s2 = list(ds["sentence1"]), list(ds["sentence2"])
        states = [{"sentence1": _text(s1[i]), "sentence2": _text(s2[i])} for i in chosen]
        gold = [labels[i] for i in chosen]
        _require_classes("semrel", code, gold)
        out.append((code, states, questions, "relatedness", SEMREL_KEYS, gold, None, "test"))
    return out


def belebele_task(langs, n, seed):
    """One shared item list. Answer texts differ by language, so each item
    is its own question. Option numbers and the correct number do not."""
    loaded = {}
    for code, config in BELEBELE_LANGS:
        if code not in langs:
            continue
        ds = _dataset()("facebook/belebele", config, split="test")
        rows = {}
        for row in ds:
            key = (row["link"], int(row["question_number"]))
            if key in rows:
                raise SystemExit("belebele %s: duplicate %s" % (code, key))
            answers = [_text(row["mc_answer%d" % i]).strip() for i in range(1, 5)]
            if any(not a for a in answers):
                continue
            rows[key] = (answers, str(row["correct_answer_num"]), _text(row["flores_passage"]), _text(row["question"]))
        loaded[code] = rows
    if not loaded:
        return []
    ref = loaded["en"] if "en" in loaded else loaded[next(iter(loaded))]
    ref_code = "en" if "en" in loaded else next(iter(loaded))
    shared = set.intersection(*(set(rows) for rows in loaded.values()))
    if len(shared) < n:
        raise SystemExit("belebele: %d items in every selected language, asked for %d" % (len(shared), n))
    ordered = sorted(shared)
    labels = []
    for key in ordered:
        number = ref[key][1]
        if number not in BELEBELE_KEYS:
            raise SystemExit("belebele: correct_answer_num %r" % number)
        for code, rows in loaded.items():
            if rows[key][1] != number:
                raise SystemExit("belebele: %s and %s disagree on the correct option for %s" % (ref_code, code, key))
        labels.append(BELEBELE_KEYS.index(number))
    chosen = stratified_indices(labels, n, seed, "belebele")
    keys_picked = [ordered[i] for i in chosen]
    out = []
    instruction = "Which option answers `question` given `passage`?"
    for code, _config in BELEBELE_LANGS:
        if code not in loaded:
            continue
        rows = loaded[code]
        triples, gold = [], []
        for key in keys_picked:
            answers, number, passage, question = rows[key]
            criteria = {BELEBELE_KEYS[i]: answers[i] for i in range(4)}
            triples.append((
                {"passage": passage, "question": question},
                _choice("answer", instruction, criteria)))
            gold.append(BELEBELE_KEYS.index(number))
        _require_classes("belebele", code, gold)
        out.append((code, triples, "answer", BELEBELE_KEYS, gold, None, "test"))
    return out


def macro(rows):
    n = len(rows)
    out = {k: round(sum(r[k] for r in rows) / n, 4) for k in ("accuracy", "ece", "nll", "brier")}
    out["seconds"] = round(sum(r["seconds"] for r in rows) / n, 1)
    return out


def print_table(records):
    suites_present = [s for s in SUITES if any(r["suite"] == s and r["lang"] != "macro" for r in records)]
    langs, seen = [], set()
    for lang in DISPLAY_ORDER:
        if any(r["lang"] == lang for r in records):
            langs.append(lang)
            seen.add(lang)
    for r in records:
        if r["lang"] not in seen and r["lang"] != "macro":
            langs.append(r["lang"])
            seen.add(r["lang"])
    by = {(r["suite"], r["lang"]): r for r in records}
    header = "Sprache".ljust(10) + "".join(s.rjust(18) for s in suites_present)
    print("\n" + header)
    print("-" * len(header))

    def cell(suite, lang):
        row = by.get((suite, lang))
        return "—".rjust(18) if row is None else ("%0.4f" % row["accuracy"]).rjust(18)

    for lang in langs:
        print(lang.ljust(10) + "".join(cell(s, lang) for s in suites_present))
    print("Makro".ljust(10) + "".join(cell(s, "macro") for s in suites_present))
    print("(Zellen = accuracy, Makro = ungewichtet über Sprachen)")
    for s in suites_present:
        m = by.get((s, "macro"))
        if m:
            print("Makro %-18s  acc %0.4f  ece %0.4f  nll %0.4f  brier %0.4f  sec %s  (%d Sprachen)" % (
                s, m["accuracy"], m["ece"], m["nll"], m["brier"], m["seconds"], m["n_langs"]))
    suite_macros = [by[(s, "macro")] for s in suites_present if (s, "macro") in by]
    if suite_macros:
        overall = macro(suite_macros)
        print("Makro über Suiten     acc %0.4f  ece %0.4f  nll %0.4f  brier %0.4f  (%d Suiten)" % (
            overall["accuracy"], overall["ece"], overall["nll"], overall["brier"], len(suite_macros)))
        return overall
    return None


def _run_fixed(url, model, states, questions, qid, keys, head):
    return run_statim(url, states, questions, qid, keys, 1,
                      api_key=os.environ.get("STATIM_API_KEY"), model=model, head_max_len=head)


def _run_varied(url, model, triples, qid, keys, head):
    probs, t0 = [], time.time()
    step = 25 if len(triples) > 25 else 0
    for i, (state, questions) in enumerate(triples):
        part, _secs = _run_fixed(url, model, [state], questions, qid, keys, head)
        probs.append(part[0])
        if step and (i + 1) % step == 0:
            print("  belebele %d/%d" % (i + 1, len(triples)), flush=True)
    return probs, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Zero-shot Statim benchmark on held-out public datasets")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    ap.add_argument("--langs", nargs="+", default=None,
                    help="language codes (default: every language configured for each suite)")
    ap.add_argument("--n", type=int, default=150, help="stratified sample size per suite and language")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", default=None, help="JSONL path, one record per suite/language plus a macro line")
    ap.add_argument("--predictions", default=None, metavar="PATH",
                    help="JSONL, one per-item gold/prediction record for paired comparisons")
    a = ap.parse_args()
    if a.langs:
        unknown = [t for t in a.langs if t.strip().lower() not in LANG_GROUPS]
        if unknown:
            raise SystemExit("unknown language(s): %s" % ", ".join(unknown))
    if a.n < 1:
        raise SystemExit("--n must be >= 1")
    n_sources = assert_held_out(a.suites)
    print("held-out ok: %d manifest source names, none of %s" % (n_sources, ", ".join(a.suites)), flush=True)

    selected = {}
    if "go_emotions" in a.suites and lang_wanted(a.langs, "en"):
        selected["go_emotions"] = ("fixed", go_emotions_task(a.n, a.seed))
    if "multi_hatecheck" in a.suites:
        langs = [code for code, _f in HATE_LANGS if lang_wanted(a.langs, code)]
        if langs:
            selected["multi_hatecheck"] = ("fixed", hate_task(langs, a.n, a.seed))
    if "sib200" in a.suites:
        langs = [code for code, _c in SIB_LANGS if lang_wanted(a.langs, code)]
        if langs:
            selected["sib200"] = ("fixed", sib_task(langs, a.n, a.seed))
    if "hwu64" in a.suites and lang_wanted(a.langs, "en"):
        selected["hwu64"] = ("fixed", hwu_task(a.n, a.seed))
    if "indonli" in a.suites and lang_wanted(a.langs, "id"):
        selected["indonli"] = ("fixed", indonli_task(a.n, a.seed))
    if "farstail" in a.suites and lang_wanted(a.langs, "fa"):
        selected["farstail"] = ("fixed", farstail_task(a.n, a.seed))
    if "belebele" in a.suites:
        langs = [code for code, _c in BELEBELE_LANGS if lang_wanted(a.langs, code)]
        if langs:
            selected["belebele"] = ("varied", belebele_task(langs, a.n, a.seed))
    if "semrel" in a.suites:
        langs = [code for code, _c in SEMREL_LANGS if lang_wanted(a.langs, code)]
        if langs:
            selected["semrel"] = ("fixed", semrel_task(langs, a.n, a.seed))
    jobs = [(suite, kind, tasks) for suite, (kind, tasks) in selected.items() if tasks]
    if not jobs:
        raise SystemExit("nothing to run: no suite/language matched")

    out = None
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        out = open(a.out, "w", encoding="utf-8")
    predictions = None
    if a.predictions:
        os.makedirs(os.path.dirname(os.path.abspath(a.predictions)) or ".", exist_ok=True)
        predictions = open(a.predictions, "w", encoding="utf-8")
    records = []
    try:
        by_suite = {}
        for suite, kind, tasks in jobs:
            for task in tasks:
                if kind == "fixed":
                    lang, states, questions, qid, keys, gold, head, split = task
                    print("run %s %s n=%d options=%d classes=%d split=%s" % (
                        suite, lang, len(gold), len(keys), len(set(gold)), split), flush=True)
                    probs, secs = _run_fixed(a.url, a.model, states, questions, qid, keys, head)
                    prediction_states, prediction_questions = states, questions
                else:
                    lang, triples, qid, keys, gold, head, split = task
                    print("run %s %s n=%d options=%d classes=%d split=%s (one request per item)" % (
                        suite, lang, len(gold), len(keys), len(set(gold)), split), flush=True)
                    probs, secs = _run_varied(a.url, a.model, triples, qid, keys, head)
                    prediction_states = [state for state, _questions in triples]
                    prediction_questions = [questions for _state, questions in triples]
                if predictions:
                    write_prediction_rows(predictions, suite, lang, prediction_states,
                                          prediction_questions, gold, probs)
                row = {"suite": suite, "lang": lang, "model": a.model, "n": len(gold),
                       "n_options": len(keys), "n_gold_labels": len(set(gold)),
                       "seed": a.seed, "split": split, "head_max_len": head}
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
            rec = {"suite": suite, "lang": "macro", "model": a.model,
                   "n": sum(r["n"] for r in rows), "n_langs": len(rows), "seed": a.seed, **m}
            records.append(rec)
            if out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
        overall = print_table(records)
        if out and overall:
            out.write(json.dumps({"suite": "all", "lang": "macro", "model": a.model,
                                  "n_suites": len(by_suite), "seed": a.seed, **overall},
                                 ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()
        if predictions:
            predictions.close()


if __name__ == "__main__":
    main()
