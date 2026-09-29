"""Offline regression tests for the v6 label fixes and the content audit (no network).

    .venv-train/bin/python -m pytest -q tools/finetune/mixture_v6/test_label_fixes.py
"""

import collections
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

from tools.finetune.mixture_v6 import audit, emotion_taxonomy, loaders, registry
from tools.finetune.mixture_v6.registry import REGISTRY_PATH, adapt
from tools.finetune.mixture_v6.templates import GLOSSES, LANGUAGES, describe


def entry(sid, config=None):
    for e in registry.ENTRIES:
        if e["id"] == sid and (config is None or e.get("config") == config):
            return e
    raise KeyError(sid)


def golds(items, kind=None):
    out = collections.Counter()
    for item in items:
        q = item["q"]
        if kind and q["type"] != kind:
            continue
        if q["type"] == "choice":
            out[list(q["criteria"])[item["target"].index(1.0)]] += 1
        else:
            out[item["target"].index(1.0)] += 1
    return out


# --------------------------------------------------------------------------- safety

def test_toxic_conversations_not_toxic_is_negative():
    rows = [{"text": "have a nice day", "label_text": "not toxic"}, {"text": "you idiot", "label_text": "toxic"}]
    items = list(adapt(entry("mteb/toxic_conversations_50k"), rows, 1))
    assert golds(items, "noul") == {0: 1, 1: 1}


def test_prosocial_casual_is_negative():
    rows = [{"context": "I want to bake a cake.", "safety_label": "__casual__"},
            {"context": "I want to hurt my cat.", "safety_label": "__needs_intervention__"}]
    items = list(adapt(entry("allenai/prosocial-dialog"), rows, 1))
    assert {it["state"]: it["target"].index(1.0) for it in items if it["q"]["type"] == "noul"} == {
        "I want to bake a cake.": 0, "I want to hurt my cat.": 1}


def test_aegis1_majority_vote_and_ambiguous_rows_dropped():
    e = entry("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")
    rows = [{"text": "bread", "labels_0": "Safe", "labels_1": "Safe", "labels_2": "Violence"},
            {"text": "harm", "labels_0": "Violence", "labels_1": "Violence", "labels_2": "Safe"},
            {"text": "tie", "labels_0": "Safe", "labels_1": "Needs Caution", "labels_2": "Violence", "labels_3": "Hate"}]
    assert registry.labels_from(e, rows[0]) == ["safe"]
    assert registry.labels_from(e, rows[1]) == ["Violence"]
    assert registry.labels_from(e, rows[2]) == []
    noul = {it["state"]: it["target"].index(1.0) for it in adapt(e, rows, 1) if it["q"]["type"] == "noul"}
    assert noul == {"bread": 0, "harm": 1}


def test_oasst2_labels_from_crowd_votes():
    e = entry("OpenAssistant/oasst2")
    benign = {"text": "How do I bake bread?", "labels": "{'name': ['spam', 'toxicity'], 'value': [0.0, 0.1], 'count': [3, 3]}"}
    toxic = {"text": "You are worthless.", "labels": {"name": ["toxicity"], "value": [0.8], "count": [3]}}
    assert registry.labels_from(e, benign) == ["safe"]
    assert registry.labels_from(e, toxic) == ["toxicity"]


# --------------------------------------------------------------------------- similarity

def test_group_pairs_have_both_classes():
    rows = [{"headline": "Flood hits town", "group_id": 1}, {"headline": "Town flooded", "group_id": 1},
            {"headline": "Budget passes", "group_id": 2}, {"headline": "Senate passes budget", "group_id": 2}]
    items = list(adapt(entry("dell-research-harvard/headlines-semantic-similarity"), rows, 1))
    assert set(golds(items, "score")) == {0, 4}


def test_qa_pairs_relevant_and_not():
    rows = [{"Question": "Q one?", "Answer": "A one."}, {"Question": "Q two?", "Answer": "A two."}]
    items = list(adapt(entry("matsuxr/JaGovFaqs-22k"), rows, 1))
    assert golds(items, "score") == {4: 2, 0: 2}


@pytest.mark.parametrize("label,level", [("Exact", 4), ("Substitute", 2), ("Complement", 1), ("Irrelevant", 0)])
def test_esci_relevance_levels(label, level):
    rows = [{"query": "bathroom fan", "product_title": "Ceiling fan with light", "esci_label": label}]
    items = list(adapt(entry("tasksource/esci"), rows, 1))
    assert golds(items, "score") == {level: 1}


def test_tapaco_pairs_need_contiguous_sets():
    rows = [{"paraphrase_set_id": "1", "paraphrase": "I ate the cheese.", "language": "en"},
            {"paraphrase_set_id": "1", "paraphrase": "I have eaten the cheese.", "language": "en"},
            {"paraphrase_set_id": "2", "paraphrase": "It is raining.", "language": "en"}]
    items = list(adapt(entry("community-datasets/tapaco"), rows, 1))
    assert set(golds(items, "score")) == {0, 4}


# --------------------------------------------------------------------------- labels

def test_class_labels_are_decoded():
    rows = [{"label": 1}, {"label": 3}, {"label": 9}, {"label": True}]
    loaders.decode_class_labels(rows, {"label": ["negative", "positive", "no_impact", "mixed"]})
    assert [r["label"] for r in rows] == ["positive", "mixed", 9, True]


@pytest.mark.parametrize("sid,col,raw,name", [
    ("YiMeng-SYSU/chinese-logic-sentiment-dataset", "label", 1, "positive"),
    ("tanaos/synthetic-sentiment-analysis-dataset-v1", "labels", "0", "very negative"),
    ("tanaos/synthetic-emotion-detection-dataset-v1", "labels", 6, "excitement"),
])
def test_card_labels(sid, col, raw, name):
    assert loaders._postprocess(sid, [{col: raw}])[0][col] == name


def test_star_ratings_become_sentiment_but_do_not_duplicate_a_sentiment_field():
    sutro = entry("sutro/synthetic-product-reviews-20k")
    assert registry.field_labels(sutro, {"review_text": "ok", "rating_out_of_5": 5}, {}) == [("sentiment", "positive")]
    assert registry.field_labels(sutro, {"review_text": "ok", "rating_out_of_5": 3}, {}) == [("sentiment", "neutral")]
    khired = entry("KhiredNetworks/synthetic-product-reviews")
    assert registry.field_labels(khired, {"text": "ok", "sentiment": "negative", "rating": 5}, {}) == [
        ("sentiment", "negative")]


def test_multi_label_rows_have_no_single_gold():
    e = entry("CohereForAI/aya_redteaming")
    one = {"prompt": "p1", "harm_category": '["Profanity"]', "language": "English"}
    two = {"prompt": "p2", "harm_category": '["Profanity", "Hate Speech"]', "language": "English"}
    other = {"prompt": "p3", "harm_category": '["Hate Speech"]', "language": "English"}
    choices = [it for it in adapt(e, [one, two, other], 1) if it["q"]["type"] == "choice"]
    assert sorted(it["state"] for it in choices) == ["p1", "p3"]


def test_special_topic_labels():
    arxiv = entry("gfissore/arxiv-abstracts-2021")
    assert registry.field_labels(arxiv, {"categories": ["math.NT cs.CR"]}, {}) == [("categories", "mathematics")]
    news = entry("Fumika/Wikinews-multilingual")
    assert registry.field_labels(news, {"categories": ["Sports", "Germany"]}, {}) == [("categories", "Sports")]
    assert registry.field_labels(news, {"categories": ["Sports", "Health"]}, {}) == []
    nhtsa = entry("vic35get/nhtsa_complaints_dataset")
    assert registry.field_labels(nhtsa, {"components": "AIR BAGS, STEERING"}, {}) == []


def test_big_patent_labels_are_cpc_section_titles():
    e = entry("NortheasternUniversity/big_patent")
    rows = [{"abstract": "A machine.", "_v6_config": "a"}, {"abstract": "A circuit.", "_v6_config": "h"}]
    items = list(adapt(e, rows, 1))
    assert golds(items, "choice") == {"human necessities": 1, "electricity": 1}


def test_covid_events_are_named():
    e = entry("joelniklaus/covid19_emergency_event")
    assert registry.labels_from(e, {"text": "x", "all_events": ["event4"]}) == ["closures or lockdown"]


def test_urgency_is_binary():
    e = entry("IDinsight/urgency_detection_maternal_health_synthetic")
    assert registry.field_labels(e, {"matching_rule": "Changes in your vision"}, {}) == [("urgency", "urgent")]
    assert registry.field_labels(e, {"matching_rule": "NOT URGENT"}, {}) == [("urgency", "not urgent")]


def test_casino_one_row_per_single_strategy_utterance():
    rows = loaders._postprocess("kchawla123/casino", [
        {"annotations": "[['Hello!', 'small-talk'], ['I need water', 'self-need,elicit-pref']]"}])
    assert [(r["utterance"], r["annotations"]) for r in rows] == [("Hello!", "small-talk")]


# --------------------------------------------------------------------------- pii and reading

def test_pii_span_lists_are_parsed_not_used_as_labels():
    e = entry("nvidia/Nemotron-PII")
    rows = [{"text": "Call Anna at 040 123.", "spans": "[{'start': 5, 'end': 9, 'label': 'first_name'}]"},
            {"text": "Mail a@b.example.", "spans": "[{'start': 5, 'end': 16, 'label': 'email'}]"}]
    items = list(adapt(e, rows, 1))
    for item in items:
        if item["q"]["type"] == "choice":
            assert set(item["q"]["criteria"]) <= {"first name", "email"}
    assert any(it.get("_task") == "pii_type" for it in items)


def test_reading_windows_have_the_same_length_for_both_classes():
    context = "".join("Sentence %d of the contract. " % i for i in range(400))
    answered = {"context": context, "question": "Q?", "answers": {"text": ["Sentence 300"], "answer_start": [context.index("Sentence 300")]}}
    unanswered = {"context": context, "question": "R?", "answers": {"text": [], "answer_start": []}}
    a, b = registry._window_context(answered), registry._window_context(unanswered)
    assert len(a["context"]) == len(b["context"]) == registry.WINDOW
    assert "Sentence 300" in a["context"]


def test_fairytale_gets_unanswerable_pairs():
    rows = [{"context": "Story A text.", "question": "Who is in A?", "answers": {"text": ["x"]}, "story_name": "a"},
            {"context": "Story B text.", "question": "Who is in B?", "answers": {"text": ["y"]}, "story_name": "b"}]
    items = list(adapt(entry("WorkInTheDark/FairytaleQA"), rows, 1))
    assert golds(items, "noul") == {1: 2, 0: 2}


def test_maud_options_come_from_the_question_group():
    rows = [{"text": "Clause A.", "question": "Q1", "subquestion": "<NONE>", "answer": "Yes"},
            {"text": "Clause B.", "question": "Q1", "subquestion": "<NONE>", "answer": "No"},
            {"text": "Clause C.", "question": "Q2", "subquestion": "<NONE>", "answer": "Only one"}]
    items = list(adapt(entry("theatticusproject/maud"), rows, 1))
    assert len(items) == 2 and all(set(it["q"]["criteria"]) == {"Yes", "No"} for it in items)


def test_complaint_sources_ask_no_constant_yes_no():
    rows = [{"text": "My parcel is late.", "label": "delivery"}, {"text": "I was charged twice.", "label": "billing"}]
    items = list(adapt(entry("liri-uzh/cfpb-complaints-mini"), rows, 1))
    assert items and all(it["q"]["type"] == "choice" for it in items)


# --------------------------------------------------------------------------- constant filter and audit

def _noul_item(yes, task="moderation"):
    return {"state": "s", "q": {"type": "noul", "instructions": "?"}, "target": [0.0, 1.0] if yes else [1.0, 0.0],
            "_task": task}


def _choice_item(options, gold, task="topic"):
    return {"state": "s", "q": {"type": "choice", "instructions": "?", "criteria": {o: o for o in options}},
            "target": [1.0 if o == gold else 0.0 for o in options], "_task": task}


def test_constant_yes_no_is_dropped_per_task():
    items = [_noul_item(True) for _ in range(30)] + [_noul_item(i % 2 == 0, "emotion") for i in range(30)]
    kept = registry.drop_constant_yes_no(items)
    assert {it["_task"] for it in kept} == {"emotion"}
    few = [_noul_item(True) for _ in range(5)]
    assert registry.drop_constant_yes_no(few) == few  # too few to call it constant


def test_audit_flags():
    flags = audit.audit_items([_choice_item(["0", "1", "2"], "1")] * 3)
    assert "numeric-option" in flags
    flags = audit.audit_items([_choice_item(["[{'start': 0, 'label': 'x'}]", "other"], "other")])
    assert "serialized-option" in flags
    flags = audit.audit_items([_noul_item(True) for _ in range(40)])
    assert "constant-yes-no" in flags
    flags = audit.audit_items([_choice_item(["a", "b"], "a") for _ in range(40)])
    assert "constant-choice" in flags
    flags = audit.audit_items([_choice_item(["event1", "event4"], "event1")])
    assert "opaque-option" in flags
    flags = audit.audit_items([_choice_item(["a", "b", "c"], "a")])
    assert "opaque-option" in flags
    flags = audit.audit_items([_choice_item(["billing", "delivery"], "billing", task="sentiment")])
    assert "option-mismatch" in flags
    clean = [_choice_item(["negative", "positive"], g, task="sentiment") for g in ["negative", "positive"] * 20]
    assert audit.audit_items(clean) == {}


def test_allow_entries_have_reasons():
    for key, flags in audit.ALLOW.items():
        assert key in {registry.source_key(e) for e in registry.ENTRIES}, key
        for flag, reason in flags.items():
            assert flag in {"serialized-option", "numeric-option", "constant-yes-no", "constant-choice",
                            "option-mismatch", "opaque-option"} and len(reason) > 20


# --------------------------------------------------------------------------- review of #20

def test_fairytale_state_holds_the_passage():
    rows = [{"context": "Once upon a time a fox lived in the woods.", "question": "Where did the fox live?",
             "answers": {"text": ["in the woods"]}, "story_name": "fox"},
            {"context": "A king had three daughters.", "question": "How many daughters?",
             "answers": {"text": ["three"]}, "story_name": "king"}]
    items = list(adapt(entry("WorkInTheDark/FairytaleQA"), rows, 1))
    assert items and all(it["state"].startswith("context: ") and "\n\nquestion: " in it["state"] for it in items)
    assert any("fox lived in the woods" in it["state"] and "Where did the fox live?" in it["state"] for it in items)


def test_aegis1_category_tie_gives_yes_no_only():
    e = entry("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")
    tie = {"text": "tie", "labels_0": "Violence", "labels_1": "Harassment", "labels_2": "Violence",
           "labels_3": "Harassment", "labels_4": "Safe"}
    assert registry.labels_from(e, tie) == [registry.UNSAFE_NO_CATEGORY]
    clear = {"text": "clear", "labels_0": "Violence", "labels_1": "Violence", "labels_2": "Safe"}
    items = list(adapt(e, [tie, clear], 1))
    tie_items = [it for it in items if it["state"] == "tie"]
    assert [it["q"]["type"] for it in tie_items] == ["noul"] and tie_items[0]["target"] == [0.0, 1.0]
    for item in items:
        if item["q"]["type"] == "choice":
            assert registry.UNSAFE_NO_CATEGORY not in item["q"]["criteria"]


def test_prosocial_possibly_needs_caution_has_no_yes_no():
    rows = [{"context": "Maybe I skip work.", "safety_label": "__possibly_needs_caution__"},
            {"context": "I will bake bread.", "safety_label": "__casual__"}]
    items = list(adapt(entry("allenai/prosocial-dialog"), rows, 1))
    assert [it["state"] for it in items if it["q"]["type"] == "noul"] == ["I will bake bread."]


def test_emotion_probes_are_balanced():
    rows = [{"sentence": "text %d" % i, "emotion": e} for i, e in enumerate(["anger", "joy", "fear", "sadness"] * 30)]
    items = list(adapt(entry("shreyaspullehf/emotion-dataset-20-emotions"), rows, 1))
    yes = [it["target"][1] for it in items if it["q"]["type"] == "noul"]
    assert 0.35 < sum(yes) / len(yes) < 0.65


def test_audit_constant_score_and_dropped_yes_no():
    score = [{"state": "s", "q": {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]},
              "target": [1.0, 0.0, 0.0], "_task": "similarity"} for _ in range(30)]
    assert "constant-score" in audit.audit_items(score)
    skewed = [_noul_item(i < 28) for i in range(30)]  # 93 % yes: survives the 97 % drop, the audit flags it
    assert registry.drop_constant_yes_no(skewed) == skewed
    assert "constant-yes-no" in audit.audit_items(skewed)


def test_sample_parquet_delivers_the_limit_when_strata_are_thin(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"i": list(range(1000))}), path)
    rows = loaders.sample_parquet(path, 700)
    assert len(rows) == 700 and len({r["i"] for r in rows}) == 700
    assert rows == loaders.sample_parquet(path, 700)


def test_contiguous_parquet_keeps_runs_long_enough(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"i": list(range(10000))}), path)
    rows = [r["i"] for r in loaders.contiguous_parquet(path, 60)]
    runs = sum(1 for a, b in zip(rows, rows[1:]) if b != a + 1) + 1
    assert len(rows) == 60 and runs <= 2


# =========================================================================== weak-category sources
# Fact-check, emotion and topic sources added for the weakest held-out categories (source_part F in
# v6-keep.json). Emotion sources that opt in map their labels to emotion_taxonomy.

ROOT = Path(__file__).resolve().parents[3]
NEW_IDS = {
    "Horizon-Labs/multilingual-zeroshot-synthetic", "sociocom:naist-life-story", "agentlans/fact-or-opinion",
    "hheiden/us-congress-bill-policy-115_117", "nlp-waseda/e_gov",
}


def _golds(items, kind):
    out = collections.Counter()
    for item in items:
        if item["q"]["type"] != kind:
            continue
        if kind == "choice":
            out[list(item["q"]["criteria"])[item["target"].index(1.0)]] += 1
        else:
            out["yes" if item["target"][1] == 1.0 else "no"] += 1
    return out


# --------------------------------------------------------------------------- registry and held-out splits

def test_new_sources_are_enabled_with_evidence():
    raw = {e["id"]: e for e in json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if e.get("use") is True}
    for sid in NEW_IDS:
        e = raw[sid]
        assert e["licence_evidence"] and e["provenance"], sid
        # commercial use, no ShareAlike / NoDerivatives / NonCommercial
        assert not re.search(r"(?i)\b(nc|sa|nd)\b|non-?commercial|share-?alike|no-?deriv", e["licence"]), sid


def _eval_categories():
    if "eval_categories" in sys.modules:
        return sys.modules["eval_categories"]
    spec = importlib.util.spec_from_file_location("eval_categories", ROOT / "bench" / "eval_categories.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_categories"] = module
    spec.loader.exec_module(module)
    return module


def test_no_new_source_is_a_held_out_source():
    ec = _eval_categories()
    held = {src.entry for sources in ec.HELD_OUT.values() for src in sources}
    held |= {src.repo for sources in ec.HELD_OUT.values() for src in sources if src.repo}
    assert not NEW_IDS & held


def test_new_loaders_read_only_training_files():
    assert all("/train-" in name for name in loaders.EGOV["files"])  # never validation-/test-
    for name in loaders.LIFE_STORY_FILES:
        spec = loaders.PINNED[name]
        assert re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) and spec["url"].startswith("https://sociocom.naist.jp/")
    for spec in (loaders.HORIZON, loaders.CONGRESS, loaders.EGOV):
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"])  # a commit, not a moving branch


# --------------------------------------------------------------------------- emotion taxonomy

def test_taxonomy_classes_have_glosses_in_every_language():
    for label in emotion_taxonomy.CLASSES:
        for lang in LANGUAGES:
            assert label in GLOSSES["emotion"][lang], (label, lang)


def test_taxonomy_maps_synonyms_and_subordinates():
    assert emotion_taxonomy.to_class("Joy") == "joy"
    assert emotion_taxonomy.to_class("pride") == "joy"
    assert emotion_taxonomy.to_class("Relief") == "joy"
    assert emotion_taxonomy.to_class("Anxiety") == "fear"
    assert emotion_taxonomy.to_class("disappointment") == "sadness"
    assert emotion_taxonomy.to_class("Sadness") == "sadness"
    for label in emotion_taxonomy.UNMAPPED:
        assert emotion_taxonomy.to_class(label) is None
    assert set(emotion_taxonomy._MAP.values()) == set(emotion_taxonomy.CLASSES)
    assert emotion_taxonomy.map_labels(["joy", "pride"]) == ["joy"]
    assert emotion_taxonomy.map_labels(["joy", "gratitude"]) is None  # a partly mapped row is dropped


def test_every_label_of_the_opted_in_sources_is_mapped_or_documented():
    horizon = ["anger", "anxiety", "disappointment", "disgust", "fear", "gratitude", "joy", "love", "neutral",
               "pride", "relief", "sadness", "surprise"]  # the fixed taxonomy of the Horizon emotion task
    for label in horizon + loaders.LIFE_STORY_EMOTIONS + ["Trust"]:
        assert emotion_taxonomy.to_class(label) or label.lower() in emotion_taxonomy.UNMAPPED, label


def test_emotion_adapter_uses_the_taxonomy():
    e = entry("Horizon-Labs/multilingual-zeroshot-synthetic")
    rows = [{"text": "Ich habe die Prüfung bestanden, ich bin so stolz!", "emotion": "pride", "_v6_lang": "de"},
            {"text": "Der Zug fällt schon wieder aus.", "emotion": "anger", "_v6_lang": "de"},
            {"text": "Danke, dass du mir geholfen hast.", "emotion": "gratitude", "_v6_lang": "de"},
            {"text": "Das Paket ist heute angekommen.", "emotion": "neutral", "_v6_lang": "de"}]
    items = list(adapt(e, rows, 7))
    options = {o for it in items if it["q"]["type"] == "choice" for o in it["q"]["criteria"]}
    assert options == {"joy", "anger", "neutral"}  # gratitude has no class: its row is dropped
    assert not any("Danke" in it["state"] for it in items)
    assert _golds(items, "choice") == {"joy": 1, "anger": 1, "neutral": 1}
    first = next(it for it in items if it["q"]["type"] == "choice")
    assert first["q"]["criteria"]["joy"] in {describe("emotion", "joy", "de"), describe("emotion", "joy", "en")}


def test_sources_without_the_taxonomy_keep_their_labels():
    e = entry("JusteLeo/French-emotion")
    assert "emotion_taxonomy" not in e
    assert registry.emotion_labels(e, {"text": "x", "label": "colere"}) == ["colere"]


# --------------------------------------------------------------------------- emotion loaders

def test_horizon_rows_keep_generated_emotion_rows_in_model_languages():
    records = [
        {"text": "Estou tão orgulhosa de você.", "lang": "Portuguese", "task": "emotion", "kind": "tax_short",
         "gold_labels": ["pride"], "text_origin": "qwen-generated"},
        {"text": "Bu haberi duyunca çok şaşırdım.", "lang": "Turkish", "task": "emotion", "kind": "tax_short",
         "gold_labels": ["surprise"], "text_origin": "qwen-generated"},
        {"text": "invented label set", "lang": "Italian", "task": "emotion", "kind": "short",
         "gold_labels": ["wistful"], "text_origin": "qwen-generated"},
        {"text": "web passage", "lang": "Dutch", "task": "emotion", "kind": "tax_short",
         "gold_labels": ["joy"], "text_origin": "fineweb"},
        {"text": "not a model language", "lang": "Swahili", "task": "emotion", "kind": "tax_short",
         "gold_labels": ["joy"], "text_origin": "qwen-generated"},
        {"text": "two labels", "lang": "Russian", "task": "emotion", "kind": "tax_short",
         "gold_labels": ["joy", "love"], "text_origin": "qwen-generated"},
        {"text": "a topic row", "lang": "Arabic", "task": "customer message topic", "kind": "tax_short",
         "gold_labels": ["cancellation"], "text_origin": "qwen-generated"},
    ]
    rows = loaders.horizon_emotion_rows(records)
    assert [(r["_v6_lang"], r["emotion"]) for r in rows] == [("pt", "pride"), ("tr", "surprise")]


def _xlsx(path, rows):
    """A minimal xlsx: shared strings for the header, inline strings and an empty cell for the body."""
    strings = rows[0]
    shared = "".join("<si><t>%s</t></si>" % s for s in strings)
    body = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(row):
            ref = "%s%d" % (chr(65 + c), r)
            if value is None:
                continue
            if r == 1:
                cells.append('<c r="%s" t="s"><v>%d</v></c>' % (ref, strings.index(value)))
            else:
                cells.append('<c r="%s" t="inlineStr"><is><t>%s</t></is></c>' % (ref, value))
        body.append('<row r="%d">%s</row>' % (r, "".join(cells)))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", "<sst %s>%s</sst>" % (ns, shared))
        zf.writestr("xl/worksheets/sheet1.xml", "<worksheet %s><sheetData>%s</sheetData></worksheet>" % (ns, "".join(body)))
    return path


LIFE_HEADER = ["Gender", "Age", "Sadness", "Anxiety", "Anger", "Disgust", "Trust", "Surprise", "Joy"]


def test_xlsx_reader_reads_shared_and_inline_strings(tmp_path):
    path = _xlsx(tmp_path / "s.xlsx", [LIFE_HEADER, ["male", "40", "飼っていた猫が亡くなって悲しかった。", None]])
    rows = list(loaders.xlsx_rows(path))
    assert rows[0] == LIFE_HEADER
    assert rows[1][:3] == ["male", "40", "飼っていた猫が亡くなって悲しかった。"]


def test_life_story_rows_filter_non_answers_and_ambiguous_texts(tmp_path):
    same = "地震のニュースを見て家族のことが心配になった。"
    table = [LIFE_HEADER,
             ["female", "30", "***飼っていた猫が亡くなって悲しかった。", same, "特になし", "家族",
              "友人はいつも約束を守ってくれるので信頼している。", "宝くじで一万円が当たって驚いた。", "なし"],
             ["male", "50", same, "来月の手術がうまくいくか不安で仕方がない。", "電車で割り込みをされて本当に腹が立った。",
              None, None, None, "孫と一緒に公園で遊んでとても楽しかった。"]]
    rows = loaders.life_story_rows([table])
    got = {r["text"]: r["emotion"] for r in rows}
    assert got == {
        "飼っていた猫が亡くなって悲しかった。": "Sadness",   # leading *** removed
        "宝くじで一万円が当たって驚いた。": "Surprise",
        "来月の手術がうまくいくか不安で仕方がない。": "Anxiety",
        "電車で割り込みをされて本当に腹が立った。": "Anger",
        "孫と一緒に公園で遊んでとても楽しかった。": "Joy",
    }  # "特になし", "家族", "なし": no episode; Trust: not read; `same`: under two emotions
    assert all(r["_v6_lang"] == "ja" for r in rows)
    items = list(adapt(entry("sociocom:naist-life-story"), rows, 3))
    assert set(_golds(items, "choice")) == {"sadness", "surprise", "fear", "anger", "joy"}


# --------------------------------------------------------------------------- fact-check

def test_fact_opinion_keeps_deepseek_rows_only():
    records = [
        {"text": "The Nile is in Africa.", "label": "Fact", "language": "en", "source": "DeepSeek"},
        {"text": "Le jazz est la meilleure musique.", "label": "Opinion", "language": "fr", "source": "DeepSeek"},
        {"text": "Brasil é o maior país da América do Sul.", "label": "Fact", "language": "pt-br", "source": "DeepSeek"},
        {"text": "hosted model text", "label": "Fact", "language": "en", "source": "ChatGPT"},
        {"text": "hosted model text", "label": "Fact", "language": "en", "source": "Claude Sonnet 4"},
        {"text": "hosted model text", "label": "Fact", "language": "en", "source": "Gemini 2.5 Flash"},
        {"text": "hosted model text", "label": "Fact", "language": "en", "source": "Le Chat"},
        {"text": "not a model language", "label": "Fact", "language": "sw", "source": "DeepSeek"},
    ]
    rows = loaders.fact_opinion_rows(records)
    assert [(r["_v6_lang"], r["label"]) for r in rows] == [("en", "Fact"), ("fr", "Opinion"), ("pt", "Fact")]


def test_claim_type_adapter_asks_both_questions_with_both_answers():
    rows = [{"text": "Water boils at 100 degrees Celsius at sea level.", "label": "Fact", "_v6_lang": "en"},
            {"text": "Rock music is better than pop music.", "label": "Opinion", "_v6_lang": "en"},
            {"text": "Paris is the capital of France and the most beautiful city.", "label": "Both", "_v6_lang": "en"},
            {"text": "Could you send me the report by Friday?", "label": "Neither", "_v6_lang": "en"},
            {"text": "no gold", "label": "", "_v6_lang": "en"}]
    items = list(adapt(entry("agentlans/fact-or-opinion"), rows, 5))
    assert _golds(items, "choice") == {"fact": 1, "opinion": 1, "fact and opinion": 1, "neither": 1}
    noul = {it["state"]: it["target"].index(1.0) for it in items if it["q"]["type"] == "noul"}
    assert noul == {rows[0]["text"]: 1, rows[1]["text"]: 0, rows[2]["text"]: 1, rows[3]["text"]: 0}
    assert all(it["_task"] == "claim_detection" for it in items)
    for item in items:
        if item["q"]["type"] == "choice":
            assert set(item["q"]["criteria"]) == set(registry.CLAIM_TYPES.values())


def test_claim_detection_glosses_cover_every_option_and_language():
    for label in registry.CLAIM_TYPES.values():
        for lang in LANGUAGES:
            assert GLOSSES["claim_detection"][lang][label].strip()


# --------------------------------------------------------------------------- topic

def test_congress_rows_drop_non_subject_areas_and_cut_the_summary():
    records = [
        {"title": "A bill to improve rural hospitals.", "summary": "S" * 5000, "policy_area": "Health"},
        {"title": "For the relief of John Doe.", "summary": "Grants residency.", "policy_area": "Private Legislation"},
        {"title": "A bill on archives.", "summary": "Archives.", "policy_area": "Social Sciences and History"},
        {"title": "A bill without a summary.", "summary": "", "policy_area": "Taxation"},
    ]
    rows = loaders.congress_rows(records)
    assert [r["policy_area"] for r in rows] == ["Health"]
    assert len(rows[0]["summary"]) == loaders.DOC_CHARS
    rows.append({"title": "A bill to cut tariffs.", "summary": "Reduces duties.", "policy_area": "Foreign Trade and International Finance",
                 "_v6_lang": "en"})
    items = list(adapt(entry("hheiden/us-congress-bill-policy-115_117"), rows, 1))
    assert _golds(items, "choice") == {"Health": 1, "Foreign Trade and International Finance": 1}
    assert all(it["_task"] == "topic" and it["state"].startswith("title: ") for it in items)


def test_egov_rows_name_the_law_field():
    categories = {"2": "刑事", "30": "厚生"}
    records = [{"text": "決闘罪ニ関スル件\n第一条 " + "条" * 3000, "metadata": {"category_id": 2}},
               {"text": "医療法\n第一条", "metadata": "{'category_id': 30, 'Era': 'Showa'}"},
               {"text": "unknown field", "metadata": {"category_id": 99}}]
    rows = loaders.egov_rows(records, categories)
    assert [r["category"] for r in rows] == ["刑事", "厚生"]
    assert len(rows[0]["text"]) == loaders.DOC_CHARS
    items = list(adapt(entry("nlp-waseda/e_gov"), rows, 1))
    assert _golds(items, "choice") == {"刑事": 1, "厚生": 1}
    assert all(it["_task"] == "topic" and it["lang"] == "ja" for it in items)


# =========================================================================== Part G sources
# Sources added for the categories where 0.7.0 trails Qwen3-8B zero-shot (source_part G in v6-keep.json).

PART_G_IDS = {"naeyn/nobody-pii-synth-de", "Powpowpow23/ru-pii-ner-data"}


def test_part_g_sources_are_enabled_with_evidence_and_pinned():
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    enabled = {e["id"]: e for e in raw if e.get("use") is True}
    for sid in PART_G_IDS:
        e = enabled[sid]
        assert e["source_part"] == "G" and e["licence_evidence"] and e["provenance"], sid
        assert not re.search(r"(?i)\b(nc|sa|nd)\b|non-?commercial|share-?alike|no-?deriv", e["licence"]), sid
        assert re.fullmatch(r"[0-9a-f]{40}", e["pinned_commit"]), sid
    assert loaders.NOBODY_PII["revision"] == enabled["naeyn/nobody-pii-synth-de"]["pinned_commit"]
    assert loaders.NOBODY_PII["file"] == "data/train.parquet"  # never validation or test
    assert loaders.RU_PII["revision"] == enabled["Powpowpow23/ru-pii-ner-data"]["pinned_commit"]
    assert loaders.RU_PII["file"] == "data/train.parquet"  # validation was the author's development split
    ru = enabled["Powpowpow23/ru-pii-ner-data"]
    assert ru["languages"] == ["ru"] and ru["label_field"].startswith("entities")
    assert ru["text_fields"] == ["text"] and ru["test_overlap_note"]
    assert ru["licence"].startswith("apache-2.0") and "DeepSeek" in ru["provenance"]
    assert all(ru["pinned_commit"] in url for url in ru["licence_evidence"][:2])
    assert "104,111" in ru["provenance"] and "2,400" in ru["provenance"]
    assert "0 exact matches" in ru["test_overlap_note"] and "29,767" in ru["test_overlap_note"]
    rejected = [e for e in raw if e.get("source_part") == "G" and e.get("use") is False]
    assert rejected and all(e["excluded_reason"] for e in rejected)


def test_no_part_g_source_is_a_held_out_source():
    ec = _eval_categories()
    held = {src.entry for sources in ec.HELD_OUT.values() for src in sources}
    held |= {src.repo for sources in ec.HELD_OUT.values() for src in sources if src.repo}
    assert not PART_G_IDS & held


def test_nobody_pii_rows_keep_types_and_retag_pii_free_rows():
    records = [
        {"text": "Rechnung an Jan Weber, IBAN DE35703188546038719758.", "lang": "de",
         "ner": "[{'start': 2, 'end': 3, 'label': 'person'}, {'start': 5, 'end': 5, 'label': 'iban'}]"},
        {"text": "The report is ready and the numbers are final.", "lang": "de", "ner": "[]"},
        {"text": "The quarterly report shows a 12 percent increase in output.", "lang": "de", "ner": "[]"},
        {"text": "Der Geschäftsführer hält auf der Messe eine Keynote und die Präsentation ist fertig.",
         "lang": "de", "ner": "[]"},
        {"text": "Het rapport is klaar en de cijfers van dit jaar zijn definitief.", "lang": "nl",
         "ner": []},
        {"text": "", "lang": "de", "ner": "[]"},
        {"text": "broken", "lang": "de", "ner": "[{'start': 1,"},
        {"text": "Bonjour, voici la facture.", "lang": "fr", "ner": "[{'start': 0, 'end': 0, 'label': 'person'}]"},
    ]
    rows = loaders.nobody_pii_rows(records)
    assert [(r["_v6_lang"], [x["label"] for x in r["entities"]]) for r in rows] == [
        ("de", ["person", "iban"]), ("en", []), ("de", []), ("nl", [])]  # one English hint only: dropped


def test_nobody_pii_adapter_asks_both_answers():
    e = entry("naeyn/nobody-pii-synth-de")
    rows = [{"text": "Kontakt: jan.weber@example.org, Tel. 0301234567, Jan Weber", "_v6_lang": "de",
             "entities": [{"label": "email"}, {"label": "phone number"}, {"label": "person"}]},
            {"text": "Die Lieferung verschiebt sich auf nächste Woche.", "_v6_lang": "de", "entities": []},
            {"text": "Please send the IBAN NL96BRDJ6008444537 to Evie Jansen.", "_v6_lang": "en",
             "entities": [{"label": "iban"}, {"label": "person"}]},
            {"text": "Het rapport is klaar en ligt ter inzage.", "_v6_lang": "nl", "entities": []}]
    items = list(adapt(e, rows * 6, 7))
    pii = [it for it in items if it.get("_task") == "pii" and it["q"]["type"] == "noul"]
    assert _golds(pii, "noul") == {"yes": 12, "no": 12}
    assert {it["lang"] for it in pii} == {"de", "en", "nl"}
    no_pii = {it["state"] for it in pii if it["target"][0] == 1.0}
    assert no_pii == {"Die Lieferung verschiebt sich auf nächste Woche.", "Het rapport is klaar en ligt ter inzage."}
    probes = [it for it in items if it.get("_task") == "pii_type"]
    assert probes and {"yes", "no"} <= set(_golds(probes, "noul"))


def test_ru_pii_rows_map_all_types_and_keep_supervision_and_negative_family():
    assert len(loaders.RU_PII_TYPES) == 25
    records = [
        {"text": "ФИО: Анна Петрова, телефон +7 000 111-22-33.",
         "entities": [{"type": "FULL_NAME", "start": 5, "end": 18},
                      {"type": "PHONE", "start": 28, "end": 44}],
         "supervised_types": ["FULL_NAME", "PHONE", "EMAIL"], "source_families": ["scenarios"]},
        {"text": "Сегодня отделение работает до шести.", "entities": [],
         "supervised_types": list(loaders.RU_PII_TYPES), "source_families": ["negative_examples"]},
        {"text": "", "entities": [], "supervised_types": ["EMAIL"], "source_families": []},
    ]
    rows = loaders.ru_pii_rows(records)
    assert rows == [
        {"text": records[0]["text"],
         "entities": [{"start": 5, "end": 18, "label": "person"},
                      {"start": 28, "end": 44, "label": "phone number"}],
         "_v6_lang": "ru", "_v6_pii_supervised": ["person", "phone number", "email"],
         "_v6_pii_negative": False},
        {"text": records[1]["text"], "entities": [], "_v6_lang": "ru",
         "_v6_pii_supervised": list(loaders.RU_PII_TYPES.values()), "_v6_pii_negative": True},
    ]


def test_ru_pii_adapter_respects_supervised_types_and_explicit_negatives():
    e = entry("Powpowpow23/ru-pii-ner-data")
    records = [
        {"text": "ФИО: Анна Петрова.",
         "entities": [{"type": "FULL_NAME", "start": 5, "end": 18}],
         "supervised_types": ["FULL_NAME"], "source_families": ["scenarios"]},
        {"text": "Напишите на user1@example.invalid.",
         "entities": [{"type": "EMAIL", "start": 12, "end": 33}],
         "supervised_types": ["EMAIL", "FULL_NAME"], "source_families": ["scenarios"]},
        {"text": "В сообщении нет целевых персональных данных.", "entities": [],
         "supervised_types": list(loaders.RU_PII_TYPES), "source_families": ["negative_examples"]},
        {"text": "Пустая разметка другого семейства.", "entities": [],
         "supervised_types": ["EMAIL"], "source_families": ["addresses"]},
    ]
    rows = loaders.ru_pii_rows(records)
    raw = list(registry.ADAPTERS[registry.source_key(e)](e, rows, 7))
    broad = [it for it in raw if it.get("_task") == "pii"]
    assert {it["state"] for it in broad} == {records[0]["text"], records[1]["text"], records[2]["text"]}
    assert _golds(broad, "noul") == {"yes": 2, "no": 1}
    probes = [it for it in raw if it.get("_task") == "pii_type"]
    # The FULL_NAME-only row cannot be asked about the email type present elsewhere in the sample.
    assert not any(it["state"] == records[0]["text"] for it in probes)
    assert probes and all(it["lang"] == "ru" for it in probes)
