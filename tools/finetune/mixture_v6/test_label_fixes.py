"""Offline regression tests for the v6 label fixes and the content audit (no network).

    .venv-train/bin/python -m pytest -q tools/finetune/mixture_v6/test_label_fixes.py
"""

import collections

import pytest

from tools.finetune.mixture_v6 import audit, loaders, registry
from tools.finetune.mixture_v6.registry import adapt


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
