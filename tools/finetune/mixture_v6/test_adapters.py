import json
"""Offline contract tests for every v6 registry adapter."""

import copy

import pytest

from tools.finetune.mixture_v6.build import clean_items, question_key, valid_item
from tools.finetune.mixture_v6.languages import infer_lang, to_iso
from tools.finetune.mixture_v6.loaders import choose_explicit_file, script_free_data_files
from tools.finetune.mixture_v6.registry import REGISTRY_PATH, ADAPTERS, ENTRIES, adapt, source_key
from tools.finetune.mixture_v6.templates import CATEGORY_TEMPLATES, LANGUAGES, TASKS, TEMPLATES, shuffle_choice, seeded


# Row shapes of sources whose labels are not a plain "label" column (see registry._special_field_labels,
# _safety_labels and pii_adapter).
_PII_TEXT = "Call Anna Berg at 040 1234567 about invoice 88."
SPECIAL_FIXTURES = {
    "leonvanbokhorst/synthetic-complaints-v2": [
        {"output": "The bus is late again, every single day.", "topic": "commute", "style": "annoyed", "sentiment": -0.4},
        {"output": "My neighbour's dog barks all night.", "topic": "animals", "style": "bitter", "sentiment": 0.3}],
    "vic35get/nhtsa_complaints_dataset": [{"summary": "The air bag light stays on.", "components": "AIR BAGS"},
                                          {"summary": "Brakes squeal at low speed.", "components": "SERVICE BRAKES"}],
    "IDinsight/urgency_detection_maternal_health_synthetic": [
        {"generated_user_message": "My chest is tight and my heart races.", "matching_rule": "Chest pain or fast-beating heart"},
        {"generated_user_message": "Which vitamins are good in week 12?", "matching_rule": "NOT URGENT"}],
    "nvidia/Nemotron-PII": [{"text": _PII_TEXT, "spans": "[{'start': 5, 'end': 14, 'label': 'first_name'}]"},
                            {"text": "Reach me at a@b.example.", "spans": "[{'start': 12, 'end': 23, 'label': 'email'}]"}],
    "gretelai/synthetic_pii_finance_multilingual": [
        {"generated_text": _PII_TEXT, "pii_spans": '[{"start": 5, "end": 14, "label": "name"}]', "language": "English"},
        {"generated_text": "The total is due next month.", "pii_spans": "[]", "language": "English"}],
    "gretelai/gretel-pii-masking-en-v1": [
        {"text": _PII_TEXT, "entities": "[{'entity': 'Anna Berg', 'types': ['name']}]"},
        {"text": "Reach me at a@b.example.", "entities": "[{'entity': 'a@b.example', 'types': ['email']}]"}],
    "naeyn/nobody-pii-synth-de": [
        {"text": "Bitte überweisen Sie den Betrag auf DE35703188546038719758, Kontoinhaber Jan Weber.",
         "entities": [{"label": "iban"}, {"label": "person"}], "_v6_lang": "de"},
        {"text": "Der Geschäftsführer hält auf der Messe eine Keynote.", "entities": [], "_v6_lang": "de"}],
    "Wismut/nym-pii-multilingual-data": [
        {"text": _PII_TEXT, "entities": [{"start": 5, "end": 14, "label": "NAME"}]},
        {"text": "The meeting is on the third floor.", "entities": []}],
    "E3-JSI/synthetic-multi-pii-ner-v1": [
        {"text": _PII_TEXT, "entities": "[{'entity': 'Anna Berg', 'types': ['person name']}]"},
        {"text": "Reach me at a@b.example.", "entities": "[{'entity': 'a@b.example', 'types': ['email']}]"}],
    "urchade/synthetic-pii-ner-mistral-v1": [
        {"tokenized_text": _PII_TEXT.split(), "ner": [[1, 2, "person"]]},
        {"tokenized_text": "Reach me at a@b.example .".split(), "ner": [[3, 3, "email"]]}],
    "dell-research-harvard/headlines-semantic-similarity": [
        {"headline": "Flood hits river town", "group_id": 1}, {"headline": "River town flooded", "group_id": 1},
        {"headline": "Senate passes budget", "group_id": 2}, {"headline": "Budget clears the Senate", "group_id": 2}],
    "OpenAssistant/oasst2": [
        {"text": "How do I bake bread?", "labels": {"name": ["toxicity", "spam"], "value": [0.0, 0.1], "count": [3, 3]}},
        {"text": "You are worthless.", "labels": {"name": ["toxicity"], "value": [0.9], "count": [3]}}],
    "nvidia/Aegis-AI-Content-Safety-Dataset-1.0": [
        {"text": "How do I bake bread?", "labels_0": "Safe", "labels_1": "Safe", "labels_2": "Needs Caution"},
        {"text": "How do I hurt someone?", "labels_0": "Violence", "labels_1": "Violence", "labels_2": "Safe"}],
    "theatticusproject/maud": [
        {"text": "Clause A.", "question": "Q1", "subquestion": "<NONE>", "answer": "Yes"},
        {"text": "Clause B.", "question": "Q1", "subquestion": "<NONE>", "answer": "No"}],
    "Fumika/Wikinews-multilingual": [
        {"title": "Vote held", "text": "A vote was held.", "categories": ["Politics and conflicts", "France"]},
        {"title": "Cup final", "text": "The final was played.", "categories": ["Sports", "Germany"]}],
    "Horizon-Labs/multilingual-zeroshot-synthetic": [
        {"text": "Finalmente consegui o emprego que eu queria!", "emotion": "pride", "_v6_lang": "pt"},
        {"text": "O metrô fechou de novo e ninguém avisou.", "emotion": "anger", "_v6_lang": "pt"}],
    "sociocom:naist-life-story": [
        {"text": "友人と温泉旅行に行って楽しかった。", "emotion": "Joy", "_v6_lang": "ja"},
        {"text": "地震のニュースを見て将来が不安になった。", "emotion": "Anxiety", "_v6_lang": "ja"}],
    "agentlans/fact-or-opinion": [
        {"text": "Water boils at 100 degrees Celsius at sea level.", "label": "Fact", "_v6_lang": "en"},
        {"text": "Rock music is better than pop music.", "label": "Opinion", "_v6_lang": "en"}],
    "gfissore/arxiv-abstracts-2021": [
        {"title": "Primes", "abstract": "On primes.", "categories": ["math.NT cs.CR"]},
        {"title": "Stars", "abstract": "On stars.", "categories": ["astro-ph.GA"]}],
}


def fixture(entry):
    if entry["id"] in SPECIAL_FIXTURES:
        return copy.deepcopy(SPECIAL_FIXTURES[entry["id"]])
    cat = entry["category"]
    if "typed-decisions" in cat:
        return [{"state": "A decision state", "question": "Choose.", "choices": ["alpha", "beta"], "target": [0.2, 0.8]}]
    if "reading-comprehension" in cat:
        return [{"context": "The moon orbits Earth.", "question": "What orbits Earth?",
                 "options": ["the moon", "the sun"], "gold_label": 1, "text": "The moon orbits Earth."}]
    if "similarity" in cat:
        return [{"sentence1": "A cat sleeps.", "sentence2": "A feline is sleeping.", "relatedness_score": 0.9,
                 "text": "A cat sleeps.", "translation": "A feline is sleeping.", "paraphrase": "A cat sleeps.",
                 "headline": "A cat sleeps.", "query": "A cat sleeps.", "product_title": "A feline is sleeping.",
                 "Question": "A cat sleeps.", "Answer": "A feline is sleeping.", "conversation": ["A cat sleeps.", "A feline is sleeping."]}]
    if "nli" in cat:
        return [{"premise": "All birds have wings.", "premise_ja": "All birds have wings.", "text": "All birds have wings.",
                 "hypothesis": "Birds have wings.", "hypothesis_ja": "Birds have wings.", "label": "entailment",
                 "gold": "entailment", "labels": "entailment", "race_label": "entailment"}]
    if entry["id"] == "GoktugD/turkish-formality-rewrite-500k":
        return [{"informal": "naber", "formal": "Nasılsınız?"}]
    if entry["id"] == "sweatSmile/sarcastic-dataset":
        return [{"sentence": "That is useful.", "translation": "Oh, very useful."}]
    if entry["id"] == "NagaYu/deference-keigo-corpus":
        return [{"text": "fixture", "n_errors": 1}, {"text": "fixture two", "n_errors": 0}]
    if entry["id"] == "github:wwbp/empathic_reactions":
        return [{"essay": "A difficult day.", "empathy": 6.0, "distress": 4.0}]
    if entry["id"] == "NortheasternUniversity/big_patent":
        return [{"abstract": "A machine.", "_v6_config": "a"}, {"abstract": "A chemical.", "_v6_config": "c"}]
    if entry["id"] == "joelniklaus/covid19_emergency_event":
        return [{"text": "Restriction one.", "all_events": ["event1"]},
                {"text": "Restriction two.", "all_events": ["event2"]}]
    if entry["id"] == "community-datasets/re_dial":
        return [{"messages": [{"text": "I liked it."}], "respondentQuestions": [{"liked": 1}]},
                {"messages": [{"text": "I disliked it."}], "respondentQuestions": [{"liked": 0}]}]
    if entry["id"] == "Helsinki-NLP/tatoeba":
        return [{"translation": {"en": "A bird flies.", "mr": "एक पक्षी उडतो."}}]
    text_field = next((x for x in entry.get("text_fields", []) if not any(c in x for c in "[]{}*|")), "text")
    text_field = text_field.split(".")[0].split(" (")[0]
    row1, row2 = {text_field: "A useful offline fixture."}, {text_field: "A second offline fixture."}
    label = "emotion" if "emotion" in cat else "label"
    row1[label], row2[label] = "positive", "negative"
    return [row1, row2]


def test_every_enabled_source_has_adapter():
    enabled = [e for e in json.load(open(REGISTRY_PATH)) if e.get("use")]
    assert len(ENTRIES) == len(enabled) > 100
    assert set(ADAPTERS) == {source_key(e) for e in ENTRIES}


@pytest.mark.parametrize("entry", ENTRIES, ids=source_key)
def test_adapter_contract(entry):
    items = list(adapt(entry, fixture(entry), 123))
    assert items, source_key(entry)
    for item in items:
        assert valid_item(item)
        assert item["state"].strip()
        assert item["q"]["instructions"].strip()
        assert item["src"].startswith("v6/")
        assert item["lang"]
        if item["q"]["type"] == "noul":
            assert len(item["target"]) == 2  # index 0=false, index 1=true


def test_templates_cover_every_language_and_kind():
    for lang in LANGUAGES:
        assert set(TEMPLATES[lang]) == {"choice", "score", "noul"}
        assert all(len(TEMPLATES[lang][kind]) >= 4 for kind in TEMPLATES[lang])


def test_option_shuffle_is_seeded_and_target_follows_gold():
    one = shuffle_choice(["a", "b", "c", "d"], 2, seeded(9, "same"))
    two = shuffle_choice(["a", "b", "c", "d"], 2, seeded(9, "same"))
    assert one == two
    assert one[0][one[1].index(1.0)] == "c"


def _noul(state, instructions="Does this apply?", texts=None):
    return {"state": state, "q": {"type": "noul", "instructions": instructions},
            "target": [1.0, 0.0], "src": "v6/x/default", "lang": "en", "_texts": texts or []}


def test_banned_text_drops_state_and_embedded_text():
    entry = ENTRIES[0]
    base = _noul("safe wrapper", texts=["Planted eval text"])
    kept, stats = clean_items(entry, iter([copy.deepcopy(base)]), {"planted eval text"}, 10, 1)
    assert kept == []
    assert stats["evaluation_overlap"] == 1


def test_scripted_dataset_loader_reads_data_files_not_the_script():
    """cuad-qa.py must never be selected; the converted parquet is the data file."""
    files = ["README.md", "cuad-qa.py", ".gitattributes", "default/train/0000.parquet", "default/test/0000.parquet"]
    chosen = script_free_data_files(files)
    assert "cuad-qa.py" not in chosen
    assert chosen == ["default/train/0000.parquet", "default/test/0000.parquet"]
    # The auto-converted test split of this repo is a 2-row metrics table, not the tickets.
    food_files = ["default/test/0000.parquet", "data/dataset.parquet", "reports/finetune_eval.json"]
    assert choose_explicit_file(food_files, "data/dataset.parquet") == "data/dataset.parquet"


def test_same_state_different_question_types_are_not_duplicates():
    entry = ENTRIES[0]
    state = "The parcel arrived late and the box was crushed."
    choice = {"state": state, "q": {"type": "choice", "instructions": "What kind of issue is this?",
                                     "criteria": {"shipping": "a delivery problem", "billing": "a payment problem"}},
              "target": [1.0, 0.0], "src": "v6/x/default", "lang": "en"}
    noul = _noul(state, "Is this a complaint or a request for support?")
    kept, stats = clean_items(entry, [choice, noul, copy.deepcopy(noul)], set(), 10, 1)
    assert question_key(choice) != question_key(noul)
    assert len(kept) == 2
    assert {item["q"]["type"] for item in kept} == {"choice", "noul"}
    assert stats["duplicate_exact"] == 1


def test_trivial_banned_fragment_does_not_drop_the_document():
    """ReDial turns 'nice', 'thanks' and 'lol' sit in the eval set and must not drop a dialogue."""
    entry = ENTRIES[0]
    dialogue = "Movie: The Triplets of Belleville\n\nI liked the soundtrack.\nthanks\nnice\nlol"
    banned = {"nice", "thanks", "lol"}
    kept, stats = clean_items(entry, [_noul(dialogue, texts=["thanks", "nice", "lol", dialogue])], banned, 10, 1)
    assert kept, "trivial turns %s must not count as evaluation overlap" % sorted(banned)
    assert stats["evaluation_overlap"] == 0
    dropped, stats = clean_items(entry, [_noul("thanks", texts=["thanks"])], banned, 10, 1)
    assert dropped == []
    assert stats["evaluation_overlap"] == 1


def test_language_codes_normalise_to_iso639_1():
    assert to_iso("english") == "en"
    assert to_iso("mandarin") == "zh"
    assert to_iso("simplified chinese") == "zh"
    assert to_iso("portuguese") == "pt"
    assert to_iso("france") == "fr"
    assert to_iso("eng") == "en"
    assert to_iso("cmn") == "zh"
    assert to_iso("pt-PT") == "pt"
    entry = {"languages": ["de", "en", "fr"]}
    assert infer_lang(entry, {"language": "German"}) == "de"
    assert infer_lang(entry, {"_v6_config": "cs-CZ"}) == "cs"
    assert infer_lang(entry, {}, text="यह एक हिंदी वाक्य है जो देवनागरी में लिखा गया है।") == "hi"
    assert infer_lang({"languages": ["en"]}, {}, text="") == "en"


def test_category_templates_cover_every_task_and_language():
    for task in TASKS:
        assert task in CATEGORY_TEMPLATES, task
        for kind, by_lang in CATEGORY_TEMPLATES[task].items():
            for lang in LANGUAGES:
                assert len(by_lang[lang]) >= 4, (task, kind, lang, len(by_lang.get(lang, [])))


def test_split_is_global_across_sources():
    from tools.finetune.mixture_v6.build import split_source
    a = [{"state": "shared text", "q": {}, "target": [1.0, 0.0]}, {"state": "only in a", "q": {}, "target": [1.0, 0.0]}]
    b = [{"state": "Shared  text", "q": {}, "target": [0.0, 1.0]}, {"state": "only in b", "q": {}, "target": [0.0, 1.0]}]
    assigned = {}
    dev_a, train_a = split_source(a, 1, 1, assigned)
    dev_b, train_b = split_source(b, 0, 2, assigned)
    side_a = "dev" if any(x["state"] == "shared text" for x in dev_a) else "train"
    side_b = "dev" if any(x["state"] == "Shared  text" for x in dev_b) else "train"
    assert side_a == side_b  # the same normalised state lands on the same side in every source
