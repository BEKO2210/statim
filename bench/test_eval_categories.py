"""Offline tests for bench/eval_categories.py (no network, no server).

    .venv-train/bin/python -m pytest -q bench/test_eval_categories.py
"""
import collections
import copy
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_categories as ec  # noqa: E402

from tools.finetune import gate  # noqa: E402
from tools.finetune.mixture_v6 import eval_texts, loaders, registry  # noqa: E402
from tools.finetune.mixture_v6.build import clean_items  # noqa: E402
from tools.finetune.mixture_v6.loaders import _DISPATCH, PINNED, _split_candidates  # noqa: E402
from tools.finetune.mixture_v6.templates import CATEGORY_TEMPLATES, TEMPLATES  # noqa: E402

ALL_SOURCES = [(suite, src) for suite, srcs in ec.HELD_OUT.items() for src in srcs]


# --------------------------------------------------------------------------- held-out split

# Registry entries with a custom v6 loader: the loader reads fixed files, checked by source text below.
CUSTOM_LOADERS = {
    "boun-tabi/nli_tr": (loaders._load_nli_tr, "multinli_tr/train/"),
    "theatticusproject/cuad-qa": (loaders._load_cuad, "default/train/"),
    "zenodo:3609356 (ClaimBuster)": (loaders._load_claimbuster, "zenodo:3609356 (ClaimBuster)"),
}


@pytest.mark.parametrize("suite,src", ALL_SOURCES, ids=[s.key for _, s in ALL_SOURCES])
def test_split_is_not_a_training_split(suite, src):
    entry = ec.entry_for(src)  # raises for a source that is not an enabled registry entry
    assert src.split != "train"
    if src.entry in _DISPATCH:
        assert src.entry in CUSTOM_LOADERS, "add the custom loader of %s to CUSTOM_LOADERS" % src.entry
        func, reads = CUSTOM_LOADERS[src.entry]
        code = inspect.getsource(func)
        assert reads in code
        assert "%s/%s/" % (src.configs[0], src.split) not in code
        if src.loader == "claimbuster":
            # same Zenodo record, a different file than the one training pins
            assert PINNED[reads]["filename"] == "groundtruth.csv" != ec.CLAIMBUSTER_CROWD["filename"]
            assert ec.CLAIMBUSTER_CROWD["url"].split("/files/")[0] == PINNED[reads]["url"].split("/files/")[0]
        return
    wanted = entry.get("split", "train")
    assert "all" not in wanted.lower().split(), "the mixture reads every split of %s" % src.entry
    # The generic loader picks splits with _split_candidates; the held-out split must never be one.
    trained = _split_candidates(wanted, ["train", "validation", "test", src.split])
    assert src.split not in trained, "%s: %s is a training split" % (src.entry, src.split)


def test_split_check_catches_a_training_split():
    """The check above would flag a source whose registry split includes the held-out one."""
    assert "test" in _split_candidates("validation+test", ["train", "validation", "test"])
    assert "test" in _split_candidates("test", ["test"])
    assert _split_candidates("train", ["train", "validation_matched"]) == ["train"]


def test_every_suite_has_sources_and_known_kinds():
    assert len(ec.SUITES) >= 14
    for suite, srcs in ec.HELD_OUT.items():
        assert srcs, suite
        for src in srcs:
            assert set(src.kinds) <= {"choice", "noul", "score"}
            assert not src.prep or src.prep in ec.PREP
            assert not src.loader or src.loader in ec.LOADERS
            assert not src.adapter or callable(getattr(registry, src.adapter))


def test_source_keys_are_unique():
    keys = [src.key for _, src in ALL_SOURCES]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- canonical adapters

def _poem():
    return ec.Source("google-research-datasets/poem_sentiment", "default", "test")


def _poem_rows():
    return [{"verse_text": "the sun is warm and kind", "label": "positive"},
            {"verse_text": "a cold and bitter night", "label": "negative"},
            {"verse_text": "the river runs to the sea", "label": "no_impact"},
            {"verse_text": "joy and grief in one breath", "label": "mixed"}]


def test_canonical_items_are_deterministic_and_sorted():
    a = ec.items_from_rows(_poem(), copy.deepcopy(_poem_rows()))
    b = ec.items_from_rows(_poem(), copy.deepcopy(_poem_rows()))
    assert a == b and len(a) == 4
    for item in a:
        keys = list(item["q"]["criteria"])
        assert keys == sorted(keys, key=lambda k: (k.casefold(), k))
        assert item["target"].count(1.0) == 1
    # every item of one source + language asks the same question: one request group
    assert len({ec.question_id(it["q"]) for it in a}) == 1
    gold = {it["state"]: ec.option_names(it)[ec.gold_index(it)] for it in a}
    assert gold["joy and grief in one breath"] == "mixed"
    assert gold["the river runs to the sea"] == "no impact"


def test_canonical_uses_first_paraphrase_in_item_language():
    item = ec.items_from_rows(_poem(), _poem_rows())[0]
    bank = CATEGORY_TEMPLATES.get("sentiment", {}).get("choice", {}).get("en") or TEMPLATES["en"]["choice"]
    assert item["q"]["instructions"] == bank[0]


def test_canonical_restores_training_behaviour():
    before = {n: getattr(registry, n) for n in ("instruction", "shuffle_choice", "_render", "_base")}
    with ec.canonical():
        assert registry.shuffle_choice is not before["shuffle_choice"]
    after = {n: getattr(registry, n) for n in before}
    assert before == after
    # training still shuffles and varies the instruction
    entry = ec.entry_for(_poem())
    orders = {tuple(it["q"]["criteria"]) for seed in range(12) for it in registry.adapt(entry, _poem_rows(), seed)}
    assert len(orders) > 1


def test_task_filter_and_kind_filter():
    src = ec.Source("stjiris/IRIS_sts", "default", "test", kinds=("noul",))
    rows = [{"sentence1": "O réu foi condenado.", "sentence2": "O arguido foi condenado.", "relatedness_score": 4.8},
            {"sentence1": "O réu foi condenado.", "sentence2": "Chove em Lisboa.", "relatedness_score": 0.2}]
    items = ec.items_from_rows(src, rows)
    assert items and all(it["q"]["type"] == "noul" for it in items)
    assert sorted(ec.gold_index(it) for it in items) == [0, 1]
    none = ec.items_from_rows(ec.Source("stjiris/IRIS_sts", "default", "test", kinds=("noul",), tasks=("nli",)), rows)
    assert none == []


# --------------------------------------------------------------------------- label repair

def test_decode_class_labels():
    rows = [{"label": 0}, {"label": 2}, {"label": 9}, {"label": True}, {"label": "x"}]
    ec.decode_class_labels(rows, {"label": ["neg", "pos", "neu"]})
    assert [r["label"] for r in rows] == ["neg", "neu", 9, True, "x"]


def test_prep_toxic_label_gives_both_classes():
    src = ec.Source("mteb/toxic_conversations_50k", "default", "test", kinds=("noul",), prep="toxic_label")
    rows = ec.PREP["toxic_label"]([{"text": "have a nice day friend", "label_text": "not toxic"},
                                   {"text": "you are a worthless idiot", "label_text": "toxic"}])
    items = ec.items_from_rows(src, rows)
    gold = {it["state"]: ec.gold_index(it) for it in items}
    assert gold == {"have a nice day friend": 0, "you are a worthless idiot": 1}


def test_prep_pii_spans_feeds_pii_adapter():
    rows = ec.PREP["pii_spans"]([
        {"generated_text": "Call Anna Berg at 040 1234567.", "pii_spans": '[{"start": 5, "end": 14, "label": "name"}]',
         "language": "English"},
        {"generated_text": "The invoice total is due next month.", "pii_spans": "[]", "language": "English"},
    ])
    assert rows[0]["ner"] == [[0, 0, "name"]] and rows[1]["ner"] == []
    src = ec.Source("gretelai/synthetic_pii_finance_multilingual", "default", "test", kinds=("noul",),
                    prep="pii_spans", adapter="pii_adapter")
    items = ec.items_from_rows(src, rows)
    assert sorted(ec.gold_index(it) for it in items) == [0, 1]


def test_topic_items_carry_cpc_section_titles():
    src = ec.HELD_OUT["topic"][0]
    rows = ec.PREP["cpc_section"]([{"abstract": "A machine.", "_v6_config": "a", "_v6_lang": "en"},
                                   {"abstract": "A circuit.", "_v6_config": "h", "_v6_lang": "en"}])
    items = ec.items_from_rows(src, rows)
    assert sorted(ec.option_names(it)[ec.gold_index(it)] for it in items) == ["electricity", "human necessities"]


# --------------------------------------------------------------------------- sampling

def _fake(source, gold, i, lang="en", options=("a", "b", "c")):
    target = [1.0 if o == gold else 0.0 for o in options]
    return {"state": "%s %s text %d" % (source, gold, i), "lang": lang, "source": source, "suite": "s",
            "q": {"type": "choice", "instructions": "Pick.", "criteria": {o: o for o in options}},
            "target": target, "_texts": []}


def _pool():
    pool = [_fake("src1", "a", i) for i in range(300)]
    pool += [_fake("src1", "b", i) for i in range(60)]
    pool += [_fake("src1", "c", i) for i in range(20)]
    pool += [_fake("src2", "a", i) for i in range(80)]
    return pool


def test_stratified_is_deterministic_and_seeded():
    pool = _pool()
    one = [it["state"] for it in ec.stratified(pool, 150, 7)]
    two = [it["state"] for it in ec.stratified(list(reversed(copy.deepcopy(pool))), 150, 7)]
    other = [it["state"] for it in ec.stratified(pool, 150, 8)]
    assert one == two            # input order does not matter
    assert one != other          # the seed does
    assert len(set(one)) == 150  # no duplicates


def test_stratified_balances_buckets():
    counts = collections.Counter((it["source"], ec.option_names(it)[ec.gold_index(it)])
                                 for it in ec.stratified(_pool(), 150, 1))
    # four buckets, 150 / 4 = 37.5; the 20-item bucket is exhausted and the rest share its remainder
    assert counts[("src1", "c")] == 20
    rest = [counts[k] for k in (("src1", "a"), ("src1", "b"), ("src2", "a"))]
    assert sum(rest) == 130 and max(rest) - min(rest) <= 1


def test_make_tasks_skips_small_and_degenerate_languages():
    pool = _pool()
    pool += [_fake("src1", "a", i, lang="de") for i in range(200)]   # single class
    pool += [_fake("src1", "b", i, lang="fr") for i in range(40)]    # too few
    tasks, skipped = ec.make_tasks(pool, ["s"], None, 150, 1)
    assert [(s, l, len(items)) for s, l, items in tasks] == [("s", "en", 150)]
    assert {(s, l) for s, l, _ in skipped} == {("s", "de"), ("s", "fr")}
    tasks2, _ = ec.make_tasks(pool, ["s"], None, 150, 1)
    assert [it["state"] for it in tasks[0][2]] == [it["state"] for it in tasks2[0][2]]


# --------------------------------------------------------------------------- banned set

def test_norm_matches_eval_texts():
    for text in ("  Hello\tWorld  ", "ÄÖÜ  ß", "a\nb"):
        assert ec._norm(text) == eval_texts.norm(text)


def test_suite_texts_reach_banned_set_and_training_drops_them():
    pool = [_fake("src1", "a", 1), _fake("src1", "b", 2)]
    pool[0]["_texts"] = ["premise shared with a training row", "hypothesis only in the suite"]
    banned = eval_texts.add_category_suites(set(), pool)
    assert ec._norm(pool[1]["state"]) in banned
    assert eval_texts.norm("premise shared with a training row") in banned
    entry = ec.entry_for(_poem())
    train = [
        {"state": pool[1]["state"].upper(), "q": {"type": "noul", "instructions": "Is it?"}, "target": [0.0, 1.0],
         "lang": "en", "_texts": []},
        {"state": "premise: premise shared with a training row\n\nhypothesis: another one",
         "q": {"type": "noul", "instructions": "Is it?"}, "target": [1.0, 0.0], "lang": "en",
         "_texts": ["premise shared with a training row", "another one"]},
        {"state": "a sentence the suites never use", "q": {"type": "noul", "instructions": "Is it?"},
         "target": [1.0, 0.0], "lang": "en", "_texts": ["a sentence the suites never use"]},
    ]
    kept, stats = clean_items(entry, train, banned, 10, 1)
    assert [it["state"] for it in kept] == ["a sentence the suites never use"]
    assert stats["evaluation_overlap"] == 2


def test_eval_texts_lists_the_category_pool():
    assert "categories:held-out-pool" in eval_texts.SUITES


def test_fingerprint_tracks_spec(monkeypatch):
    before = ec.fingerprint()
    assert before == ec.fingerprint()
    monkeypatch.setattr(ec, "POOL_ROWS", ec.POOL_ROWS + 1)
    assert ec.fingerprint() != before


# --------------------------------------------------------------------------- scoring and gate

def test_probabilities_per_question_type():
    choice = _fake("s", "b", 0)
    assert ec.probabilities({"probabilities": {"c": 0.1, "a": 0.2, "b": 0.7}}, choice) == [0.2, 0.7, 0.1]
    noul = {"q": {"type": "noul", "instructions": "?"}, "target": [0.0, 1.0]}
    assert ec.probabilities({"noul": 0.8}, noul) == pytest.approx([0.2, 0.8])
    score = {"q": {"type": "score", "instructions": "?", "criteria": ["x", "y", "z"]}, "target": [0, 0, 1.0]}
    assert ec.probabilities({"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}, score) == [0.1, 0.2, 0.7]


def test_score_pools_mixed_widths():
    items = [_fake("s", "a", 0), {"q": {"type": "noul", "instructions": "?"}, "target": [0.0, 1.0]}]
    probs = [[0.6, 0.3, 0.1], [0.8, 0.2]]
    m = ec.score(items, probs)
    assert m["accuracy"] == 0.5
    assert 0 <= m["ece"] <= 1 and m["nll"] > 0


def test_gate_prefixes_category_suites():
    assert gate.suite_key({"family": "categories", "suite": "sentiment"}) == "categories:sentiment"
    assert gate.suite_key({"suite": "multilingual_sentiments"}) == "multilingual_sentiments"


def test_gate_compare_reports_the_categories_family(tmp_path, capsys):
    import json
    champ, chall = tmp_path / "champ", tmp_path / "chall"
    for d, acc in ((champ, 0.80), (chall, 0.74)):
        d.mkdir()
        heldout = {"categories:%s/en" % s: {"acc": acc, "n": 150} for s in ("sentiment", "nli", "emotion", "intent")}
        json.dump({"validation": {"v": 0.5 if d == champ else 0.6}, "heldout": heldout}, open(d / "eval.json", "w"))
    assert gate.compare(str(champ), str(chall)) is False
    out = capsys.readouterr().out
    assert "family categories" in out and "REGRESSION" in out


def test_template_fields_stay_in_state_but_not_in_banned_texts():
    src = ec.Source("theatticusproject/cuad-qa", "default", "test", kinds=("noul",), template_fields=("question",))
    question = 'Highlight the parts (if any) of this contract related to "Governing Law".'
    rows = [{"context": "This agreement is governed by the laws of Delaware.", "question": question,
             "answers": {"text": ["the laws of Delaware"], "answer_start": [30]}},
            {"context": "The supplier ships goods within ten days of each order.", "question": question,
             "answers": {"text": [], "answer_start": []}}]
    items = ec.items_from_rows(src, rows)
    assert sorted(ec.gold_index(it) for it in items) == [0, 1]
    banned = ec.suite_texts(items)
    assert all(question in it["state"] for it in items)
    assert ec._norm(question) not in banned
    assert ec._norm("This agreement is governed by the laws of Delaware.") in banned


# --------------------------------------------------------------------------- review follow-ups

def _item(state, texts=(), suite="s"):
    return {"state": state, "_texts": list(texts), "suite": suite}


def test_mixture_contamination_exact_and_containment():
    long_a = "the parcel arrived two weeks late and the box was completely crushed on arrival"
    long_b = "our printer on the third floor keeps dropping off the network every afternoon"
    pool = [
        _item("Exact Match Text"),                                  # exact (normalised)
        _item("premise: %s\n\nhypothesis: late" % long_a, [long_a]),  # its source text sits inside a training state
        _item("ticket: %s. please help" % long_b),                  # a training state sits inside it
        _item("a clean text nobody trained on, long enough to be checked by containment"),
        _item("short"),                                             # short: exact only
    ]
    states = ["exact   match text", "complaint: " + long_a + " and nobody answered", long_b, "shor"]
    assert ec.contaminated(pool, [ec._norm(s) for s in states]) == {0, 1, 2}


def test_mixture_contamination_without_spaces():
    zh = "这家餐厅的服务态度非常差，菜也很难吃，等了一个多小时才上菜，我再也不会来了，真的太失望了"
    assert len(zh) >= ec.CONTAIN_CHARS
    pool = [_item(zh), _item("完全不同的一句话，和训练数据没有任何关系，只是用来对照的文本而已")]
    assert ec.contaminated(pool, [ec._norm("评论：" + zh + "。")]) == {0}


def test_exclude_mixture_reads_the_built_file(tmp_path):
    import gzip
    import json
    path = tmp_path / "mixture.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"state": "Trained Text", "q": {}, "target": []}) + "\n")
        fh.write(json.dumps({"state": {"text": "structured"}, "q": {}, "target": []}) + "\n")
    pool = [_item("trained text"), _item("held out text")]
    kept, fp = ec.exclude_mixture(pool, path, log=lambda m: None)
    assert [it["state"] for it in kept] == ["held out text"] and len(fp) == 16


def test_gate_notes():
    assert ec.gate_note("emotion", "pt") and ec.gate_note("emotion", "ru")
    assert ec.gate_note("emotion", "de") is None
    assert ec.gate_note("nli", "en") is None

    class Old:
        pass

    class New:
        WINDOW = 1600
        CPC_SECTIONS = {}

    assert ec._reading_windows(Old) and ec._reading_windows(New) is None
    assert ec._topic_labels(Old) and ec._topic_labels(New) is None
    New._special_field_labels = staticmethod(lambda sid, row: None)
    assert ec._urgency_labels(Old) and ec._urgency_labels(New) is None


def test_balanced_accuracy_does_not_reward_a_constant_answer():
    items = [{"q": {"type": "noul", "instructions": "?"}, "target": [0.0, 1.0]} for _ in range(8)]
    items += [{"q": {"type": "noul", "instructions": "?"}, "target": [1.0, 0.0]} for _ in range(2)]
    always_yes = [[0.1, 0.9]] * 10
    m = ec.score(items, always_yes)
    assert m["accuracy"] == 0.8 and m["balanced_accuracy"] == 0.5


def test_majority_cap_skips_a_skewed_language():
    pool = [_fake("src1", "a", i, lang="fr") for i in range(200)] + [_fake("src1", "b", i, lang="fr") for i in range(30)]
    tasks, skipped = ec.make_tasks(pool, ["s"], None, 150, 1)
    assert tasks == [] and skipped[0][:2] == ("s", "fr")


def test_single_emotion_rows_only():
    rows = [{"text": "a", "anger": "1", "joy": "0"}, {"text": "b", "anger": "1", "joy": "1"}, {"text": "c"}]
    kept = ec.PREP["single_emotion"](rows)
    assert [(r["text"], r["emotions"]) for r in kept] == [("a", ["anger"])]


def test_incomplete_pool_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "build_pool", lambda log=print: ([_item("x")], {"src": "HTTPError: 503"}))
    cache = tmp_path / "pool.jsonl.gz"
    pool, failures = ec.load_pool(cache=cache, log=lambda m: None)
    assert failures and not cache.exists()


def test_fingerprint_tracks_registry_file(tmp_path, monkeypatch):
    import shutil
    for rel in ("tools/finetune/mixture_v6", "tools/finetune/sources"):
        shutil.copytree(ec.ROOT / rel, tmp_path / rel)
    monkeypatch.setattr(ec, "ROOT", tmp_path)
    before = ec.fingerprint()
    keep = tmp_path / "tools/finetune/sources/v6-keep.json"
    keep.write_text(keep.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert ec.fingerprint() != before


def _gate_pair(tmp_path, pools):
    import json
    dirs = []
    for name, acc, pool in (("champ", 0.80, pools[0]), ("chall", 0.60, pools[1])):
        d = tmp_path / name
        d.mkdir(parents=True)
        heldout = {"categories:nli/en": {"acc": acc, "n": 150, "pool": pool}}
        json.dump({"validation": {"v": 0.5 if name == "champ" else 0.6}, "heldout": heldout}, open(d / "eval.json", "w"))
        dirs.append(str(d))
    return dirs


def test_gate_compares_category_cells_only_on_the_same_pool(tmp_path, capsys):
    # A 20-point drop on a different pool is not compared at all, so it is no regression.
    gate.compare(*_gate_pair(tmp_path / "different", ("p1", "p2")))
    out = capsys.readouterr().out
    assert "1 cells not compared" in out
    assert "(held-out regression)" not in out and "REGRESSION" not in out
    # The same drop on the same pool is a regression.
    assert gate.compare(*_gate_pair(tmp_path / "same", ("p1", "p1"))) is False
    out = capsys.readouterr().out
    assert "cells not compared" not in out
    assert "categories:nli/en" in out
