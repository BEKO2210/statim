import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from corpora import CORPORA, EUR_LANGUAGES, iter_corpus, split_passages  # noqa: E402
from grounded import PILOT_LANGUAGES, run_pilot  # noqa: E402
from leakage import LeakageGuard  # noqa: E402
from ollama_http import OllamaHTTP  # noqa: E402


def words(prefix, n=80):
    return " ".join("%s%d" % (prefix, i) for i in range(n)) + "."


def test_fixture_loaders_split_dedupe_and_pin(tmp_path):
    assert len(EUR_LANGUAGES) == 24
    assert {"de", "fr", "es", "it", "pt", "nl", "pl"} <= set(EUR_LANGUAGES.values())
    duplicate = words("bill")
    bills = [{"title": "B1", "text": duplicate + "\n\n" + duplicate}]
    got = list(iter_corpus("billsum", tmp_path / "cache", rows=bills))
    assert len(got) == 1
    assert got[0][1] == "en" and got[0][3] == CORPORA["billsum"]["revision"]
    assert 60 <= len(got[0][0].split()) <= 400

    gov = [{"id": "1", "report": [{"paragraphs": [words("gao")], "subsections": []}]}]
    row = next(iter_corpus("gov_report", tmp_path / "cache2", rows=gov))
    assert row[2] == "launch/gov_report:train:1"

    eur = [{"celex_id": "X", "reference": words("gesetz"), "_lang": "de"}]
    row = next(iter_corpus("eur_lex_sum", tmp_path / "cache3", languages=("de",), rows=eur))
    assert row[1] == "de" and row[2].endswith(":de:X")
    assert not split_passages("too short")


def test_cache_below_data_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        list(iter_corpus("billsum", HERE.parents[1] / "data" / "bad", rows=[]))


def test_eight_gram_leakage_guard():
    guard = LeakageGuard(["zero one two three four five six seven eight nine"])
    assert guard.overlap("before one two three four five six seven eight after")
    assert not guard.overlap("one two three four five six seven changed")


class FakeClient:
    def chat(self, messages, schema, temperature, num_predict, seed=None):
        text = messages[-1]["content"]
        if "Requested urgency label:" in text:
            label = text.split("Requested urgency label:", 1)[1].splitlines()[0].strip()
            token = text.splitlines()[1].split()[0]
            content = json.dumps({"request": "request words show a grounded situation requiring action " + label + " " + token,
                                  "label": label})
        elif "Write exactly three short hypotheses" in text:
            token = text.splitlines()[1].split()[0]
            content = json.dumps({"items": [
                {"hypothesis": "hypothesis entailed unique " + token, "label": "entailment"},
                {"hypothesis": "hypothesis contradicted unique " + token, "label": "contradiction"},
                {"hypothesis": "hypothesis neutral unique " + token, "label": "neutral"}]})
        elif "Choose exactly one: not urgent" in text:
            request = text.splitlines()[1]
            content = json.dumps({"answer": next(x for x in ("not urgent", "soon", "critical")
                                                       if (" " + x + " ") in request)})
        else:
            label = ("entailment" if "entailed" in text else
                     "contradiction" if "contradicted" in text else "neutral")
            content = json.dumps({"answer": label})
        return {"content": content, "thinking": "", "eval_count": 1,
                "prompt_eval_count": 1, "total_s": 0.01}


def test_fake_end_to_end_provenance_and_samples(tmp_path):
    passages = {}
    for lang in PILOT_LANGUAGES:
        revision = CORPORA["billsum" if lang == "en" else "eur_lex_sum"]["revision"]
        passages[lang] = []
        for i in range(2):
            passage = words(lang + str(i))
            source = ("FiscalNote/billsum:train:%s:%d" % (lang, i) if lang == "en"
                      else "dennlinger/eur-lex-sum:train:%s:X%d" % (lang, i))
            passages[lang].append((passage, lang, source, revision))
    out = tmp_path / "out"
    meta = {"digest": "a" * 64}
    manifest = run_pilot(out, tmp_path / "cache", FakeClient(), FakeClient(), LeakageGuard([]),
                         per_capability=48, concurrency=4, passages=passages,
                         generator_meta=meta, verifier_meta={"digest": "b" * 64})
    assert manifest["items_per_capability"] == 48
    assert (out / "samples.md").read_text().count("### urgency") == 30
    assert (out / "samples.md").read_text().count("### nli") == 30
    for task in ("urgency", "nli"):
        with gzip.open(out / (task + ".jsonl.gz"), "rt") as handle:
            assert len(list(handle)) == 48
        with gzip.open(out / (task + ".provenance.jsonl.gz"), "rt") as handle:
            rows = [json.loads(line) for line in handle]
        assert len(rows) == 48
        row = rows[0]
        assert set(row["seed"]) == {"source_id", "revision", "passage_sha256", "lang", "target"}
        assert row["seed_passage"]
        assert row["generator"] == [
            {"model": "Qwen/Qwen3-8B", "role": "text"},
            {"model": "microsoft/Phi-4-mini-instruct", "role": "labels"}]
        assert row["ollama_model_digests"] == {"qwen3:8b": "a" * 64, "phi4-mini": "b" * 64}
        assert row["prompt_version"] == "grounded-pilot-1"


def test_ollama_thinking_is_disabled():
    client = OllamaHTTP("http://invalid", "qwen3:8b", -1)
    seen = {}
    client._post = lambda path, body, timeout=None: seen.update(body) or {"message": {"content": "{}"}}
    client.chat([], {"type": "object"}, 0, 1)
    assert seen["think"] is False
