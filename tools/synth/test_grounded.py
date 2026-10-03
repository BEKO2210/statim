import gzip
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import corpora  # noqa: E402
import leakage  # noqa: E402
from common import sha256_text  # noqa: E402
from corpora import CORPORA, EUR_LANGUAGES, PILOT_LANGUAGES, iter_corpus, split_passages  # noqa: E402
from generate import digest_from_show  # noqa: E402
from grounded import (_contains_label, _generate_job, _quota, run_pilot,
                      validate_model_meta)  # noqa: E402
from grounded_prompts import NLI_LABELS, URGENCY_LABELS, generation_request, verification_request  # noqa: E402
from grounded_tasks import PILOT_LANGS, TASKS, local_labels, task_labels  # noqa: E402
from leakage import LeakageGuard  # noqa: E402
from ollama_http import OllamaHTTP  # noqa: E402
from ready import assert_ready  # noqa: E402


def words(prefix, n=80):
    return " ".join("%s%d" % (prefix, i) for i in range(n)) + "."


def valid_eval_payload(texts):
    suites, categories, _ = leakage._eval_definition()
    return {"version": 1, "normalisation": "build_mixture.norm", "suites": suites,
            "categories": categories, "texts": set(texts)}


def local_files(tmp_path, texts=("a protected evaluation phrase with enough words for the guard",)):
    cache = tmp_path / "eval-texts.pkl"
    cache.write_bytes(pickle.dumps(valid_eval_payload(texts)))
    s1 = tmp_path / "s1bench"
    s1.mkdir()
    (s1 / "index.json").write_text(json.dumps({"subsets": {"tiny": {"items": 1}}}))
    (s1 / "tiny.json").write_text(json.dumps({"items": [{"state": "s1 protected example text"}]}))
    return cache, s1


def test_fixture_loaders_require_native_ids_and_do_not_stamp_injected_rows(tmp_path):
    assert len(EUR_LANGUAGES) == 24
    duplicate = words("bill")
    bills = [{"bill_id": "B-1", "text": duplicate + "\n\n" + duplicate},
             {"title": "not-an-id", "text": words("skip")},
             {"title": "no text"}]
    got = list(iter_corpus("billsum", tmp_path / "cache", rows=bills))
    # the real BillSum has no id column: the full text's SHA-256 is the id, never the title
    assert [g[2] for g in got] == [
        "FiscalNote/billsum:train:B-1",
        "FiscalNote/billsum:train:sha256-" + hashlib.sha256(words("skip").encode()).hexdigest()]
    assert got[0][3] is None
    assert 60 <= len(got[0][0].split()) <= 400

    gov = [{"id": "1", "reports": {"paragraphs": [words("crs")], "subsections": []}},
           {"report": [{"paragraphs": [words("missing")]}]}]
    row = next(iter_corpus("gov_report", tmp_path / "cache2", rows=gov))
    assert row[2] == "launch/gov_report:train:1" and row[3] is None

    eur = [{"celex_id": "X", "reference": words("gesetz"), "_lang": "de"},
           {"reference": words("skip"), "_lang": "de"}]
    rows = list(iter_corpus("eur_lex_sum", tmp_path / "cache3", languages=("de",), rows=eur))
    assert len(rows) == 1 and rows[0][2].endswith(":de:X") and rows[0][3] is None
    assert not split_passages("too short")


def test_raw_loaders_use_only_pinned_train_urls(monkeypatch, tmp_path):
    calls = []

    def fake_load_dataset(kind, data_files, **kwargs):
        calls.extend(data_files["train"] if isinstance(data_files["train"], list)
                     else [data_files["train"]])
        if kind == "parquet":
            return [{"bill_id": "B", "text": words("bill")}]
        return [{"id": "G", "report": [{"paragraphs": [words("gov")]}]}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    list(corpora._raw_rows("billsum", tmp_path, ("en",)))
    list(corpora._raw_rows("gov_report", tmp_path, ("en",)))
    # Avoid interpreting the JSON fixture; this test is about addressed artefacts.
    list(corpora._raw_rows("eur_lex_sum", tmp_path, ("de",)))
    expected = [
        corpora._url(CORPORA["billsum"], "data/train-00000-of-00001.parquet"),
        corpora._url(CORPORA["gov_report"], "data/gao_train.jsonl"),
        corpora._url(CORPORA["gov_report"], "data/crs_train.jsonl"),
        corpora._url(CORPORA["eur_lex_sum"], "data/german/train.json"),
    ]
    assert calls == expected
    filenames = [url.rsplit("/", 1)[-1] for url in calls]
    assert all("test" not in name and "ca_test" not in name for name in filenames)


def test_cache_below_data_is_rejected():
    with pytest.raises(ValueError):
        list(iter_corpus("billsum", HERE.parents[1] / "data" / "bad", rows=[]))


def test_leakage_rejects_word_short_cjk_exact_and_containment():
    cjk = "这是一个必须严格保护的中文评估句子绝不能复制到训练数据中"
    short = "reset my frozen account please"
    long = "the annual committee report contains protected findings for every regional office"
    guard = LeakageGuard([cjk, short, long])
    assert guard.overlap("prefix one two three four five six seven eight suffix") is False
    assert guard.overlap("前缀" + cjk[4:28] + "后缀")
    assert guard.overlap(short)
    assert guard.overlap("Note: " + long + " today")
    # an excerpt of 8+ words of a long held-out text is caught through its word anchor
    assert guard.overlap("intro " + " ".join(long.split()[1:10]) + " outro")
    # same limit as bench/eval_categories.contaminated (the gate's own check): an excerpt of fewer
    # than 8 words is anchored by characters while long pool texts are indexed by words
    assert guard.overlap(long[4:55]) is False


def test_from_local_fails_closed_and_records_fingerprints(tmp_path):
    with pytest.raises(FileNotFoundError):
        LeakageGuard.from_local(tmp_path / "missing.pkl", tmp_path / "s1")
    cache, s1 = local_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        LeakageGuard.from_local(cache, tmp_path / "missing-s1")
    bad = tmp_path / "bad.pkl"
    payload = valid_eval_payload({"x"})
    payload["categories"] = "stale"
    bad.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        LeakageGuard.from_local(bad, s1)
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(pickle.dumps(valid_eval_payload(set())))
    with pytest.raises(ValueError, match="empty"):
        LeakageGuard.from_local(empty, s1)
    guard = LeakageGuard.from_local(cache, s1)
    assert guard.texts >= 2
    assert len(guard.metadata["suite_fingerprint"]) == 64
    assert guard.metadata["s1bench_subsets"] == 1


def test_s1bench_requires_index_and_every_listed_subset(tmp_path):
    cache, s1 = local_files(tmp_path)
    (s1 / "tiny.json").unlink()
    with pytest.raises(FileNotFoundError, match="subset"):
        LeakageGuard.from_local(cache, s1)
    (s1 / "index.json").unlink()
    with pytest.raises(FileNotFoundError, match="index"):
        LeakageGuard.from_local(cache, s1)


def test_digest_source_and_model_allowlist():
    top = digest_from_show({"digest": "a" * 64, "details": {"family": "qwen3",
                           "parameter_size": "8.2B"}})
    assert top["digest_source"] == "show.digest"
    validate_model_meta("qwen3:8b", top)
    fallback = digest_from_show({"modelfile": "FROM sha256-" + "b" * 64,
        "details": {"family": "phi4", "parameter_size": "3.8B"}})
    assert fallback["digest_source"] == "show.modelfile FROM sha256"
    validate_model_meta("phi4-mini", fallback)
    with pytest.raises(SystemExit, match="allowlist"):
        validate_model_meta("qwen3:8b", dict(top, family="llama"))
    with pytest.raises(SystemExit, match="allowlist"):
        validate_model_meta("phi4-mini", dict(fallback, parameter_size="14B"))


LANG_TEXT = {
    "en": "the citizen with this permit has ample time calmcase before filing documents",
    "de": "der Bürger hat mit diesem Antrag viel Zeit calmcase für alle Unterlagen",
    "fr": "le citoyen dispose avec cette demande de beaucoup de temps calmcase administratif",
    "es": "el ciudadano tiene con esta solicitud mucho tiempo calmcase para documentos administrativos",
    "it": "il cittadino ha con questa domanda molto tempo calmcase per documenti amministrativi",
    "pt": "o cidadão tem com esta solicitação muito tempo calmcase para documentos administrativos",
    "nl": "de burger heeft met deze aanvraag ruim tijd calmcase voor alle documenten",
    "pl": "ten obywatel ma dużo czasu calmcase i dokumenty dla urzędu lokalnego",
}
NLI_TEXT = {
    "en": "the agency confirms ecase this measure for citizens",
    "de": "die Behörde bestätigt ecase diese Maßnahme für Bürger",
    "fr": "la commission confirme ecase cette mesure avec précision",
    "es": "la agencia confirma ecase esta medida para ciudadanos",
    "it": "la commissione conferma ecase questa misura per cittadini",
    "pt": "a comissão confirma ecase esta medida com clareza",
    "nl": "de instantie bevestigt ecase deze maatregel voor burgers",
    "pl": "ta komisja potwierdza ecase ten środek dla obywateli",
}
_LANG_NAMES = {code: name for code, name in [
    ("en", "English"), ("de", "German"), ("fr", "French"), ("es", "Spanish"),
    ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish")]}
def _lang_template(marker):
    return {lang: LANG_TEXT[lang].replace("calmcase", marker) for lang in PILOT_LANGS}


def _nli_template(marker):
    return {lang: NLI_TEXT[lang].replace("ecase", marker) for lang in PILOT_LANGS}
_SINGLE_MARKERS = {
    "urgency": {"not urgent": "calmcase", "soon": "daycase", "critical": "harmcase"},
    "spam": {"spam": "junkcase", "not spam": "legitcase"},
    "sarcasm": {"sarcastic": "irncase", "sincere": "frankcase"},
    "emotion": {"anger": "angrcase", "fear": "fearcase", "joy": "joycase", "sadness": "sadcase",
                "surprise": "surpcase", "disgust": "disgcase", "neutral": "neutcase"},
    "claim": {"checkable claim": "factcase", "opinion": "viewcase", "no claim": "chatcase"},
}
_PAIR_MARKERS = {
    "nli": {"entailment": "ecase", "contradiction": "xcase", "neutral": "ucase"},
    "stance": {"favour": "procase", "against": "concase", "neutral": "midcase"},
    "reading": {"yes": "yescase", "no": "nocase", "not answerable": "unkcase"},
}
_MARKER_TO_ANSWER = {}
for task, mapping in {**_SINGLE_MARKERS, **_PAIR_MARKERS}.items():
    for label, marker in mapping.items():
        _MARKER_TO_ANSWER[marker] = label
_ALL_MARKERS = tuple(_MARKER_TO_ANSWER)


def _lang_from_prompt(text):
    return next(code for code, name in _LANG_NAMES.items() if "(" + name + ")" in text)


class BlindFakeClient:
    """Answers from semantic marker text; it never receives or remembers a gold label."""
    def __init__(self):
        self.verification_prompts = []

    def chat(self, messages, schema, temperature, num_predict, seed=None):
        text = messages[-1]["content"]
        if "Requested urgency label:" in text:
            label = text.split("Requested urgency label:", 1)[1].splitlines()[0].strip()
            lang = _lang_from_prompt(text)
            marker = _SINGLE_MARKERS["urgency"][label]
            request = LANG_TEXT[lang].replace("calmcase", marker)
            return self._result({"request": request, "label": label})
        if "Write exactly three short hypotheses" in text:
            lang = _lang_from_prompt(text)
            base = NLI_TEXT[lang]
            return self._pair_items("hypothesis", base, "ecase", _PAIR_MARKERS["nli"])
        if "Write one short message" in text:
            return self._single_lang(text, "message", _SINGLE_MARKERS["spam"])
        if "Write one short comment" in text and "citizen comments" not in text:
            return self._single_lang(text, "comment", _SINGLE_MARKERS["sarcasm"])
        if "Write one short statement" in text:
            return self._single_lang(text, "statement", _SINGLE_MARKERS["emotion"])
        if "Write one short utterance" in text:
            return self._single_lang(text, "utterance", _SINGLE_MARKERS["claim"])
        if "citizen comments" in text:
            return self._pair_lang(text, "comment", _PAIR_MARKERS["stance"])
        if "yes/no questions" in text:
            return self._pair_lang(text, "question", _PAIR_MARKERS["reading"])
        self.verification_prompts.append(text)
        marker = next(value for value in _ALL_MARKERS if value in text)
        return self._result({"answer": _MARKER_TO_ANSWER[marker]})

    def _single_lang(self, text, field, markers):
        label = text.split("Requested label:", 1)[1].splitlines()[0].strip()
        lang = _lang_from_prompt(text)
        body = _lang_template(markers[label])[lang]
        return self._result({field: body, "label": label})

    def _pair_lang(self, text, field, markers):
        lang = _lang_from_prompt(text)
        items = [{field: _nli_template(marker)[lang], "label": label}
                 for label, marker in markers.items()]
        return self._result({"items": items})

    def _pair_items(self, field, base, placeholder, markers):
        items = []
        for label, marker in markers.items():
            hypothesis = base.replace(placeholder, marker)
            items.append({field: hypothesis, "label": label})
        return self._result({"items": items})

    @staticmethod
    def _result(value):
        return {"content": json.dumps(value), "thinking": "", "eval_count": 1,
                "prompt_eval_count": 1, "total_s": 0.01}


def fixture_passages(count=4):
    passages = {}
    for lang in PILOT_LANGUAGES:
        revision = CORPORA["billsum" if lang == "en" else "eur_lex_sum"]["revision"]
        passages[lang] = []
        for i in range(count):
            passage = words("seed%s%d" % (lang, i))
            source = ("FiscalNote/billsum:train:B%d" % i if lang == "en"
                      else "dennlinger/eur-lex-sum:train:%s:X%d" % (lang, i))
            passages[lang].append((passage, lang, source, revision))
    return passages


def model_meta(digest, family, size):
    return {"digest": digest * 64, "digest_source": "show.digest", "family": family,
            "parameter_size": size}


def test_fake_end_to_end_is_blind_language_marked_and_balanced(tmp_path):
    fake = BlindFakeClient()
    guard = LeakageGuard(["unrelated protected evaluation words zero one two three four five six"])
    out = tmp_path / "out"
    manifest = run_pilot(out, tmp_path / "cache", fake, fake, guard,
        per_capability=48, concurrency=4, passages=fixture_passages(),
        generator_meta=model_meta("a", "qwen3", "8B"),
        verifier_meta=model_meta("b", "phi4", "3.8B"))
    assert manifest["leakage"]["checked"] is True
    assert assert_ready(out) == manifest
    assert not list(out.glob("*.part"))
    assert fake.verification_prompts
    for task, labels in (("urgency", URGENCY_LABELS), ("nli", NLI_LABELS)):
        expected = _quota(48, PILOT_LANGUAGES, labels)
        assert manifest["stats"][task]["counts"] == {
            "%s/%s" % cell: count for cell, count in sorted(expected.items())}
        with gzip.open(out / (task + ".jsonl.gz"), "rt", encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle]
        with gzip.open(out / (task + ".provenance.jsonl.gz"), "rt", encoding="utf-8") as handle:
            provs = [json.loads(line) for line in handle]
        assert len(items) == len(provs) == 48
        assert {row["id"] for row in items} == {row["id"] for row in provs}
        assert all(row["src"].startswith(("FiscalNote/", "dennlinger/")) for row in items)
        assert all(len(row["passage_sha256"]) == 64 for row in items)
        for row in provs:
            assert row["ollama_models"]["qwen3:8b"]["digest_source"] == "show.digest"
            assert row["ollama_model_digests"] == {"qwen3:8b": "a" * 64,
                                                    "phi4-mini": "b" * 64}
    assert (out / "samples.md").read_text().count("### urgency") == 30
    assert (out / "samples.md").read_text().count("### nli") == 30


def test_verifier_request_is_blind_and_label_words_are_rejected():
    request = verification_request("urgency", {"request": "the service stopped today"}, "en")
    payload = request[0][-1]["content"]
    assert "target" not in payload.casefold() and "gold" not in payload.casefold()
    assert _contains_label("This is critical for us", "urgency", "en")
    assert _contains_label("This is urgent for us", "urgency", "en")
    assert _contains_label("Das ist kritisch für uns", "urgency", "de")
    assert _contains_label("A neutral statement", "nli", "en")


class StaticClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seeds = []

    def chat(self, messages, schema, temperature, num_predict, seed=None):
        self.seeds.append(seed)
        value = self.replies.pop(0)
        return {"content": value if isinstance(value, str) else json.dumps(value)}


def test_label_leak_seed_copy_and_language_mismatch_drop_before_verifier():
    passage = " ".join(["the", "agency", "with", "this", "permit", "has", "ample", "time"] +
                       ["seed%d" % i for i in range(70)])
    verifier = StaticClient([{"answer": "soon"}] * 4)
    for request, reason in [
        ("the citizen says this critical problem requires help from the agency", "label_leak"),
        ("the agency with this permit has ample time and the citizen waits", "seed_copy"),
    ]:
        generator = StaticClient([{"request": request, "label": "soon"}])
        rows, metrics = _generate_job("urgency", (passage, "en", "FiscalNote/billsum:train:B", None),
            "soon", generator, verifier, 0.4, {}, LeakageGuard(["unrelated protected text"] ))
        assert not rows and metrics[reason] == 1
    german_passage = words("gesetz")
    generator = StaticClient([{"request": LANG_TEXT["en"], "label": "soon"}])
    rows, metrics = _generate_job("urgency", (german_passage, "de", "eur:de:X", None),
        "soon", generator, verifier, 0.4, {}, LeakageGuard(["unrelated protected text"]))
    assert not rows and metrics["bad_text"] == 1
    assert not verifier.seeds


def test_json_failure_retries_once_with_a_different_seed():
    generator = StaticClient(["not-json", {"request": LANG_TEXT["en"], "label": "not urgent"}])
    verifier = StaticClient([{"answer": "not urgent"}])
    rows, metrics = _generate_job("urgency", (words("seed"), "en", "FiscalNote/billsum:train:B", None),
        "not urgent", generator, verifier, 0.4, {}, LeakageGuard(["protected eval text"]))
    assert len(rows) == 1 and metrics["generate_retry"] == 1
    assert len(generator.seeds) == 2 and generator.seeds[0] != generator.seeds[1]


def test_generation_path_drops_planted_heldout_copy():
    planted = LANG_TEXT["en"]
    generator = StaticClient([{"request": planted, "label": "not urgent"}])
    verifier = StaticClient([{"answer": "not urgent"}])
    rows, metrics = _generate_job(
        "urgency", (words("seed"), "en", "FiscalNote/billsum:train:B", None),
        "not urgent", generator, verifier, 0.4, {}, LeakageGuard([planted]))
    assert not rows and metrics["leakage_reject"] == 1 and metrics["verify_agree"] == 1


def test_existing_directory_requires_resume_and_complete_run_resumes(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    args = (out, tmp_path / "cache", BlindFakeClient(), BlindFakeClient(),
            LeakageGuard(["protected evaluation text"] ))
    with pytest.raises(FileExistsError):
        run_pilot(*args, per_capability=24, passages=fixture_passages())
    meta_g, meta_v = model_meta("a", "qwen3", "8B"), model_meta("b", "phi4", "3.8B")
    run_pilot(*args, per_capability=24, passages=fixture_passages(), resume=True,
              generator_meta=meta_g, verifier_meta=meta_v)
    # A completed directory can be resumed without duplicating rows.
    run_pilot(*args, per_capability=24, passages=fixture_passages(), resume=True,
              generator_meta=meta_g, verifier_meta=meta_v)
    with gzip.open(out / "nli.jsonl.gz", "rt", encoding="utf-8") as handle:
        assert len(list(handle)) == 24


def test_filter_joins_by_id_rewrites_samples_and_ready_gate(tmp_path):
    held = "the committee shall report the findings of the annual review to the board"
    cache, s1 = local_files(tmp_path, (held,))
    src = tmp_path / "colab"
    src.mkdir()
    clean_item = {"id": "keep", "state": "a fresh request about a broken heating system",
                  "q": {"instructions": "question"}, "target": [1], "src": "source",
                  "passage_sha256": "a" * 64}
    leaky_item = dict(clean_item, id="drop", state="Note: " + held + " today")
    provenance = {
        "keep": {"id": "keep", "seed": {"source_id": "source", "lang": "en",
                  "target": "x"}, "seed_passage": "clean passage"},
        "drop": {"id": "drop", "seed": {"source_id": "source", "lang": "en",
                  "target": "x"}, "seed_passage": "leaky passage"},
    }
    # Deliberately reverse sidecar order: a positional join would retain the wrong provenance.
    for name, rows in (("nli.jsonl.gz", [clean_item, leaky_item]),
                       ("nli.provenance.jsonl.gz", [provenance["drop"], provenance["keep"]])):
        with gzip.open(src / name, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    (src / "samples.md").write_text("STALE LEAK " + held)
    (src / "manifest.json").write_text(json.dumps({"seed": 1, "leakage": {"checked": False}}))
    with pytest.raises(ValueError):
        assert_ready(src)
    removed = leakage.filter_dir(src, tmp_path / "clean", cache, s1)
    assert removed == {"nli": 1}
    assert assert_ready(tmp_path / "clean")["leakage"]["checked"] is True
    with gzip.open(tmp_path / "clean" / "nli.provenance.jsonl.gz", "rt") as handle:
        assert [json.loads(line)["id"] for line in handle] == ["keep"]
    samples = (tmp_path / "clean" / "samples.md").read_text()
    assert "clean passage" in samples and held not in samples and "STALE" not in samples


def test_filter_rejects_mismatched_ids(tmp_path):
    cache, s1 = local_files(tmp_path)
    src = tmp_path / "bad"
    src.mkdir()
    with gzip.open(src / "nli.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"id": "item", "state": "safe"}) + "\n")
    with gzip.open(src / "nli.provenance.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"id": "other"}) + "\n")
    (src / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="id sets"):
        leakage.filter_dir(src, tmp_path / "dst", cache, s1)


def test_ollama_thinking_is_disabled():
    client = OllamaHTTP("http://invalid", "qwen3:8b", -1)
    seen = {}
    client._post = lambda path, body, timeout=None: seen.update(body) or {"message": {"content": "{}"}}
    client.chat([], {"type": "object"}, 0, 1)
    assert seen["think"] is False


def test_phased_batches_keep_the_same_items(tmp_path):
    # generate-then-verify per batch (8 GB GPU) must not change which items are kept
    guard = LeakageGuard(["unrelated protected evaluation words zero one two three four five six"])
    kwargs = dict(per_capability=48, concurrency=4, passages=fixture_passages(),
                  generator_meta=model_meta("a", "qwen3", "8B"),
                  verifier_meta=model_meta("b", "phi4", "3.8B"))
    plain = run_pilot(tmp_path / "plain", tmp_path / "cache", BlindFakeClient(), BlindFakeClient(),
                      guard, **kwargs)
    phased = run_pilot(tmp_path / "phased", tmp_path / "cache", BlindFakeClient(), BlindFakeClient(),
                       guard, phased_batch=16, **kwargs)
    assert phased["phased_batch"] == 16
    for task in ("urgency", "nli"):
        with gzip.open(tmp_path / "plain" / (task + ".jsonl.gz"), "rt") as a, \
                gzip.open(tmp_path / "phased" / (task + ".jsonl.gz"), "rt") as b:
            assert sorted(json.loads(x)["id"] for x in a) == sorted(json.loads(x)["id"] for x in b)
    assert plain["stats"]["nli"]["counts"] == phased["stats"]["nli"]["counts"]
    only = run_pilot(tmp_path / "only", tmp_path / "cache", BlindFakeClient(), BlindFakeClient(),
                     guard, tasks=("nli",), **kwargs)
    assert list(only["stats"]) == ["nli"] and not (tmp_path / "only" / "urgency.jsonl.gz").exists()


def test_openai_client_requests_schema_output_without_thinking_and_checks_the_served_model():
    from openai_http import OpenAIHTTP
    client = OpenAIHTTP("http://vllm", "microsoft/phi-4", "b" * 40)
    sent = {}
    client._post = lambda path, body, timeout=None: sent.update(path=path, body=body) or {
        "choices": [{"message": {"content": '{"answer": "soon"}'}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 5, "prompt_tokens": 50}}
    out = client.chat([{"role": "user", "content": "x"}], {"type": "object"}, 0, 32, seed=7)
    assert sent["path"] == "/v1/chat/completions" and out["content"] == '{"answer": "soon"}'
    body = sent["body"]
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["seed"] == 7
    client._get = lambda path: {"data": [{"id": "microsoft/phi-4"}]}
    assert client.show() == {"hf_id": "microsoft/phi-4", "revision": "b" * 40,
                             "backend": "vllm-openai", "served_models": ["microsoft/phi-4"]}
    client._get = lambda path: {"data": [{"id": "some/other-model"}]}
    with pytest.raises(Exception, match="serves"):
        client.show()


def test_hf_models_need_allowlist_and_exact_revision():
    import grounded
    meta = {"hf_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "revision": "a" * 40}
    assert grounded.validate_hf_meta(meta) is meta
    with pytest.raises(SystemExit, match="allowlist"):
        grounded.validate_hf_meta({"hf_id": "meta-llama/Llama-3.1-8B", "revision": "a" * 40})
    with pytest.raises(SystemExit, match="revision"):
        grounded.validate_hf_meta({"hf_id": "microsoft/phi-4", "revision": "main"})


def test_two_stage_generate_then_verify(tmp_path):
    import grounded
    guard = LeakageGuard(["unrelated protected evaluation words zero one two three four five six"])
    cands = tmp_path / "cands.jsonl.gz"
    passages = fixture_passages(count=40)
    first = grounded.stage_generate(cands, tmp_path / "cache", BlindFakeClient(), ("urgency", "nli"),
                                    48, 8, 0.4, oversample=2.0, passages=passages)
    assert first["short"] == {} and first["candidates"] == 2 * 48 * 2
    again = grounded.stage_generate(cands, tmp_path / "cache", BlindFakeClient(), ("urgency", "nli"),
                                    48, 8, 0.4, oversample=2.0, passages=passages)
    assert again["candidates"] == first["candidates"]  # resume: targets already met, nothing added
    meta = {"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": {"hf_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                                                     "revision": "a" * 40},
            "microsoft/phi-4": {"hf_id": "microsoft/phi-4", "revision": "b" * 40}}
    out = tmp_path / "out"
    manifest = grounded.stage_verify(cands, out, BlindFakeClient(), guard, ("urgency", "nli"), 48, 8, meta)
    assert manifest["models"] == [{"model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "role": "text"},
                                  {"model": "microsoft/phi-4", "role": "labels"}]
    assert assert_ready(out) == manifest and manifest["stages"] == "generate+verify"
    for task in ("urgency", "nli"):
        with gzip.open(out / (task + ".provenance.jsonl.gz"), "rt") as f:
            rows = [json.loads(x) for x in f]
        assert len(rows) == 48 and rows[0]["generator"] == manifest["models"]
        assert all(set(c.split("/")[0] for c in manifest["stats"][task]["counts"]) ==
                   set(PILOT_LANGUAGES) for _ in [0])
    # too few candidates: a clear message, never padding
    few = tmp_path / "few.jsonl.gz"
    grounded.stage_generate(few, tmp_path / "cache", BlindFakeClient(), ("nli",), 48, 8, 0.4,
                            oversample=0.5, passages=passages)
    with pytest.raises(SystemExit, match="short cells"):
        grounded.stage_verify(few, tmp_path / "out2", BlindFakeClient(), guard, ("nli",), 48, 8, meta)


def test_truncated_candidate_tail_and_finished_task_stats_survive_resume(tmp_path):
    import grounded
    guard = LeakageGuard(["unrelated protected evaluation words zero one two three four five six"])
    cands = tmp_path / "cands.jsonl.gz"
    passages = fixture_passages(count=40)
    grounded.stage_generate(cands, tmp_path / "cache", BlindFakeClient(), ("urgency", "nli"), 48, 8, 0.4,
                            oversample=2.0, passages=passages)
    whole = cands.read_bytes()
    cands.write_bytes(whole[:-40])  # a crash cut into the last gzip member
    rows = grounded._read_candidates(cands)
    assert len(rows) == 2 * 48 * 2 - 1 and len(grounded._read_candidates(cands)) == len(rows)
    grounded.stage_generate(cands, tmp_path / "cache", BlindFakeClient(), ("urgency", "nli"), 48, 8, 0.4,
                            oversample=2.0, passages=passages)
    assert len(grounded._read_candidates(cands)) == 2 * 48 * 2  # the lost row was generated again
    meta = {"Qwen/Qwen3-30B-A3B-GPTQ-Int4": {"hf_id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "revision": "a" * 40},
            "microsoft/phi-4": {"hf_id": "microsoft/phi-4", "revision": "b" * 40}}
    out = tmp_path / "out"
    first = grounded.stage_verify(cands, out, BlindFakeClient(), guard, ("urgency", "nli"), 48, 8, meta)
    again = grounded.stage_verify(cands, out, BlindFakeClient(), guard, ("urgency", "nli"), 48, 8, meta,
                                  resume=True)
    assert again["stats"] == first["stats"] and first["stats"]["urgency"]["metrics"]["verify_called"] > 0


def test_urgency_nli_ids_match_pilot_fixture(tmp_path):
    fixture = json.loads((HERE / "testdata" / "grounded_pilot_fixture.json").read_text())
    fake = BlindFakeClient()
    guard = LeakageGuard(["unrelated protected evaluation words zero one two three four five six"])
    out = tmp_path / "out"
    manifest = run_pilot(out, tmp_path / "cache", fake, fake, guard,
        per_capability=48, concurrency=4, passages=fixture_passages(),
        generator_meta=model_meta("a", "qwen3", "8B"), verifier_meta=model_meta("b", "phi4", "3.8B"),
        tasks=("urgency", "nli"))
    for task in ("urgency", "nli"):
        with gzip.open(out / (task + ".jsonl.gz"), "rt") as handle:
            ids = sorted(json.loads(line)["id"] for line in handle)
        assert ids == fixture[task]["ids"]
        sample = json.loads(next(gzip.open(out / (task + ".jsonl.gz"), "rt")))
        assert sample["q"] == fixture[task]["sample"]["q"]
    assert manifest["prompt_versions"]["urgency"] == manifest["prompt_versions"]["nli"] == "grounded-pilot-2"


def test_registry_locales_cover_every_language():
    for name, spec in TASKS.items():
        for lang in PILOT_LANGS:
            loc = spec["locales"][lang]
            assert "question" in loc and "local_labels" in loc
            assert len(loc["local_labels"]) == len(spec["labels"])
            if spec["shape"] == "pair" or not spec.get("ordinal"):
                assert len(loc["criteria"]) == len(spec["labels"])
            if spec["shape"] == "pair":
                assert len(loc["pair_fields"]) == 2


def test_label_leak_catches_each_label_in_each_language():
    for task in TASKS:
        labels = task_labels(task)
        for lang in PILOT_LANGS:
            for label in labels:
                sample = ("plain text with %s inside" % label if " " in label
                          else "plain %s word" % label)
                assert _contains_label(sample, task, lang)
            for local in local_labels(task, lang):
                if local.casefold() in {l.casefold() for l in labels}:
                    continue
                sample = ("plain text with %s inside" % local if " " in local
                          else "plain %s word" % local)
                assert _contains_label(sample, task, lang)


def test_verification_requests_never_mention_target():
    passage = words("seed")
    for task in TASKS:
        spec = TASKS[task]
        for lang in PILOT_LANGS:
            if spec["shape"] == "single":
                field = spec["verify_field"]
                item = {field: "neutral body text without label words here please"}
            else:
                pfield, gfield = spec["verify_fields"]
                item = {pfield: passage, gfield: "neutral body text without label words here please"}
            payload = verification_request(task, item, lang)[0][-1]["content"].casefold()
            assert "target" not in payload and "gold" not in payload
            assert "requested label" not in payload


def test_all_tasks_two_stage_run_fills_every_cell(tmp_path):
    import grounded
    all_tasks = tuple(TASKS)
    guard = LeakageGuard(["unrelated protected evaluation words zero one two three four five six"])
    cands = tmp_path / "cands.jsonl.gz"
    passages = fixture_passages(count=80)
    per = 16
    grounded.stage_generate(cands, tmp_path / "cache", BlindFakeClient(), all_tasks,
                            per, 8, 0.4, oversample=6.0, max_jobs_factor=60, passages=passages)
    meta = {"gen": {"hf_id": "Qwen/Qwen3-8B", "revision": "a" * 40},
            "microsoft/phi-4": {"hf_id": "microsoft/phi-4", "revision": "b" * 40}}
    model_meta = {"Qwen/Qwen3-8B": meta["gen"], "microsoft/phi-4": meta["microsoft/phi-4"]}
    out = tmp_path / "out"
    manifest = grounded.stage_verify(cands, out, BlindFakeClient(), guard, all_tasks, per, 8, model_meta)
    assert set(manifest["tasks"]) == set(all_tasks)
    for task in all_tasks:
        labels = task_labels(task)
        quota = _quota(per, PILOT_LANGUAGES, labels)
        assert sum(manifest["stats"][task]["counts"].values()) == per
        for cell, need in quota.items():
            key = "%s/%s" % cell
            assert manifest["stats"][task]["counts"].get(key, 0) == need
        assert manifest["prompt_versions"][task] == TASKS[task]["prompt_version"]
