"""Build the single normalised exclusion set for every Statim evaluation.

The pickle is a dict, not a bare set, so a stale cache cannot silently omit a
new suite.  Text normalisation is exactly build_mixture.norm.
"""

from __future__ import annotations

import json
import pickle
import random
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "data" / "eval-texts.pkl"
SUITES = [
    "ag_news:test", "ag_news:train(eval_dev)", "dair_emotion:test",
    "dair_emotion:validation", "banking77:test", "banking77:train-dev500",
    "massive:test-all-languages", "massive:validation-eval-languages",
    "multilingual_sentiments:test", "multilingual_sentiments:valid",
    "typed_decisions:test", "go_emotions:test", "multi_hatecheck:test",
    "flores200:dev+devtest", "belebele:test-passages", "sib200:test",
    "hwu64:test", "indonli:test_expert", "farstail:test", "semrel:test",
]


def norm(text):
    return " ".join(str(text).split()).lower()


def _strings(value):
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _add(out, rows, fields):
    for row in rows:
        for field in fields:
            for text in _strings(row.get(field)):
                out.add(norm(text))


def _load_json(url, split="x", **kwargs):
    from datasets import load_dataset
    return load_dataset("json", data_files={split: url}, split=split, **kwargs)


def _add_flores(out):
    """Read official FLORES files directly; the Meta Hub repo is gated."""
    url = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    archive = Path(tempfile.gettempdir()) / "statim-flores200_dataset.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(archive)  # only a complete download gets the final name
    try:
        tf = tarfile.open(archive, "r:gz")
    except (tarfile.TarError, EOFError, OSError):
        archive.unlink(missing_ok=True)  # never reuse a broken archive
        raise
    with tf:
        for member in tf:
            name = member.name
            if not member.isfile() or not ("/dev/" in name or "/devtest/" in name):
                continue
            if not (name.endswith(".dev") or name.endswith(".devtest")):
                continue
            fh = tf.extractfile(member)
            if fh:
                for line in fh:
                    text = line.decode("utf-8").strip()
                    if text:
                        out.add(norm(text))


def build_set():
    from datasets import get_dataset_config_names, load_dataset
    from tools.finetune.train_multitask import MASSIVE_URL, SENT_LANGS, SENT_URL

    out = set()
    _add(out, load_dataset("fancyzhx/ag_news", split="test"), ["text"])
    _add(out, load_dataset("fancyzhx/ag_news", split="train"), ["text"])
    _add(out, load_dataset("dair-ai/emotion", "split", split="test"), ["text"])
    _add(out, load_dataset("dair-ai/emotion", "split", split="validation"), ["text"])
    bank_train = list(load_dataset("mteb/banking77", split="train"))
    random.Random(20260926).shuffle(bank_train)
    _add(out, bank_train[:500], ["text"])
    _add(out, load_dataset("mteb/banking77", split="test"), ["text"])

    api = "https://huggingface.co/api/datasets/mteb/amazon_massive_intent/tree/main/test"
    langs = [x["path"].split("/")[-1].replace(".json.gz", "")
             for x in json.load(urllib.request.urlopen(api))]
    for lang in langs:
        _add(out, _load_json(MASSIVE_URL % ("test", lang)), ["text"])
    for lang in ("de", "en", "fr", "es", "it", "tr", "pl", "ru", "ja", "zh-CN", "ar", "hi"):
        _add(out, _load_json(MASSIVE_URL % ("validation", lang)), ["text"])
    for lang in SENT_LANGS:
        from datasets import load_dataset as ld
        _add(out, ld("csv", data_files={"x": SENT_URL % (lang, "test")}, split="x"), ["text"])
        _add(out, ld("csv", data_files={"x": SENT_URL % (lang, "valid")}, split="x"), ["text"])

    typed = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    _add(out, typed, ["state"])
    # Also protect plain text embedded in the JSON state.
    for row in typed:
        try:
            for text in _strings(json.loads(row["state"])):
                out.add(norm(text))
        except (TypeError, json.JSONDecodeError):
            pass
    _add(out, load_dataset("google-research-datasets/go_emotions", "simplified", split="test"), ["text"])

    for code in ("eng", "deu", "fra", "spa", "ita", "nld", "pol", "por", "cmn", "ara", "hin"):
        url = "https://huggingface.co/datasets/mteb/multi-hatecheck/resolve/main/test/%s.jsonl.gz" % code
        _add(out, _load_json(url), ["text"])

    # Belebele passages are separately included because the suites use their
    # exact, sometimes postprocessed, passage strings.
    _add_flores(out)
    for config in get_dataset_config_names("facebook/belebele"):
        _add(out, load_dataset("facebook/belebele", config, split="test"), ["flores_passage"])

    for config in ("eng_Latn", "deu_Latn", "arb_Arab", "hin_Deva"):
        _add(out, load_dataset("Davlan/sib200", config, split="test"), ["text"])
    _add(out, load_dataset("DeepPavlov/hwu64", split="test"), ["utterance"])
    _add(out, _load_json("https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/test_expert.jsonl"),
         ["premise", "hypothesis"])
    _add(out, load_dataset("csv", data_files={"x": "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv"},
                           split="x", delimiter="\t"), ["premise", "hypothesis"])
    for config in ("eng", "arb", "hin"):
        _add(out, load_dataset("SemRel/SemRel2024", config, split="test"), ["sentence1", "sentence2"])
    out.discard("")
    return out


def load(cache=DEFAULT_CACHE, rebuild=False):
    cache = Path(cache)
    if cache.exists() and not rebuild:
        payload = pickle.loads(cache.read_bytes())
        if isinstance(payload, dict) and payload.get("suites") == SUITES and isinstance(payload.get("texts"), set):
            return payload["texts"]
    texts = build_set()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({"version": 1, "normalisation": "build_mixture.norm",
                                    "suites": SUITES, "texts": texts}, protocol=pickle.HIGHEST_PROTOCOL))
    return texts


if __name__ == "__main__":
    print("evaluation texts:", len(load()))
