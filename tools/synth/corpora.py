"""Pinned, licence-clean grounding corpora for synthetic-data generation.

The public iterator yields ``(text, lang, source_id, revision)``.  Downloads are
streamed through Hugging Face datasets and its cache is forced below ``cache``;
callers must pass an explicit path or accept a directory below ``$TMPDIR``.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path


CORPORA = {
    "billsum": {
        "repo": "FiscalNote/billsum",
        "revision": "3d8510441c06a3d9dfb32eb0d7f80151730bcc4f",
        "licence": "CC0-1.0",
        "licence_evidence": "https://huggingface.co/datasets/FiscalNote/billsum/blob/3d8510441c06a3d9dfb32eb0d7f80151730bcc4f/README.md",
    },
    "gov_report": {
        "repo": "launch/gov_report",
        "revision": "32feeaede49fed993aef070bc4da09263fd0429a",
        "licence": "CC-BY-4.0",
        "licence_evidence": "https://huggingface.co/datasets/launch/gov_report/blob/32feeaede49fed993aef070bc4da09263fd0429a/README.md",
    },
    "eur_lex_sum": {
        "repo": "dennlinger/eur-lex-sum",
        "revision": "33ecb2d630298e3f912d067aaaa71aaf4ee92404",
        "licence": "CC-BY-4.0",
        "licence_evidence": "https://huggingface.co/datasets/dennlinger/eur-lex-sum/blob/33ecb2d630298e3f912d067aaaa71aaf4ee92404/README.md",
    },
}

EUR_LANGUAGES = {
    "bulgarian": "bg", "croatian": "hr", "czech": "cs", "danish": "da",
    "dutch": "nl", "english": "en", "estonian": "et", "finnish": "fi",
    "french": "fr", "german": "de", "greek": "el", "hungarian": "hu",
    "irish": "ga", "italian": "it", "latvian": "lv", "lithuanian": "lt",
    "maltese": "mt", "polish": "pl", "portuguese": "pt", "romanian": "ro",
    "slovak": "sk", "slovenian": "sl", "spanish": "es", "swedish": "sv",
}
PILOT_LANGUAGES = ("en", "de", "fr", "es", "it", "pt", "nl", "pl")
_SPACE = re.compile(r"[ \t]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def default_cache() -> Path:
    return Path(tempfile.gettempdir()) / "statim-synth-corpora"


def validate_cache(cache) -> Path:
    path = Path(cache) if cache else default_cache()
    data = (Path(__file__).resolve().parents[2] / "data").resolve()
    resolved = path.expanduser().resolve()
    if resolved == data or data in resolved.parents:
        raise ValueError("corpus cache must not be under data/")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _words(text):
    return text.split()


def _split_long(text, maximum):
    sentences = _SENTENCE.split(text)
    chunks, current = [], []
    for sentence in sentences:
        words = _words(sentence)
        if len(words) > maximum:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(" ".join(words[i:i + maximum]) for i in range(0, len(words), maximum))
        elif len(current) + len(words) > maximum:
            chunks.append(" ".join(current))
            current = words
        else:
            current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_passages(text, minimum=60, maximum=400):
    """Make paragraph-sized passages, combining short and splitting long ones."""
    raw = re.split(r"\n\s*\n|(?<=\n)(?=\S)", str(text or "").replace("\r", "\n"))
    paragraphs = [_SPACE.sub(" ", p.replace("\n", " ")).strip() for p in raw]
    units = [chunk for p in paragraphs if p for chunk in _split_long(p, maximum)]
    out, current = [], []
    for unit in units:
        words = _words(unit)
        if current and len(current) + len(words) > maximum:
            if len(current) >= minimum:
                out.append(" ".join(current))
                current = []
        current.extend(words)
        if len(current) >= minimum:
            out.append(" ".join(current))
            current = []
    if current:
        if out and len(_words(out[-1])) + len(current) <= maximum:
            out[-1] += " " + " ".join(current)
        elif len(current) >= minimum:
            out.append(" ".join(current))
    return [x for x in out if minimum <= len(_words(x)) <= maximum]


def _url(spec, filename):
    return "https://huggingface.co/datasets/%s/resolve/%s/%s" % (
        spec["repo"], spec["revision"], filename)


def _load_json(urls, cache):
    from datasets import load_dataset
    return load_dataset("json", data_files={"train": urls}, split="train", streaming=True,
                        cache_dir=str(cache))


def _raw_rows(name, cache, languages):
    spec = CORPORA[name]
    from datasets import load_dataset
    if name == "billsum":
        url = _url(spec, "data/train-00000-of-00001.parquet")
        rows = load_dataset("parquet", data_files={"train": url}, split="train", streaming=True,
                            cache_dir=str(cache))
        for row in rows:
            yield dict(row, _pinned_revision=spec["revision"])
    elif name == "gov_report":
        for row in _load_json([_url(spec, "data/gao_train.jsonl"),
                               _url(spec, "data/crs_train.jsonl")], cache):
            yield dict(row, _pinned_revision=spec["revision"])
    else:
        wanted = set(languages or EUR_LANGUAGES.values())
        for dirname, lang in EUR_LANGUAGES.items():
            if lang not in wanted:
                continue
            for row in _load_json(_url(spec, "data/%s/train.json" % dirname), cache):
                row = dict(row)
                row["_lang"] = lang
                row["_pinned_revision"] = spec["revision"]
                yield row


def _sections(section):
    if not isinstance(section, dict):
        return
    for paragraph in section.get("paragraphs") or []:
        yield paragraph
    for child in section.get("subsections") or []:
        yield from _sections(child)


def _row_text(name, row):
    if name == "billsum":
        ident = row.get("bill_id") or row.get("id")
        return row.get("text"), "en", (
            "FiscalNote/billsum:train:%s" % ident if ident else None)
    if name == "eur_lex_sum":
        ident = row.get("celex_id")
        return row.get("reference"), row.get("_lang"), (
            "dennlinger/eur-lex-sum:train:%s:%s" % (row.get("_lang"), ident)
            if ident else None)
    ident = row.get("id")
    if row.get("report") is not None:
        text = "\n\n".join(p for section in row["report"] for p in _sections(section))
    else:
        text = "\n\n".join(_sections(row.get("reports") or {}))
    return text, "en", "launch/gov_report:train:%s" % ident if ident else None


def iter_corpus(name, cache=None, languages=None, rows=None):
    """Yield deduplicated passages. ``rows`` is an injection point for tiny tests."""
    if name not in CORPORA:
        raise ValueError("unknown corpus %r" % name)
    cache = validate_cache(cache)
    seen = set()
    fetched = rows is None
    source = rows if rows is not None else _raw_rows(name, cache, languages)
    for row in source:
        text, lang, source_id = _row_text(name, row)
        if not source_id:
            continue
        if languages and lang not in set(languages):
            continue
        for passage in split_passages(text):
            digest = hashlib.sha256(" ".join(passage.lower().split()).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            # Only _raw_rows can prove that the row came from the immutable URL.
            yield passage, lang, source_id, row.get("_pinned_revision") if fetched else None


def iter_pilot(cache=None, languages=PILOT_LANGUAGES):
    """English government text plus the seven requested EUR-Lex languages."""
    yield from iter_corpus("billsum", cache, languages=("en",))
    yield from iter_corpus("gov_report", cache, languages=("en",))
    yield from iter_corpus("eur_lex_sum", cache, languages=tuple(x for x in languages if x != "en"))
