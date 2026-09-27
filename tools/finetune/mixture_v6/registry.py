"""Source registry and row adapters for mixture v6.

Adapters consume a small materialised sample because label vocabularies and
negative pairs are source-local.  They never load data themselves; build.py is
the sole loader and therefore the single enforcement point for ``use: true``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .languages import infer_lang as _infer_lang, to_iso
from .templates import (aspect_name, describe, instruction, score_levels, seeded, shuffle_choice)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "sources" / "v6-keep.json"


def _entries():
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    raw = raw if isinstance(raw, list) else raw["sources"]
    return [r for r in raw if r.get("use") is True]


ENTRIES = _entries()


def source_key(entry):
    return "%s::%s" % (entry["id"], entry.get("config", "default"))


def source_name(entry):
    return "v6/%s/%s" % (entry["id"], entry.get("config") or "default")


def metadata(entry):
    licence = entry.get("licence", "")
    attribution = ""
    if "cc-by" in licence.lower() or "cc by" in licence.lower():
        attribution = "%s; see tools/finetune/sources/v6-research.md" % entry["id"]
    return {"licence": licence, "languages": entry.get("languages", []),
            "category": entry.get("category", ""), "attribution": attribution}


def human(value):
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return text
    text = re.sub(r"[_./-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _get(row, path):
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _paths(entry, which):
    values = entry.get(which, [])
    if isinstance(values, str):
        values = re.split(r"\s*;\s*", values)
    out = []
    for value in values:
        # Documentation after parentheses is not a field name.
        value = re.sub(r"\s*\(.*", "", str(value)).strip()
        if value and not any(c in value for c in "*{}|"):
            out.append(value)
    return out


def _flatten_strings(value):
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, dict):
        preferred = [value[k] for k in ("text", "content", "utterance", "message") if k in value]
        for v in preferred or value.values():
            yield from _flatten_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten_strings(v)


def row_texts(entry, row):
    texts = []
    if entry["id"] == "github:asappresearch/abcd":
        original = row.get("original")
        if isinstance(original, list):
            texts = [str(turn[1]).strip() for turn in original
                     if isinstance(turn, list) and len(turn) > 1 and turn[0] == "customer" and str(turn[1]).strip()]
        elif isinstance(original, str) and original.strip():
            texts = [original.strip()]
    if isinstance(row.get("translation"), dict):
        texts.extend(_flatten_strings(row["translation"]))
    if any("<lang>" in str(x) for x in entry.get("text_fields", [])):
        for key, value in row.items():
            if key.startswith("text_"):
                texts.extend(_flatten_strings(value))
    paths = [] if entry["id"] == "github:asappresearch/abcd" else _paths(entry, "text_fields")
    for path in paths:
        value = _get(row, path)
        if value is None:
            # Nested documentation commonly names the parent with [] syntax.
            root = path.split("[")[0].split(".")[0]
            value = row.get(root) if isinstance(row, dict) else None
        texts.extend(_flatten_strings(value))
    if not texts:
        for key in ("text", "sentence", "utterance", "prompt", "input", "content",
                    "question", "summary", "description", "statement", "clause"):
            texts.extend(_flatten_strings(row.get(key)))
    if entry["id"] == "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1" and row.get("injection"):
        texts = list(_flatten_strings(row["injection"]))
    # Preserve order while removing exact repeats.
    return list(dict.fromkeys(t for t in texts if t))


def state_from(entry, row, texts=None):
    texts = texts or row_texts(entry, row)
    names = _paths(entry, "text_fields")
    if len(texts) == 1:
        return texts[0]
    return "\n\n".join("%s: %s" % (human(names[i]) if i < len(names) else "text %d" % (i + 1), t)
                       for i, t in enumerate(texts))


def infer_lang(entry, row, index=0, text=""):
    return _infer_lang(entry, row, text)


_CANON = {
    "positive": "positive", "negative": "negative", "neutral": "neutral", "mixed": "mixed",
    "hate": "hate", "toxic": "toxic", "nothate": "not hate", "not hate": "not hate",
    "non hateful": "not hate",
}
_MINDS14 = [
    "abroad", "address", "app_error", "atm_limit", "balance", "business_loan",
    "card_issues", "cash_deposit", "direct_debit", "freeze", "high_value_payment",
    "joint_account", "latest_transactions", "pay_bill",
]


def canon(label):
    text = human(label)
    return _CANON.get(text.lower(), text)


def _documented_paths(entry):
    documented = entry.get("label_field", "")
    paths = []
    for raw in re.split(r"\s*;\s*|\s+/\s+", documented):
        raw = re.sub(r"\s*\(.*", "", raw).strip().replace("labels.{", "").rstrip("}")
        if raw and not raw.startswith(("derived:", "none ", "config ", "soft label", "positives ")):
            if not any(c in raw for c in "*{}|"):
                paths.append(raw)
    return paths


def _coerce_labels(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value[:1] in "[{":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return [canon(value)]
    if isinstance(value, dict):
        return [canon(k) for k, v in value.items() if v is True or isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0.5]
    if isinstance(value, (list, tuple, set)):
        return [canon(v) for v in value if v not in (None, "", -1)]
    if isinstance(value, bool) or value is None:
        return []
    return [canon(value)]


def task_for(entry):
    cat = entry.get("category") or ""
    sid = entry["id"]
    fields = " ".join(entry.get("text_fields") or []).lower()
    if sid in ("community-datasets/re_dial", "ai4bharat/IndicSentiment"):
        return "aspect_sentiment"
    if "prompt-injection" in cat:
        return "prompt_injection"
    if "5-spam" in cat:
        return "spam"
    if "5-toxicity-hate" in cat:
        return "toxicity"
    if "5-safety" in cat or "5-toxicity-moderation" in cat:
        return "moderation"
    if "1-sentiment" in cat:
        return "aspect_sentiment" if "aspect" in fields else "sentiment"
    if "2-emotion" in cat:
        return "emotion"
    if "3-complaint" in cat:
        return "complaint_category"
    if "10-urgency" in cat:
        return "urgency"
    if "4-nli" in cat:
        return "nli"
    if "7-similarity" in cat:
        if any(token in fields for token in ("query", "passage", "product_title")):
            return "relevance"
        return "similarity"
    if "8-topic" in cat:
        return "topic"
    if "9-intent" in cat:
        if any(token in sid.lower() for token in ("multi_woz", "crosswoz", "bitod", "casino")):
            return "dialogue_act"
        return "intent"
    if "10-stance" in cat or "10-argument" in cat:
        return "stance"
    if "10-sarcasm" in cat or "10-humor" in cat:
        return "sarcasm"
    if "10-formality" in cat:
        return "formality"
    if "10-fact-check" in cat or "10-claim" in cat:
        return "fact_check"
    if "10-pii" in cat:
        return "pii"
    if "10-language-id" in cat:
        return "language_id"
    if "6-reading" in cat:
        return "reading"
    return None


def task_for_field(entry, path):
    name = (path or "").lower()
    if "sentiment" in name or name.endswith("liked"):
        return "aspect_sentiment" if entry["id"] in ("community-datasets/re_dial", "ai4bharat/IndicSentiment") or "aspect" in name else "sentiment"
    if "urgent" in name or "priority" in name:
        return "urgency"
    if "emotion" in name:
        return "emotion"
    if name in {"dialogue_act", "dialog_act"} or name.endswith("dialogue_act"):
        return "dialogue_act"
    if "intent" in name:
        return "intent"
    if name == "domain":
        return "topic"
    if "hate" in name or "toxic" in name:
        return "toxicity"
    return task_for(entry)


def _render(kind, lang, rng, task, fmt=None):
    text, ilang = instruction(kind, lang, rng, task)
    if not fmt or "{" not in text:
        return text, ilang
    filled = {}
    for key, value in fmt.items():
        if key == "emotion":
            filled[key] = describe("emotion", value, ilang)
        elif key == "aspect":
            filled[key] = aspect_name(value, ilang)
        else:
            filled[key] = value
    try:
        text = text.format(**filled)
    except (KeyError, IndexError, ValueError):
        pass
    return text, ilang


def _label_fields(entry, row):
    documented = entry.get("label_field", "")
    candidates = []
    for raw in re.split(r"\s*;\s*|\s+/\s+", documented):
        raw = re.sub(r"\s*\(.*", "", raw).strip()
        raw = raw.replace("labels.{", "").rstrip("}")
        if raw and not raw.startswith(("derived:", "none ", "config ", "soft label", "positives ")):
            candidates.append(raw)
    for fallback in ("label", "labels", "category", "intent", "sentiment", "emotion", "domain",
                     "gold", "target", "Answer", "Intent_Type", "LABEL", "Label"):
        if fallback not in candidates:
            candidates.append(fallback)
    return [(path, _get(row, path)) for path in candidates if _get(row, path) is not None]


def labels_from(entry, row):
    sid = entry["id"]
    if sid == "community-datasets/re_dial":
        qs = row.get("respondentQuestions") or row.get("initiatorQuestions") or []
        liked = [q.get("liked") for q in qs if isinstance(q, dict) and q.get("liked") in (0, 1)]
        return ["liked" if sum(liked) >= len(liked) / 2 else "not liked"] if liked else []
    if sid == "NagaYu/deference-keigo-corpus" and row.get("n_errors") is not None:
        return ["contains keigo errors" if int(row["n_errors"]) > 0 else "correct keigo"]
    if sid == "joelniklaus/covid19_emergency_event":
        events = row.get("all_events")
        if isinstance(events, str):
            try:
                events = json.loads(events.replace("'", '"'))
            except json.JSONDecodeError:
                events = []
        return [human(x) for x in (events or [])]
    if sid == "NortheasternUniversity/big_patent":
        return [human(row.get("_v6_config", ""))]
    if "multi_woz_v22" in sid and row.get("services"):
        return [human(row["services"][0])]
    if "CrossWOZ" in sid:
        goal = row.get("goal") or []
        domains = [str(x[1]) for x in goal if isinstance(x, list) and len(x) > 1]
        if domains:
            return [human(domains[0])]
    if row.get("emotion_eng"):
        return [human(row["emotion_eng"])]
    fields = _label_fields(entry, row)
    if not fields:
        return []
    value = fields[0][1]
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        if v[:1] in "[{":
            try:
                value = json.loads(v)
            except json.JSONDecodeError:
                pass
    if isinstance(value, dict):
        return [human(k) for k, v in value.items() if v is True or isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0.5]
    if isinstance(value, (list, tuple, set)):
        return [human(v) for v in value if v not in (None, "", -1)]
    return [human(value)]


def _sentiment_scheme(rows):
    vals = set()
    for row in rows:
        value = row.get("sentiment")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        vals.add(int(value))
    if vals and vals <= {-1, 0, 1} and -1 in vals:
        return {-1: "negative", 0: "neutral", 1: "positive"}
    return {}


def field_labels(entry, row, scheme):
    sid = entry["id"]
    if sid == "PolyAI/minds14" and isinstance(row.get("intent_class"), (int, float)) and not isinstance(row.get("intent_class"), bool):
        index = int(row["intent_class"])
        if 0 <= index < len(_MINDS14):
            return [("issue", _MINDS14[index])]
    if "ClaimBuster" in sid and row.get("Verdict") not in (None, ""):
        try:
            verdict = {-1: "non-factual", 0: "unimportant factual", 1: "check-worthy"}[int(float(row["Verdict"]))]
            return [("Verdict", verdict)]
        except (KeyError, TypeError, ValueError):
            pass
    if "BiToD" in sid or sid.startswith("github:HLTCHKUST"):
        intent = row.get("action-inform_intent-intent")
        if isinstance(intent, str) and intent not in ("", "none"):
            return [("intent", human(intent))]
        acts = [human(key[len("action-"):]) for key, value in row.items()
                if key.startswith("action-") and isinstance(value, str) and value not in ("", "none")]
        if acts:
            return [("dialogue_act", acts[0])]
    if "multi_woz" in sid and (row.get("intent") or row.get("dialogue_act")) and not row.get("services"):
        found = []
        if row.get("dialogue_act"):
            found.append(("dialogue_act", human(row["dialogue_act"])))
        if row.get("intent"):
            label = human(row["intent"])
            if not found or found[0][1] != label:
                found.append(("intent", label))
        if found:
            return found
    found = []
    for path in _documented_paths(entry):
        value = _get(row, path)
        if value is None:
            continue
        if scheme and "sentiment" in path.lower() and isinstance(value, (int, float)) and not isinstance(value, bool):
            mapped = scheme.get(int(value))
            if mapped:
                found.append((path, mapped))
                continue
        labels = _coerce_labels(value)
        if labels:
            found.append((path, labels[0]))
    if found:
        return found
    labels = labels_from(entry, row)
    if labels:
        return [("", canon(labels[0]))]
    return []


def _base(entry, row, state, q, target, lang, texts):
    return {"state": state, "q": q, "target": target, "src": source_name(entry),
            "lang": lang, "_texts": texts}


def _choice(entry, row, state, options, gold, lang, rng, texts, task=None, prompt=None):
    options = list(dict.fromkeys(canon(x) for x in options))
    gold = canon(gold)
    if len(options) < 2 or gold not in options:
        return None
    pos = options.index(gold)
    options, target = shuffle_choice(options, pos, rng)
    text, ilang = _render("choice", lang, rng, task)
    q = {"type": "choice", "instructions": prompt or text,
         "criteria": {x: describe(task, x, ilang) for x in options}}
    return _base(entry, row, state, q, target, lang, texts)


def _noul(entry, row, state, truth, lang, rng, texts, task=None, fmt=None, statement=None):
    text, _ilang = _render("noul", lang, rng, task, fmt)
    q = {"type": "noul", "instructions": statement or text}
    return _base(entry, row, state, q, [0.0, 1.0] if truth else [1.0, 0.0], lang, texts)


def _score(entry, row, state, levels, gold, lang, rng, texts, task=None, fmt=None, prompt=None):
    if len(levels) < 2 or not 0 <= gold < len(levels):
        return None
    text, ilang = _render("score", lang, rng, task, fmt)
    levels = score_levels(task, ilang, levels)
    q = {"type": "score", "instructions": prompt or text, "criteria": list(levels)}
    return _base(entry, row, state, q, [1.0 if i == gold else 0.0 for i in range(len(levels))], lang, texts)


def _vocabulary(entry, rows, limit=80):
    counts = {}
    for row in rows:
        for label in labels_from(entry, row):
            label = canon(label)
            counts[label] = counts.get(label, 0) + 1
    return [x for x, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def classification_adapter(entry, rows, seed):
    rows = list(rows)
    scheme = _sentiment_scheme(rows)
    parsed = [field_labels(entry, row, scheme) for row in rows]
    buckets = {}
    for fields in parsed:
        for path, label in fields:
            buckets.setdefault(path, {})
            buckets[path][label] = buckets[path].get(label, 0) + 1
    vocabs = {path: [x for x, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:80]]
              for path, counts in buckets.items()}
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        if not texts:
            continue
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
        for path, label in parsed[i]:
            vocab = vocabs.get(path) or []
            if label not in vocab:
                continue
            task = task_for_field(entry, path)
            rng = seeded(seed, source_key(entry), i, path, state)
            item = _choice(entry, row, state, vocab, label, lang, rng, texts, task=task)
            if item:
                yield item


def emotion_adapter(entry, rows, seed):
    rows = list(rows)
    vocab = _vocabulary(entry, rows)
    for i, row in enumerate(rows):
        texts, labels = row_texts(entry, row), [canon(x) for x in labels_from(entry, row)]
        if not texts or not labels:
            continue
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
        rng = seeded(seed, source_key(entry), i, state)
        item = _choice(entry, row, state, vocab, labels[0], lang, rng, texts, task="emotion")
        if item:
            yield item
        if vocab:
            probe = rng.choice(vocab)
            yield _noul(entry, row, state, probe in labels, lang, rng, texts, task="emotion", fmt={"emotion": probe})


NLI = ["entailment", "neutral", "contradiction"]


def nli_adapter(entry, rows, seed):
    aliases = {"0": "entailment", "1": "neutral", "2": "contradiction", "e": "entailment",
               "n": "neutral", "c": "contradiction", "entails": "entailment", "contradictory": "contradiction"}
    for i, row in enumerate(rows):
        texts, labels = row_texts(entry, row), labels_from(entry, row)
        if len(texts) < 2:
            texts = list(dict.fromkeys(texts + [str(row[k]) for k in ("premise", "premise_ja", "text", "hypothesis", "hypothesis_ja")
                                                if row.get(k)]))
        if len(texts) < 2 or not labels:
            continue
        gold = aliases.get(labels[0].lower(), labels[0].lower())
        if gold not in NLI:
            continue
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts[:2])
        rng = seeded(seed, source_key(entry), i, state)
        yield _choice(entry, row, state, NLI, gold, lang, rng, texts, task="nli")
        yield _noul(entry, row, state, gold == "entailment", lang, rng, texts, task="nli")


SIM_LEVELS = ["unrelated", "weakly related", "moderately related", "closely related", "same meaning"]


def _tapaco_pairs(rows):
    if not rows or not any(row.get("paraphrase_set_id") and row.get("paraphrase") for row in rows):
        return None
    groups = {}
    for row in rows:
        text = row.get("paraphrase")
        if not text or row.get("paraphrase_set_id") in (None, ""):
            continue
        lang = row.get("_v6_lang") or row.get("language") or ""
        groups.setdefault((lang, str(row["paraphrase_set_id"])), []).append(str(text))
    made, seeds = [], {}
    for (lang, _set_id), sents in groups.items():
        uniq = list(dict.fromkeys(sents))
        seeds.setdefault(lang, []).append(uniq[0])
        if len(uniq) >= 2:
            made.append({"sentence1": uniq[0], "sentence2": uniq[1], "relatedness_score": 1.0,
                         "language": lang, "_v6_lang": lang})
    for lang, texts in seeds.items():
        if len(texts) < 2:
            continue
        for i, text in enumerate(texts):
            other = texts[(i + max(1, len(texts) // 2)) % len(texts)]
            if other != text:
                made.append({"sentence1": text, "sentence2": other, "relatedness_score": 0.0,
                             "language": lang, "_v6_lang": lang})
    return made or None


def similarity_adapter(entry, rows, seed):
    rows = list(rows)
    paired = _tapaco_pairs(rows)
    if paired:
        rows = paired
    task = task_for(entry) or "similarity"
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        if len(texts) < 2:
            texts = list(dict.fromkeys(texts + [str(row[k]) for k in ("sentence1", "sentence2", "translation", "answer",
                                                                       "product_title", "product_description") if row.get(k)]))
        if len(texts) < 2 and i + 1 < len(rows):
            other = row_texts(entry, rows[i + 1])
            if other:
                texts.append(other[0])
        if len(texts) < 2:
            continue
        raw = next((v for _k, v in _label_fields(entry, row) if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if raw is None and isinstance(row.get("relatedness_score"), (int, float)):
            raw = row["relatedness_score"]
        if raw is None:
            label = (labels_from(entry, row) or [""])[0].lower()
            same = label in {"1", "true", "yes", "relevant", "duplicate", "paraphrase", "same"}
            score = 4 if same else 0
        else:
            value = float(raw)
            value = value / 5.0 if value > 1 else value
            score = max(0, min(4, round(value * 4)))
            same = value >= 0.8
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts[:2])
        rng = seeded(seed, source_key(entry), i, state, score)
        item = _score(entry, row, state, SIM_LEVELS, score, lang, rng, texts, task=task)
        if item:
            yield item
        if task == "similarity":
            yield _noul(entry, row, state, same, lang, rng, texts, task="similarity")


def complaint_adapter(entry, rows, seed):
    yield from classification_adapter(entry, rows, seed)
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        if not texts:
            continue
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
        rng = seeded(seed, source_key(entry), "complaint", i, state)
        yield _noul(entry, row, state, True, lang, rng, texts, task="complaint_detection")


def safety_adapter(entry, rows, seed):
    rows = list(rows)
    negative_words = {"safe", "valid", "benign", "not hate", "nothate", "non hateful", "0", "false"}
    task = task_for(entry) or "moderation"
    vocab_counts = {}
    prepared = []
    for row in rows:
        labels = [canon(x).lower() for x in labels_from(entry, row)]
        prepared.append(labels)
        for label in labels:
            vocab_counts[label] = vocab_counts.get(label, 0) + 1
    vocab = [x for x, _ in sorted(vocab_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        if not texts:
            continue
        labels = prepared[i]
        truth = not labels or not any(x in negative_words for x in labels)
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
        rng = seeded(seed, source_key(entry), i, state)
        if labels and len(vocab) >= 2 and labels[0] in vocab and not all(str(x).isdigit() for x in vocab):
            item = _choice(entry, row, state, vocab, labels[0], lang, rng, texts, task=task)
            if item:
                yield item
        yield _noul(entry, row, state, truth, lang, rng, texts, task=task)


def positive_adapter(entry, rows, seed):
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        if texts:
            lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
            rng = seeded(seed, source_key(entry), i, state)
            text, _ilang = _render("noul", lang, rng, "sarcasm")
            if _ilang == "en":
                text = "Is this text humorous?"
            yield _noul(entry, row, state, True, lang, rng, texts, statement=text)


def keigo_adapter(entry, rows, seed):
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        if texts and row.get("n_errors") is not None:
            lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
            rng = seeded(seed, source_key(entry), i, state)
            yield _noul(entry, row, state, int(row["n_errors"]) == 0, lang, rng, texts, task="formality")


def empathic_adapter(entry, rows, seed):
    levels = ["very low", "low", "moderate", "high", "very high"]
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        for field in ("empathy", "distress"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            gold = max(0, min(4, round((value - 1) * 4 / 6)))
            lang, state = "en", state_from(entry, row, texts)
            rng = seeded(seed, source_key(entry), i, field)
            item = _score(entry, row, state, levels, gold, lang, rng, texts, task="emotion", fmt={"aspect": field})
            if item:
                yield item


def language_adapter(entry, rows, seed):
    if entry["id"] != "Helsinki-NLP/tatoeba":
        yield from classification_adapter(entry, rows, seed)
        return
    options = sorted({lang for row in rows for lang in (row.get("translation") or {})})
    for i, row in enumerate(rows):
        for lang, text in (row.get("translation") or {}).items():
            if not text:
                continue
            code = to_iso(lang) or lang
            rng = seeded(seed, source_key(entry), i, lang)
            choices = [to_iso(x) or x for x in options]
            item = _choice(entry, row, str(text), choices, code, code if len(code) == 2 else "en", rng, [str(text)],
                           task="language_id")
            if item:
                yield item


def _window_context(row):
    context = row.get("context")
    if not isinstance(context, str) or len(context) <= 1800:
        return row
    cloned = dict(row)
    answers = row.get("answers") if isinstance(row.get("answers"), dict) else {}
    starts = answers.get("answer_start") or []
    if starts:
        try:
            start = int(starts[0])
        except (TypeError, ValueError):
            start = 0
        cloned["context"] = context[max(0, start - 700):start + 900]
    else:
        cloned["context"] = context[:1500]
    return cloned


def reading_adapter(entry, rows, seed):
    for i, row in enumerate(rows):
        row = _window_context(row)
        texts = row_texts(entry, row)
        if not texts:
            continue
        lang, state, rng = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts), seeded(seed, source_key(entry), i)
        options = row.get("options") or row.get("choices")
        if isinstance(options, dict):
            options = options.get("text") or list(options.values())
        gold = row.get("gold_label", row.get("label"))
        if isinstance(options, list) and len(options) >= 2 and isinstance(gold, str) and human(gold) in [human(x) for x in options]:
            item = _choice(entry, row, state, [human(x) for x in options], human(gold), lang, rng, texts, task="reading")
            if item:
                yield item
                continue
        if isinstance(options, list) and len(options) >= 2 and gold is not None:
            try:
                gi = int(gold) - (1 if int(gold) >= 1 else 0)
                if 0 <= gi < len(options):
                    yield _choice(entry, row, state, [human(x) for x in options], human(options[gi]), lang, rng, texts,
                                  task="reading")
                    continue
            except (TypeError, ValueError):
                pass
        answers = row.get("answers")
        present = bool(answers and (not isinstance(answers, dict) or answers.get("text")))
        yield _noul(entry, row, state, present, lang, rng, texts, task="reading")


def paired_style_adapter(entry, rows, seed):
    fields = _paths(entry, "text_fields")
    task = "formality" if "formality" in entry.get("category", "") else "sarcasm"
    for i, row in enumerate(rows):
        values = [(f, _get(row, f)) for f in fields if isinstance(_get(row, f), str) and _get(row, f).strip()]
        for j, (field, value) in enumerate(values):
            lang, rng = infer_lang(entry, row, text=value), seeded(seed, source_key(entry), i, j)
            if task == "formality":
                truth = "formal" in field.lower() and "informal" not in field.lower()
            else:
                truth = field.lower() == "translation"
            yield _noul(entry, row, value.strip(), truth, lang, rng, [value.strip()], task=task)


def typed_adapter(entry, rows, seed):
    for i, row in enumerate(rows):
        state = row.get("state")
        question = row.get("question")
        choices = row.get("choices") or row.get("options")
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                try:
                    choices = json.loads(choices.replace("'", '"'))
                except json.JSONDecodeError:
                    choices = None
        target = row.get("target") or row.get("label") or row.get("soft_label") or row.get("probs") or row.get("gold")
        if isinstance(target, str) and target.startswith("["):
            try:
                target = json.loads(target)
            except json.JSONDecodeError:
                pass
        if not isinstance(state, str) or not state.strip() or not isinstance(choices, list) or len(choices) < 2:
            continue
        if isinstance(target, list) and len(target) == len(choices):
            gold = max(range(len(target)), key=target.__getitem__)
        else:
            try:
                gold = int(target)
            except (TypeError, ValueError):
                continue
        lang, rng = infer_lang(entry, row, text=state), seeded(seed, source_key(entry), i, state)
        item = _choice(entry, row, state, choices, choices[gold], lang, rng, [state], prompt=str(question) if question else None)
        if item:
            yield item


def redial_adapter(entry, rows, seed):
    options = ["liked", "not liked", "did not say"]
    liked_name = {0: "not liked", 1: "liked", 2: "did not say"}
    for i, row in enumerate(rows):
        names = {}
        for mention in row.get("movieMentions") or []:
            if isinstance(mention, dict) and mention.get("movieId") is not None:
                names[str(mention["movieId"])] = mention.get("movieName") or str(mention["movieId"])
        lines = [m.get("text").strip() for m in (row.get("messages") or []) if isinstance(m, dict) and str(m.get("text") or "").strip()]
        dialogue = "\n".join(lines)[:2000]
        if not dialogue:
            continue
        questions = list(row.get("respondentQuestions") or []) or list(row.get("initiatorQuestions") or [])
        seen = set()
        for question in questions:
            if not isinstance(question, dict):
                continue
            label = liked_name.get(question.get("liked"))
            movie_id = str(question.get("movieId"))
            if not label or movie_id in seen:
                continue
            seen.add(movie_id)
            title = names.get(movie_id) or movie_id
            state = "Movie: %s\n\n%s" % (title, dialogue)
            rng = seeded(seed, source_key(entry), i, movie_id)
            item = _choice(entry, row, state, options, label, "en", rng, [dialogue, title], task="aspect_sentiment")
            if item:
                yield item


def fact_adapter(entry, rows, seed):
    rows = list(rows)
    if not any(row.get("correct_answers") and row.get("incorrect_answers") and row.get("question") for row in rows):
        yield from classification_adapter(entry, rows, seed)
        return
    for i, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        correct = row.get("correct_answers") or []
        incorrect = row.get("incorrect_answers") or []
        if isinstance(correct, str):
            correct = [correct]
        if isinstance(incorrect, str):
            incorrect = [incorrect]
        if not question or not correct or not incorrect:
            continue
        lang = infer_lang(entry, row, text=question)
        pairs = [(True, str(correct[0]).strip()), (False, str(incorrect[0]).strip())]
        for truth, answer in pairs:
            if not answer:
                continue
            state = "Question: %s\n\nAnswer: %s" % (question, answer)
            rng = seeded(seed, source_key(entry), i, truth, answer)
            yield _noul(entry, row, state, truth, lang, rng, [question, answer], task="fact_check")


def pii_adapter(entry, rows, seed):
    rows = list(rows)
    if not any(isinstance(row.get("tokenized_text"), list) for row in rows):
        yield from classification_adapter(entry, rows, seed)
        return
    prepared, counts = [], {}
    for row in rows:
        tokens = [str(tok) for tok in (row.get("tokenized_text") or [])]
        text = " ".join(tokens).strip()
        kinds = []
        for span in row.get("ner") or []:
            if isinstance(span, (list, tuple)) and len(span) >= 3 and str(span[2]).strip():
                kinds.append(human(span[2]))
        prepared.append((text, kinds))
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
    vocab = [x for x, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
    for i, (text, kinds) in enumerate(prepared):
        if not text:
            continue
        lang = infer_lang(entry, rows[i], text=text)
        rng = seeded(seed, source_key(entry), i, text[:80])
        yield _noul(entry, rows[i], text, bool(kinds), lang, rng, [text], task="pii")
        if kinds and len(vocab) >= 2 and kinds[0] in vocab:
            item = _choice(entry, rows[i], text, vocab, kinds[0], lang, rng, [text], task="pii")
            if item:
                yield item


def adapter_for(entry):
    cat, sid = entry["category"], entry["id"]
    if sid == "community-datasets/re_dial":
        return redial_adapter
    if sid == "Eurolingua/truthfulqax":
        return fact_adapter
    if sid == "urchade/synthetic-pii-ner-mistral-v1":
        return pii_adapter
    if "typed-decisions" in cat:
        return typed_adapter
    if "language-id" in cat:
        return language_adapter
    if "reading-comprehension" in cat:
        return reading_adapter
    if "similarity" in cat:
        return similarity_adapter
    if "nli" in cat:
        return nli_adapter
    if "emotion" in cat:
        if sid == "github:wwbp/empathic_reactions":
            return empathic_adapter
        return emotion_adapter
    if "complaint" in cat:
        return complaint_adapter
    if any(x in cat for x in ("safety", "toxicity", "prompt-injection", "spam")):
        return safety_adapter
    if sid in ("GoktugD/turkish-formality-rewrite-500k", "sweatSmile/sarcastic-dataset"):
        return paired_style_adapter
    if sid == "NagaYu/deference-keigo-corpus":
        return keigo_adapter
    if sid == "yoonholee/humor-greats-public-domain":
        return positive_adapter
    return classification_adapter


ADAPTERS = {source_key(entry): adapter_for(entry) for entry in ENTRIES}


def adapt(entry, rows, seed):
    yield from ADAPTERS[source_key(entry)](entry, list(rows), seed)
