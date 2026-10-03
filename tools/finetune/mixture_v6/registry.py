"""Source registry and row adapters for mixture v6.

Adapters consume a small materialised sample because label vocabularies and
negative pairs are source-local.  They never load data themselves; build.py is
the sole loader and therefore the single enforcement point for ``use: true``.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from . import emotion_taxonomy
from .languages import infer_lang as _infer_lang, to_iso
from .templates import (aspect_name, describe, instruction, score_levels, seeded, shuffle_choice)
from tools.finetune.source_policy import exclusion_matches, row_is_excluded

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "sources" / "v6-keep.json"
POLICY_PATH = REGISTRY_PATH.with_name("policy.json")


def _entries():
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    raw = raw if isinstance(raw, list) else raw["sources"]
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    excluded = [x for x in policy["exclusions"] if x["registry"] == "v6" and x["scope"] == "source"]
    for entry in raw:
        matches = [x for x in excluded if exclusion_matches(
            "v6", entry["id"], entry.get("config", "default"), x)]
        if matches and entry.get("use") is True:
            raise ValueError("licence-excluded v6 source is enabled: %s::%s" %
                             (entry["id"], entry.get("config", "default")))
    return raw


ENTRIES = _entries()
ENABLED_ENTRIES = [entry for entry in ENTRIES if entry.get("use") is True]


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


def _parse_structured(text):
    """JSON, or a Python literal such as "['Theft']" (several sources store lists that way)."""
    import ast
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


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
            value = _parse_structured(value)
            if isinstance(value, str):
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
        if any(token in fields for token in ("query", "passage", "product_title", "answer")):
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
    if name in ("domain", "topic"):
        return "topic"
    if name == "style" and entry["id"] == "leonvanbokhorst/synthetic-complaints-v2":
        return "emotion"  # the tone the complaint was written in (annoyed, bitter, ...)
    if "hate" in name or "toxic" in name:
        return "toxicity"
    if "3-complaint" in entry.get("category", ""):
        # "3-complaint; 1-sentiment" sources: only a sentiment field asks for sentiment (above);
        # a category field is the kind of issue.
        return "complaint_category"
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


# big_patent configs are CPC section letters; the letter alone is not a topic a reader can pick.
CPC_SECTIONS = {
    "a": "human necessities", "b": "performing operations; transporting", "c": "chemistry; metallurgy",
    "d": "textiles; paper", "e": "fixed constructions",
    "f": "mechanical engineering; lighting; heating; weapons; blasting", "g": "physics", "h": "electricity",
    "y": "general tagging of new technological developments",
}
# joelniklaus/covid19_emergency_event card: event1..event8 name COVID-19 measures.
COVID_EVENTS = {
    "event1": "state of emergency", "event2": "restrictions of fundamental rights and civil liberties",
    "event3": "restrictions of daily liberties", "event4": "closures or lockdown",
    "event5": "suspension of international cooperation and commitments", "event6": "police mobilization",
    "event7": "army mobilization", "event8": "government oversight",
}
AEGIS1_SAFE = {"safe", "needs caution"}
# Unsafe, but the annotators disagree on the category: a yes/no item, no category choice.
UNSAFE_NO_CATEGORY = "unsafe category disputed"  # survives canon() unchanged
# prosocial-dialog labels that are neither safe nor clearly unsafe: no yes/no item.
UNDECIDED = {"possibly needs caution"}


def _safety_labels(sid, row):
    """Labels for safety sources whose label columns the generic parser cannot read. Returns None
    for any other source. "safe" is a negative word in safety_adapter."""
    if sid == "nvidia/Aegis-AI-Content-Safety-Dataset-1.0":
        # labels_0..labels_4: one annotation each ("None" = no annotator). Majority vote; a "Needs
        # Caution" majority or a safe/unsafe tie yields no label (the row is dropped). An unsafe
        # majority split between categories is unsafe without a category: the yes/no item only.
        votes = [str(row.get("labels_%d" % i) or "").strip() for i in range(5)]
        votes = [v for v in votes if v and v != "None"]
        safe = sum(v.lower() == "safe" for v in votes)
        unsafe = [v for v in votes if v.lower() not in AEGIS1_SAFE]
        if safe * 2 > len(votes):
            return ["safe"]
        if len(unsafe) * 2 > len(votes):
            counts = collections.Counter(unsafe).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                return [UNSAFE_NO_CATEGORY]
            return [counts[0][0]]
        return []
    if sid == "OpenAssistant/oasst2":
        # labels = {"name": [...], "value": [...], "count": [...]}, value = mean crowd vote in [0, 1]
        labels = row.get("labels")
        if isinstance(labels, str):
            labels = _parse_structured(labels)
        if not isinstance(labels, dict) or not labels.get("name"):
            return []
        scores = dict(zip(labels.get("name") or [], labels.get("value") or []))
        harmful = ("toxicity", "hate_speech", "not_appropriate", "sexual_content", "violence")
        worst = max(harmful, key=lambda k: scores.get(k) or 0.0)
        return [human(worst)] if (scores.get(worst) or 0.0) >= 0.5 else ["safe"]
    return None


def labels_from(entry, row):
    sid = entry["id"]
    safety = _safety_labels(sid, row)
    if safety is not None:
        return safety
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
        return [COVID_EVENTS.get(str(x), human(x)) for x in (events or [])]
    if sid == "NortheasternUniversity/big_patent":
        section = CPC_SECTIONS.get(str(row.get("_v6_config", "")).lower())
        return [section] if section else []
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
            value = _parse_structured(v)
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


# arXiv archives (the part of the primary category before the dot) as readable topics.
ARXIV_ARCHIVES = {
    "astro-ph": "astrophysics", "cond-mat": "condensed matter physics", "cs": "computer science",
    "econ": "economics", "eess": "electrical engineering and systems science", "gr-qc": "general relativity",
    "hep-ex": "particle physics", "hep-lat": "particle physics", "hep-ph": "particle physics",
    "hep-th": "particle physics", "math": "mathematics", "math-ph": "mathematical physics",
    "nlin": "nonlinear sciences", "nucl-ex": "nuclear physics", "nucl-th": "nuclear physics",
    "physics": "physics", "q-bio": "quantitative biology", "q-fin": "quantitative finance",
    "quant-ph": "quantum physics", "stat": "statistics",
}
# English Wikinews top-level topic categories. The other categories of an article are people,
# places and maintenance tags, so a row is used only when exactly one of these is present.
WIKINEWS_TOPICS = {
    "Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business",
    "Education", "Environment", "Health", "Obituaries", "Politics and conflicts", "Science and technology",
    "Sports", "Weather",
}


def _polarity(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return "negative" if value < -0.1 else ("positive" if value > 0.1 else "neutral")


def _stars(value):
    """A 1-5 star rating as sentiment; the digits are not options a reader can interpret."""
    try:
        stars = float(value)
    except (TypeError, ValueError):
        return None
    if not 1 <= stars <= 5:
        return None
    return "negative" if stars <= 2 else ("neutral" if stars < 4 else "positive")


def _special_field_labels(sid, row):
    """(path, label) pairs for sources whose documented label field needs interpretation. None for
    every other source."""
    if sid == "IDinsight/urgency_detection_maternal_health_synthetic":
        # matching_rule names one of 35 warning signs or NOT URGENT; the decision is urgent or not.
        rule = str(row.get("matching_rule") or "").strip()
        return [("urgency", "not urgent" if rule.upper() == "NOT URGENT" else "urgent")] if rule else []
    if sid == "leonvanbokhorst/synthetic-complaints-v2":
        found = []
        if row.get("topic"):
            found.append(("topic", human(row["topic"])))
        if row.get("style"):
            found.append(("style", human(row["style"])))
        polarity = _polarity(row.get("sentiment"))  # TextBlob polarity in [-1, 1]
        if polarity:
            found.append(("sentiment", polarity))
        return found
    if sid == "gfissore/arxiv-abstracts-2021":
        cats = row.get("categories")
        if isinstance(cats, str):
            cats = _parse_structured(cats) if cats[:1] == "[" else cats
        first = (cats[0] if isinstance(cats, list) and cats else cats) or ""
        primary = str(first).split()[0] if str(first).split() else ""
        archive = ARXIV_ARCHIVES.get(primary.split(".")[0])
        return [("categories", archive)] if archive else []
    if sid == "Fumika/Wikinews-multilingual":
        cats = row.get("categories")
        if isinstance(cats, str):
            cats = _parse_structured(cats)
        topics = [c for c in (cats or []) if c in WIKINEWS_TOPICS] if isinstance(cats, list) else []
        return [("categories", topics[0])] if len(topics) == 1 else []
    if sid == "vic35get/nhtsa_complaints_dataset":
        components = str(row.get("components") or "").strip()
        return [("components", components)] if components and "," not in components else []
    return None


def field_labels(entry, row, scheme):
    sid = entry["id"]
    special = _special_field_labels(sid, row)
    if special is not None:
        return special
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
    found, ambiguous, stars = [], False, None
    for path in _documented_paths(entry):
        value = _get(row, path)
        if value is None:
            continue
        if scheme and "sentiment" in path.lower() and isinstance(value, (int, float)) and not isinstance(value, bool):
            mapped = scheme.get(int(value))
            if mapped:
                found.append((path, mapped))
                continue
        if "rating" in path.lower() and "1-sentiment" in entry.get("category", ""):
            stars = stars or _stars(value)  # used only when the row has no sentiment field
            continue
        labels = list(dict.fromkeys(_coerce_labels(value)))
        if len(labels) == 1:
            found.append((path, labels[0]))
        elif labels:
            ambiguous = True  # several labels on one row: no single gold option for this field
    if stars and not any("sentiment" in path.lower() for path, _label in found):
        found.append(("sentiment", stars))
    if found or ambiguous:
        return found
    labels = list(dict.fromkeys(canon(x) for x in labels_from(entry, row)))
    if len(labels) == 1:
        return [("", labels[0])]
    return []


def _base(entry, row, state, q, target, lang, texts, task=None):
    # _task (the adapter task) is internal like _texts: build.py removes both before writing.
    return {"state": state, "q": q, "target": target, "src": source_name(entry),
            "lang": lang, "_texts": texts, "_task": task}


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
    return _base(entry, row, state, q, target, lang, texts, task)


def _noul(entry, row, state, truth, lang, rng, texts, task=None, fmt=None, statement=None):
    text, _ilang = _render("noul", lang, rng, task, fmt)
    q = {"type": "noul", "instructions": statement or text}
    return _base(entry, row, state, q, [0.0, 1.0] if truth else [1.0, 0.0], lang, texts, task)


def _score(entry, row, state, levels, gold, lang, rng, texts, task=None, fmt=None, prompt=None):
    if len(levels) < 2 or not 0 <= gold < len(levels):
        return None
    text, ilang = _render("score", lang, rng, task, fmt)
    levels = score_levels(task, ilang, levels)
    q = {"type": "score", "instructions": prompt or text, "criteria": list(levels)}
    return _base(entry, row, state, q, [1.0 if i == gold else 0.0 for i in range(len(levels))], lang, texts, task)


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


def emotion_labels(entry, row):
    """Emotion labels of one row. A source with "emotion_taxonomy" gets the classes of that taxonomy
    (emotion_taxonomy.py); a row with a label outside it yields no labels and is skipped."""
    labels = [canon(x) for x in labels_from(entry, row)]
    if entry.get("emotion_taxonomy") == emotion_taxonomy.NAME:
        return emotion_taxonomy.map_labels(labels) or []
    return labels


def emotion_adapter(entry, rows, seed):
    rows = list(rows)
    if entry.get("emotion_taxonomy") == emotion_taxonomy.NAME:
        counts = collections.Counter(x for row in rows for x in emotion_labels(entry, row))
        vocab = [x for x, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    else:
        vocab = _vocabulary(entry, rows)
    for i, row in enumerate(rows):
        texts, labels = row_texts(entry, row), emotion_labels(entry, row)
        if not texts or not labels:
            continue
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
        rng = seeded(seed, source_key(entry), i, state)
        # A multi-label row has no single gold emotion; it still answers the yes/no probe below.
        item = (_choice(entry, row, state, vocab, labels[0], lang, rng, texts, task="emotion")
                if len(set(labels)) == 1 else None)
        if item:
            yield item
        if vocab:
            # Half the probes name an emotion the row has, half one it has not: with a random probe
            # over a 20-emotion vocabulary the answer was "no" about 95 % of the time.
            absent = [x for x in vocab if x not in labels]
            probe = rng.choice(labels) if (rng.random() < 0.5 or not absent) else rng.choice(absent)
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


# Graded relevance labels (ESCI: Exact/Substitute/Complement/Irrelevant; WANDS: Exact/Partial/Irrelevant)
# on the SIM_LEVELS scale.
RELEVANCE_LEVELS = {"exact": 4, "substitute": 2, "partial": 2, "complement": 1, "irrelevant": 0}


def _group_pairs(rows, group_key, text_key):
    """Pairs from a grouped source: neighbours in one group are related (1.0); each text is also
    paired with a text of another group (0.0)."""
    groups = {}
    for row in rows:
        text = row.get(text_key)
        if isinstance(text, str) and text.strip() and row.get(group_key) not in (None, ""):
            groups.setdefault(str(row[group_key]), []).append(row)
    keys = sorted(groups)
    made = []
    for n, key in enumerate(keys):
        members = groups[key]
        other = groups[keys[(n + max(1, len(keys) // 2)) % len(keys)]] if len(keys) > 1 else []
        for i, row in enumerate(members):
            base = {k: v for k, v in row.items() if k.startswith("_v6_")}
            if i + 1 < len(members) and members[i + 1][text_key] != row[text_key]:
                made.append({**base, "sentence1": row[text_key], "sentence2": members[i + 1][text_key],
                             "relatedness_score": 1.0})
            if other and other[i % len(other)][text_key] != row[text_key]:
                made.append({**base, "sentence1": row[text_key], "sentence2": other[i % len(other)][text_key],
                             "relatedness_score": 0.0})
    return made


def _qa_pairs(rows, question_key, answer_key):
    """A question with its own answer is relevant (1.0); with the answer of another row, not (0.0)."""
    usable = [r for r in rows if isinstance(r.get(question_key), str) and isinstance(r.get(answer_key), str)
              and r[question_key].strip() and r[answer_key].strip()]
    made = []
    for i, row in enumerate(usable):
        base = {k: v for k, v in row.items() if k.startswith("_v6_")}
        made.append({**base, "sentence1": row[question_key], "sentence2": row[answer_key], "relatedness_score": 1.0})
        other = usable[(i + max(1, len(usable) // 2)) % len(usable)]
        if other[answer_key] != row[answer_key]:
            made.append({**base, "sentence1": row[question_key], "sentence2": other[answer_key],
                         "relatedness_score": 0.0})
    return made


def similarity_adapter(entry, rows, seed):
    rows = list(rows)
    derived = entry.get("label_field", "")
    paired = _tapaco_pairs(rows)
    if paired:
        rows = paired
    elif derived.startswith("derived: same group_id"):
        rows = _group_pairs(rows, "group_id", (_paths(entry, "text_fields") or ["text"])[0])
    elif derived.startswith("derived: question-answer pair"):
        fields = _paths(entry, "text_fields")
        rows = _qa_pairs(rows, fields[0], fields[1])
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
            if label in RELEVANCE_LEVELS:
                score = RELEVANCE_LEVELS[label]
                same = score == 4
            else:
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
    # No "is the writer reporting a problem?" item: every row of a complaint source is a complaint,
    # so that question would always be answered yes.
    yield from classification_adapter(entry, rows, seed)


def safety_adapter(entry, rows, seed):
    rows = list(rows)
    # "not toxic": mteb/toxic_conversations_50k; "casual": prosocial-dialog's __casual__ label.
    negative_words = {"safe", "valid", "benign", "not hate", "nothate", "non hateful", "not toxic", "non toxic",
                      "casual", "0", "false"}
    task = task_for(entry) or "moderation"
    vocab_counts = {}
    prepared = []
    for row in rows:
        labels = [canon(x).lower() for x in labels_from(entry, row)]
        prepared.append(labels)
        for label in labels:
            if label != UNSAFE_NO_CATEGORY:
                vocab_counts[label] = vocab_counts.get(label, 0) + 1
    vocab = [x for x, _ in sorted(vocab_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        if not texts:
            continue
        labels = prepared[i]
        if not labels and _safety_labels(entry["id"], row) is not None:
            continue  # a source with explicit labels, and this row's are ambiguous
        truth = not labels or not any(x in negative_words for x in labels)
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
        rng = seeded(seed, source_key(entry), i, state)
        if (len(set(labels)) == 1 and len(vocab) >= 2 and labels[0] in vocab
                and not all(str(x).isdigit() for x in vocab)):
            item = _choice(entry, row, state, vocab, labels[0], lang, rng, texts, task=task)
            if item:
                yield item
        if not UNDECIDED & set(labels):
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


WINDOW = 1600  # characters of a long context kept around the evidence


def _window_context(row):
    """A WINDOW-character slice of a long context. Rows with and without an answer get slices of the
    same length at a pseudo-random offset (seeded by the question), so neither the length nor the
    position of the slice reveals whether the question can be answered."""
    context = row.get("context")
    if not isinstance(context, str) or len(context) <= WINDOW:
        return row
    cloned = dict(row)
    answers = row.get("answers") if isinstance(row.get("answers"), dict) else {}
    starts = answers.get("answer_start") or []
    texts = answers.get("text") or []
    rng = seeded("window", row.get("question") or "", len(context))
    if starts:
        try:
            start = int(starts[0])
        except (TypeError, ValueError):
            start = 0
        length = len(str(texts[0])) if texts else 0
        lead = rng.randint(0, max(0, WINDOW - length - 1))  # answer anywhere inside the slice
        begin = min(max(0, start - lead), len(context) - WINDOW)
    else:
        begin = rng.randint(0, len(context) - WINDOW)
    cloned["context"] = context[begin:begin + WINDOW]
    return cloned


MAUD_MAX_OPTION = 120


def _maud_options(rows):
    """MAUD rows carry the answer text only; the options are the answers seen for the same
    (question, subquestion) in the loaded rows."""
    groups = {}
    for row in rows:
        answer = str(row.get("answer") or "").strip()
        if answer and len(answer) <= MAUD_MAX_OPTION:  # longer answers are lists of several clauses
            groups.setdefault((row.get("question"), row.get("subquestion")), set()).add(answer)
    return {key: sorted(values) for key, values in groups.items() if len(values) >= 2}


def _unanswerable_pairs(rows):
    """For a source where every question has an answer (FairytaleQA): the same question against a
    section of a different story, which cannot answer it."""
    made = []
    usable = [r for r in rows if r.get("context") and r.get("question")]
    for i, row in enumerate(usable):
        for step in range(1, len(usable)):
            other = usable[(i + step * max(1, len(usable) // 7)) % len(usable)]
            if other.get("story_name") != row.get("story_name") and other["context"] != row["context"]:
                made.append({**row, "context": other["context"], "answers": {"text": []}})
                break
    return made


def reading_adapter(entry, rows, seed):
    rows = list(rows)
    maud = _maud_options(rows) if entry["id"] == "theatticusproject/maud" else None
    if entry["id"] == "WorkInTheDark/FairytaleQA":
        rows = rows + _unanswerable_pairs(rows)
    for i, row in enumerate(rows):
        if maud is not None:
            options = maud.get((row.get("question"), row.get("subquestion")))
            answer = str(row.get("answer") or "").strip()
            if options and answer in options:
                row = dict(row, options=options, gold_label=answer)
            else:
                continue  # MAUD has no answerability label; without options there is no item
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
        if not 0 <= gold < len(choices):
            continue  # a negative index would pick a wrong option silently
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


# agentlans/fact-or-opinion labels -> options (templates.GLOSSES["claim_detection"]).
CLAIM_TYPES = {"fact": "fact", "opinion": "opinion", "both": "fact and opinion", "neither": "neither"}
STATES_A_FACT = {"fact", "fact and opinion"}


def claim_type_adapter(entry, rows, seed):
    """Fact vs opinion: which kind of statement (4 options), and does it state a verifiable fact (yes/no).
    The yes/no side is balanced in the source: fact + both against opinion + neither."""
    options = list(CLAIM_TYPES.values())
    for i, row in enumerate(rows):
        texts = row_texts(entry, row)
        label = CLAIM_TYPES.get(str(row.get("label") or "").strip().lower())
        if not texts or not label:
            continue
        lang, state = infer_lang(entry, row, text=texts[0]), state_from(entry, row, texts)
        rng = seeded(seed, source_key(entry), i, state)
        item = _choice(entry, row, state, options, label, lang, rng, texts, task="claim_detection")
        if item:
            yield item
        yield _noul(entry, row, state, label in STATES_A_FACT, lang, rng, texts, task="claim_detection")


PII_SPAN_KEYS = ("pii_spans", "spans", "entities", "privacy_mask", "span_labels")


def _span_kind(span):
    """PII type of one span: [start, end, type] lists or dicts with label / type / types."""
    if isinstance(span, (list, tuple)) and len(span) >= 3:
        return str(span[2]).strip()
    if isinstance(span, dict):
        for key in ("label", "type", "entity_type"):
            if span.get(key):
                return str(span[key]).strip()
        types = span.get("types")
        if isinstance(types, list) and types:
            return str(types[0]).strip()
    return ""


def _pii_row(entry, row):
    """(text, [pii types]) for a span-annotated row, or None when the row has no span column. The
    types come from the spans; a document-level field such as document_type is not a PII type."""
    if isinstance(row.get("tokenized_text"), list):
        text = " ".join(str(tok) for tok in row["tokenized_text"]).strip()
        spans = row.get("ner") or []
    else:
        key = next((k for k in PII_SPAN_KEYS if k in row), None)
        if key is None:
            return None
        spans = row.get(key)
        if isinstance(spans, str):
            spans = _parse_structured(spans) if spans.strip() else []
        if not isinstance(spans, list):
            return None
        texts = row_texts(entry, row)
        text = texts[0] if texts else ""
    kinds = [human(k) for k in (_span_kind(s) for s in spans) if k]
    return text, kinds


def pii_adapter(entry, rows, seed):
    rows = list(rows)
    prepared, counts = [], {}
    for row in rows:
        parsed = _pii_row(entry, row)
        prepared.append(parsed)
        for kind in (parsed or ("", []))[1]:
            counts[kind] = counts.get(kind, 0) + 1
    vocab = [x for x, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
    for i, parsed in enumerate(prepared):
        if not parsed or not parsed[0]:
            continue
        text, kinds = parsed
        lang = infer_lang(entry, rows[i], text=text)
        rng = seeded(seed, source_key(entry), i, text[:80])
        # Most PII sources are exhaustively annotated, so an empty span list is a document-level
        # negative. Sources with explicit supervision metadata opt out: there, only a row marked
        # by the loader as an intentional negative may answer the broad PII question with "no".
        bounded = "_v6_pii_supervised" in rows[i]
        if kinds or not bounded or rows[i].get("_v6_pii_negative"):
            yield _noul(entry, rows[i], text, bool(kinds), lang, rng, [text], task="pii")
        # Is one given type present? Unambiguous also when a text holds several types (most do).
        # Half the probes name a type the text holds, half one it does not, so the answer is balanced.
        supervised = set(rows[i].get("_v6_pii_supervised", vocab))
        absent = [k for k in vocab if k not in kinds and k in supervised]
        if kinds and absent:
            probe = rng.choice(kinds) if rng.random() < 0.5 else rng.choice(absent)
            yield _noul(entry, rows[i], text, probe in kinds, lang, rng, [text], task="pii_type",
                        statement="Does this text contain personal data of the type '%s'?" % probe)
        # Which type of personal data: only when the text holds exactly one type, so the gold is unique.
        distinct = list(dict.fromkeys(kinds))
        if len(distinct) == 1 and len(vocab) >= 2 and distinct[0] in vocab:
            item = _choice(entry, rows[i], text, vocab, distinct[0], lang, rng, [text], task="pii")
            if item:
                yield item


def adapter_for(entry):
    cat, sid = entry["category"], entry["id"]
    if sid == "community-datasets/re_dial":
        return redial_adapter
    if sid == "Eurolingua/truthfulqax":
        return fact_adapter
    if sid == "agentlans/fact-or-opinion":
        return claim_type_adapter
    if sid == "urchade/synthetic-pii-ner-mistral-v1" or "10-pii" in cat:
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


CONSTANT_SHARE = 0.97  # a yes/no question answered the same way this often within a source is dropped
CONSTANT_MIN = 20      # ... once the source yields at least this many of them


def drop_constant_yes_no(items):
    """Remove the yes/no items of a task whose answer is (nearly) constant within one source: a
    harmful-only safety set asked "does this violate a policy?", a PII set where every text has
    PII. Training on them teaches the answer, not the decision. Choice items are kept."""
    counts = {}
    for item in items:
        if item["q"]["type"] == "noul":
            yes = item["target"][1] == 1.0
            c = counts.setdefault(item.get("_task"), [0, 0])
            c[yes] += 1
    constant = {task for task, (no, yes) in counts.items()
                if no + yes >= CONSTANT_MIN and max(no, yes) >= CONSTANT_SHARE * (no + yes)}
    return [item for item in items if not (item["q"]["type"] == "noul" and item.get("_task") in constant)]


def adapt(entry, rows, seed):
    rows = list(rows)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    rows = [row for row in rows if not row_is_excluded(
        "v6", entry["id"], entry.get("config", "default"), row, policy)]
    # Tests may exercise an excluded adapter from its raw registry entry. The
    # Production builders can only pass entries from ENABLED_ENTRIES.
    items = [item for item in adapter_for(entry)(entry, rows, seed) if item]
    yield from drop_constant_yes_no(items)
