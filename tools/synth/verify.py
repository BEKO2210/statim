#!/usr/bin/env python3
"""Second pass and structural rejects.

The verifier sees only the situation, the question, and the options. The gold
label and the rationale are not in that prompt. An item is kept only when this
pass agrees with the gold. Score items need the exact level string.

Also rejects a leaked answer, duplicate options, the wrong script, a business
or reading situation under 15 words, and emotion-classification questions on
the workplace tasks.

Sentiment, emotion, and complaint categories are closed label sets: the gold
key must be the label the seed requested. Urgency is an ordinal score whose
level index must match the seed. Sarcasm and complaint yes/no items must match
the requested boolean. Sarcasm is rejected when the text uses an emoji or /s.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import random
import re
import sys

from common import has_key, language_ok, nfc, norm, one_hot, script_counts, training_item, word_count
from prompts import verify_request
from seeds import COMPLAINT_LABELS, EMOTION_LABELS, SENTIMENT_LABELS

_SARCASM_TAG = re.compile(r"/s\b|\\s\b", re.I)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
_SARCASM_WORD = re.compile(r"\b(sarcasm|sarcastic)\b", re.I)

# Obvious emotion-classification questions, not a label inventory.
_EMOTION_Q = (
    "which emotion", "what emotion", "emotion is expressed", "emotion expressed",
    "welche emotion", "welches gefühl", "welche gefühl",
    "quelle émotion", "quelle emotion", "qué emoción", "que emocion",
    "quale emozione", "qual emoção", "qual emocao", "welke emotie",
    "jaka emocja", "hangi duygu", "какая эмоция", "какую эмоцию",
    "ما الشعور", "ما العاطفة", "कौन सी भावना", "どの感情", "什么情绪", "哪种情绪",
    "sentiment of", "what is the sentiment", "welche stimmung",
)

_META = (
    "correct answer", "the answer is", "gold label", "intended answer",
    "richtige antwort", "respuesta correcta", "réponse correcte", "reponse correcte",
    "risposta corretta", "resposta correta", "doğru cevap", "dogru cevap",
    "правильный ответ", "الإجابة الصحيحة", "सही उत्तर", "正しい答え", "正确答案",
)


def parse_model_json(content):
    text = nfc(content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _options_from(parsed):
    raw = parsed.get("options")
    options = []
    if isinstance(raw, dict):
        raw = [{"key": k, "description": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        return None
    for row in raw:
        if isinstance(row, str):
            options.append({"key": nfc(row), "description": ""})
            continue
        if not isinstance(row, dict):
            return None
        key = row.get("key", row.get("label", row.get("name")))
        desc = row.get("description", row.get("desc", row.get("text")))
        options.append({"key": nfc(key), "description": nfc(desc)})
    return options


def _leaked(question, pieces):
    q = norm(question)
    for piece in pieces:
        piece = nfc(piece)
        if len(piece) < 4:
            continue
        if len(piece) >= 12 and norm(piece) in q:
            return True
        if re.search(r"(?<!\w)" + re.escape(piece) + r"(?!\w)", question, flags=re.I):
            return True
    return False


def _meta(text):
    hay = norm(text)
    return any(phrase in hay for phrase in _META)


def _emotion(instructions):
    hay = norm(instructions)
    return any(phrase in hay for phrase in _EMOTION_Q)


def _sarcasm_cheat(state):
    """The classified text, not the question. A copied rule in the question is not the sarcasm device."""
    if _SARCASM_TAG.search(state) or _EMOJI.search(state) or _SARCASM_WORD.search(state):
        return True
    return False


def _script_ok(text, lang):
    """language_ok, with the long-text quotas relaxed for a short label."""
    counts = script_counts(nfc(text))
    total = sum(counts.values())
    if total < 4:
        return True
    if lang == "zh" and counts["cjk"] < 8:
        return counts["cjk"] / total >= 0.5 and counts["kana"] <= 1
    if lang == "ja" and counts["kana"] < 4:
        return (counts["kana"] + counts["cjk"]) / total >= 0.5 and counts["arabic"] / total < 0.1
    return language_ok(text, lang)


def _fold_key(text):
    text = nfc(text).casefold().replace("-", " ").replace("_", " ")
    return " ".join(text.split())


def _match_closed(options, allowed):
    if not options:
        return None, "bad_options"
    folded = {_fold_key(label): label for label in allowed}
    out = []
    seen = set()
    for opt in options:
        canon = folded.get(_fold_key(opt["key"]))
        if canon is None:
            return None, "bad_key"
        if canon in seen:
            return None, "duplicate_options"
        seen.add(canon)
        out.append({"key": canon, "description": opt["description"]})
    if seen != set(allowed):
        return None, "option_count"
    return out, None


def _dup_options(options):
    keys = [norm(o["key"]) for o in options]
    descs = [norm(o["description"]) for o in options if o["description"]]
    if any(not k for k in keys) or len(set(keys)) != len(keys):
        return True
    if descs and len(set(descs)) != len(descs):
        return True
    return False


def _lang_fields(lang, fields):
    judged = 0
    for text in fields:
        if sum(script_counts(text).values()) < 4:
            continue
        judged += 1
        if not language_ok(text, lang):
            return False
    return judged > 0


def build_item(seed, parsed):
    """Turn model JSON into a training item, or a structural rejection.

    The returned item never contains the rationale.
    """
    if not isinstance(parsed, dict):
        return _rej("bad_json")
    rationale = nfc(parsed.get("rationale"))
    task = seed["task"]
    lang = seed["lang"]
    if task == "sentiment":
        return _build_closed_choice(seed, parsed, rationale, lang, SENTIMENT_LABELS)
    if task == "emotion":
        return _build_closed_choice(seed, parsed, rationale, lang, EMOTION_LABELS)
    if task == "complaint" and seed.get("form") == "choice":
        return _build_closed_choice(seed, parsed, rationale, lang, COMPLAINT_LABELS)
    if task in {"complaint", "sarcasm"}:
        return _build_noul(seed, parsed, rationale, lang)
    if task == "urgency":
        return _build_urgency(seed, parsed, rationale, lang)
    if task == "business" and seed["form"] == "noul":
        return _build_noul(seed, parsed, rationale, lang)
    if task == "score":
        return _build_score(seed, parsed, rationale, lang)
    return _build_choice(seed, parsed, rationale, lang, reading=(task == "reading"))


def _rej(reason, rationale=""):
    return {"status": "rejected", "reason": reason, "item": None, "view": None,
            "rationale": rationale, "gold": None}


def _finish(item, view, rationale, gold):
    if has_key(item, "rationale"):
        return _rej("rationale_in_item", rationale)
    if rationale and len(rationale) >= 40 and norm(rationale) in norm(item["state"] + "\n" + item["q"]["instructions"]):
        return _rej("rationale_leaked", rationale)
    return {"status": "ok", "reason": None, "item": training_item(item), "view": view,
            "rationale": rationale, "gold": gold}


def _build_choice(seed, parsed, rationale, lang, reading):
    options = _options_from(parsed)
    if not options:
        return _rej("bad_options", rationale)
    lo, hi = (3, 4) if reading else (3, 12)
    if not (lo <= len(options) <= hi):
        return _rej("option_count", rationale)
    if _dup_options(options):
        return _rej("duplicate_options", rationale)
    for opt in options:
        if not opt["key"] or len(opt["key"]) > 80:
            return _rej("bad_key", rationale)
        if not opt["description"] or len(opt["description"]) > 400:
            return _rej("missing_description", rationale)
        if norm(opt["description"]) == norm(opt["key"]):
            return _rej("missing_description", rationale)
        if not reading and len(opt["description"]) < 8:
            return _rej("missing_description", rationale)
    if reading:
        state = nfc(parsed.get("passage") or parsed.get("state"))
    else:
        state = nfc(parsed.get("state"))
    instructions = nfc(parsed.get("instructions"))
    if not state or not instructions:
        return _rej("missing_text", rationale)
    if _meta(state) or _meta(instructions):
        return _rej("meta_leak", rationale)
    if _emotion(instructions):
        return _rej("emotion", rationale)
    words = word_count(state)
    if reading:
        if words < 50 or words > 300:
            return _rej("passage_length", rationale)
    elif words < 15:
        return _rej("short_state", rationale)
    lang_fields = [state, instructions] + [o["description"] for o in options]
    if not reading:
        lang_fields.extend(o["key"] for o in options)
    if not _lang_fields(lang, lang_fields):
        return _rej("wrong_language", rationale)
    gold = nfc(parsed.get("gold"))
    keys = [o["key"] for o in options]
    if gold not in keys:
        folded = {k.casefold(): k for k in keys}
        gold = folded.get(gold.casefold(), gold)
    if gold not in keys:
        return _rej("gold_not_in_options", rationale)
    gold_opt = next(o for o in options if o["key"] == gold)
    if _leaked(instructions, [gold_opt["key"], gold_opt["description"]]):
        return _rej("answer_leaked", rationale)
    # Generators put the correct option first far more often than chance. Shuffle per seed so the
    # position carries no signal; reading options are relabelled A, B, C, D in their new order.
    options = list(options)
    random.Random(seed.get("id") or json.dumps(seed, sort_keys=True)).shuffle(options)
    if reading:
        gold = "ABCD"[[o["key"] for o in options].index(gold)]
        options = [{"key": "ABCD"[i], "description": o["description"]} for i, o in enumerate(options)]
    keys = [o["key"] for o in options]
    criteria = {o["key"]: o["description"] for o in options}
    # Dict insertion order is the choice target order.
    item = {
        "state": state,
        "q": {"type": "choice", "instructions": instructions, "criteria": criteria},
        "target": one_hot(len(options), keys.index(gold)),
        "src": "synth-v1/%s/%s" % ("reading" if reading else "business", lang),
    }
    view = {"type": "choice", "state": state, "instructions": instructions, "options": [
        {"key": o["key"], "description": o["description"]} for o in options]}
    return _finish(item, view, rationale, gold)


def _build_closed_choice(seed, parsed, rationale, lang, allowed):
    options, reason = _match_closed(_options_from(parsed), allowed)
    if reason:
        return _rej(reason, rationale)
    if _dup_options(options):
        return _rej("duplicate_options", rationale)
    for opt in options:
        if not opt["description"] or len(opt["description"]) > 400 or len(opt["description"]) < 2:
            return _rej("missing_description", rationale)
        if norm(opt["description"]) == norm(opt["key"]):
            return _rej("missing_description", rationale)
    state = nfc(parsed.get("state"))
    instructions = nfc(parsed.get("instructions"))
    if not state or not instructions:
        return _rej("missing_text", rationale)
    if _meta(state) or _meta(instructions):
        return _rej("meta_leak", rationale)
    if word_count(state) < 15:
        return _rej("short_state", rationale)
    # Keys stay English on purpose. Short descriptions use a relaxed script check.
    if not _lang_fields(lang, [state, instructions]):
        return _rej("wrong_language", rationale)
    if not all(_script_ok(opt["description"], lang) for opt in options):
        return _rej("wrong_language", rationale)
    folded = {_fold_key(label): label for label in allowed}
    gold = folded.get(_fold_key(parsed.get("gold")))
    if gold is None:
        return _rej("gold_not_in_options", rationale)
    if gold != seed.get("target"):
        return _rej("target_mismatch", rationale)
    gold_opt = next(opt for opt in options if opt["key"] == gold)
    if _leaked(instructions, [gold_opt["key"], gold_opt["description"]]):
        return _rej("answer_leaked", rationale)
    options = list(options)
    random.Random(seed.get("id") or json.dumps(seed, sort_keys=True)).shuffle(options)
    keys = [opt["key"] for opt in options]
    criteria = {opt["key"]: opt["description"] for opt in options}
    item = {
        "state": state,
        "q": {"type": "choice", "instructions": instructions, "criteria": criteria},
        "target": one_hot(len(options), keys.index(gold)),
        "src": "synth-v1/%s/%s" % (seed["task"], lang),
    }
    view = {"type": "choice", "state": state, "instructions": instructions, "options": [
        {"key": opt["key"], "description": opt["description"]} for opt in options]}
    return _finish(item, view, rationale, gold)


def _build_noul(seed, parsed, rationale, lang):
    state = nfc(parsed.get("state"))
    instructions = nfc(parsed.get("instructions"))
    false_d = nfc(parsed.get("false_description"))
    true_d = nfc(parsed.get("true_description"))
    if not state or not instructions or len(false_d) < 8 or len(true_d) < 8:
        return _rej("missing_text", rationale)
    if norm(false_d) == norm(true_d):
        return _rej("duplicate_options", rationale)
    if seed.get("task") == "sarcasm" and _sarcasm_cheat(state):
        return _rej("sarcasm_marker", rationale)
    if word_count(state) < 15:
        return _rej("short_state", rationale)
    if _meta(state) or _meta(instructions) or _emotion(instructions):
        return _rej("emotion" if _emotion(instructions) else "meta_leak", rationale)
    if not _lang_fields(lang, [state, instructions, false_d, true_d]):
        return _rej("wrong_language", rationale)
    gold = parsed.get("gold")
    if isinstance(gold, str):
        low = gold.strip().casefold()
        if low in {"true", "yes"}:
            gold = True
        elif low in {"false", "no"}:
            gold = False
    if not isinstance(gold, bool):
        return _rej("bad_gold", rationale)
    if "target" in seed and gold != (seed["target"] == "true"):
        return _rej("target_mismatch", rationale)
    if seed.get("task") in {"complaint", "sarcasm"} and _leaked(instructions, [true_d if gold else false_d]):
        return _rej("answer_leaked", rationale)
    # noul target order is always [false, true].
    item = {
        "state": state,
        "q": {"type": "noul", "instructions": instructions,
              "criteria": {"false": false_d, "true": true_d}},
        "target": [0.0, 1.0] if gold else [1.0, 0.0],
        "src": "synth-v1/%s/%s" % (seed.get("task") or "business", lang),
    }
    view = {"type": "noul", "state": state, "instructions": instructions, "options": [
        {"key": "false", "description": false_d},
        {"key": "true", "description": true_d}]}
    return _finish(item, view, rationale, gold)


def _build_score(seed, parsed, rationale, lang):
    state = nfc(parsed.get("state"))
    instructions = nfc(parsed.get("instructions"))
    levels_raw = parsed.get("levels")
    if not isinstance(levels_raw, list):
        return _rej("bad_options", rationale)
    levels = [nfc(x) for x in levels_raw]
    if not (3 <= len(levels) <= 5) or any(not lv or len(lv) > 120 for lv in levels):
        return _rej("option_count", rationale)
    if len({norm(lv) for lv in levels}) != len(levels):
        return _rej("duplicate_options", rationale)
    if not state or not instructions:
        return _rej("missing_text", rationale)
    if word_count(state) < 5:
        return _rej("missing_text", rationale)
    if seed.get("aspect") == "similarity" and "---" not in state:
        return _rej("missing_text", rationale)
    if _meta(state) or _meta(instructions) or _emotion(instructions):
        return _rej("emotion" if _emotion(instructions) else "meta_leak", rationale)
    if not _lang_fields(lang, [state, instructions, *levels]):
        return _rej("wrong_language", rationale)
    gold = nfc(parsed.get("gold"))
    if gold not in levels:
        return _rej("gold_not_in_options", rationale)
    if "target" in seed and levels.index(gold) + 1 != seed["target"]:
        return _rej("target_mismatch", rationale)
    if _leaked(instructions, [gold]):
        return _rej("answer_leaked", rationale)
    item = {
        "state": state,
        "q": {"type": "score", "instructions": instructions, "criteria": levels},
        "target": one_hot(len(levels), levels.index(gold)),
        "src": "synth-v1/score/%s" % lang,
    }
    view = {"type": "score", "state": state, "instructions": instructions, "levels": list(levels)}
    return _finish(item, view, rationale, gold)


def _build_urgency(seed, parsed, rationale, lang):
    state = nfc(parsed.get("state"))
    instructions = nfc(parsed.get("instructions"))
    levels_raw = parsed.get("levels")
    if not isinstance(levels_raw, list):
        return _rej("bad_options", rationale)
    levels = [nfc(x) for x in levels_raw]
    n_levels = seed.get("n_levels")
    if n_levels not in (3, 4) or len(levels) != n_levels or any(not lv or len(lv) > 120 for lv in levels):
        return _rej("option_count", rationale)
    if len({norm(lv) for lv in levels}) != len(levels):
        return _rej("duplicate_options", rationale)
    if not state or not instructions:
        return _rej("missing_text", rationale)
    if word_count(state) < 15:
        return _rej("short_state", rationale)
    if _meta(state) or _meta(instructions):
        return _rej("meta_leak", rationale)
    if _emotion(instructions):
        return _rej("emotion", rationale)
    if not _lang_fields(lang, [state, instructions]):
        return _rej("wrong_language", rationale)
    if not all(_script_ok(level, lang) for level in levels):
        return _rej("wrong_language", rationale)
    gold = nfc(parsed.get("gold"))
    if gold not in levels:
        folded = {lv.casefold(): lv for lv in levels}
        gold = folded.get(gold.casefold(), gold)
    if gold not in levels:
        return _rej("gold_not_in_options", rationale)
    if levels.index(gold) + 1 != seed.get("target"):
        return _rej("target_mismatch", rationale)
    if _leaked(instructions, levels):
        return _rej("answer_leaked", rationale)
    item = {
        "state": state,
        "q": {"type": "score", "instructions": instructions, "criteria": levels},
        "target": one_hot(len(levels), levels.index(gold)),
        "src": "synth-v1/urgency/%s" % lang,
    }
    view = {"type": "score", "state": state, "instructions": instructions, "levels": list(levels)}
    return _finish(item, view, rationale, gold)


def agrees(item, answer):
    """Gold agreement. Score requires the exact level string, not a neighbour."""
    kind = item["q"]["type"]
    ans = nfc(answer)
    if kind == "score":
        levels = item["q"]["criteria"]
        gold = levels[item["target"].index(1.0)]
        return ans == gold
    if kind == "noul":
        gold = item["target"][1] == 1.0
        low = ans.casefold()
        if low not in {"true", "false"}:
            return False
        return (low == "true") == gold
    keys = list(item["q"]["criteria"])
    gold = keys[item["target"].index(1.0)]
    if ans == gold:
        return True
    folded = {}
    for key in keys:
        folded.setdefault(key.casefold(), key)
    mapped = folded.get(ans.casefold())
    if mapped is not None:
        return mapped == gold
    # A unique description match is the same option, not a different one.
    desc_hits = [key for key, desc in item["q"]["criteria"].items() if nfc(desc) == ans or norm(desc) == norm(ans)]
    return desc_hits == [gold]


def verify_view(client, view, seed_int):
    messages, schema, npredict = verify_request(view)
    blob = json.dumps(messages, ensure_ascii=False)
    if "rationale" in blob or '"gold"' in blob:
        raise RuntimeError("verifier prompt contains gold or rationale")
    resp = client.chat(messages, schema, temperature=0, num_predict=npredict, seed=0)
    parsed = parse_model_json(resp["content"])
    if not isinstance(parsed, dict) or "answer" not in parsed:
        raise ValueError("verifier JSON has no answer")
    return parsed["answer"], resp


def run_checks():
    from collections import Counter

    from prompts import GEN_SYSTEM, GEN_SYSTEM_CLASS, PROMPT_VERSION, VER_SYSTEM, generate_request, verify_request
    from seeds import (
        COMPLAINT_LANGS, EMOTION_LANGS, LANGS, SENTIMENT_LANGS, URGENCY_LANGS, plan,
    )
    assert GEN_SYSTEM != VER_SYSTEM
    assert "rationale" not in VER_SYSTEM
    view = {"type": "choice", "state": "A clerk filed two copies of the same invoice on Monday morning at the desk.",
            "instructions": "Which queue should take this?",
            "options": [{"key": "billing", "description": "invoices and payments"},
                        {"key": "access", "description": "badges and accounts"}]}
    messages, _schema, _n = verify_request(view)
    blob = json.dumps(messages)
    assert "rationale" not in blob and "gold" not in blob
    assert language_ok(view["state"], "en")
    assert not language_ok(view["state"], "ar")
    assert language_ok("هذه جملة عربية طويلة للاختبار", "ar")
    assert language_ok("これは日本語の試験文です。今日の手続きを確認します。", "ja")
    assert language_ok("这是一段用于脚本检查的中文句子内容。", "zh")
    assert not language_ok("これは日本語の試験文です。今日の手続きを確認します。", "zh")
    assert word_count("one two three") == 3
    assert word_count("你好世界") == 4
    seed = {"task": "business", "form": "choice", "lang": "en", "i": 0}
    parsed = {
        "state": "The caller says the March invoice was charged twice and asks the desk to send the duplicate back to the card today before the books close for the week.",
        "instructions": "Which desk should own this request?",
        "options": [
            {"key": "billing", "description": "invoices, card charges, and refunds"},
            {"key": "access", "description": "badges, passwords, and accounts"},
            {"key": "facilities", "description": "rooms, keys, and building issues"},
        ],
        "gold": "billing",
        "rationale": "The caller describes a duplicate card charge.",
    }
    # Force the state over 15 words.
    parsed["state"] = ("The caller says the March invoice was charged twice and asks the desk "
                       "to send the duplicate amount back to the card today before the books close.")
    built = build_item(seed, parsed)
    assert built["status"] == "ok", built["reason"]
    crit = list(built["item"]["q"]["criteria"])
    assert sorted(crit) == ["access", "billing", "facilities"]
    assert built["item"]["target"] == one_hot(3, crit.index("billing"))
    assert build_item(seed, parsed)["item"] == built["item"], "shuffle must be deterministic per seed"
    # Gold always first in the raw output must end up spread over all positions.
    pos = [0, 0, 0]
    for i in range(300):
        got = build_item(dict(seed, id="%08x" % i), parsed)["item"]
        pos[got["target"].index(1.0)] += 1
    assert min(pos) > 70, pos
    reading_seed = {"task": "reading", "lang": "en", "i": 2}
    reading = {
        "passage": " ".join(["The archive team moved the paper records to room four on Tuesday."] * 8),
        "instructions": "Where are the paper records now?",
        "options": [{"key": "A", "description": "room four"}, {"key": "B", "description": "the basement"},
                    {"key": "C", "description": "the main office"}, {"key": "D", "description": "a storage unit"}],
        "gold": "A",
        "rationale": "The passage says they moved to room four.",
    }
    for i in range(20):
        rgot = build_item(dict(reading_seed, id="%08x" % i), reading)
        assert rgot["status"] == "ok", rgot["reason"]
        rc = rgot["item"]["q"]["criteria"]
        assert list(rc) == ["A", "B", "C", "D"]
        assert rc["ABCD"[rgot["item"]["target"].index(1.0)]] == "room four"
    assert "rationale" not in json.dumps(built["item"])
    assert built["item"]["q"]["criteria"]["billing"].startswith("invoices")
    leaked = dict(parsed, instructions="Should billing handle the duplicate charge?")
    assert build_item(seed, leaked)["reason"] == "answer_leaked"
    short = dict(parsed, state="Too short to keep.")
    assert build_item(seed, short)["reason"] == "short_state"
    dup = dict(parsed)
    dup["options"] = [dict(parsed["options"][0]), dict(parsed["options"][0]), parsed["options"][2]]
    assert build_item(seed, dup)["reason"] == "duplicate_options"
    score_seed = {"task": "score", "lang": "en", "aspect": "urgency", "i": 0}
    score = {
        "state": "A payroll file failed at 16:40 and salaries go out tomorrow morning, so the desk still has the evening.",
        "instructions": "How soon does this need action?",
        "levels": ["can wait", "today", "immediately"],
        "gold": "today",
        "rationale": "Salaries go out tomorrow, so this evening is enough.",
    }
    got = build_item(score_seed, score)
    assert got["status"] == "ok", got["reason"]
    assert got["item"]["target"] == [0.0, 1.0, 0.0]
    assert agrees(got["item"], "today")
    assert not agrees(got["item"], "immediately")
    assert not agrees(got["item"], "Today")
    noul_seed = {"task": "business", "form": "noul", "lang": "en", "i": 1}
    noul = {
        "state": parsed["state"],
        "instructions": "Does the caller ask for the duplicate charge to be returned?",
        "false_description": "the caller does not ask for money back",
        "true_description": "the caller asks for the duplicate charge to be returned",
        "gold": True,
        "rationale": "They ask for the duplicate amount to be sent back.",
    }
    ngot = build_item(noul_seed, noul)
    assert ngot["status"] == "ok", ngot["reason"]
    assert ngot["item"]["target"] == [0.0, 1.0]
    assert agrees(ngot["item"], "true")
    assert not agrees(ngot["item"], "false")
    messages, _schema, _npred = generate_request(score_seed | {
        "persona": "a clerk", "aspect_help": "how soon", "n_levels": 3, "lang": "en",
        "industry": "a retail bank", "register": "a letter", "length": "short: 30 to 60 words", "target": 2})
    # A generator that ignores the requested answer is rejected, so the label balance holds.
    assert build_item(dict(noul_seed, target="false"), noul)["reason"] == "target_mismatch"
    assert build_item(dict(noul_seed, target="true"), noul)["status"] == "ok"
    assert build_item(dict(score_seed, target=3), score)["reason"] == "target_mismatch"
    assert build_item(dict(score_seed, target=2), score)["status"] == "ok"
    assert "gold" not in messages[1]["content"] or "gold (" in messages[1]["content"]
    assert ngot["item"]["src"] == "synth-v1/business/en"
    assert build_item(seed, dict(parsed, instructions="Which emotion is expressed by the caller in this note?"))["reason"] == "emotion"
    assert PROMPT_VERSION == "synth-prompts-5"
    assert GEN_SYSTEM != GEN_SYSTEM_CLASS
    assert "Do not write an emotion, sentiment, or feeling classification." in GEN_SYSTEM
    assert "Do not write an emotion" not in GEN_SYSTEM_CLASS

    def _closed_parsed(task_labels, gold, state):
        return {
            "state": state,
            "instructions": "Which class fits this text?",
            "options": [{"key": key, "description": "means the class %s here" % key} for key in task_labels],
            "gold": gold,
            "rationale": "The seed requested this class.",
        }

    state = parsed["state"]
    sent = _closed_parsed(SENTIMENT_LABELS, "Positive", state)
    sent_seed = {"task": "sentiment", "lang": "en", "target": "positive", "id": "s1"}
    sgot = build_item(sent_seed, sent)
    assert sgot["status"] == "ok", sgot["reason"]
    assert sgot["item"]["src"] == "synth-v1/sentiment/en"
    assert sgot["item"]["target"].count(1.0) == 1
    assert list(sgot["item"]["q"]["criteria"])[sgot["item"]["target"].index(1.0)] == "positive"
    assert "gold" not in sgot["view"] and "rationale" not in sgot["view"]
    sblob = json.dumps(verify_request(sgot["view"])[0])
    assert "rationale" not in sblob and '"gold"' not in sblob
    assert build_item(dict(sent_seed, target="negative"), sent)["reason"] == "target_mismatch"
    assert build_item(sent_seed, dict(sent, instructions="Is the class positive here?"))["reason"] == "answer_leaked"
    assert build_item(sent_seed, dict(sent, state="Too short to keep."))["reason"] == "short_state"
    assert build_item(dict(sent_seed, lang="ar"), sent)["reason"] == "wrong_language"
    assert build_item(sent_seed, dict(sent, options=sent["options"][:3]))["reason"] == "option_count"
    dup_opts = [dict(opt) for opt in sent["options"]]
    dup_opts[1]["description"] = dup_opts[0]["description"]
    assert build_item(sent_seed, dict(sent, options=dup_opts))["reason"] == "duplicate_options"
    bad_key = [dict(opt) for opt in sent["options"]]
    bad_key[0] = dict(bad_key[0], key="happy")
    assert build_item(sent_seed, dict(sent, options=bad_key))["reason"] == "bad_key"
    positions = set()
    for i in range(30):
        placed = build_item(dict(sent_seed, id="%08x" % i), sent)["item"]
        positions.add(placed["target"].index(1.0))
    assert len(positions) > 1, positions

    emo = _closed_parsed(EMOTION_LABELS, "joy", state)
    emo["instructions"] = "Which emotion fits this note?"
    emo_seed = {"task": "emotion", "lang": "en", "target": "joy", "id": "e1"}
    egot = build_item(emo_seed, emo)
    assert egot["status"] == "ok", egot["reason"]
    assert egot["item"]["src"] == "synth-v1/emotion/en"
    assert build_item(dict(emo_seed, target="anger"), emo)["reason"] == "target_mismatch"
    eblob = json.dumps(verify_request(egot["view"])[0])
    assert "rationale" not in eblob and '"gold"' not in eblob

    comp = _closed_parsed(COMPLAINT_LABELS, "product defect", state)
    comp["instructions"] = "Which category fits this message?"
    comp_seed = {"task": "complaint", "form": "choice", "lang": "en", "target": "product_defect", "id": "c1"}
    cgot = build_item(comp_seed, comp)
    assert cgot["status"] == "ok", cgot["reason"]
    assert cgot["item"]["src"] == "synth-v1/complaint/en"
    assert "product_defect" in cgot["item"]["q"]["criteria"]
    assert build_item(dict(comp_seed, target="refund"), comp)["reason"] == "target_mismatch"
    cnoul_seed = {"task": "complaint", "form": "noul", "lang": "en", "target": "true"}
    cnoul = build_item(cnoul_seed, noul)
    assert cnoul["status"] == "ok", cnoul["reason"]
    assert cnoul["item"]["src"] == "synth-v1/complaint/en"
    assert build_item(dict(cnoul_seed, target="false"), noul)["reason"] == "target_mismatch"
    cblob = json.dumps(verify_request(cnoul["view"])[0])
    assert "rationale" not in cblob and '"gold"' not in cblob

    urg_seed = {"task": "urgency", "lang": "en", "n_levels": 4, "target": 4, "id": "u1"}
    urgent = {
        "state": score["state"],
        "instructions": "How soon does this need action?",
        "levels": ["can wait", "this week", "today", "immediately"],
        "gold": "immediately",
        "rationale": "The requested level is the last one.",
    }
    ugot = build_item(urg_seed, urgent)
    assert ugot["status"] == "ok", ugot["reason"]
    assert ugot["item"]["src"] == "synth-v1/urgency/en"
    assert ugot["item"]["target"] == [0.0, 0.0, 0.0, 1.0]
    assert ugot["view"]["type"] == "score" and "gold" not in ugot["view"]
    ublob = json.dumps(verify_request(ugot["view"])[0])
    assert "rationale" not in ublob and '"gold"' not in ublob
    assert build_item(dict(urg_seed, target=1), urgent)["reason"] == "target_mismatch"
    assert build_item(dict(urg_seed, n_levels=3), urgent)["reason"] == "option_count"
    assert build_item(urg_seed, dict(urgent, instructions="Should this be handled immediately now?"))["reason"] == "answer_leaked"
    assert build_item(urg_seed, dict(urgent, state="Too short to keep."))["reason"] == "short_state"
    assert build_item(dict(urg_seed, lang="ar"), urgent)["reason"] == "wrong_language"
    assert build_item({"task": "urgency", "lang": "en", "n_levels": 3, "target": 2}, score)["status"] == "ok"
    zh_sentence = "这是一段用于脚本检查的中文句子内容。"
    zh_urgent = {
        "state": zh_sentence * 2,
        "instructions": zh_sentence,
        "levels": ["可以等待", "今天处理", "立即处理"],
        "gold": "今天处理",
        "rationale": "The middle level is the requested one.",
    }
    zh_got = build_item({"task": "urgency", "lang": "zh", "n_levels": 3, "target": 2}, zh_urgent)
    assert zh_got["status"] == "ok", zh_got["reason"]
    short_desc = [dict(opt) for opt in sent["options"]]
    short_desc[1]["description"] = "repulsa"
    assert build_item(sent_seed, dict(sent, options=short_desc))["status"] == "ok"

    sar_seed = {"task": "sarcasm", "lang": "en", "target": "true", "id": "r1"}
    sargot = build_item(sar_seed, noul)
    assert sargot["status"] == "ok", sargot["reason"]
    assert sargot["item"]["src"] == "synth-v1/sarcasm/en"
    assert build_item(dict(sar_seed, target="false"), noul)["reason"] == "target_mismatch"
    assert build_item(sar_seed, dict(noul, state=noul["state"] + " /s"))["reason"] == "sarcasm_marker"
    assert build_item(sar_seed, dict(noul, state=noul["state"] + " \U0001F600"))["reason"] == "sarcasm_marker"
    assert build_item(sar_seed, dict(noul, state=noul["state"] + " sarcastic"))["reason"] == "sarcasm_marker"
    rblob = json.dumps(verify_request(sargot["view"])[0])
    assert "rationale" not in rblob and '"gold"' not in rblob

    sent_rows = plan("sentiment", 12, 20260927)
    assert Counter(row["target"] for row in sent_rows) == {key: 3 for key in SENTIMENT_LABELS}
    assert sum(row["variant"] == "aspect" for row in sent_rows) == 6
    pairs = Counter((row["target"], row["variant"]) for row in sent_rows)
    for key in SENTIMENT_LABELS:
        assert pairs[(key, "overall")] >= 1 and pairs[(key, "aspect")] >= 1
    for row in sent_rows:
        for attr in ("industry", "register", "length", "difficulty", "persona"):
            assert row[attr]
        if row["variant"] == "aspect":
            assert row["focus"] in row["aspects"] and len(row["aspects"]) == 3
        else:
            assert row["aspects"] == [] and row["focus"] == ""
    assert plan("sentiment", 7, 20260927) == sent_rows[:7]
    cycle = Counter(row["lang"] for row in plan("sentiment", len(SENTIMENT_LANGS), 9))
    assert cycle["de"] == 2 and cycle["en"] == 1 and cycle["ja"] == 2 and cycle["zh"] == 1 and cycle["hi"] == 1
    assert set(cycle) == set(LANGS)

    emo_cycle = Counter(row["lang"] for row in plan("emotion", len(EMOTION_LANGS), 9))
    assert set(emo_cycle) == {"de", "es", "it", "pt", "nl", "pl", "tr", "ru", "ar", "ja"}
    assert set(emo_cycle.values()) == {2}
    assert Counter(row["target"] for row in plan("emotion", 16, 9)) == {key: 2 for key in EMOTION_LABELS}

    comp_rows = plan("complaint", 28, 9)
    comp_noul = [row for row in comp_rows if row["form"] == "noul"]
    comp_choice = [row for row in comp_rows if row["form"] == "choice"]
    assert len(comp_noul) == 14 and len(comp_choice) == 14
    assert sum(row["target"] == "true" for row in comp_noul) == 7
    assert Counter(row["target"] for row in comp_choice) == {key: 2 for key in COMPLAINT_LABELS}
    comp_cycle = Counter(row["lang"] for row in plan("complaint", len(COMPLAINT_LANGS), 9))
    assert set(comp_cycle) == {"fr", "it", "nl", "pl", "tr", "ar", "hi", "ja", "zh"}
    assert set(comp_cycle.values()) == {2}

    urg_rows = plan("urgency", 24, 9)
    four = [row for row in urg_rows if row["n_levels"] == 4]
    three = [row for row in urg_rows if row["n_levels"] == 3]
    assert Counter(row["target"] for row in four) == {1: 3, 2: 3, 3: 3, 4: 3}
    assert Counter(row["target"] for row in three) == {1: 4, 2: 4, 3: 4}
    assert all(len(row["levels_brief"]) == row["n_levels"] for row in urg_rows)
    urg_cycle = Counter(row["lang"] for row in plan("urgency", len(URGENCY_LANGS), 9))
    assert "en" not in urg_cycle and set(urg_cycle) == set(LANGS) - {"en"} and set(urg_cycle.values()) == {2}

    assert Counter(row["target"] for row in plan("sarcasm", 12, 9)) == {"true": 6, "false": 6}
    sar_cycle = Counter(row["lang"] for row in plan("sarcasm", len(LANGS), 9))
    assert set(sar_cycle) == set(LANGS) and set(sar_cycle.values()) == {1}

    biz_msgs, _, _ = generate_request(plan("business", 1, 1)[0])
    assert "Do not write an emotion, sentiment, or feeling classification." in biz_msgs[0]["content"]
    wide = plan("sentiment", 16, 4)
    mixed = next(row for row in wide if row["target"] == "mixed")
    neutral = next(row for row in wide if row["target"] == "neutral")
    aspect = next(row for row in wide if row["variant"] == "aspect")
    mixed_text = generate_request(mixed)[0][1]["content"]
    assert "explicit praise" in mixed_text and "explicit complaint" in mixed_text
    assert "no opinion" in generate_request(neutral)[0][1]["content"]
    assert aspect["focus"] in generate_request(aspect)[0][1]["content"]
    assert "only this aspect" in generate_request(aspect)[0][1]["content"]
    assert "Do not write an emotion" not in generate_request(mixed)[0][0]["content"]
    for label in SENTIMENT_LABELS:
        assert label in mixed_text
    emo_row = plan("emotion", 8, 4)[0]
    emo_text = generate_request(emo_row)[0][1]["content"]
    assert emo_row["genre"] in emo_text and "first person" in emo_text
    for label in EMOTION_LABELS:
        assert label in emo_text
    comp_prompt_rows = plan("complaint", 4, 4)
    false_row = next(row for row in comp_prompt_rows if row["form"] == "noul" and row["target"] == "false")
    choice_row = next(row for row in comp_prompt_rows if row["form"] == "choice")
    assert "neutral request" in generate_request(false_row)[0][1]["content"]
    choice_text = generate_request(choice_row)[0][1]["content"]
    for label in COMPLAINT_LABELS:
        assert label in choice_text
    urg_row = plan("urgency", 1, 4)[0]
    urg_text = generate_request(urg_row)[0][1]["content"]
    assert "deadline" in urg_text and "absence" in urg_text and str(urg_row["target"]) in urg_text
    sar_row = plan("sarcasm", 1, 4)[0]
    assert sar_row["target"] == "true"
    sar_text = generate_request(sar_row)[0][1]["content"]
    assert "praise wording" in sar_text and "emoji" in sar_text and "letter s" in sar_text
    assert "full text to classify" in generate_request(mixed)[0][0]["content"]
    print("verify checks ok")


def main():
    ap = argparse.ArgumentParser(description="Structural checks and the verifier prompt")
    ap.add_argument("--check", action="store_true", help="run local checks, no model call")
    a = ap.parse_args()
    if a.check:
        run_checks()
        return
    raise SystemExit("verify runs inside generate.py; use --check for the local checks")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
