#!/usr/bin/env python3
"""Second pass and structural rejects.

The verifier sees only the situation, the question, and the options. The gold
label and the rationale are not in that prompt. An item is kept only when this
pass agrees with the gold. Score items need the exact level string.

Also rejects a leaked answer, duplicate options, the wrong script, a business
or reading situation under 15 words, and emotion-classification questions.
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


def _build_noul(seed, parsed, rationale, lang):
    state = nfc(parsed.get("state"))
    instructions = nfc(parsed.get("instructions"))
    false_d = nfc(parsed.get("false_description"))
    true_d = nfc(parsed.get("true_description"))
    if not state or not instructions or len(false_d) < 8 or len(true_d) < 8:
        return _rej("missing_text", rationale)
    if norm(false_d) == norm(true_d):
        return _rej("duplicate_options", rationale)
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
    # noul target order is always [false, true].
    item = {
        "state": state,
        "q": {"type": "noul", "instructions": instructions,
              "criteria": {"false": false_d, "true": true_d}},
        "target": [0.0, 1.0] if gold else [1.0, 0.0],
        "src": "synth-v1/business/%s" % lang,
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
    from prompts import GEN_SYSTEM, VER_SYSTEM, generate_request, verify_request
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
