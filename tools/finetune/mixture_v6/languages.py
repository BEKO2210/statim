"""ISO 639-1 language codes and a script heuristic.

The character classes and thresholds match ``tools/synth/common.py``
``language_ok``. The logic is copied so this package does not import the
synthetic-data pipeline.
"""

from __future__ import annotations

import re

_LATIN = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_DEVA = re.compile(r"[\u0900-\u097F]")
_CYR = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")
_KANA = re.compile(r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9D]")
_CJK = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")

# English names, common mistakes, and ISO 639-2/3 codes → 639-1.
_NAMES = {
    "english": "en", "american english": "en", "british english": "en",
    "mandarin": "zh", "chinese": "zh", "simplified chinese": "zh", "traditional chinese": "zh",
    "portuguese": "pt", "brazilian portuguese": "pt", "european portuguese": "pt",
    "french": "fr", "france": "fr", "german": "de", "spanish": "es", "castilian": "es",
    "italian": "it", "dutch": "nl", "polish": "pl", "turkish": "tr", "russian": "ru",
    "arabic": "ar", "standard arabic": "ar", "najdi arabic": "ar", "moroccan arabic": "ar",
    "hindi": "hi", "japanese": "ja", "korean": "ko", "czech": "cs", "greek": "el",
    "hebrew": "he", "swedish": "sv", "danish": "da", "finnish": "fi", "hungarian": "hu",
    "romanian": "ro", "bulgarian": "bg", "ukrainian": "uk", "vietnamese": "vi", "thai": "th",
    "indonesian": "id", "malay": "ms", "standard malay": "ms", "persian": "fa",
    "iranian persian": "fa", "farsi": "fa", "bengali": "bn", "tamil": "ta", "telugu": "te",
    "urdu": "ur", "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "odia": "or",
    "oriya": "or", "punjabi": "pa", "panjabi": "pa", "assamese": "as", "marathi": "mr",
    "amharic": "am", "swahili": "sw", "filipino": "tl", "tagalog": "tl", "cebuano": "ceb",
    "us": "en", "uk": "en", "gb": "en", "au": "en", "ca": "en",
    "eng": "en", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es", "esp": "es",
    "spn": "es", "por": "pt", "nld": "nl", "dut": "nl", "pol": "pl", "tur": "tr", "rus": "ru",
    "ara": "ar", "arb": "ar", "hin": "hi", "jpn": "ja", "zho": "zh", "cmn": "zh", "yue": "zh",
    "mar": "mr", "ind": "id", "fas": "fa", "kor": "ko", "ces": "cs", "cze": "cs", "ell": "el",
    "gre": "el", "ukr": "uk", "heb": "he", "ben": "bn", "tam": "ta", "tel": "te", "urd": "ur",
    "guj": "gu", "kan": "kn", "mal": "ml", "ori": "or", "pan": "pa", "asm": "as", "kat": "ka",
    "vie": "vi", "tha": "th", "swe": "sv", "dan": "da", "fin": "fi", "hun": "hu", "ron": "ro",
    "rum": "ro", "bul": "bg", "slk": "sk", "slo": "sk", "slv": "sl", "hrv": "hr", "lit": "lt",
    "lav": "lv", "est": "et", "gle": "ga", "iri": "ga", "mlt": "mt", "afr": "af", "swa": "sw",
    "amh": "am", "fil": "tl", "tgl": "tl", "srp": "sr", "bos": "bs", "cat": "ca", "eus": "eu",
    "glg": "gl", "isl": "is", "nob": "no", "nno": "no", "nor": "no",
}

# Function words used only to separate Latin-script languages. Order is the
# priority list; a hit is a point, and the best candidate wins.
_LATIN_HINTS = {
    "de": (" und ", " der ", " die ", " nicht ", " ich ", " ein "),
    "fr": (" le ", " la ", " les ", " est ", " pas ", " une "),
    "es": (" el ", " los ", " las ", " una ", " que ", " por "),
    "it": (" il ", " che ", " non ", " una ", " per ", " sono "),
    "pt": (" não ", " uma ", " para ", " com ", " você ", " não"),
    "nl": (" het ", " een ", " van ", " niet ", " zijn ", " dat "),
    "pl": (" nie ", " się ", " jest ", " na ", " to ", " czy "),
    "tr": (" bir ", " ve ", " için ", " bu ", " değil ", " ile "),
    "en": (" the ", " and ", " is ", " to ", " of ", " that "),
}


def script_counts(text):
    counts = {"latin": 0, "arabic": 0, "devanagari": 0, "cyrillic": 0, "kana": 0, "cjk": 0}
    for ch in text or "":
        if _LATIN.match(ch):
            counts["latin"] += 1
        elif _ARABIC.match(ch):
            counts["arabic"] += 1
        elif _DEVA.match(ch):
            counts["devanagari"] += 1
        elif _CYR.match(ch):
            counts["cyrillic"] += 1
        elif _KANA.match(ch):
            counts["kana"] += 1
        elif _CJK.match(ch):
            counts["cjk"] += 1
    return counts


def script_family(text):
    """Return ar, hi, ru, ja, zh, latin, or None. Latin languages are not split here."""
    counts = script_counts(text)
    total = sum(counts.values())
    if total < 4:
        return None
    latin = counts["latin"] / total
    arabic = counts["arabic"] / total
    deva = counts["devanagari"] / total
    cyr = counts["cyrillic"] / total
    kana = counts["kana"] / total
    cjk = counts["cjk"] / total
    if arabic >= 0.6 and latin < 0.25:
        return "ar"
    if deva >= 0.6 and latin < 0.25:
        return "hi"
    if cyr >= 0.6 and latin < 0.25:
        return "ru"
    if counts["kana"] >= 4 and kana >= 0.05 and (kana + cjk) >= 0.5 and arabic < 0.1:
        return "ja"
    if counts["cjk"] >= 8 and cjk >= 0.5 and counts["kana"] <= 1 and kana < 0.02:
        return "zh"
    foreign = arabic + deva + cyr + kana + cjk
    if latin >= 0.7 and foreign <= 0.15:
        return "latin"
    return None


def to_iso(value):
    """Normalise a language name or code to ISO 639-1 when one exists."""
    if value is None:
        return ""
    text = str(value).strip().lower().replace("_", "-")
    if not text or text in {"mixed", "...", "unknown", "none", "null"}:
        return ""
    if text in _NAMES:
        return _NAMES[text]
    primary = text.split("-")[0]
    if primary in _NAMES:
        return _NAMES[primary]
    if len(primary) == 2 and primary.isalpha():
        return primary
    return primary if len(primary) == 3 and primary.isalpha() else text


def lang_from_config(value):
    """Pull a language out of a dataset config or file name such as ``cs-CZ`` or ``translation-hi``."""
    if not value:
        return ""
    text = str(value).strip().split("~")[0]
    if text.startswith("translation-"):
        return to_iso(text.split("-", 1)[1])
    parts = re.split(r"[./]", text)
    if len(parts) >= 2 and parts[-1] == "clean" and 2 <= len(parts[-2]) <= 3 and parts[-2].isalpha():
        return to_iso(parts[-2])
    if re.fullmatch(r"[A-Za-z]{2,3}[-_][A-Za-z]{2}", text):
        return to_iso(text)
    if re.fullmatch(r"[A-Za-z]{2,3}", text):
        return to_iso(text)
    return ""


def _latin_guess(text, candidates):
    padded = " " + " ".join(str(text).lower().split()) + " "
    scores = []
    allowed = set(candidates) if candidates else set(_LATIN_HINTS)
    for lang, hints in _LATIN_HINTS.items():
        if allowed and lang not in allowed:
            continue
        scores.append((sum(1 for hint in hints if hint in padded), lang))
    scores.sort(reverse=True)
    if scores and scores[0][0] >= 2 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
        return scores[0][1]
    return ""


def guess_lang(text, candidates=()):
    """Infer a language from script, then from function words when several Latin candidates remain."""
    family = script_family(text)
    cleaned = [to_iso(c) for c in candidates if to_iso(c)]
    if family in {"ar", "hi", "ru", "ja", "zh"}:
        return family
    if family == "latin":
        guessed = _latin_guess(text, cleaned)
        if guessed:
            return guessed
        latin = [c for c in cleaned if c not in {"ar", "hi", "ru", "ja", "zh"}]
        if len(latin) == 1:
            return latin[0]
    return ""


def infer_lang(entry, row, text=""):
    """Column, then config or path, then script. Falls back to the entry's first language."""
    row = row or {}
    for key in ("lang", "language", "locale", "language_code", "_v6_lang"):
        code = to_iso(row.get(key))
        if code and len(code) == 2:
            return code
    for key in ("_v6_config",):
        code = lang_from_config(row.get(key))
        if code and len(code) == 2:
            return code
    path = str(row.get("path") or "")
    if path:
        code = lang_from_config(path.split("~")[0].split("/")[-1])
        if code and len(code) == 2:
            return code
    candidates = []
    for raw in entry.get("languages") or []:
        code = to_iso(raw)
        if code and len(code) == 2 and code not in candidates:
            candidates.append(code)
    guessed = guess_lang(text or "", candidates)
    if guessed:
        return guessed
    return candidates[0] if candidates else "en"
