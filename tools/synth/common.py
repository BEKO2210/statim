"""Shared helpers for the synthetic typed-decision pipeline.

Training text is produced only by a local model. This module has no examples.
"""
import hashlib
import json
import random
import re
import unicodedata

NEAR_DUP = 0.8
NGRAM_N = 5
MINHASH_PERMS = 64
MINHASH_BANDS = 32  # 2 rows each; low LSH threshold, Jaccard confirms
_PRIME = (1 << 61) - 1

# Letters by script. Latin includes Turkish and other Latin-extended orthographies.
_LATIN = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_DEVA = re.compile(r"[\u0900-\u097F]")
_CYR = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")
_KANA = re.compile(r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9D]")
_CJK = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")
_WORDISH = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def nfc(text):
    return unicodedata.normalize("NFC", "" if text is None else str(text)).strip()


def norm(text):
    """Same whitespace normalisation as tools/finetune/build_mixture.py."""
    return " ".join(str(text).split()).lower()


def word_count(text):
    """Whitespace words, with each CJK or kana character counted as one word."""
    text = nfc(text)
    cjk = len(_WORDISH.findall(text))
    rest = _WORDISH.sub(" ", text)
    return cjk + len(rest.split())


def script_counts(text):
    counts = {"latin": 0, "arabic": 0, "devanagari": 0, "cyrillic": 0, "kana": 0, "cjk": 0}
    for ch in text:
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


def language_ok(text, lang):
    """Script heuristic. Latin languages are not separated from each other."""
    counts = script_counts(text)
    total = sum(counts.values())
    if total < 4:
        return False
    latin = counts["latin"] / total
    arabic = counts["arabic"] / total
    deva = counts["devanagari"] / total
    cyr = counts["cyrillic"] / total
    kana = counts["kana"] / total
    cjk = counts["cjk"] / total
    if lang == "ar":
        return arabic >= 0.6 and latin < 0.25
    if lang == "hi":
        return deva >= 0.6 and latin < 0.25
    if lang == "ru":
        return cyr >= 0.6 and latin < 0.25
    if lang == "ja":
        return counts["kana"] >= 4 and kana >= 0.05 and (kana + cjk) >= 0.5 and arabic < 0.1
    if lang == "zh":
        return counts["cjk"] >= 8 and cjk >= 0.5 and counts["kana"] <= 1 and kana < 0.02
    # en de fr es it pt nl pl tr
    foreign = arabic + deva + cyr + kana + cjk
    return latin >= 0.7 and foreign <= 0.15


def one_hot(n, index):
    return [1.0 if i == index else 0.0 for i in range(n)]


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_id(payload):
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(blob)


def char_ngrams(text, n=NGRAM_N):
    t = norm(text)
    if not t:
        return set()
    if len(t) <= n:
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    inter = sum(1 for g in a if g in b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def _gram_hash(gram):
    return int.from_bytes(hashlib.blake2s(gram.encode("utf-8"), digest_size=8).digest(), "little")


class MinHash:
    """64-permutation MinHash. Bands are only a candidate filter; Jaccard decides."""

    def __init__(self, nperm=MINHASH_PERMS, seed=20260927):
        rng = random.Random(seed)
        self.ab = [(rng.randrange(1, _PRIME - 1), rng.randrange(0, _PRIME - 1)) for _ in range(nperm)]
        self.bands = MINHASH_BANDS
        self.rows = nperm // MINHASH_BANDS

    def signature(self, grams):
        if not grams:
            return tuple(0 for _ in self.ab)
        hashes = [_gram_hash(g) % _PRIME for g in grams]
        out = []
        for a, b in self.ab:
            out.append(min((a * h + b) % _PRIME for h in hashes))
        return tuple(out)

    def band_keys(self, sig):
        keys = []
        for band in range(self.bands):
            sl = sig[band * self.rows:(band + 1) * self.rows]
            keys.append((band, sl))
        return keys


def item_fingerprint(item):
    q = item["q"]
    parts = [item.get("state") or "", q.get("instructions") or ""]
    crit = q.get("criteria")
    if isinstance(crit, dict):
        for key, value in crit.items():
            parts.append(str(key))
            if value:
                parts.append(str(value))
    elif isinstance(crit, list):
        parts.extend(str(x) for x in crit)
    return "\n".join(parts)


def training_item(item):
    """The four fields the mixture trainer reads. No rationale, no seed."""
    q = item["q"]
    out_q = {"type": q["type"], "instructions": q["instructions"]}
    if "criteria" in q:
        out_q["criteria"] = q["criteria"]
    return {"state": item["state"], "q": out_q, "target": list(item["target"]), "src": item["src"]}


def has_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(has_key(v, key) for v in obj)
    return False
