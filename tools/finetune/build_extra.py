#!/usr/bin/env python3
"""Build an audited, CPU-only multilingual supplement to Statim's mixture.

Only the pinned sources below are training inputs. Dataset tags are NOT an
upstream rights audit: safety prompts must additionally be traceable, by ID and
exact English text, to crowdwritten Anthropic HH training prompts (MIT).
Translations/annotations are CC-BY-4.0. No responses, redactions, cultural
adaptations, jailbreaks, untraceable prompts, or evaluation sources are trained.

All downloads/evidence stay under data/. Existing outputs are never overwritten.
The cap applies to the entire dataset, across languages and question types.
Arabic/Hindi receive extra sampling weight; classes are sampled evenly within
each language. Each distinct text produces only one decision.

Usage: .venv-train/bin/python tools/finetune/build_extra.py --out data/extra-v1.jsonl.gz
"""

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import tarfile
import time
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED = 20260927
NVIDIA = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
INDIC = "l3cube-pune/IndicGuard"
AEGIS = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
HH = "Anthropic/hh-rlhf"
MINDS = "PolyAI/minds14"
SNIPS = "benayas/snips"

# Deliberately no --dataset argument: new training sources require a fresh audit.
REVISIONS = {
    NVIDIA: "a3f7ecb3433d1933701a83f18de16c36934a7f51",
    INDIC: "f60250767f0ab6ae8f1d99831516fe8a6b453f8d",
    AEGIS: "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c",
    HH: "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
    MINDS: "40ce77cb32a384e4d50a568e1ec39ac804019d33",
    SNIPS: "16915b895754dd4028068abe52b6851a904977d7",
    "HebArabNlpProject/ArabicSentimentDataSet": "b1884f3399bf55b22206920a6f309fb512e3dac1",
    "khalidalt/SANAD": "cc04efc6edd44fc890b7625b82e36e023a353c59",
    "Divyanshu/indicxnli": "7092c27872e919f31d0496fb8b9c47bd2cba3f6c",
    "Tobi-Bueck/customer-support-tickets": "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb",
}
EXPECTED_LICENSES = {NVIDIA: "cc-by-4.0", INDIC: "cc-by-4.0",
                     AEGIS: "cc-by-4.0", HH: "mit", MINDS: "cc-by-4.0",
                     SNIPS: "apache-2.0"}
ALLOWED = {"apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
           "bsd-3-clause-clear", "cc0-1.0", "odc-by"}
FORBIDDEN = ["go_emotions", "any emotion dataset", "multi-hatecheck", "SIB-200",
             "Flores", "HWU64", "IndoNLI", "FarsTail", "Belebele", "SemRel2024",
             "AG News", "DAIR Emotion", "Banking77", "MASSIVE",
             "tyqiangz/multilingual-sentiments", "cardiffnlp tweet data",
             "LocalLLaMA/typed-decisions", "all derivatives of these sources"]

INDIC_LANGS = {"Hindi": "hi", "Bengali": "bn", "Gujarati": "gu",
               "Marathi": "mr", "Punjabi": "pa", "Tamil": "ta", "Telugu": "te",
               "Kannada": "kn", "Malyalam": "ml", "Urdu": "ur", "English": "en"}
NVIDIA_LANGS = ["ar", "hi", "de", "en", "es", "fr", "it", "ja", "ko", "nl", "th", "zh"]
MINDS_LOCALES = ["cs-CZ", "de-DE", "en-AU", "en-GB", "en-US", "es-ES", "fr-FR",
                 "it-IT", "ko-KR", "nl-NL", "pl-PL", "pt-PT", "ru-RU", "zh-CN"]
MINDS_OPTIONS = {
    "abroad": "Using a bank card or account abroad",
    "address": "Changing the account holder's address",
    "app_error": "An error in the banking app",
    "atm_limit": "The ATM cash withdrawal limit",
    "balance": "Checking the account balance",
    "business_loan": "Applying for a business loan",
    "card_issues": "A problem with a bank card",
    "cash_deposit": "Depositing cash into an account",
    "direct_debit": "Managing a direct debit",
    "freeze": "Freezing an account or card",
    "high_value_payment": "Making a large payment",
    "joint_account": "Opening or managing a joint account",
    "latest_transactions": "Checking recent transactions",
    "pay_bill": "Paying a bill",
}
SNIPS_OPTIONS = {
    "AddToPlaylist": "Add a song to a playlist",
    "BookRestaurant": "Reserve a table at a restaurant",
    "GetWeather": "Ask about the weather",
    "PlayMusic": "Play a song, album, or artist",
    "RateBook": "Assign a rating to a book",
    "SearchCreativeWork": "Find a book, song, film, or other creative work",
    "SearchScreeningEvent": "Find a movie screening or showtime",
}
QUESTIONS = {
    "safety": {
        "ar": ["هل تحتوي رسالة المستخدم على محتوى غير آمن؟",
               "هل ينبغي تصنيف هذه الرسالة على أنها غير آمنة؟",
               "هل تتضمن هذه الرسالة محتوى ضارًا أو غير آمن؟"],
        "hi": ["क्या उपयोगकर्ता के इस संदेश में असुरक्षित सामग्री है?",
               "क्या इस संदेश को असुरक्षित माना जाना चाहिए?",
               "क्या इस संदेश में हानिकारक या असुरक्षित सामग्री है?"],
        "en": ["Does this user message contain unsafe content?",
               "Should this user message be classified as unsafe?",
               "Does this message contain harmful or unsafe material?"],
    },
    "safety_score": {
        "ar": ["صنّف سلامة هذه الرسالة على مقياس من مستويين: آمن أو غير آمن.",
               "أي مستوى من مستويَي السلامة يطابق هذه الرسالة؟"],
        "hi": ["इस संदेश को दो स्तरों वाले पैमाने पर आँकें: सुरक्षित या असुरक्षित।",
               "सुरक्षा के इन दो स्तरों में से कौन सा इस संदेश पर लागू होता है?"],
        "en": ["Rate this message on the two-level scale: safe, unsafe.",
               "Which level on the safe-to-unsafe scale describes this message?"],
    },
    "banking": {"en": ["Which banking support topic best fits this request?",
                         "What banking issue does this customer need help with?",
                         "Which intent should this banking request be routed to?"]},
    "intent": {"en": ["What does the user want to do?",
                        "Which intent best describes this request?",
                        "Which action is the user asking for?"]},
}

# These are exclusion-only inputs. None can enter the training candidate path.
EVALUATION_DATASETS = [
    "google-research-datasets/go_emotions", "mteb/multi-hatecheck", "Davlan/sib200",
    "FastFit/hwu_64", "DeepPavlov/hwu64", "DeepPavlov/hwu64-translated",
    "DeepPavlov/hwu64_ru", "mteb/indonli", "Solmazp/farstail", "facebook/belebele",
    "SemRel/SemRel2024", "fancyzhx/ag_news", "dair-ai/emotion", "mteb/banking77",
    "mteb/amazon_massive_intent", "tyqiangz/multilingual-sentiments",
    "cardiffnlp/tweet_eval", "cardiffnlp/tweet_sentiment_multilingual",
    "LocalLLaMA/typed-decisions",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    # A superset of literal exact matching, consistent with build_mixture.py.
    return " ".join(text.split()).lower()


def text_hash(text):
    return digest(norm(text).encode("utf-8"))


def permissive(licence):
    return licence in ALLOWED or bool(re.fullmatch(r"cc-by-(?:1\.0|2\.0|2\.5|3\.0|4\.0)", licence))


class Cache:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = {}

    def fetch(self, url, relative=None):
        relative = relative or ("http/" + digest(url.encode()))
        path = self.root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=(15, 90))
                    r.raise_for_status()
                    # Exclusive creation also makes concurrent duplicate fetches safe.
                    try:
                        with path.open("xb") as f:
                            f.write(r.content)
                    except FileExistsError:
                        pass
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
        data = path.read_bytes()
        self.records[url] = {"url": url, "path": str(path.relative_to(ROOT)),
                             "sha256": digest(data), "bytes": len(data)}
        return path

    def json(self, url, relative=None):
        return json.loads(self.fetch(url, relative).read_text(encoding="utf-8"))

    def api(self, repo):
        rev = REVISIONS[repo]
        # Do not reuse a mutable-main API response as pinned evidence.
        return self.json(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}",
                         f"{repo}/api-{rev}.json")

    def hf(self, repo, filename):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{REVISIONS[repo]}/{urllib.parse.quote(filename)}"
        return self.fetch(url, f"{repo}/{filename}")


def json_rows(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from json.load(f)
        else:
            # Iterating the file handles literal LF only; str.splitlines() would
            # incorrectly split JSON strings containing Unicode U+2028/U+0085.
            for line in f:
                if line.strip():
                    yield json.loads(line)


def audit(cache):
    metadata = {}
    for repo in REVISIONS:
        a = cache.api(repo)
        if a["sha"] != REVISIONS[repo]:
            raise ValueError(f"Revision mismatch: {repo}")
        card = cache.hf(repo, "README.md").read_text(encoding="utf-8")
        licences = a.get("cardData", {}).get("license")
        licences = licences if isinstance(licences, list) else [licences]
        metadata[repo] = {"id": repo, "revision": a["sha"], "api_licence": licences,
                          "licence_evidence_url": f"https://huggingface.co/datasets/{repo}/blob/{a['sha']}/README.md",
                          "api_evidence_url": f"https://huggingface.co/api/datasets/{repo}/revision/{a['sha']}"}
        if repo in EXPECTED_LICENSES:
            expected = EXPECTED_LICENSES[repo]
            if licences != [expected] or not permissive(expected) or expected not in card:
                raise ValueError(f"Licence verification failed: {repo}: {licences}")

    evidence = {
        "hh_license": ("https://raw.githubusercontent.com/anthropics/hh-rlhf/master/LICENSE", "MIT License"),
        "hh_repo": ("https://raw.githubusercontent.com/anthropics/hh-rlhf/master/README.md", "crowdworker"),
        "snips_license": ("https://raw.githubusercontent.com/sonos/nlu-benchmark/master/LICENSE", "CC0 1.0 Universal"),
        "snips_repo": ("https://raw.githubusercontent.com/sonos/nlu-benchmark/master/2017-06-custom-intent-engines/README.md", "crowdsourcing"),
        "minds_paper": ("https://aclanthology.org/2021.emnlp-main.591.pdf", None),
        "culture_paper": ("https://arxiv.org/html/2508.01710v3", "CultureGuard"),
        "indic_paper": ("https://arxiv.org/html/2606.22841v1", "IndicGuard"),
        "indicxnli_repo": ("https://raw.githubusercontent.com/divyanshuaggarwal/IndicXNLI/main/README.md", "IndicXNLI"),
        "sanad_upstream": ("https://data.mendeley.com/datasets/57zpx667y9/2", None),
    }
    for key, (url, marker) in evidence.items():
        path = cache.fetch(url)
        if marker and marker.lower() not in path.read_text(encoding="utf-8").lower():
            raise ValueError(f"Upstream evidence changed: {key}")
    guidelines = cache.hf("HebArabNlpProject/ArabicSentimentDataSet", "Guidelines for Arabic Sentiment.docx")
    with zipfile.ZipFile(guidelines) as z:
        guide = " ".join(ET.fromstring(z.read("word/document.xml")).itertext())
    if "tweets" not in guide.lower():
        raise ValueError("Arabic sentiment provenance needs re-auditing")

    rejected = []
    for repo, reason, urls in [
        ("HebArabNlpProject/ArabicSentimentDataSet", "Annotation guidelines identify the source texts as tweets; CC-BY label/card does not remove the prohibited text provenance.",
         [f"https://huggingface.co/datasets/HebArabNlpProject/ArabicSentimentDataSet/blob/{REVISIONS['HebArabNlpProject/ArabicSentimentDataSet']}/Guidelines%20for%20Arabic%20Sentiment.docx"]),
        ("khalidalt/SANAD", "News articles scraped from AlKhaleej, AlArabiya and Akhbarona; prohibited text origin despite CC-BY-4.0 metadata.",
         [evidence["sanad_upstream"][0]]),
        ("Divyanshu/indicxnli", "Card body explicitly restricts repository contents to noncommercial research under CC-BY-NC-4.0; conflicts with CC0 API metadata. Train split is also excluded.",
         [evidence["indicxnli_repo"][0]]),
        ("Tobi-Bueck/customer-support-tickets", "CC-BY-NC-4.0; noncommercial licence is not allowed.", []),
    ]:
        rejected.append({**metadata[repo], "reason": reason, "upstream_evidence_urls": urls})
    return metadata, evidence, rejected


def strings(value):
    """All string values, including nested sentence pairs/choices; never keys."""
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from strings(v)


def evaluation_hashes(cache):
    banned, report = set(), []

    def parquet_dataset(repo):
        hashes, rows, failures = set(), 0, []
        try:
            index = cache.json("https://datasets-server.huggingface.co/parquet?" + urllib.parse.urlencode({"dataset": repo}))
            files = [f for f in index["parquet_files"] if "test" in f["split"].lower()]
            if not files:
                raise ValueError("No test split exposed by parquet endpoint")
            for entry in files:
                try:
                    table = pq.read_table(cache.fetch(entry["url"]))
                    for row in table.to_pylist():
                        hashes.update(text_hash(t) for t in strings(row))
                        rows += 1
                except Exception as exc:
                    failures.append({"config": entry["config"], "split": entry["split"], "error": str(exc)[:350]})
            return hashes, {"id": repo, "status": "partial" if failures else "loaded",
                            "test_files": len(files), "rows": rows, "hashes": len(hashes), "failures": failures}
        except Exception as exc:
            return hashes, {"id": repo, "status": "unavailable", "error": str(exc)[:350]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for hashes, entry in pool.map(parquet_dataset, EVALUATION_DATASETS):
            banned.update(hashes)
            report.append(entry)
            print(f"evaluation {entry['id']}: {entry['status']} ({entry.get('rows', 0)} rows)", flush=True)

    # Original IndoNLI exposes two test sets through a legacy loading script.
    for split in ("test_lay", "test_expert"):
        url = f"https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/{split}.jsonl"
        try:
            rows = list(json_rows(cache.fetch(url)))
            for row in rows:
                banned.update(text_hash(t) for t in strings(row))
            report.append({"id": "afaji/indonli", "split": split, "url": url, "status": "loaded", "rows": len(rows)})
        except Exception as exc:
            report.append({"id": "afaji/indonli", "split": split, "status": "unavailable", "error": str(exc)[:350]})

    # FLORES public evaluation is devtest; the hidden test set is not public.
    url = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    try:
        count = 0
        with tarfile.open(cache.fetch(url), "r:gz") as archive:
            for member in archive:
                if member.isfile() and "/devtest/" in member.name:
                    for line in archive.extractfile(member):
                        banned.add(text_hash(line.decode("utf-8")))
                        count += 1
        if not count:
            raise ValueError("No FLORES devtest texts in archive")
        report.append({"id": "FLORES-200", "split": "devtest", "url": url, "status": "loaded", "rows": count})
    except Exception as exc:
        report.append({"id": "FLORES-200", "split": "devtest", "status": "unavailable", "error": str(exc)[:350]})
    return banned, report


def safety_lineage(cache, banned):
    human = set()
    for row in json_rows(cache.hf(HH, "harmless-base/train.jsonl.gz")):
        for key in ("chosen", "rejected"):
            human.update(t.strip() for t in re.findall(r"\n\nHuman: (.*?)(?=\n\nAssistant:)", row[key], re.S))
    verified = {}
    for row in json_rows(cache.hf(AEGIS, "train.json")):
        prompt = row["prompt"]
        if prompt.strip() in human and text_hash(prompt) not in banned:
            verified[row["id"]] = {"prompt": prompt, "label": row["prompt_label"],
                                    "english_sha256": digest(prompt.strip().encode())}
    # Check that the current English source really uses the same IDs/text/labels.
    english = {}
    for row in json_rows(cache.hf(NVIDIA, "en/train.jsonl")):
        base = verified.get(row["id"])
        if base and row["tag"] == "generic" and row["prompt"].strip() == base["prompt"].strip() and row["prompt_label"] == base["label"]:
            english[row["id"]] = base
    if not english:
        raise ValueError("No safety prompts passed the upstream provenance join")
    return english


def safety_candidates(cache, repo, lineage, stats):
    configs = [(lang, f"{lang}/train.jsonl") for lang in NVIDIA_LANGS] if repo == NVIDIA else [
        (lang, f"{folder}/train.json") for folder, lang in INDIC_LANGS.items()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda x: cache.hf(repo, x[1]), configs))
    for (lang, filename), path in zip(configs, paths):
        for row in json_rows(path):
            stats["raw_rows"] += 1
            base = lineage.get(row.get("id"))
            if row.get("tag", "").lower() != "generic" or not base:
                stats["unverified_lineage_or_non_generic"] += 1
                continue
            reconstruction = row.get("reconstruction_id_if_redacted")
            if reconstruction is not None and not (isinstance(reconstruction, float) and math.isnan(reconstruction)):
                stats["redacted"] += 1
                continue
            prompt = row.get("prompt")
            label = row.get("prompt_label")
            if not isinstance(prompt, str) or "REDACTED" in prompt or label not in ("safe", "unsafe") or label != base["label"]:
                stats["invalid_or_conflicting_label"] += 1
                continue
            if lang == "en" and prompt.strip() != base["prompt"].strip():
                stats["changed_english_prompt"] += 1
                continue
            yield {"text": prompt, "label": label, "lang": lang, "task": "safety",
                   "row_id": str(row["id"]), "file": filename,
                   "english_sha256": base["english_sha256"], "english_text": base["prompt"]}


def minds_candidates(cache, stats):
    # Read only text/label columns: audio is never decoded or used for training.
    api = cache.api(MINDS)
    infos = api["cardData"]["dataset_info"]
    info = next(i for i in infos if i["config_name"] == "all")
    features = {f["name"]: f for f in info["features"]}
    names = features["intent_class"]["dtype"]["class_label"]["names"]
    labels = [names[str(i)] for i in range(len(names))] if isinstance(names, dict) else names
    if labels != list(MINDS_OPTIONS):
        raise ValueError("MINDS intent class order changed")
    for locale in MINDS_LOCALES:
        files = sorted(s["rfilename"] for s in api["siblings"] if s["rfilename"].startswith(locale + "/train-") and s["rfilename"].endswith(".parquet"))
        if not files:
            raise ValueError(f"Missing MINDS train files for {locale}")
        for filename in files:
            rows = pq.read_table(cache.hf(MINDS, filename), columns=["transcription", "english_transcription", "intent_class", "path"]).to_pylist()
            for row in rows:
                stats["raw_rows"] += 1
                yield {"text": row["transcription"], "english_text": row["english_transcription"],
                       "label": labels[row["intent_class"]], "lang": locale[:2],
                       "locale": locale, "task": "banking", "row_id": row["path"], "file": filename}


def snips_candidates(cache, stats):
    # The HF repackaging differs from the author's published training corpus.
    # Require an exact text AND intent match instead of trusting its Apache tag.
    original = {}
    for label in SNIPS_OPTIONS:
        url = f"https://raw.githubusercontent.com/sonos/nlu-benchmark/master/2017-06-custom-intent-engines/{label}/train_{label}_full.json"
        raw = cache.fetch(url, f"snips-upstream/{label}.json").read_bytes()
        # Historical SNIPS JSON sometimes encodes an emoji as CESU-8 surrogate
        # pairs. Decode those losslessly; never use replacement characters.
        source = json.loads(raw.decode("utf-8", errors="surrogatepass"))
        original[label] = set()
        for row in source[label]:
            text = "".join(part["text"] for part in row["data"])
            try:
                text = text.encode("utf-16", errors="surrogatepass").decode("utf-16")
            except UnicodeError:
                stats["invalid_upstream_unicode"] += 1
                continue
            original[label].add(text)
    filename = "data/train-00000-of-00001.parquet"
    for n, row in enumerate(pq.read_table(cache.hf(SNIPS, filename)).to_pylist()):
        stats["raw_rows"] += 1
        if row["category"] not in SNIPS_OPTIONS:
            raise ValueError(f"Unknown SNIPS label: {row['category']}")
        if row["text"] not in original[row["category"]]:
            stats["not_exact_upstream_training_text_and_label"] += 1
            continue
        yield {"text": row["text"], "label": row["category"], "lang": "en",
               "task": "intent", "row_id": str(n), "file": filename}


def choose_rows(rows, cap, repo, banned, seen, stats):
    rng = random.Random(f"{SEED}:{repo}")
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    unique = set()
    for row in rows:
        text = row["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 6000:
            stats["empty_or_too_long"] += 1
            continue
        h = text_hash(text)
        if h in banned or (row.get("english_text") and text_hash(row["english_text"]) in banned):
            stats["evaluation_overlap"] += 1
            continue
        if h in seen or h in unique:
            stats["duplicate_text"] += 1
            continue
        unique.add(h)
        buckets[row["lang"]][row["label"]].append(row)
    queues = {}
    for lang, classes in sorted(buckets.items()):
        for values in classes.values():
            rng.shuffle(values)
        queue = []
        while any(classes.values()):
            labels = sorted(label for label in classes if classes[label])
            rng.shuffle(labels)
            for label in labels:
                queue.append(classes[label].pop())
        queues[lang] = collections.deque(queue)
    # Weighted round-robin gives ar/hi priority without starving other languages.
    cycle = [lang for lang in sorted(queues) for _ in range(20 if lang in {"ar", "hi"} else 1)]
    selected = []
    while len(selected) < cap and any(queues.values()):
        for lang in cycle:
            if queues[lang] and len(selected) < cap:
                row = queues[lang].popleft()
                selected.append(row)
                seen.add(text_hash(row["text"]))
    rng.shuffle(selected)
    stats["eligible_unique_rows"] = len(unique)
    return selected


def to_item(row, repo, rng):
    task, lang = row["task"], row["lang"]
    if task == "safety":
        # Both targets express the original binary annotation, never an invented
        # multi-level severity score. noul is always [false, true].
        kind = "score" if rng.randrange(3) == 0 else "noul"
        prompts = QUESTIONS["safety_score" if kind == "score" else "safety"]
        q = {"type": kind, "instructions": rng.choice(prompts.get(lang, prompts["en"])),
             "criteria": ["safe", "unsafe"] if kind == "score" else {}}
        target = [float(row["label"] == "safe"), float(row["label"] == "unsafe")]
    else:
        options = dict(MINDS_OPTIONS if task == "banking" else SNIPS_OPTIONS)
        q = {"type": "choice", "instructions": rng.choice(QUESTIONS[task]["en"]), "criteria": options}
        target = [float(label == row["label"]) for label in options]
    return {"state": row["text"], "q": q, "target": target, "src": f"{repo}/{lang}/train"}


def validate(item):
    if set(item) != {"state", "q", "target", "src"} or not isinstance(item["state"], str) or not item["state"].strip():
        raise ValueError("Invalid training item")
    q, target = item["q"], item["target"]
    if not isinstance(q["instructions"], str) or not q["instructions"]:
        raise ValueError("Missing question")
    if q["type"] == "noul":
        size = 2
        if q["criteria"] != {}:
            raise ValueError("noul does not have named choices")
    elif q["type"] == "choice" and isinstance(q["criteria"], dict):
        size = len(q["criteria"])
    elif q["type"] == "score" and isinstance(q["criteria"], list):
        size = len(q["criteria"])
    else:
        raise ValueError("Unknown decision type")
    if size < 2 or len(target) != size or any(t not in (0.0, 1.0) for t in target) or sum(target) != 1.0:
        raise ValueError("Target is not a correctly sized one-hot vector")


def write_gzip(path, rows):
    # Reproducible bytes, independent of wall clock and output basename.
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dataset", type=int, default=3000)
    args = ap.parse_args()
    if args.per_dataset < 1:
        ap.error("--per-dataset must be positive")
    out = Path(args.out).resolve()
    manifest_path = Path(str(out) + ".manifest.json")
    provenance_path = Path(str(out) + ".provenance.jsonl.gz")
    if not out.is_relative_to(DATA.resolve()) or not str(out).endswith(".jsonl.gz"):
        ap.error("--out must be a .jsonl.gz file under data/")
    for path in (out, manifest_path, provenance_path):
        if path.exists():
            ap.error(f"Refusing to modify existing file: {path}")
    cache = Cache(DATA / "extra-v1.sources")
    metadata, evidence, rejected = audit(cache)
    print("Licence/provenance audit passed for the restricted training sources.", flush=True)
    banned, evaluation_report = evaluation_hashes(cache)
    lineage = safety_lineage(cache, banned)
    print(f"Evaluation text hashes: {len(banned)}; verified safety IDs: {len(lineage)}", flush=True)

    items, provenance, taken, seen = [], [], [], set()
    language_counts, kind_counts = collections.Counter(), collections.Counter()
    for repo in (NVIDIA, INDIC, MINDS, SNIPS):
        stats = collections.Counter()
        print(f"Building {repo} ...", flush=True)
        if repo in (NVIDIA, INDIC):
            candidates = safety_candidates(cache, repo, lineage, stats)
        elif repo == MINDS:
            candidates = minds_candidates(cache, stats)
        else:
            candidates = snips_candidates(cache, stats)
        selected = choose_rows(candidates, args.per_dataset, repo, banned, seen, stats)
        if not selected:
            raise ValueError(f"No usable rows for audited dataset {repo}")
        rng = random.Random(f"{SEED}:{repo}:questions")
        for row in selected:
            item = to_item(row, repo, rng)
            validate(item)
            items.append(item)
            provenance.append({"item_sha256": digest(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()),
                               "id": repo, "revision": REVISIONS[repo], "file": row["file"],
                               "row_id": row["row_id"], "language": row["lang"],
                               "label": row["label"], "state_sha256": digest(row["text"].encode()),
                               **({"upstream_english_sha256": row["english_sha256"],
                                   "upstream_datasets": [HH, AEGIS, NVIDIA]} if "english_sha256" in row else {})})
            language_counts[row["lang"]] += 1
            kind_counts[item["q"]["type"]] += 1
        entry = {**metadata[repo], "licence": EXPECTED_LICENSES[repo],
                 "text_licence": ["MIT", "CC-BY-4.0"] if repo in (NVIDIA, INDIC) else (["CC0-1.0", "Apache-2.0"] if repo == SNIPS else ["CC-BY-4.0"]),
                 "labels_licence": "CC-BY-4.0" if repo != SNIPS else "CC0-1.0 (upstream); Apache-2.0 (HF packaging)",
                 "languages": sorted({r["lang"] for r in selected}), "items": len(selected),
                 "counts_per_language": dict(sorted(collections.Counter(r["lang"] for r in selected).items())),
                 "split": "train", "stats": dict(stats)}
        if repo in (NVIDIA, INDIC):
            entry.update({"selection": "Generic prompt-only rows; exact HH harmless-base/train human-turn to AEGIS train to Nemotron English train ID/text/label join; consistent binary labels. Exclude all adapted/jailbreak/redacted/untraceable rows and all responses.",
                          "attribution": "Anthropic (2022); Ghosh et al. (2025), AEGIS2.0; Joshi et al. (2025), CultureGuard; " + ("Bramhecha et al. (2026), IndicGuard" if repo == INDIC else "NVIDIA Corporation"),
                          "upstream_evidence_urls": [metadata[HH]["licence_evidence_url"], evidence["hh_license"][0], evidence["hh_repo"][0], metadata[AEGIS]["licence_evidence_url"], evidence["culture_paper"][0]] + ([evidence["indic_paper"][0]] if repo == INDIC else [])})
        elif repo == MINDS:
            entry.update({"selection": "Native transcriptions only; original crowd-recorded utterances; no audio or translated English added as training items.",
                          "attribution": "Gerz et al. (2021), Multilingual and Cross-Lingual Intent Detection from Spoken Data, PolyAI Limited.",
                          "upstream_evidence_urls": [evidence["minds_paper"][0]]})
        else:
            entry.update({"selection": "HF train rows with an exact text and intent match in the author's CC0 train_intent_full.json files; unmatched repackaged rows excluded.",
                          "attribution": "Coucke et al. (2018), Snips Voice Platform; Snips/Sonos; benayas (HF packaging).",
                          "upstream_evidence_urls": [evidence["snips_license"][0], evidence["snips_repo"][0]]})
        taken.append(entry)
        print(f"  {len(selected)} items; {entry['counts_per_language']}", flush=True)

    random.Random(SEED).shuffle(items)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gzip(out, items)
    write_gzip(provenance_path, provenance)
    manifest = {
        "output": str(out.relative_to(ROOT)), "items": len(items), "seed": SEED,
        "per_dataset_cap": args.per_dataset, "datasets": taken, "rejected": rejected,
        "counts_per_language": dict(sorted(language_counts.items())), "kinds": dict(kind_counts),
        "forbidden_training_sources": FORBIDDEN, "evaluation_exclusion": {
            "method": "SHA-256 of lowercased whitespace-normalized string values; includes raw text, sentence-pair components, nested choices, and available English source text.",
            "hashes": len(banned), "datasets": evaluation_report,
            "limitations": "Only accessible public test/devtest texts can be checked; unavailable sets are reported. This is exact/normalized matching, not semantic or translation deduplication. All emotion datasets and all named forbidden families/derivatives are excluded as training sources by the closed source whitelist."},
        "sampling": "Seeded shuffle within language/label buckets; class-balanced round-robin within languages; language weights ar=20, hi=20, others=1; single decision per distinct normalized text; final seeded shuffle.",
        "schema": "noul targets [false,true]; choice targets follow criteria insertion order; score targets follow levels [safe,unsafe] (binary safety only, not inferred severity).",
        "changes_to_source": "Keep original prompt/transcription text; replace annotation fields with typed questions and one-hot targets; short option descriptions and question paraphrases are authored here.",
        "coverage_limits": "No scraped sentiment/news data or noncommercial IndicXNLI admitted. Tasks covered: safety, intent, banking support routing; no clean NLI or sentiment source admitted in this audit.",
        "provenance": str(provenance_path.relative_to(ROOT)), "sha256": digest(out.read_bytes()),
        "provenance_sha256": digest(provenance_path.read_bytes()),
        "evidence_and_downloads": sorted(cache.records.values(), key=lambda x: x["url"]),
    }
    with manifest_path.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "items": len(items),
                      "counts_per_language": manifest["counts_per_language"], "kinds": manifest["kinds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
