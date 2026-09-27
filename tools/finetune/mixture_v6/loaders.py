"""Row loaders for mixture v6.

Dataset scripts are never executed. Script-backed Hub repos are read from the
auto-converted parquet revision or from raw data files. GitHub and Zenodo
files are pinned to a commit or record and checked against a SHA-256 stored
on the registry entry.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw"
PARQUET_REV = "refs/convert/parquet"

# Intent order matches the ClassLabel used by PolyAI/minds14 (alphabetical;
# path ``ATM_LIMIT`` is class 3).
MINDS14_INTENTS = [
    "abroad", "address", "app_error", "atm_limit", "balance", "business_loan",
    "card_issues", "cash_deposit", "direct_debit", "freeze", "high_value_payment",
    "joint_account", "latest_transactions", "pay_bill",
]
MINDS14_CONFIGS = [
    "cs-CZ", "de-DE", "en-AU", "en-GB", "en-US", "es-ES", "fr-FR", "it-IT",
    "ko-KR", "nl-NL", "pl-PL", "pt-PT", "ru-RU", "zh-CN",
]
TAPACO_CONFIGS = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "tr", "ru", "ar", "hi", "ja", "cmn"]
JOB_LANGS = ["de", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "zh"]
INDIC_LANGS = ["as", "bd", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
TATOEBA_PAIRS = ["en-mr", "eo-nl", "es-gl", "es-pt", "fr-ru"]
EURLEX_LANGS = ["DE", "EN", "ES", "FR", "IT", "NL", "PL", "PT"]
TRUTHFUL_LANGS = ["DE", "FR", "ES", "IT", "PT-PT", "NL", "PL"]

PINNED = {
    "github:asappresearch/abcd": {
        "url": "https://raw.githubusercontent.com/asappresearch/abcd/0548f35968a93f407035bc0c8a6334c6cf5d07ef/data/abcd_v1.1.json.gz",
        "sha256": "2bdf53ac359543dcdc38d55bc6513e78df120363f8f44870716e909f4606de15",
        "filename": "abcd_v1.1.json.gz",
        "cache": "abcd",
    },
    "github:wwbp/empathic_reactions": {
        "url": "https://raw.githubusercontent.com/wwbp/empathic_reactions/931501798177d4c51b9a7b0360d39497e7a1bb0e/data/responses/data/messages.csv",
        "sha256": "380d2670254565629f81616df57cc0729de96ac87c056d05e39c370624ae8f64",
        "filename": "messages.csv",
        "cache": "empathic_reactions",
    },
    "github:bvidgen/Dynamically-Generated-Hate-Speech-Dataset (v0.2.3.csv; NOT tasksource/dynahate mirror tagged gpl)": {
        "url": "https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/03eb1c60af1f213ae19a16ff1d139d88229c2b7b/Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv",
        "sha256": "6e0aa7e598ab5b648cb1f0b48626a3e5f2e39a5abbd9fee998efa5a0b6cab4ca",
        "filename": "Dynamically_Generated_Hate_Dataset_v0.2.3.csv",
        "cache": "dynahate",
    },
    "jagoldz/gahd (filter via GitHub jagol/gahd gahd_disaggregated.csv)": {
        "url": "https://raw.githubusercontent.com/jagol/gahd/ca08f691b3d6c4ee5ea41b8be00431f4ac3bc294/gahd_disaggregated.csv",
        "sha256": "e9cf8c06c572419a127f868bc45d41e00a0c0e04fe35b5f1ad9dc6e514f3703c",
        "filename": "gahd_disaggregated.csv",
        "cache": "gahd",
    },
    "zenodo:3609356 (ClaimBuster)": {
        "url": "https://zenodo.org/api/records/3609356/files/groundtruth.csv/content",
        "sha256": "64890c51e1092fec0d361ca770b3efccc0c889dd569b46b35922d94c44fa99ea",
        "filename": "groundtruth.csv",
        "cache": "claimbuster",
    },
}


def script_free_data_files(files):
    """Data files only. A dataset script (``*.py``) is never selected."""
    out = []
    for name in files:
        base = name.rsplit("/", 1)[-1]
        if base.endswith(".py") or base.startswith(".") or base.lower().startswith("readme"):
            continue
        if name.endswith((".parquet", ".json", ".jsonl", ".csv", ".tsv", ".json.gz")):
            out.append(name)
    return out


def choose_explicit_file(files, wanted):
    """Pick ``wanted`` out of a repo listing, never a metrics split or a script."""
    if wanted in files:
        return wanted
    raise RuntimeError("data file %s is not in the repo" % wanted)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_verified(url, dest, digest):
    """Download ``url`` to ``dest`` unless a copy with the right SHA-256 is cached. A cached or
    fresh file with the wrong digest (tampered, truncated, partial) is deleted, so the next build
    fetches it again instead of failing on the same file forever."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == digest:
        return dest
    dest.unlink(missing_ok=True)
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(3):  # a proxy can cut a large download short (ContentTooShortError)
        try:
            urllib.request.urlretrieve(url, partial)
            break
        except urllib.error.ContentTooShortError:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
    if sha256_file(partial) != digest:
        partial.unlink(missing_ok=True)
        raise RuntimeError("SHA-256 mismatch for %s" % dest.name)
    partial.replace(dest)
    return dest


def pinned_path(spec):
    """Download a pinned URL into ``data/raw/<cache>/`` and verify its SHA-256."""
    dest_dir = RAW_ROOT / spec["cache"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec["filename"]
    expected = spec.get("sha256") or ""
    if dest.exists():
        got = sha256_file(dest)
        if expected and got != expected:
            dest.unlink()
        else:
            return dest
    urllib.request.urlretrieve(spec["url"], dest)
    got = sha256_file(dest)
    if expected and got != expected:
        dest.unlink()
        raise RuntimeError("SHA-256 mismatch for %s" % spec["filename"])
    return dest


def _hub(repo, filename, revision=None):
    from huggingface_hub import hf_hub_download
    kwargs = {"repo_type": "dataset"}
    if revision:
        kwargs["revision"] = revision
    return hf_hub_download(repo, filename, **kwargs)


def sample_parquet(path, limit, columns=None, drop=("audio", "audio_path")):
    """Read at most ``limit`` rows spread across the file so blocked labels are not missed: one row
    at a seeded random offset inside each of ``limit`` equal strata. A fixed stride would alias with
    a file whose labels cycle (turkish-intent repeats 20 intents, so every 1,000th row has the same one)."""
    import bisect
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = list(columns) if columns else [n for n in pf.schema_arrow.names if n not in drop]
    total = pf.metadata.num_rows or 0
    if total <= 0:
        return []
    if total <= limit:
        want = list(range(total))
    else:
        import random
        rng = random.Random("%s|%d|%d" % (Path(path).name, total, limit))
        step = total / float(limit)
        want = sorted({min(total - 1, int(i * step + rng.random() * step)) for i in range(limit)})
    rows, seen = [], 0
    for batch in pf.iter_batches(batch_size=4096, columns=names):
        n = batch.num_rows
        lo, hi = bisect.bisect_left(want, seen), bisect.bisect_left(want, seen + n)
        if hi > lo:
            rows.extend(batch.take([w - seen for w in want[lo:hi]]).to_pylist())
        seen += n
        if seen > want[-1]:
            break
    return rows[:limit]


def contiguous_parquet(path, limit, blocks=8):
    """``limit`` rows as ``blocks`` contiguous runs at evenly spaced offsets. Pair builders need
    neighbouring rows (a paraphrase set, a day's headlines); an evenly spread sample splits them."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows or 0
    if total <= limit:
        return sample_parquet(path, limit)
    run = max(1, limit // blocks)
    spans = [(start, start + run) for start in
             (int(i * (total - run) / max(1, blocks - 1)) for i in range(blocks))]
    rows, seen = [], 0
    names = [n for n in pf.schema_arrow.names if n not in ("audio", "audio_path")]
    for batch in pf.iter_batches(batch_size=1024, columns=names):
        lo, hi = seen, seen + batch.num_rows
        if any(a < hi and b > lo for a, b in spans):
            for j, row in enumerate(batch.to_pylist()):
                if any(a <= lo + j < b for a, b in spans):
                    rows.append(row)
        seen = hi
        if seen >= spans[-1][1]:
            break
    return rows[:limit]


def _tag(rows, **fields):
    for row in rows:
        row.update(fields)
    return rows


def _read_json_bytes(raw):
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _csv_rows(path, limit, predicate=None):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            if predicate and not predicate(row):
                continue
            rows.append(dict(row))
            if len(rows) >= limit:
                break
    return rows


def _balanced_take(groups, limit):
    """Round-robin across groups so one language or class does not fill the cap."""
    keys = list(groups)
    out, index = [], 0
    while len(out) < limit and any(groups[k] for k in keys):
        key = keys[index % len(keys)]
        index += 1
        if groups[key]:
            out.append(groups[key].pop(0))
    return out


def _load_abcd(limit):
    path = pinned_path(PINNED["github:asappresearch/abcd"])
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    train = data.get("train") if isinstance(data, dict) else data
    return [row for row in train if isinstance(row, dict)][:limit]


def _load_empathic(limit):
    path = pinned_path(PINNED["github:wwbp/empathic_reactions"])
    return _csv_rows(path, limit)


def _load_dynahate(limit):
    path = pinned_path(PINNED["github:bvidgen/Dynamically-Generated-Hate-Speech-Dataset (v0.2.3.csv; NOT tasksource/dynahate mirror tagged gpl)"])
    return _csv_rows(path, limit, lambda row: (row.get("split") or "train") == "train")


def _load_gahd(limit):
    path = pinned_path(PINNED["jagoldz/gahd (filter via GitHub jagol/gahd gahd_disaggregated.csv)"])
    keep = {"dynabench", "contrastive", "translation"}
    return _csv_rows(path, limit, lambda row: row.get("source") in keep and (row.get("split") or "train") == "train")


def _load_claimbuster(limit):
    path = pinned_path(PINNED["zenodo:3609356 (ClaimBuster)"])
    return _csv_rows(path, limit)


def _load_indic(limit):
    groups = {lang: [] for lang in INDIC_LANGS}
    per = max(1, limit // len(INDIC_LANGS))
    for lang in INDIC_LANGS:
        bucket = []
        for split in ("validation", "test"):
            filename = "translation-%s/%s/0000.parquet" % (lang, split)
            try:
                path = _hub("ai4bharat/IndicSentiment", filename, PARQUET_REV)
            except Exception:
                continue
            bucket.extend(sample_parquet(path, per))
            if len(bucket) >= per:
                break
        groups[lang] = _tag(bucket[:per], _v6_config="translation-%s" % lang, _v6_lang=lang)
    return _balanced_take(groups, limit)


def _load_truthful(limit):
    groups = {}
    per = max(1, limit // len(TRUTHFUL_LANGS))
    iso = {"PT-PT": "pt"}
    for lang in TRUTHFUL_LANGS:
        filename = "truthfulqa_gen_%s_validation.jsonl" % lang
        path = _hub("Eurolingua/truthfulqax", filename)
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= per:
                    break
        code = iso.get(lang, lang.lower())
        groups[code] = _tag(rows, _v6_lang=code, _v6_config=lang)
    return _balanced_take(groups, limit)


def _load_nli_tr(limit):
    path = _hub("boun-tabi/nli_tr", "multinli_tr/train/0000.parquet", PARQUET_REV)
    return _tag(sample_parquet(path, limit), _v6_lang="tr", _v6_config="multinli_tr")


def _load_cuad(limit):
    path = _hub("theatticusproject/cuad-qa", "default/train/0000.parquet", PARQUET_REV)
    return _tag(sample_parquet(path, limit), _v6_lang="en")


def _load_eurlex(limit):
    meta_path = _hub("ddrg/super_eurlex", "meta_data/1.parquet")
    import pyarrow.parquet as pq
    meta = {}
    for row in pq.read_table(meta_path, columns=["celex_id", "subject_matter", "eurovoc", "directory_code"]).to_pylist():
        cid = row.get("celex_id")
        cid = cid[0] if isinstance(cid, list) and cid else cid
        if cid:
            meta[str(cid)] = row
    groups = {}
    per = max(1, limit // len(EURLEX_LANGS))
    for lang in EURLEX_LANGS:
        path = _hub("ddrg/super_eurlex", "text_data/%s/1_clean.parquet" % lang)
        rows = []
        for row in sample_parquet(path, per * 3, columns=["celex_id", "text_cleaned"]):
            cid = row.get("celex_id")
            cid = cid[0] if isinstance(cid, list) and cid else cid
            info = meta.get(str(cid), {})
            label = info.get("subject_matter") or info.get("eurovoc") or info.get("directory_code")
            text = row.get("text_cleaned") or ""
            if isinstance(text, str) and len(text) > 1500:
                text = text[:1500]
            if not text or not label:
                continue
            rows.append({"text": text, "subject_matter": label, "_v6_lang": lang.lower(), "_v6_config": "1.%s.clean" % lang})
            if len(rows) >= per:
                break
        groups[lang] = rows
    return _balanced_take(groups, limit)


def _load_multiwoz(limit):
    path = _hub("pfb30/multi_woz_v22", "v2.2/train/0000.parquet", PARQUET_REV)
    dialogs = sample_parquet(path, max(limit, 200))
    rows = []
    for dialog in dialogs:
        turns = dialog.get("turns") or {}
        utts = turns.get("utterance") or []
        speakers = turns.get("speaker") or []
        acts = turns.get("dialogue_acts") or []
        frames = turns.get("frames") or []
        for i, utt in enumerate(utts):
            if not isinstance(utt, str) or not utt.strip():
                continue
            # Speaker 0 is the user in MultiWOZ 2.2.
            if i < len(speakers) and speakers[i] not in (0, "0", "USER", "user"):
                continue
            act = ""
            if i < len(acts) and isinstance(acts[i], dict):
                types = ((acts[i].get("dialog_act") or {}).get("act_type")) or []
                act = types[0] if types else ""
            intent = ""
            if i < len(frames) and isinstance(frames[i], dict):
                for state in frames[i].get("state") or []:
                    if isinstance(state, dict) and state.get("active_intent") not in (None, "", "NONE"):
                        intent = state["active_intent"]
                        break
            rows.append({"utterance": utt.strip(), "dialogue_act": act, "intent": intent or act,
                         "_v6_lang": "en", "_v6_config": "v2.2"})
            if len(rows) >= limit:
                return rows
    return rows


def _load_crosswoz(limit):
    path = _hub("ConvLab/crosswoz", "data.zip")
    with zipfile.ZipFile(path) as zf:
        dialogs = json.loads(zf.read("data/dialogues.json"))
    rows = []
    for dialog in dialogs:
        if dialog.get("data_split") not in (None, "train"):
            continue
        for turn in dialog.get("turns") or []:
            if turn.get("speaker") not in ("user", "usr"):
                continue
            utt = turn.get("utterance") or ""
            if not str(utt).strip():
                continue
            acts = (turn.get("dialogue_acts") or {}).get("binary") or []
            chosen = next((a for a in acts if isinstance(a, dict) and a.get("domain") not in (None, "", "General")), None)
            chosen = chosen or (acts[0] if acts else {})
            intent = (chosen or {}).get("intent") or ""
            domain = (chosen or {}).get("domain") or ""
            rows.append({"utterance": str(utt).strip(), "dialog_act": intent, "domain": domain,
                         "_v6_lang": "zh", "_v6_config": "zh"})
            if len(rows) >= limit:
                return rows
    return rows


def _load_bitod(limit):
    groups = {}
    per = max(1, limit // 2)
    for lang in ("en", "zh"):
        path = _hub("DeepPavlov/BiToD", "%s/train/0000.parquet" % lang, PARQUET_REV)
        rows = sample_parquet(path, per)
        groups[lang] = _tag(rows, _v6_lang=lang, _v6_config=lang)
    return _balanced_take(groups, limit)


def _load_minds14(limit):
    groups = {}
    per = max(1, limit // len(MINDS14_CONFIGS))
    cols = ["path", "transcription", "english_transcription", "intent_class", "lang_id"]
    for config in MINDS14_CONFIGS:
        path = _hub("PolyAI/minds14", "%s/train/0000.parquet" % config, PARQUET_REV)
        rows = sample_parquet(path, per, columns=cols)
        groups[config] = _tag(rows, _v6_config=config, _v6_lang=config.split("-")[0].lower())
    return _balanced_take(groups, limit)


def _load_tapaco(limit):
    groups = {}
    per = max(1, limit // len(TAPACO_CONFIGS))
    for lang in TAPACO_CONFIGS:
        path = _hub("community-datasets/tapaco", "%s/train/0000.parquet" % lang, PARQUET_REV)
        rows = contiguous_parquet(path, per)  # whole paraphrase sets, so positive pairs exist
        groups[lang] = _tag(rows, _v6_config=lang, _v6_lang="zh" if lang == "cmn" else lang)
    return _balanced_take(groups, limit)


def _load_headlines(limit):
    """Contiguous runs of the date-ordered file, so one day's stories (group_id) come together."""
    repo = "dell-research-harvard/headlines-semantic-similarity"
    shards = sorted(f for f in _parquet_listing(repo) if f.startswith("default/train/"))
    if len(shards) > MAX_SHARDS:
        shards = [shards[int(i * len(shards) / MAX_SHARDS)] for i in range(MAX_SHARDS)]
    per = max(1, limit // max(1, len(shards)))
    rows = []
    for shard in shards:
        rows.extend(contiguous_parquet(_hub(repo, shard, PARQUET_REV), per))
    return _tag(rows, _v6_lang="en", _v6_config="default")


def _load_job_titles(limit):
    groups = {}
    per = max(2, limit // len(JOB_LANGS))
    for lang in JOB_LANGS:
        queries = sample_parquet(_hub("Avature/Job-Title-Similarity", "%s/queries/0000.parquet" % lang, PARQUET_REV), 200)
        corpus = [row.get("text") or "" for row in sample_parquet(
            _hub("Avature/Job-Title-Similarity", "%s/corpus/0000.parquet" % lang, PARQUET_REV), 100000)]
        rows = []
        for query in queries:
            labels = [i for i in (query.get("labels") or []) if isinstance(i, int) and 0 <= i < len(corpus)]
            if not labels or not query.get("text"):
                continue
            banned = set(labels)
            negative = next((i for i in range(len(corpus)) if i not in banned and corpus[i]), None)
            rows.append({"sentence1": query["text"], "sentence2": corpus[labels[0]], "relatedness_score": 1.0,
                         "_v6_lang": lang, "_v6_config": lang})
            if negative is not None:
                rows.append({"sentence1": query["text"], "sentence2": corpus[negative], "relatedness_score": 0.0,
                             "_v6_lang": lang, "_v6_config": lang})
            if len(rows) >= per:
                break
        groups[lang] = rows[:per]
    return _balanced_take(groups, limit)


def _load_pii_json(limit):
    path = _hub("urchade/synthetic-pii-ner-mistral-v1", "data.json")
    # The file is one JSON list. Stream objects without holding every record twice.
    decoder = json.JSONDecoder()
    raw = Path(path).read_text(encoding="utf-8")
    index = 0
    while index < len(raw) and raw[index] not in "[{":
        index += 1
    if index >= len(raw):
        return []
    if raw[index] == "{":
        obj, _ = decoder.raw_decode(raw, index)
        return [obj]
    index += 1
    rows = []
    n = len(raw)
    while index < n and len(rows) < limit:
        while index < n and raw[index] in " \t\r\n,":
            index += 1
        if index >= n or raw[index] == "]":
            break
        obj, end = decoder.raw_decode(raw, index)
        rows.append(obj)
        index = end
    return rows


def _load_food(limit):
    path = _hub("OrSabbach/food-delivery-support-tickets", "data/dataset.parquet")
    return _tag(sample_parquet(path, limit), _v6_lang="en")


def _load_novora(limit):
    path = _hub("Novora/Tri-Class-Sentiment-Synthetic", "default/train/0000.parquet", PARQUET_REV)
    return _tag(sample_parquet(path, limit), _v6_lang="en")


def _load_tatoeba(entry, limit):
    groups = {}
    per = max(1, limit // len(TATOEBA_PAIRS))
    lid = "language-id" in (entry.get("category") or "")
    for pair in TATOEBA_PAIRS:
        path = _hub("Helsinki-NLP/tatoeba", "%s/train/0000.parquet" % pair, PARQUET_REV)
        left, right = pair.split("-")
        aligned = []
        for row in sample_parquet(path, max(per, 40)):
            translation = row.get("translation") or {}
            if not isinstance(translation, dict):
                translation = {left: row.get(left), right: row.get(right)}
            a, b = translation.get(left), translation.get(right)
            if a and b:
                aligned.append((str(a), str(b)))
        taken = []
        if lid:
            for a, b in aligned:
                taken.append({"translation": {left: a}, "_v6_lang": left})
                taken.append({"translation": {right: b}, "_v6_lang": right})
                if len(taken) >= per:
                    break
        else:
            for i, (a, b) in enumerate(aligned):
                taken.append({"sentence1": a, "sentence2": b, "relatedness_score": 1.0,
                              "language": left, "_v6_lang": left})
                other = aligned[(i + len(aligned) // 2) % len(aligned)][1]
                if other != b:
                    taken.append({"sentence1": a, "sentence2": other, "relatedness_score": 0.0,
                                  "language": left, "_v6_lang": left})
                if len(taken) >= per:
                    break
        groups[pair] = taken[:per]
    return _balanced_take(groups, limit)


def _spread(rows, limit):
    if len(rows) <= limit:
        return rows
    step = len(rows) / float(limit)
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(limit)]


def _load_hh_redteam(limit):
    path = _hub("Anthropic/hh-rlhf", "red-team-attempts/red_team_attempts.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for row in _spread(list(data), limit):
        if isinstance(row, dict) and str(row.get("transcript") or "").strip():
            copied = dict(row)
            copied["_v6_lang"] = "en"
            rows.append(copied)
    return rows


def _load_quality(limit):
    path = _hub("emozilla/quality", "default/train/0000.parquet", PARQUET_REV)
    rows = []
    for row in sample_parquet(path, limit, columns=["article", "question", "options", "answer"]):
        article = row.get("article") or ""
        question = row.get("question") or ""
        options = row.get("options") or []
        if not article or not question or not isinstance(options, list) or len(options) < 2:
            continue
        try:
            gold = int(row.get("answer"))
        except (TypeError, ValueError):
            continue
        if not 0 <= gold < len(options):
            continue
        # The mirror stores a 0-based index. Pass the option text so the adapter does not treat it as 1-based.
        rows.append({"context": article, "question": question, "options": options, "gold_label": options[gold],
                     "_v6_lang": "en", "_v6_config": "default"})
    return rows


def _load_fairytale(limit):
    path = _hub("WorkInTheDark/FairytaleQA", "plain_text/train/0000.parquet", PARQUET_REV)
    rows = []
    for row in sample_parquet(path, limit):
        answer = str(row.get("answer1") or "").strip()
        story = row.get("story_section") or ""
        question = row.get("question") or ""
        if not story or not question:
            continue
        rows.append({"context": story, "question": question,
                     "answers": {"text": [answer]} if answer else {"text": []},
                     "story_name": row.get("story_name") or "",
                     "_v6_lang": "en", "_v6_config": "plain_text"})
    return rows


NLUPP_SHA = {
    "banking-fold0.json": "bc9544ba197f5f073d07beab056466997ad986077d1ee46b434a251568034d0c",
    "banking-fold1.json": "ebe7cb30185e0884c9f6a20c8650636bb7ed0419d1a1a5608d63eb323f5df2ec",
    "banking-fold2.json": "9ce7cc6605cfa7dcb07be2b5f6ecb6cc075db5f344c998963a7a0be34727db99",
    "banking-fold3.json": "caff1b2b46503f85ae0ea2add66498857b341c87e87fa3dfbd3e8f12ee092fcd",
    "banking-fold4.json": "2c5121f7bcdca72c5a790b6a5ab1de6e0d89f5e9c6038bfa36507942f91aae5e",
    "banking-fold5.json": "fbc4252be25dfe5fd81b246eb51bcd9fccf509e43d29968832b99dcd0595449e",
    "banking-fold6.json": "fc2cbe872188462202f0fa4abdc4ee3bb38df130530500cb29f0c58c7f130cf1",
    "banking-fold7.json": "3029b8f9364ce6fc5ea202dc3b5d00b89a678cc90e49be0b865ace4cd944cea9",
    "banking-fold8.json": "7c76964657343991ed2d2c874db11b679b35cbe9fbf1ab05259667c02fd87335",
    "banking-fold9.json": "2d26f835b622d2c8bf83f4f29ebfc2cf5b1615f9cf7082e341f2cd77fd5d6a89",
    "banking-fold10.json": "a61de5477bd09ac0baa4152705be278f819de5ae60226c65c19d341b01330f6f",
    "banking-fold11.json": "cc01ca2a90f3da8c0e9d3762e59d1a84329bd357e5e9f17f490108ce5450b3a6",
    "banking-fold12.json": "d47c38523d56a9ffff0ffc1e4daa14339a6e42dfbf7b29ede030f1cc659dd612",
    "banking-fold13.json": "cb4c26ebecc7e68b8be0fafb4a691c9199fe6bb1368f0b549d29c6a1d6438fd1",
    "banking-fold14.json": "2b5350380c0d13e9cdef211a02b7406d97be2a766081bdab9beba12035948b48",
    "banking-fold15.json": "48ab8cf2a87cadfa3be845de6071f91cf22089b6618c471eceb749e6c719739e",
    "banking-fold16.json": "29e3bce3fa74908b85ee8c595b17419ae8161414cd4eb682fc41a79abe93c2bc",
    "banking-fold17.json": "4bf5dfe9d5114ef0300f970ed4c473eb7a9090900223a01af018e6a3e2a17bee",
    "banking-fold18.json": "f4f514ae92501dbe009ef639f229558d966eb63066f6273327558b51b9fcf69a",
    "banking-fold19.json": "c24f0fb42c4714d27f8639351c22df75e714e1e3e2451032d73452e4faa3942c",
    "hotels-fold0.json": "16724351a44b5c8b32c84a827dcd8530f4d248c6fac4d309d88fd21245fca448",
    "hotels-fold1.json": "123aeb21f3c73a47ce7df75ff28ba2cd318c9782f1c779e75e8e9322a36e68e7",
    "hotels-fold2.json": "517732b73e40ac30993ec9ca7e7531ca2e7abac829258391961836dd7ebdc2d6",
    "hotels-fold3.json": "555561c4484df605b2d180f762559d013bc864b304a6619a172d582eff15f859",
    "hotels-fold4.json": "a6c7459c085a09646a1c77de4c70a288076d2426a7f2f15c8838dc4e3d4ee9a6",
    "hotels-fold5.json": "489b91cb4132083fc975b85cce9774602e9aa7af46db2b7561884af17e281453",
    "hotels-fold6.json": "052e52bf41f87b24c5bbf246e871f6456b105230bd84a8f7f13ff65bedcc2cad",
    "hotels-fold7.json": "aa7869f8031cd3caa94675cc92052afd8526cbaa81b400de665fb95ba4b62c6c",
    "hotels-fold8.json": "1d20d1b1b0ff349f33649d219b2359828977fd9663ba758e4e9e67672a0954a2",
    "hotels-fold9.json": "0346a8658c7e47750d7a8589cd8b3b3e22f8beed3074b6488daa4213f47b5966",
    "hotels-fold10.json": "b5ed876c74026c5be1aa6ae3956134a89894e01baa410c1ded1e52e72750839e",
    "hotels-fold11.json": "582d1de5ac593226182b5e32c146750b6678a9d6198aef4c0cdfe9ad8b077161",
    "hotels-fold12.json": "f36349037466f2cc01430fef6b39ffadb0d4222396c7beae028cf07f333f897a",
    "hotels-fold13.json": "ee9773b91c2e37ea566b4f09687b6951a5c44d36e22292cead93eb221904e06a",
    "hotels-fold14.json": "722562aa47b087cbd432930064da2e080df0ada23000783a9493786125a22dad",
    "hotels-fold15.json": "f2d34e8c75d1e49f29b905de61d6690ae6f85f4e040f6f72882e161a09e57c24",
    "hotels-fold16.json": "dac95faf7ba946819714ec5e4e3014d0cf3e054634cc10f1ac9f7693a90280cb",
    "hotels-fold17.json": "3e0a95d9c8097d40d8919b5405938c0088a80926d262f03c9234fc93a20a5d33",
    "hotels-fold18.json": "1034017ae94f7eb50ecf2cfb4ef850c53e362506b69b02ee7c99f346eec16437",
    "hotels-fold19.json": "b0140eecf03dd9e6cb251c1a4b42edff2f4ea2c74079a9988777ee7181ebb3e7"
}


def _load_nlupp(limit):
    import ast
    root = RAW_ROOT / "nlupp"
    rows = []
    per = max(1, limit // len(NLUPP_SHA))
    commit = "452c61c86596b4865bf71327d689dc1822ee3a92"
    for name, digest in NLUPP_SHA.items():
        dest = root / name
        domain, fold = name.split("-fold")
        fold = fold.replace(".json", "")
        url = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/%s/nlupp/data/%s/fold%s.json"
               % (commit, domain, fold))
        _fetch_verified(url, dest, digest)
        data = json.loads(dest.read_text(encoding="utf-8"))
        for row in data:
            intents = row.get("intents")
            if isinstance(intents, str) and intents[:1] in "[(":
                try:
                    intents = ast.literal_eval(intents)
                except (ValueError, SyntaxError):
                    pass
            text = row.get("text") or ""
            if not str(text).strip():
                continue
            rows.append({"text": str(text).strip(), "intents": intents, "_v6_lang": "en", "_v6_config": domain})
            if len(rows) >= limit:
                return rows
    return rows


def _load_taskmaster(limit):
    files = {
        "self-dialogs.json": (
            "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/335324ca7afee220b6114e72d90f994e07a888c4/TM-1-2019/self-dialogs.json",
            "1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d",
        ),
        "woz-dialogs.json": (
            "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/335324ca7afee220b6114e72d90f994e07a888c4/TM-1-2019/woz-dialogs.json",
            "cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f",
        ),
    }
    root = RAW_ROOT / "taskmaster"
    root.mkdir(parents=True, exist_ok=True)
    dialogs = []
    for name, (url, digest) in files.items():
        dest = root / name
        _fetch_verified(url, dest, digest)
        dialogs.extend(json.loads(dest.read_text(encoding="utf-8")))
    rows = []
    for dialog in dialogs:
        instruction = str(dialog.get("instruction_id") or "")
        domain = instruction.split("-")[0] if instruction else ""
        for turn in dialog.get("utterances") or []:
            if str(turn.get("speaker") or "").upper() not in ("USER", "0"):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            rows.append({"text": text, "instruction_id": instruction, "domain": domain,
                         "_v6_lang": "en", "_v6_config": "TM-1"})
            if len(rows) >= limit:
                return rows
    return rows

_DISPATCH = {
    "ai4bharat/IndicSentiment": lambda entry, limit: _load_indic(limit),
    "Novora/Tri-Class-Sentiment-Synthetic": lambda entry, limit: _load_novora(limit),
    "github:asappresearch/abcd": lambda entry, limit: _load_abcd(limit),
    "PolyAI/minds14": lambda entry, limit: _load_minds14(limit),
    "OrSabbach/food-delivery-support-tickets": lambda entry, limit: _load_food(limit),
    "github:wwbp/empathic_reactions": lambda entry, limit: _load_empathic(limit),
    "Eurolingua/truthfulqax": lambda entry, limit: _load_truthful(limit),
    "zenodo:3609356 (ClaimBuster)": lambda entry, limit: _load_claimbuster(limit),
    "urchade/synthetic-pii-ner-mistral-v1": lambda entry, limit: _load_pii_json(limit),
    "Helsinki-NLP/tatoeba": _load_tatoeba,
    "boun-tabi/nli_tr": lambda entry, limit: _load_nli_tr(limit),
    "community-datasets/tapaco": lambda entry, limit: _load_tapaco(limit),
    "dell-research-harvard/headlines-semantic-similarity": lambda entry, limit: _load_headlines(limit),
    "Avature/Job-Title-Similarity": lambda entry, limit: _load_job_titles(limit),
    "github:bvidgen/Dynamically-Generated-Hate-Speech-Dataset (v0.2.3.csv; NOT tasksource/dynahate mirror tagged gpl)": lambda entry, limit: _load_dynahate(limit),
    "jagoldz/gahd (filter via GitHub jagol/gahd gahd_disaggregated.csv)": lambda entry, limit: _load_gahd(limit),
    "theatticusproject/cuad-qa": lambda entry, limit: _load_cuad(limit),
    "ddrg/super_eurlex": lambda entry, limit: _load_eurlex(limit),
    "pfb30/multi_woz_v22 (github budzianowski/multiwoz)": lambda entry, limit: _load_multiwoz(limit),
    "ConvLab/crosswoz (github thu-coai/CrossWOZ)": lambda entry, limit: _load_crosswoz(limit),
    "github:HLTCHKUST/BiToD (mirror DeepPavlov/BiToD)": lambda entry, limit: _load_bitod(limit),
    "github:PolyAI-LDN/task-specific-datasets/nlupp": lambda entry, limit: _load_nlupp(limit),
    "google-research-datasets/taskmaster1|2|3 (github Taskmaster TM-1..TM-4)": lambda entry, limit: _load_taskmaster(limit),
    "Anthropic/hh-rlhf": lambda entry, limit: _load_hh_redteam(limit),
    "nyu-mll/quality (github; mirror emozilla/quality)": lambda entry, limit: _load_quality(limit),
    "WorkInTheDark/FairytaleQA": lambda entry, limit: _load_fairytale(limit),
}


def _simple_config(entry):
    config = entry.get("config", "default")
    if config in ("", "default") or any(c in str(config) for c in "<>*{}|") or str(config).startswith(("data/", "data_dir=", "per ", "language ")):
        return None
    if "," in str(config) or " + " in str(config):
        return None
    return config


def _split_candidates(wanted, available):
    wanted = (wanted or "").lower()
    choices = []
    for name in ("train", "validation", "test", "all", "intents"):
        if name in wanted and name in available:
            choices.append(name)
    if not choices:
        choices = ["train"] if "train" in available else list(available[:1])
    return choices


ALIASES = {
    "nyu-mll/quality (github; mirror emozilla/quality)": "emozilla/quality",
}
CONFIGS = {
    "llm-for-emotion/Cultural-Emo": ["ara", "deu", "eng", "hin", "spn"],
    "MoritzLaurer/synthetic_zeroshot_mixtral_v0.1": [
        "mixtral_written_texts_for_tasks", "mixtral_written_texts_for_tasks_v2",
        "mixtral_written_texts_for_tasks_v3", "mixtral_written_texts_for_tasks_v4",
    ],
    "bench-llms/or-bench": ["or-bench-80k", "or-bench-hard-1k", "or-bench-toxic"],
    "NortheasternUniversity/big_patent": list("abcdefghy"),  # the CPC sections the dataset has
    "amyrmahdy/decima-synthetic-decisions": ["short", "long", "relabel"],
}


# Parquet shards read per config/split; bounds the download for multi-GB datasets. The CI audit sets 1.
MAX_SHARDS = int(os.environ.get("STATIM_V6_MAX_SHARDS", "4"))


def class_label_names(path):
    """{column: [names]} for ClassLabel columns, from the datasets metadata in a parquet schema."""
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata or {}
    try:
        features = json.loads(meta[b"huggingface"])["info"]["features"]
    except (KeyError, ValueError, TypeError):
        return {}
    return {col: list(feat["names"]) for col, feat in features.items()
            if isinstance(feat, dict) and feat.get("_type") == "ClassLabel" and feat.get("names")}


def decode_class_labels(rows, names):
    """ClassLabel ints -> their names. An adapter would otherwise offer "0", "1", "2" as options."""
    for row in rows:
        for col, labels in names.items():
            value = row.get(col)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(labels):
                row[col] = labels[value]
    return rows


_PARQUET_LISTING = {}


def _parquet_listing(dataset_id):
    if dataset_id not in _PARQUET_LISTING:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(dataset_id, repo_type="dataset", revision=PARQUET_REV)
        _PARQUET_LISTING[dataset_id] = [f for f in files if f.endswith(".parquet")]
    return _PARQUET_LISTING[dataset_id]


def _parquet_config(dataset_id, config):
    dirs = sorted({f.split("/")[0] for f in _parquet_listing(dataset_id)})
    if config is not None:
        return config if config in dirs else None
    if "default" in dirs:
        return "default"
    return dirs[0] if len(dirs) == 1 else None


def _parquet_split_rows(dataset_id, config, split, limit):
    """``limit`` rows spread evenly over the split (at most MAX_SHARDS shards, evenly spaced), so a
    split sorted by label still yields every label. ClassLabel ints are decoded to names."""
    shards = sorted(f for f in _parquet_listing(dataset_id) if f.startswith("%s/%s/" % (config, split)))
    if len(shards) > MAX_SHARDS:
        step = len(shards) / float(MAX_SHARDS)
        shards = [shards[int(i * step)] for i in range(MAX_SHARDS)]
    per = max(1, limit // max(1, len(shards)))
    rows = []
    for shard in shards:
        path = _hub(dataset_id, shard, PARQUET_REV)
        rows.extend(decode_class_labels(sample_parquet(path, per), class_label_names(path)))
    return rows[:limit]


def _load_hf_parquet(entry, limit, dataset_id, configs):
    """Rows from the auto-converted parquet revision. Configs without a parquet conversion are skipped
    (as streaming skips configs that fail); returns None when none has one."""
    names = [(config, _parquet_config(dataset_id, config)) for config in configs]
    names = [(config, name) for config, name in names if name is not None]
    if not names:
        return None
    rows = []
    per_config = max(1, limit // len(names))
    for config, name in names:
        available = sorted({f.split("/")[1] for f in _parquet_listing(dataset_id) if f.startswith(name + "/")})
        splits = _split_candidates(entry.get("split", "train"), available)
        for split in splits:
            part = _parquet_split_rows(dataset_id, name, split, per_config)
            for record in part:
                record.pop("audio", None)
                record.setdefault("_v6_config", config or "default")
            rows.extend(part)
    return rows[:limit]


def _load_hf_streaming(entry, limit):
    """Ordinary Hub datasets that already ship data files, not a loading script. The parquet
    revision is read first (rows spread over the split); streaming the head of the split is the
    fallback, and it decodes ClassLabels from the dataset features."""
    from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset

    original = entry["id"]
    dataset_id = ALIASES.get(original, original)
    if original in CONFIGS:
        configs = CONFIGS[original]
    else:
        config = _simple_config(entry)
        if config is None and any(x in entry.get("config", "") for x in ("48 configs", "<lang>", "eng +")):
            configs = get_dataset_config_names(dataset_id)
        else:
            configs = [config]
    try:
        rows = _load_hf_parquet(entry, limit, dataset_id, configs)
    except Exception:  # listing or download failed: fall back to streaming
        rows = None
    if rows:
        return rows, []
    rows, failures = [], []
    per_config = max(1, limit // max(1, len(configs)))
    for config in configs:
        if len(rows) >= limit:
            break
        try:
            available = get_dataset_split_names(dataset_id, config)
            splits = _split_candidates(entry.get("split", "train"), available)
            for split in splits:
                try:
                    ds = load_dataset(dataset_id, config, split=split, streaming=True)
                except Exception:
                    ds = load_dataset(dataset_id, config, split=split)
                columns = getattr(ds, "column_names", None) or []
                if "audio" in columns:
                    ds = ds.cast_column("audio", Audio(decode=False))
                    ds = ds.remove_columns(["audio"])
                take = min(per_config, limit - len(rows))
                features = getattr(ds, "features", None) or {}
                names = {k: list(v.names) for k, v in features.items() if type(v).__name__ == "ClassLabel"}
                for i, row in enumerate(ds):
                    if i >= take:
                        break
                    record = {k: v for k, v in dict(row).items() if k != "audio"}
                    decode_class_labels([record], names)
                    record.setdefault("_v6_config", config or "default")
                    rows.append(record)
        except Exception as exc:
            failures.append("%s: %s" % (config or "default", str(exc).splitlines()[0]))
    if not rows:
        raise RuntimeError("; ".join(failures) or "no rows returned")
    return rows, failures


# Integer labels that are plain ints (not ClassLabel), named on the dataset card.
CARD_LABELS = {
    # "0 表示负面，1 表示正面"
    "YiMeng-SYSU/chinese-logic-sentiment-dataset": {"label": ["negative", "positive"]},
    # card table: 0 very_negative ... 4 very_positive
    "tanaos/synthetic-sentiment-analysis-dataset-v1": {
        "labels": ["very negative", "negative", "neutral", "positive", "very positive"]},
    # card table: 0 joy, 1 anger, 2 fear, 3 sadness, 4 surprise, 5 disgust, 6 excitement, 7 neutral
    "tanaos/synthetic-emotion-detection-dataset-v1": {
        "labels": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "excitement", "neutral"]},
}


def _explode_casino(rows):
    """One row per annotated utterance: annotations = [[utterance, "strategy,strategy"], ...]. The
    first strategy is the label; the whole dialogue is not the decision text."""
    out = []
    for row in rows:
        annotations = row.get("annotations")
        if isinstance(annotations, str):
            try:
                import ast
                annotations = ast.literal_eval(annotations)
            except (ValueError, SyntaxError):
                continue
        for pair in annotations or []:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2 and str(pair[0]).strip() and str(pair[1]).strip():
                out.append({"utterance": str(pair[0]).strip(), "annotations": str(pair[1]).split(",")[0].strip(),
                            "_v6_lang": "en", "_v6_config": row.get("_v6_config", "default")})
    return out


ROW_TRANSFORMS = {"kchawla123/casino": _explode_casino}


def _postprocess(sid, rows):
    for col, names in CARD_LABELS.get(sid, {}).items():
        for row in rows:
            value = row.get(col)
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                value = int(value)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(names):
                row[col] = names[value]
    if sid in ROW_TRANSFORMS:
        rows = ROW_TRANSFORMS[sid](rows)
    return rows


def load_rows(entry, limit):
    """Load at most ``limit`` rows. Returns ``(rows, warnings)``."""
    sid = entry["id"]
    if sid in _DISPATCH:
        rows = _DISPATCH[sid](entry, limit)
        if not rows:
            raise RuntimeError("loader returned no rows")
        return _postprocess(sid, rows), []
    if sid.startswith(("github:", "zenodo:")) or "|" in sid:
        raise RuntimeError("non-Hugging-Face source needs a pinned raw-file loader")
    rows, warnings = _load_hf_streaming(entry, limit)
    return _postprocess(sid, rows), warnings
