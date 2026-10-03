#!/usr/bin/env python3
"""Generate the grounded urgency + multilingual NLI pilot with two local models."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import canonical_id, language_ok, one_hot, sha256_text, word_count  # noqa: E402
from corpora import CORPORA, PILOT_LANGUAGES, iter_corpus, validate_cache  # noqa: E402
from generate import digest_from_show  # noqa: E402
from grounded_prompts import (NLI_LABELS, PROMPT_VERSION, URGENCY_LABELS, generation_request,
                              prompts_digest, verification_request)  # noqa: E402
from leakage import DeferredGuard, LeakageGuard, grams  # noqa: E402
from ollama_http import OllamaError, OllamaHTTP  # noqa: E402
from openai_http import OpenAIHTTP  # noqa: E402
from verify import parse_model_json  # noqa: E402

GENERATOR = {"model": "Qwen/Qwen3-8B", "role": "text"}
VERIFIER = {"model": "microsoft/Phi-4-mini-instruct", "role": "labels"}
OLLAMA_TO_HF = {"qwen3:8b": GENERATOR["model"], "phi4-mini": VERIFIER["model"]}
# models a vLLM (OpenAI-compatible) backend may serve; each id is approved in
# tools/finetune/sources/policy.json and recorded with its exact revision
HF_ALLOWLIST = {"Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-8B", "microsoft/phi-4",
                "microsoft/Phi-4-mini-instruct"}
MODEL_ALLOWLIST = {
    "qwen3:8b": {"families": {"qwen3"}, "parameter": re.compile(r"\b8(?:\.\d+)?\s*b\b", re.I)},
    "phi4-mini": {"families": {"phi3", "phi4"},
                  "parameter": re.compile(r"\b(?:3\.8|3\.75)\s*b\b", re.I)},
}

_WORDS = {
    "en": ("Premise", "Hypothesis", "How are the premise and hypothesis related?", "How urgent is this request?"),
    "de": ("Prämisse", "Hypothese", "Wie hängen Prämisse und Hypothese zusammen?", "Wie dringend ist diese Anfrage?"),
    "fr": ("Prémisse", "Hypothèse", "Quel est le lien entre la prémisse et l’hypothèse ?", "Quel est le degré d’urgence de cette demande ?"),
    "es": ("Premisa", "Hipótesis", "¿Qué relación hay entre la premisa y la hipótesis?", "¿Qué grado de urgencia tiene esta solicitud?"),
    "it": ("Premessa", "Ipotesi", "Qual è il rapporto tra premessa e ipotesi?", "Quanto è urgente questa richiesta?"),
    "pt": ("Premissa", "Hipótese", "Qual é a relação entre a premissa e a hipótese?", "Qual é a urgência deste pedido?"),
    "nl": ("Premisse", "Hypothese", "Wat is het verband tussen de premisse en de hypothese?", "Hoe dringend is dit verzoek?"),
    "pl": ("Przesłanka", "Hipoteza", "Jaki jest związek między przesłanką a hipotezą?", "Jak pilna jest ta prośba?"),
}
_NLI_DESC = {
    "en": ("follows from the premise", "conflicts with the premise", "is neither supported nor contradicted"),
    "de": ("folgt aus der Prämisse", "widerspricht der Prämisse", "wird weder gestützt noch widerlegt"),
    "fr": ("découle de la prémisse", "contredit la prémisse", "n’est ni étayée ni contredite"),
    "es": ("se deduce de la premisa", "contradice la premisa", "no está respaldada ni contradicha"),
    "it": ("deriva dalla premessa", "contraddice la premessa", "non è né sostenuta né contraddetta"),
    "pt": ("decorre da premissa", "contradiz a premissa", "não é apoiada nem contradita"),
    "nl": ("volgt uit de premisse", "spreekt de premisse tegen", "wordt niet ondersteund of tegengesproken"),
    "pl": ("wynika z przesłanki", "jest sprzeczna z przesłanką", "nie jest ani potwierdzona, ani obalona"),
}
_LOCAL_LABELS = {
    "urgency": {
        "en": ("not urgent", "soon", "critical"), "de": ("nicht dringend", "bald", "kritisch"),
        "fr": ("pas urgent", "bientôt", "critique"), "es": ("no urgente", "pronto", "crítico"),
        "it": ("non urgente", "presto", "critico"), "pt": ("não urgente", "breve", "crítico"),
        "nl": ("niet dringend", "binnenkort", "kritiek"), "pl": ("niepilne", "wkrótce", "krytyczne"),
    },
    "nli": {
        "en": NLI_LABELS, "de": ("folgerung", "widerspruch", "neutral"),
        "fr": ("implication", "contradiction", "neutre"),
        "es": ("implicación", "contradicción", "neutral"),
        "it": ("implicazione", "contraddizione", "neutrale"),
        "pt": ("implicação", "contradição", "neutro"),
        "nl": ("gevolgtrekking", "tegenspraak", "neutraal"),
        "pl": ("wynikanie", "sprzeczność", "neutralne"),
    },
}


def validate_model_meta(tag, meta):
    rule = MODEL_ALLOWLIST[tag]
    family = str(meta.get("family") or "").casefold()
    parameter = str(meta.get("parameter_size") or "")
    digest = str(meta.get("digest") or "")
    if family not in rule["families"] or not rule["parameter"].search(parameter):
        raise SystemExit("Ollama tag %s does not match allowlist: family=%r parameter_size=%r" %
                         (tag, family, parameter))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise SystemExit("Ollama tag %s has no valid model digest" % tag)
    if not meta.get("digest_source"):
        raise SystemExit("Ollama tag %s has no digest source" % tag)
    return meta


def validate_hf_meta(meta):
    hf_id, revision = meta.get("hf_id"), str(meta.get("revision") or "")
    if hf_id not in HF_ALLOWLIST:
        raise SystemExit("model %r is not on the synthesis allowlist" % hf_id)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("model %s needs its exact 40-hex revision, got %r" % (hf_id, revision))
    return meta


def _roles(model_meta):
    """[generator, verifier] as policy model ids; model_meta is ordered generator first."""
    out = []
    for (tag, meta), role in zip(model_meta.items(), ("text", "labels")):
        out.append({"model": (meta or {}).get("hf_id") or OLLAMA_TO_HF.get(tag, tag), "role": role})
    return out


def _chat(client, request, temperature, seed):
    messages, schema, predict = request
    return client.chat(messages, schema, temperature=temperature, num_predict=predict, seed=seed)


def _request_json(client, request, temperature, seed, metrics, prefix):
    for retry in range(2):
        mixed_seed = (seed + retry * 0x9E3779B1) % (2 ** 31 - 1)
        try:
            response = _chat(client, request, temperature, mixed_seed)
            return parse_model_json(response["content"])
        except (OllamaError, KeyError, ValueError, json.JSONDecodeError) as exc:
            metrics[prefix + "_error:" + type(exc).__name__] += 1
            if retry == 0:
                metrics[prefix + "_retry"] += 1
    return None


def _quota(total, languages, labels):
    cells = [(lang, label) for lang in languages for label in labels]
    return {cell: total // len(cells) + (i < total % len(cells)) for i, cell in enumerate(cells)}


def _source_spec(source_id):
    if source_id.startswith("FiscalNote/"):
        return "billsum"
    if source_id.startswith("launch/"):
        return "gov_report"
    return "eur_lex_sum"


def _provenance(ident, task, lang, label, passage, source_id, revision, model_meta, verify):
    return {
        "id": ident,
        "seed": {"source_id": source_id, "revision": revision,
                 "passage_sha256": sha256_text(passage), "lang": lang, "target": label},
        "seed_passage": passage,
        "generator": _roles(model_meta),
        "ollama_models": {tag: dict(meta or {}) for tag, meta in model_meta.items()},
        "ollama_model_digests": {tag: (meta or {}).get("digest") for tag, meta in model_meta.items()},
        "prompt_version": PROMPT_VERSION,
        "capability": task,
        "verify": verify,
    }


def _base_item(task, lang, label, passage, generated):
    if task == "urgency":
        return {"state": generated, "q": {"type": "score", "instructions": _WORDS[lang][3],
                "criteria": list(URGENCY_LABELS)}, "target": one_hot(3, URGENCY_LABELS.index(label))}
    state = "%s: %s\n\n%s: %s" % (_WORDS[lang][0], passage, _WORDS[lang][1], generated)
    criteria = {name: _NLI_DESC[lang][i] for i, name in enumerate(NLI_LABELS)}
    return {"state": state, "q": {"type": "choice", "instructions": _WORDS[lang][2],
            "criteria": criteria}, "target": one_hot(3, NLI_LABELS.index(label))}


def _contains_label(text, task, lang):
    labels = set(URGENCY_LABELS if task == "urgency" else NLI_LABELS)
    labels.update(_LOCAL_LABELS[task][lang])
    words = set(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE))
    label_words = {word for label in labels
                   for word in re.findall(r"[^\W_]+", label.casefold(), re.UNICODE)}
    return bool(words & label_words)


def _generate_candidates(task, passage_row, target, generator, temperature):
    """Phase 1 (generator only): checked candidate texts for one seed passage."""
    passage, lang, source_id, revision = passage_row
    seed = int(hashlib.sha256((task + source_id + passage + str(target)).encode()).hexdigest()[:8], 16)
    metrics = collections.Counter()
    parsed = _request_json(generator, generation_request(task, passage, lang, target),
                           temperature, seed, metrics, "generate")
    if parsed is None:
        return [], dict(metrics)
    raw = []
    if task == "urgency":
        if parsed.get("label") != target or not str(parsed.get("request") or "").strip():
            return [], {**dict(metrics), "bad_generation": 1}
        raw.append((target, str(parsed["request"]).strip()))
    else:
        rows = parsed.get("items") if isinstance(parsed, dict) else None
        if not isinstance(rows, list) or len(rows) != 3:
            return [], {**dict(metrics), "bad_generation": 1}
        labels = [row.get("label") for row in rows if isinstance(row, dict)]
        if sorted(labels) != sorted(NLI_LABELS):
            return [], {**dict(metrics), "bad_generation": 1}
        raw.extend((row["label"], str(row.get("hypothesis") or "").strip()) for row in rows)
    candidates = []
    for label, text in raw:
        limits = (8, 120) if task == "urgency" else (3, 80)
        if not limits[0] <= word_count(text) <= limits[1] or not language_ok(text, lang):
            metrics["bad_text"] += 1
            continue
        if _contains_label(text, task, lang):
            metrics["label_leak"] += 1
            continue
        if grams(text) & grams(passage):
            metrics["seed_copy"] += 1
            continue
        candidates.append({"task": task, "label": label, "text": text, "passage_row": passage_row,
                           "seed": seed})
    return candidates, dict(metrics)


def _verify_candidate(candidate, verifier, model_meta, guard):
    """Phase 2 (verifier only): a blind check, then the leakage guard."""
    task, label, text, seed = candidate["task"], candidate["label"], candidate["text"], candidate["seed"]
    passage, lang, source_id, revision = candidate["passage_row"]
    metrics = collections.Counter()
    blind = {"request": text} if task == "urgency" else {"premise": passage, "hypothesis": text}
    metrics["verify_called"] += 1
    answer_obj = _request_json(verifier, verification_request(task, blind, lang), 0,
                               seed ^ 0x5A5A5A5A, metrics, "verify")
    if answer_obj is None:
        metrics["verify_error"] += 1
        return None, dict(metrics)
    answer = answer_obj.get("answer")
    if answer != label:
        metrics["verify_disagree"] += 1
        return None, dict(metrics)
    metrics["verify_agree"] += 1
    base = _base_item(task, lang, label, passage, text)
    passage_hash = sha256_text(passage)
    ident = canonical_id({"item": base, "source_id": source_id,
                          "passage_sha256": passage_hash, "target": label})
    item = dict(base, id=ident, src=source_id, passage_sha256=passage_hash)
    if guard.overlap(item):
        metrics["leakage_reject"] += 1
        return None, dict(metrics)
    verify = {"answer": answer, "agree": True}
    provenance = _provenance(ident, task, lang, label, passage, source_id, revision,
                             model_meta, verify)
    return {"id": ident, "item": item, "provenance": provenance}, dict(metrics)


def _generate_job(task, passage_row, target, generator, verifier, temperature, model_meta, guard):
    candidates, metrics = _generate_candidates(task, passage_row, target, generator, temperature)
    metrics = collections.Counter(metrics)
    accepted = []
    for candidate in candidates:
        row, more = _verify_candidate(candidate, verifier, model_meta, guard)
        metrics.update(more)
        if row is not None:
            accepted.append(row)
    return accepted, dict(metrics)


def _passages(cache, per_language):
    out = {lang: [] for lang in PILOT_LANGUAGES}
    for lang in PILOT_LANGUAGES:
        names = ("billsum", "gov_report") if lang == "en" else ("eur_lex_sum",)
        per_source = (per_language + len(names) - 1) // len(names)
        for name in names:
            for row in iter_corpus(name, cache, languages=(lang,)):
                out[lang].append(row)
                if sum(1 for value in out[lang] if _source_spec(value[2]) == name) >= per_source:
                    break
            if len(out[lang]) >= per_language:
                break
        out[lang] = out[lang][:per_language]
    missing = [lang for lang, rows in out.items() if not rows]
    if missing:
        raise RuntimeError("no grounding passages for: %s" % ", ".join(missing))
    return out


def _run_task(task, total, passages, generator, verifier, concurrency, temperature, model_meta,
              guard, max_jobs, initial=None, on_accept=None, phased_batch=0):
    """With ``phased_batch`` > 0, each batch of that many seed jobs is generated first and verified
    afterwards, so a GPU too small for both models (8 GB) swaps models twice per batch instead of
    on every request."""
    labels = URGENCY_LABELS if task == "urgency" else NLI_LABELS
    quota = _quota(total, PILOT_LANGUAGES, labels)
    kept = list(initial or [])
    counts = collections.Counter((row["provenance"]["seed"]["lang"],
                                  row["provenance"]["seed"]["target"]) for row in kept)
    positions = collections.Counter()
    for lang in PILOT_LANGUAGES:
        accepted = sum(counts[(lang, label)] for label in labels)
        positions[lang] = accepted if task == "urgency" else (accepted + len(labels) - 1) // len(labels)
    seen_ids = {row["id"] for row in kept}
    reasons, jobs = collections.Counter(), 0
    started, start_kept = time.monotonic(), len(kept)
    _progress(task, len(kept), total, reasons, started, start_kept)
    while len(kept) < total and jobs < max_jobs:
        batch = []
        for lang in PILOT_LANGUAGES:
            need = [label for label in labels if counts[(lang, label)] < quota[(lang, label)]]
            if not need:
                continue
            passage = passages[lang][positions[lang] % len(passages[lang])]
            positions[lang] += 1
            target = need[0] if task == "urgency" else None
            batch.append((task, passage, target, generator, verifier, temperature, model_meta, guard))
            if len(batch) >= (phased_batch or concurrency * 2):
                break
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            if phased_batch:
                generated = list(pool.map(
                    lambda args: _generate_candidates(args[0], args[1], args[2], args[3], args[5]), batch))
                flat = [c for candidates, _ in generated for c in candidates]
                verified = list(pool.map(
                    lambda c: _verify_candidate(c, verifier, model_meta, guard), flat))
                gen_metrics = collections.Counter()
                for _, m in generated:
                    gen_metrics.update(m)
                results = [([row for row, _ in verified if row is not None], dict(gen_metrics))]
                for _, m in verified:
                    results.append(([], m))
            else:
                results = list(pool.map(lambda args: _generate_job(*args), batch))
        jobs += len(batch)
        for candidates, job_metrics in results:
            reasons.update(job_metrics)
            for row in candidates:
                cell = (row["provenance"]["seed"]["lang"], row["provenance"]["seed"]["target"])
                if counts[cell] >= quota[cell] or len(kept) >= total:
                    continue
                if row["id"] in seen_ids:
                    reasons["duplicate"] += 1
                    continue
                seen_ids.add(row["id"])
                counts[cell] += 1
                kept.append(row)
                if on_accept:
                    on_accept(row)
        _progress(task, len(kept), total, reasons, started, start_kept)
    if len(kept) != total:
        raise RuntimeError("%s kept %d/%d after %d jobs; reasons=%s" %
                           (task, len(kept), total, jobs, dict(reasons)))
    called, agreed = reasons["verify_called"], reasons["verify_agree"]
    return kept, {"jobs": jobs, "metrics": dict(reasons),
                  "verify_acceptance_rate": (agreed / called) if called else None,
                  "counts": {"%s/%s" % cell: count for cell, count in sorted(counts.items())}}


def _progress(task, kept, total, reasons, started, start_kept, width=30, stream=sys.stdout):
    """One live line per batch: bar, kept/target, verifier agreement, kept items per hour, ETA."""
    done = kept / total if total else 1.0
    bar = "#" * int(width * done) + "-" * (width - int(width * done))
    called, agreed = reasons["verify_called"], reasons["verify_agree"]
    agree = ("%3.0f%%" % (100.0 * agreed / called)) if called else "  - "
    elapsed = time.monotonic() - started
    new = kept - start_kept
    rate = new * 3600.0 / elapsed if elapsed > 0 and new else 0.0
    eta = ("%d min" % round((total - kept) / rate * 60)) if rate else "-"
    print("[%-7s] |%s| %4d/%-4d %3.0f%%  verifier agrees %s  %6.0f kept/h  ETA %s" % (
        task, bar, kept, total, 100 * done, agree, rate, eta), file=stream, flush=True)


def _append_gzip(path, row):
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resume_rows(out_dir, task):
    item_final = out_dir / (task + ".jsonl.gz")
    prov_final = out_dir / (task + ".provenance.jsonl.gz")
    item_part = out_dir / (task + ".jsonl.gz.part")
    prov_part = out_dir / (task + ".provenance.jsonl.gz.part")
    if item_final.exists() or prov_final.exists():
        if not (item_final.exists() and prov_final.exists()):
            raise RuntimeError("incomplete finalized %s output" % task)
        items, provs = _read_gzip(item_final), _read_gzip(prov_final)
        return _join_rows(task, items, provs), item_final, prov_final, True
    if item_part.exists() or prov_part.exists():
        if not (item_part.exists() and prov_part.exists()):
            raise RuntimeError("incomplete %s part files" % task)
        return _join_rows(task, _read_gzip(item_part), _read_gzip(prov_part)), item_part, prov_part, False
    return [], item_part, prov_part, False


def _join_rows(task, items, provs):
    by_id = {row.get("id"): row for row in provs}
    ids = [row.get("id") for row in items]
    if None in ids or None in by_id or len(ids) != len(set(ids)) or len(by_id) != len(provs):
        raise RuntimeError("%s resume files have missing/duplicate ids" % task)
    if set(ids) != set(by_id):
        raise RuntimeError("%s resume item/provenance ids differ" % task)
    return [{"id": row["id"], "item": row, "provenance": by_id[row["id"]]} for row in items]


def write_samples(path, by_task, n, seed):
    rng = random.Random(seed)
    lines = ["# Grounded synthetic pilot: review sample", ""]
    for task, rows in by_task.items():
        lines.extend(("## %s" % task, ""))
        for i, row in enumerate(rng.sample(rows, min(n, len(rows))), 1):
            item, prov = row["item"], row["provenance"]
            lines.extend(("### %s %d" % (task, i), "", "- id: `%s`" % row["id"],
                          "- source: `%s`" % prov["seed"]["source_id"],
                          "- language / target: `%s` / `%s`" %
                          (prov["seed"]["lang"], prov["seed"]["target"]), "",
                          "**Grounding passage**", "", prov["seed_passage"], "",
                          "**Training item**", "", item["state"], "",
                          "Question: " + item["q"]["instructions"], ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pilot(out_dir, cache, generator, verifier, guard, per_capability=200, concurrency=4,
              seed=20261003, temperature=0.4, max_jobs_factor=12, passages=None,
              generator_meta=None, verifier_meta=None, resume=False, tasks=("urgency", "nli"),
              phased_batch=0):
    out_dir = Path(out_dir)
    if out_dir.exists() and not resume:
        raise FileExistsError("refusing existing output directory without --resume: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_meta = {(generator_meta or {}).get("hf_id", "qwen3:8b"): dict(generator_meta or {}),
                  (verifier_meta or {}).get("hf_id", "phi4-mini"): dict(verifier_meta or {})}
    passages = passages or _passages(cache, max(20, per_capability * max_jobs_factor // 8))
    started = time.monotonic()
    by_task, stats = {}, {}
    for task in tasks:
        initial, item_path, prov_path, finalized = _resume_rows(out_dir, task) if resume else (
            [], out_dir / (task + ".jsonl.gz.part"),
            out_dir / (task + ".provenance.jsonl.gz.part"), False)
        if len(initial) > per_capability:
            raise RuntimeError("%s resume output exceeds requested count" % task)
        if finalized and len(initial) != per_capability:
            raise RuntimeError("finalized %s output has wrong count" % task)

        def accept(row):
            _append_gzip(item_path, row["item"])
            _append_gzip(prov_path, row["provenance"])

        rows, task_stats = _run_task(
            task, per_capability, passages, generator, verifier, concurrency, temperature,
            model_meta, guard, per_capability * max_jobs_factor, initial=initial,
            on_accept=None if finalized else accept, phased_batch=phased_batch)
        by_task[task], stats[task] = rows, task_stats
        if not finalized:
            item_path.replace(out_dir / (task + ".jsonl.gz"))
            prov_path.replace(out_dir / (task + ".provenance.jsonl.gz"))
    return _finish(out_dir, by_task, stats, model_meta, guard, per_capability, concurrency, tasks,
                   seed, started, {"phased_batch": phased_batch})


def _finish(out_dir, by_task, stats, model_meta, guard, per_capability, concurrency, tasks, seed,
            started, extra):
    write_samples(out_dir / "samples.md", by_task, 30, seed)
    leakage_meta = dict(getattr(guard, "metadata", {}) or {})
    leakage_meta.update({"word_shingle": 8, "char_shingle": 20, "contain_chars": 40,
                         "heldout_texts_indexed": getattr(guard, "texts", None),
                         "checked": not getattr(guard, "deferred", False)})
    manifest = {
        "kind": "grounded-synthetic-pilot",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items_per_capability": per_capability, "languages": list(PILOT_LANGUAGES),
        "models": _roles(model_meta), "ollama": model_meta,
        "prompt_version": PROMPT_VERSION, "prompts_sha256": prompts_digest(),
        "corpora": CORPORA, "leakage": leakage_meta, "concurrency": concurrency,
        "tasks": list(tasks), **extra,
        "seed": seed, "stats": stats, "elapsed_s": time.monotonic() - started,
    }
    manifest_part = out_dir / "manifest.json.part"
    manifest_part.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_part.replace(out_dir / "manifest.json")
    return manifest


# --------------------------------------------------------------------------- two-stage runs
# For a GPU that serves one large model at a time (vLLM on an A100): stage "generate" writes
# checked candidates to a file with the generator loaded; stage "verify" then loads the verifier and
# applies the blind check, the leakage guard and the quotas. Both stages resume.

def _read_candidates(path):
    """Candidates appended one gzip member per row; a crash can cut the last member. Keep every
    complete row, and rewrite the file without the broken tail so appending can continue."""
    import zlib
    rows, broken = [], False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (EOFError, OSError, zlib.error, json.JSONDecodeError, UnicodeDecodeError):
        broken = True
    if broken:
        part = Path(str(path) + ".part")
        with gzip.open(part, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        part.replace(path)
        print("candidates: dropped a truncated tail, kept %d complete rows" % len(rows), flush=True)
    return rows


def _cell_targets(tasks, per_capability, oversample):
    targets = {}
    for task in tasks:
        labels = URGENCY_LABELS if task == "urgency" else NLI_LABELS
        for cell, n in _quota(per_capability, PILOT_LANGUAGES, labels).items():
            targets[(task,) + cell] = math.ceil(n * oversample)
    return targets


def _stage_line(stage, task, done, total, started, start_done, extra=""):
    frac = done / total if total else 1.0
    bar = "#" * int(30 * frac) + "-" * (30 - int(30 * frac))
    elapsed = time.monotonic() - started
    rate = (done - start_done) * 3600.0 / elapsed if elapsed > 0 and done > start_done else 0.0
    eta = ("%d min" % round((total - done) / rate * 60)) if rate else "-"
    print("[%-8s %-7s] |%s| %6d/%-6d %3.0f%%  %7.0f/h  ETA %s%s" % (
        stage, task, bar, done, total, 100 * frac, rate, eta, extra), flush=True)


def stage_generate(candidates, cache, generator, tasks, per_capability, concurrency, temperature,
                   oversample=2.5, max_jobs_factor=12, passages=None):
    """Append generated candidates to ``candidates`` (jsonl.gz) until every (task, language,
    label) cell holds per-cell quota x ``oversample`` candidates. Resumes from the file."""
    candidates = Path(candidates)
    state_path = Path(str(candidates) + ".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"positions": {}, "jobs": {}}
    have = collections.Counter()
    if candidates.exists():
        for row in _read_candidates(candidates):
            have[(row["task"], row["passage_row"][1], row["label"])] += 1
    targets = _cell_targets(tasks, per_capability, oversample)
    passages = passages or _passages(cache, max(20, per_capability * max_jobs_factor // 8))
    metrics = collections.Counter()
    for task in tasks:
        labels = URGENCY_LABELS if task == "urgency" else NLI_LABELS
        total = sum(n for cell, n in targets.items() if cell[0] == task)
        done = lambda: sum(min(have[c], n) for c, n in targets.items() if c[0] == task)
        started, start_done = time.monotonic(), done()
        jobs = state["jobs"].get(task, 0)
        positions = collections.Counter(state["positions"].get(task, {}))
        max_jobs = per_capability * max_jobs_factor
        _stage_line("generate", task, done(), total, started, start_done)
        while done() < total and jobs < max_jobs:
            batch = []
            for lang in PILOT_LANGUAGES:
                need = [l for l in labels if have[(task, lang, l)] < targets[(task, lang, l)]]
                if not need:
                    continue
                for _ in range(max(1, concurrency // len(PILOT_LANGUAGES))):
                    passage = passages[lang][positions[lang] % len(passages[lang])]
                    positions[lang] += 1
                    target = need[positions[lang] % len(need)] if task == "urgency" else None
                    batch.append((task, passage, target))
            if not batch:
                break
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(
                    lambda a: _generate_candidates(a[0], a[1], a[2], generator, temperature), batch))
            jobs += len(batch)
            for rows, m in results:
                metrics.update(m)
                for row in rows:
                    cell = (task, row["passage_row"][1], row["label"])
                    if have[cell] >= targets[cell]:
                        continue
                    have[cell] += 1
                    _append_gzip(candidates, dict(row, passage_row=list(row["passage_row"])))
            state["jobs"][task], state["positions"][task] = jobs, dict(positions)
            state_path.write_text(json.dumps(state))
            _stage_line("generate", task, done(), total, started, start_done)
    short = {"%s/%s/%s" % c: n - have[c] for c, n in targets.items() if have[c] < n}
    return {"candidates": sum(have.values()), "short": short, "metrics": dict(metrics)}


def stage_verify(candidates, out_dir, verifier, guard, tasks, per_capability, concurrency,
                 model_meta, seed=20261003, resume=False):
    """Blind-verify candidates in file order and keep them up to the per-cell quotas; writes the
    same outputs and manifest as run_pilot. A short cell is reported, never padded."""
    out_dir = Path(out_dir)
    if out_dir.exists() and not resume:
        raise FileExistsError("refusing existing output directory without --resume: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_all = _read_candidates(candidates)
    previous = {}
    if resume and (out_dir / "manifest.json").exists():
        previous = json.loads((out_dir / "manifest.json").read_text()).get("stats", {})
    state_path = out_dir / "verify.state.json"
    state = json.loads(state_path.read_text()) if resume and state_path.exists() else {}
    started = time.monotonic()
    by_task, stats = {}, {}
    for task in tasks:
        labels = URGENCY_LABELS if task == "urgency" else NLI_LABELS
        quota = _quota(per_capability, PILOT_LANGUAGES, labels)
        initial, item_path, prov_path, finalized = _resume_rows(out_dir, task) if resume else (
            [], out_dir / (task + ".jsonl.gz.part"), out_dir / (task + ".provenance.jsonl.gz.part"), False)
        kept = list(initial)
        counts = collections.Counter((r["provenance"]["seed"]["lang"], r["provenance"]["seed"]["target"])
                                     for r in kept)
        seen = {r["id"] for r in kept}
        reasons = collections.Counter()
        todo = [dict(r, passage_row=tuple(r["passage_row"])) for r in rows_all if r["task"] == task]
        pos = state.get(task, 0)
        t0, k0 = time.monotonic(), len(kept)
        _stage_line("verify", task, len(kept), per_capability, t0, k0)
        while not finalized and len(kept) < per_capability and pos < len(todo):
            chunk = []
            while pos < len(todo) and len(chunk) < concurrency * 4:
                cand = todo[pos]
                pos += 1
                if counts[(cand["passage_row"][1], cand["label"])] < quota[(cand["passage_row"][1], cand["label"])]:
                    chunk.append(cand)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                verified = list(pool.map(lambda c: _verify_candidate(c, verifier, model_meta, guard), chunk))
            for row, m in verified:
                reasons.update(m)
                if row is None:
                    continue
                cell = (row["provenance"]["seed"]["lang"], row["provenance"]["seed"]["target"])
                if counts[cell] >= quota[cell] or len(kept) >= per_capability:
                    continue
                if row["id"] in seen:
                    reasons["duplicate"] += 1
                    continue
                seen.add(row["id"])
                counts[cell] += 1
                kept.append(row)
                _append_gzip(item_path, row["item"])
                _append_gzip(prov_path, row["provenance"])
            state[task] = pos
            state_path.write_text(json.dumps(state))
            called, agreed = reasons["verify_called"], reasons["verify_agree"]
            agree = "  verifier agrees %3.0f%%" % (100.0 * agreed / called) if called else ""
            _stage_line("verify", task, len(kept), per_capability, t0, k0, agree)
        if len(kept) < per_capability:
            short = {"%s/%s" % c: quota[c] - counts[c] for c in quota if counts[c] < quota[c]}
            raise SystemExit("%s: %d/%d kept, candidates exhausted; short cells %s. Run "
                             "--stage generate with a higher --oversample, then --stage verify "
                             "--resume." % (task, len(kept), per_capability, short))
        if not finalized:
            item_path.replace(out_dir / (task + ".jsonl.gz"))
            prov_path.replace(out_dir / (task + ".provenance.jsonl.gz"))
        called, agreed = reasons["verify_called"], reasons["verify_agree"]
        by_task[task] = kept
        if finalized and task in previous:  # resumed after this task was complete: keep its numbers
            stats[task] = previous[task]
            continue
        stats[task] = {"jobs": pos, "metrics": dict(reasons),
                       "verify_acceptance_rate": (agreed / called) if called else None,
                       "counts": {"%s/%s" % c: n for c, n in sorted(counts.items())}}
    return _finish(out_dir, by_task, stats, model_meta, guard, per_capability, concurrency, tasks,
                   seed, started, {"stages": "generate+verify", "candidates": str(candidates)})


def _client_and_meta(args, role):
    """Client plus validated provenance for the generator or the verifier."""
    model = args.generator_model if role == "generator" else args.verifier_model
    if args.backend == "ollama":
        client = OllamaHTTP(args.host, model, args.num_gpu)
        meta = validate_model_meta(model, digest_from_show(client.show()))
        meta["requested_model"] = model
        return client, model, meta
    revision = args.generator_revision if role == "generator" else args.verifier_revision
    client = OpenAIHTTP(args.host, model, revision)
    return client, model, validate_hf_meta(client.show())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, help="output directory (stage single or verify)")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--per-capability", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="simultaneous requests (Ollama: set OLLAMA_NUM_PARALLEL >= this value)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--backend", choices=("ollama", "openai"), default="ollama",
                    help="openai = an OpenAI-compatible server such as vLLM serving one HF model")
    ap.add_argument("--generator-model", default="qwen3:8b",
                    help="Ollama tag (qwen3:8b) or, with --backend openai, an allowlisted HF repo id")
    ap.add_argument("--verifier-model", default="phi4-mini")
    ap.add_argument("--generator-revision", default=None, help="exact HF commit sha (--backend openai)")
    ap.add_argument("--verifier-revision", default=None, help="exact HF commit sha (--backend openai)")
    ap.add_argument("--num-gpu", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--max-jobs-factor", type=int, default=12)
    ap.add_argument("--eval-cache", type=Path, default=None)
    ap.add_argument("--s1bench-dir", type=Path, default=None)
    ap.add_argument("--defer-leakage-check", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tasks", default="urgency,nli", help="comma-separated capabilities to generate")
    ap.add_argument("--phased-batch", type=int, default=0, metavar="N",
                    help="generate N seed jobs, then verify them (for GPUs that cannot hold both "
                         "models, e.g. 8 GB); 0 = generate and verify each job together")
    ap.add_argument("--stage", choices=("single", "generate", "verify"), default="single",
                    help="generate: only the generator runs and candidates go to --candidates; "
                         "verify: only the verifier runs over --candidates (one model per GPU)")
    ap.add_argument("--candidates", type=Path, default=None, help="candidate file (jsonl.gz)")
    ap.add_argument("--oversample", type=float, default=2.5,
                    help="stage generate: candidates per kept item to aim for")
    args = ap.parse_args(argv)
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    if not tasks or any(t not in ("urgency", "nli") for t in tasks):
        ap.error("--tasks must name urgency and/or nli")
    if args.per_capability < 1 or args.concurrency < 1 or args.max_jobs_factor < 1:
        ap.error("counts, concurrency and max-jobs-factor must be positive")
    if args.stage != "single" and not args.candidates:
        ap.error("--stage %s needs --candidates" % args.stage)
    if args.stage != "generate" and not args.out_dir:
        ap.error("--out-dir is required")
    cache = validate_cache(args.cache)

    if args.stage == "generate":
        generator, _, gmeta = _client_and_meta(args, "generator")
        state = Path(str(args.candidates) + ".state.json")
        saved = json.loads(state.read_text()) if state.exists() else {}
        if saved.get("generator_meta") not in (None, gmeta):
            raise SystemExit("candidate file was started with another generator: %s" % saved["generator_meta"])
        saved.setdefault("positions", {}), saved.setdefault("jobs", {})
        saved["generator_meta"], saved["generator_tag"] = gmeta, args.generator_model
        state.write_text(json.dumps(saved))
        result = stage_generate(args.candidates, cache, generator, tasks, args.per_capability,
                                args.concurrency, args.temperature, args.oversample, args.max_jobs_factor)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    guard = DeferredGuard() if args.defer_leakage_check else LeakageGuard.from_local(
        args.eval_cache, args.s1bench_dir)
    if args.stage == "verify":
        verifier, vtag, vmeta = _client_and_meta(args, "verifier")
        saved = json.loads(Path(str(args.candidates) + ".state.json").read_text())
        model_meta = {saved["generator_tag"]: saved["generator_meta"], vtag: vmeta}
        manifest = stage_verify(args.candidates, args.out_dir, verifier, guard, tasks,
                                args.per_capability, args.concurrency, model_meta, args.seed,
                                resume=args.resume)
    else:
        generator, _, gmeta = _client_and_meta(args, "generator")
        verifier, _, vmeta = _client_and_meta(args, "verifier")
        manifest = run_pilot(args.out_dir, cache, generator, verifier, guard, args.per_capability,
                             args.concurrency, args.seed, args.temperature, args.max_jobs_factor,
                             generator_meta=gmeta, verifier_meta=vmeta, resume=args.resume,
                             tasks=tasks, phased_batch=args.phased_batch)
    print(json.dumps({"out_dir": str(args.out_dir), "stats": manifest["stats"],
                      "elapsed_s": manifest["elapsed_s"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
