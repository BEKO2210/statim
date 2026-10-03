#!/usr/bin/env python3
"""Generate the grounded urgency + multilingual NLI pilot with two local models."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
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
from verify import parse_model_json  # noqa: E402

GENERATOR = {"model": "Qwen/Qwen3-8B", "role": "text"}
VERIFIER = {"model": "microsoft/Phi-4-mini-instruct", "role": "labels"}
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
        "generator": [dict(GENERATOR), dict(VERIFIER)],
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


def _generate_job(task, passage_row, target, generator, verifier, temperature, model_meta, guard):
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
    accepted = []
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
        blind = {"request": text} if task == "urgency" else {"premise": passage, "hypothesis": text}
        metrics["verify_called"] += 1
        answer_obj = _request_json(verifier, verification_request(task, blind, lang), 0,
                                   seed ^ 0x5A5A5A5A, metrics, "verify")
        if answer_obj is None:
            metrics["verify_error"] += 1
            continue
        answer = answer_obj.get("answer")
        if answer != label:
            metrics["verify_disagree"] += 1
            continue
        metrics["verify_agree"] += 1
        base = _base_item(task, lang, label, passage, text)
        passage_hash = sha256_text(passage)
        ident = canonical_id({"item": base, "source_id": source_id,
                              "passage_sha256": passage_hash, "target": label})
        item = dict(base, id=ident, src=source_id, passage_sha256=passage_hash)
        if guard.overlap(item):
            metrics["leakage_reject"] += 1
            continue
        verify = {"answer": answer, "agree": True}
        provenance = _provenance(ident, task, lang, label, passage, source_id, revision,
                                 model_meta, verify)
        accepted.append({"id": ident, "item": item, "provenance": provenance})
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
              guard, max_jobs, initial=None, on_accept=None):
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
            if len(batch) >= concurrency * 2:
                break
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
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
    if len(kept) != total:
        raise RuntimeError("%s kept %d/%d after %d jobs; reasons=%s" %
                           (task, len(kept), total, jobs, dict(reasons)))
    called, agreed = reasons["verify_called"], reasons["verify_agree"]
    return kept, {"jobs": jobs, "metrics": dict(reasons),
                  "verify_acceptance_rate": (agreed / called) if called else None,
                  "counts": {"%s/%s" % cell: count for cell, count in sorted(counts.items())}}


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
              generator_meta=None, verifier_meta=None, resume=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and not resume:
        raise FileExistsError("refusing existing output directory without --resume: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_meta = {"qwen3:8b": dict(generator_meta or {}),
                  "phi4-mini": dict(verifier_meta or {})}
    passages = passages or _passages(cache, max(20, per_capability * max_jobs_factor // 8))
    started = time.monotonic()
    by_task, stats = {}, {}
    for task in ("urgency", "nli"):
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
            on_accept=None if finalized else accept)
        by_task[task], stats[task] = rows, task_stats
        if not finalized:
            item_path.replace(out_dir / (task + ".jsonl.gz"))
            prov_path.replace(out_dir / (task + ".provenance.jsonl.gz"))
    write_samples(out_dir / "samples.md", by_task, 30, seed)
    leakage_meta = dict(getattr(guard, "metadata", {}) or {})
    leakage_meta.update({"word_shingle": 8, "char_shingle": 20, "contain_chars": 40,
                         "heldout_texts_indexed": getattr(guard, "texts", None),
                         "checked": not getattr(guard, "deferred", False)})
    manifest = {
        "kind": "grounded-synthetic-pilot",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items_per_capability": per_capability, "languages": list(PILOT_LANGUAGES),
        "models": [GENERATOR, VERIFIER], "ollama": model_meta,
        "prompt_version": PROMPT_VERSION, "prompts_sha256": prompts_digest(),
        "corpora": CORPORA, "leakage": leakage_meta, "concurrency": concurrency,
        "seed": seed, "stats": stats, "elapsed_s": time.monotonic() - started,
    }
    manifest_part = out_dir / "manifest.json.part"
    manifest_part.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_part.replace(out_dir / "manifest.json")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--per-capability", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="simultaneous requests; set OLLAMA_NUM_PARALLEL >= this value")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--generator-model", default="qwen3:8b", choices=("qwen3:8b",))
    ap.add_argument("--verifier-model", default="phi4-mini", choices=("phi4-mini",))
    ap.add_argument("--num-gpu", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--max-jobs-factor", type=int, default=12)
    ap.add_argument("--eval-cache", type=Path, default=None)
    ap.add_argument("--s1bench-dir", type=Path, default=None)
    ap.add_argument("--defer-leakage-check", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if args.per_capability < 1 or args.concurrency < 1 or args.max_jobs_factor < 1:
        ap.error("counts, concurrency and max-jobs-factor must be positive")
    cache = validate_cache(args.cache)
    guard = DeferredGuard() if args.defer_leakage_check else LeakageGuard.from_local(
        args.eval_cache, args.s1bench_dir)
    generator = OllamaHTTP(args.host, args.generator_model, args.num_gpu)
    verifier = OllamaHTTP(args.host, args.verifier_model, args.num_gpu)
    gmeta = validate_model_meta(args.generator_model, digest_from_show(generator.show()))
    vmeta = validate_model_meta(args.verifier_model, digest_from_show(verifier.show()))
    gmeta["requested_model"], vmeta["requested_model"] = args.generator_model, args.verifier_model
    manifest = run_pilot(args.out_dir, cache, generator, verifier, guard, args.per_capability,
                         args.concurrency, args.seed, args.temperature, args.max_jobs_factor,
                         generator_meta=gmeta, verifier_meta=vmeta, resume=args.resume)
    print(json.dumps({"out_dir": str(args.out_dir), "stats": manifest["stats"],
                      "elapsed_s": manifest["elapsed_s"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
