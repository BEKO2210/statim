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
from leakage import DeferredGuard, LeakageGuard  # noqa: E402
from ollama_http import OllamaError, OllamaHTTP  # noqa: E402
from verify import parse_model_json  # noqa: E402

GENERATOR = {"model": "Qwen/Qwen3-8B", "role": "text"}
VERIFIER = {"model": "microsoft/Phi-4-mini-instruct", "role": "labels"}

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


def _chat(client, request, temperature, seed):
    messages, schema, predict = request
    return client.chat(messages, schema, temperature=temperature, num_predict=predict, seed=seed)


def _quota(total, languages, labels):
    cells = [(lang, label) for lang in languages for label in labels]
    return {cell: total // len(cells) + (i < total % len(cells)) for i, cell in enumerate(cells)}


def _source_spec(source_id):
    if source_id.startswith("FiscalNote/"):
        return "billsum"
    if source_id.startswith("launch/"):
        return "gov_report"
    return "eur_lex_sum"


def _provenance(task, lang, label, passage, source_id, revision, digests):
    return {
        "seed": {"source_id": source_id, "revision": revision,
                 "passage_sha256": sha256_text(passage), "lang": lang, "target": label},
        "seed_passage": passage,
        "generator": [dict(GENERATOR), dict(VERIFIER)],
        "ollama_model_digests": digests,
        "prompt_version": PROMPT_VERSION,
        "capability": task,
    }


def _item(task, lang, label, passage, generated, source_id):
    if task == "urgency":
        return {"state": generated, "q": {"type": "score", "instructions": _WORDS[lang][3],
                "criteria": list(URGENCY_LABELS)}, "target": one_hot(3, URGENCY_LABELS.index(label)),
                "src": "grounded/urgency/%s/%s" % (lang, source_id)}
    premise, hypothesis = passage, generated
    state = "%s: %s\n\n%s: %s" % (_WORDS[lang][0], premise, _WORDS[lang][1], hypothesis)
    desc = _NLI_DESC[lang]
    criteria = {name: desc[i] for i, name in enumerate(NLI_LABELS)}
    return {"state": state, "q": {"type": "choice", "instructions": _WORDS[lang][2],
            "criteria": criteria}, "target": one_hot(3, NLI_LABELS.index(label)),
            "src": "grounded/nli/%s/%s" % (lang, source_id)}


def _generate_job(task, passage_row, target, generator, verifier, temperature, digests, guard):
    passage, lang, source_id, revision = passage_row
    seed = int(hashlib.sha256((task + source_id + passage + str(target)).encode()).hexdigest()[:8], 16)
    try:
        response = _chat(generator, generation_request(task, passage, lang, target), temperature, seed)
        parsed = parse_model_json(response["content"])
    except (OllamaError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return [], {"generate_error:%s" % type(exc).__name__: 1}
    metrics = collections.Counter()
    raw = []
    if task == "urgency":
        if parsed.get("label") != target or not str(parsed.get("request") or "").strip():
            return [], {"bad_generation": 1}
        raw.append((target, str(parsed["request"]).strip()))
    else:
        rows = parsed.get("items") if isinstance(parsed, dict) else None
        if not isinstance(rows, list) or len(rows) != 3:
            return [], {"bad_generation": 1}
        labels = [row.get("label") for row in rows if isinstance(row, dict)]
        if sorted(labels) != sorted(NLI_LABELS):
            return [], {"bad_generation": 1}
        raw.extend((row["label"], str(row.get("hypothesis") or "").strip()) for row in rows)
    accepted = []
    for label, text in raw:
        limits = (8, 120) if task == "urgency" else (3, 80)
        if not limits[0] <= word_count(text) <= limits[1] or not language_ok(text, lang):
            metrics["bad_text"] += 1
            continue
        blind = {"request": text} if task == "urgency" else {"premise": passage, "hypothesis": text}
        try:
            metrics["verify_called"] += 1
            vresp = _chat(verifier, verification_request(task, blind, lang), 0, 0)
            answer = parse_model_json(vresp["content"]).get("answer")
        except (OllamaError, KeyError, ValueError, json.JSONDecodeError):
            metrics["verify_error"] += 1
            continue
        if answer != label:
            metrics["verify_disagree"] += 1
            continue
        metrics["verify_agree"] += 1
        item = _item(task, lang, label, passage, text, source_id)
        if guard.overlap(item):
            metrics["leakage_reject"] += 1
            continue
        provenance = _provenance(task, lang, label, passage, source_id, revision, digests)
        ident = canonical_id({"item": item, "provenance": provenance["seed"]})
        accepted.append({"id": ident, "item": item, "provenance": provenance,
                         "verify": {"answer": answer, "agree": True}})
    return accepted, dict(metrics)


def _passages(cache, per_language):
    out = {lang: [] for lang in PILOT_LANGUAGES}
    for lang in PILOT_LANGUAGES:
        names = ("billsum", "gov_report") if lang == "en" else ("eur_lex_sum",)
        per_source = (per_language + len(names) - 1) // len(names)
        for name in names:
            for row in iter_corpus(name, cache, languages=(lang,)):
                out[lang].append(row)
                if sum(1 for x in out[lang] if _source_spec(x[2]) == name) >= per_source:
                    break
            if len(out[lang]) >= per_language:
                break
        out[lang] = out[lang][:per_language]
    missing = [lang for lang, rows in out.items() if not rows]
    if missing:
        raise RuntimeError("no grounding passages for: %s" % ", ".join(missing))
    return out


def _run_task(task, total, passages, generator, verifier, concurrency, temperature, digests, guard,
              max_jobs):
    labels = URGENCY_LABELS if task == "urgency" else NLI_LABELS
    quota, counts = _quota(total, PILOT_LANGUAGES, labels), collections.Counter()
    positions = collections.Counter()
    kept, seen_ids, reasons, jobs = [], set(), collections.Counter(), 0
    while len(kept) < total and jobs < max_jobs:
        batch = []
        for lang in PILOT_LANGUAGES:
            need = [label for label in labels if counts[(lang, label)] < quota[(lang, label)]]
            if not need:
                continue
            passage = passages[lang][positions[lang] % len(passages[lang])]
            positions[lang] += 1
            # NLI produces all three labels; urgency has one uniformly quota-driven target.
            target = need[0] if task == "urgency" else None
            batch.append((task, passage, target, generator, verifier, temperature, digests, guard))
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
    if len(kept) != total:
        raise RuntimeError("%s kept %d/%d after %d jobs; reasons=%s" %
                           (task, len(kept), total, jobs, dict(reasons)))
    called, agreed = reasons["verify_called"], reasons["verify_agree"]
    return kept, {"jobs": jobs, "metrics": dict(reasons),
                  "verify_acceptance_rate": (agreed / called) if called else None,
                  "counts": {"%s/%s" % cell: count for cell, count in sorted(counts.items())}}


def _write_jsonl(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_samples(path, by_task, n, seed):
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
              generator_meta=None, verifier_meta=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {"qwen3:8b": (generator_meta or {}).get("digest"),
               "phi4-mini": (verifier_meta or {}).get("digest")}
    passages = passages or _passages(cache, max(20, per_capability * max_jobs_factor // 8))
    started = time.monotonic()
    by_task, stats = {}, {}
    for task in ("urgency", "nli"):
        rows, task_stats = _run_task(task, per_capability, passages, generator, verifier, concurrency,
                                     temperature, digests, guard, per_capability * max_jobs_factor)
        by_task[task], stats[task] = rows, task_stats
        _write_jsonl(out_dir / (task + ".jsonl.gz"), [row["item"] for row in rows])
        _write_jsonl(out_dir / (task + ".provenance.jsonl.gz"),
                     [dict(row["provenance"], item_id=row["id"], verify=row["verify"]) for row in rows])
    _write_samples(out_dir / "samples.md", by_task, 30, seed)
    manifest = {
        "kind": "grounded-synthetic-pilot", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items_per_capability": per_capability, "languages": list(PILOT_LANGUAGES),
        "models": [GENERATOR, VERIFIER], "ollama": {"qwen3:8b": generator_meta,
        "phi4-mini": verifier_meta}, "prompt_version": PROMPT_VERSION,
        "prompts_sha256": prompts_digest(), "corpora": CORPORA, "leakage": {
            "n_gram": 8, "heldout_texts_indexed": getattr(guard, "texts", None),
            "checked": not getattr(guard, "deferred", False)},
        "concurrency": concurrency, "seed": seed, "stats": stats,
        "elapsed_s": time.monotonic() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="corpus cache (default: $TMPDIR/statim-synth-corpora)")
    ap.add_argument("--per-capability", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="simultaneous Ollama requests; set OLLAMA_NUM_PARALLEL >= this value on the server")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--generator-model", default="qwen3:8b", choices=("qwen3:8b",))
    ap.add_argument("--verifier-model", default="phi4-mini", choices=("phi4-mini",))
    ap.add_argument("--num-gpu", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--max-jobs-factor", type=int, default=12)
    ap.add_argument("--eval-cache", type=Path, default=None)
    ap.add_argument("--s1bench-dir", type=Path, default=None)
    ap.add_argument("--defer-leakage-check", action="store_true",
                    help="no held-out texts on this machine (Colab): mark output unchecked; run "
                         "`leakage.py --filter` locally before use")
    args = ap.parse_args(argv)
    if args.per_capability < 1 or args.concurrency < 1 or args.max_jobs_factor < 1:
        ap.error("counts, concurrency and max-jobs-factor must be positive")
    cache = validate_cache(args.cache)
    guard = DeferredGuard() if args.defer_leakage_check else LeakageGuard.from_local(
        args.eval_cache, args.s1bench_dir)
    generator = OllamaHTTP(args.host, args.generator_model, args.num_gpu)
    verifier = OllamaHTTP(args.host, args.verifier_model, args.num_gpu)
    gmeta, vmeta = digest_from_show(generator.show()), digest_from_show(verifier.show())
    if not gmeta.get("digest") or not vmeta.get("digest"):
        raise SystemExit("Ollama did not expose both model digests; refusing incomplete provenance")
    gmeta["requested_model"], vmeta["requested_model"] = args.generator_model, args.verifier_model
    manifest = run_pilot(args.out_dir, cache, generator, verifier, guard, args.per_capability,
                         args.concurrency, args.seed, args.temperature, args.max_jobs_factor,
                         generator_meta=gmeta, verifier_meta=vmeta)
    print(json.dumps({"out_dir": str(args.out_dir), "stats": manifest["stats"],
                      "elapsed_s": manifest["elapsed_s"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
