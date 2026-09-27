#!/usr/bin/env python3
"""Synthetic typed decisions for Statim.

The only writer of situation text, questions, options, and gold labels is a
local Ollama model (default qwen3:8b, Apache-2.0). This script does not embed
training examples.

    .venv-train/bin/python tools/synth/generate.py --n 6 --num-gpu 0 --workers 1 \
        --out data/synth-smoke.jsonl.gz

Outputs next to --out (default data/synth-v1.jsonl.gz):
    <stem>.jsonl.gz              training items
    <stem>.provenance.jsonl.gz   rationale, seed, verify answer, rejects
    <stem>.manifest.json         model digest, prompt version, counts, verify rate

Resume is the default: an id whose latest provenance status is accepted or
rejected is not generated again. --stage filter rebuilds the jsonl from the
provenance file. Every chat request sends options.num_gpu (default -1).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import collections
import gzip
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import has_key, training_item  # noqa: E402
from filter import BanIndex, apply_filter, load_ban_strings, public_item  # noqa: E402
from ollama_http import OllamaError, OllamaHTTP  # noqa: E402
from prompts import PROMPT_VERSION, generate_request, prompts_digest  # noqa: E402
from seeds import TASKS, acceptance_from_rows, load_target_weights, plan  # noqa: E402
from verify import agrees, build_item, parse_model_json, verify_view  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RETRYABLE = {"bad_json", "transport"}


def derive_paths(out):
    out = Path(out)
    name = out.name
    if name.endswith(".jsonl.gz"):
        stem = out.with_name(name[: -len(".jsonl.gz")])
        items = out
    else:
        stem = out
        items = Path(str(out) + ".jsonl.gz")
    return items, Path(str(stem) + ".provenance.jsonl.gz"), Path(str(stem) + ".manifest.json")


def read_jsonl_gz(path):
    if not path.exists():
        return []
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl_gz(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def latest_by_id(rows):
    last = {}
    for row in rows:
        last[row["id"]] = row
    return last


def digest_from_show(show):
    modelfile = show.get("modelfile") or ""
    found = re.search(r"sha256-([0-9a-f]{64})", modelfile)
    details = show.get("details") or {}
    info = show.get("model_info") or {}
    license_field = info.get("general.license")
    if isinstance(license_field, str) and len(license_field) > 80:
        license_field = license_field[:80]
    return {
        "digest": found.group(1) if found else None,
        "digest_source": "sha256 in the FROM line of /api/show modelfile "
                         "(this Ollama build does not return a top-level digest)",
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "general_license": license_field,
        "modified_at": show.get("modified_at"),
    }


def _record(seed, status, reason, item, rationale, raw, verify, usage, attempts,
            generated_instructions=None):
    item_out = training_item(item) if item else None
    if item_out is not None and has_key(item_out, "rationale"):
        status, reason, item_out = "rejected", "rationale_in_item", None
    record = {
        "id": seed["id"],
        "seed": seed,
        "status": status,
        "reason": reason,
        "item": item_out,
        "rationale": rationale or None,
        "verify": verify,
        "raw": (raw or "")[:8000],
        "usage": usage,
        "attempts": attempts,
        "prompt_version": PROMPT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if generated_instructions:
        # The question the model wrote, before the seeded paraphrase replaced it.
        record["generated_instructions"] = generated_instructions
    return record


def generate_once(client, seed, temperature):
    messages, schema, npredict = generate_request(seed)
    seed_int = int(seed["id"][:8], 16)
    try:
        resp = client.chat(messages, schema, temperature=temperature,
                           num_predict=npredict, seed=seed_int)
    except OllamaError as exc:
        return _record(seed, "error", "transport", None, None, str(exc), None, None, 1)
    if resp.get("thinking"):
        # think:false should keep this empty. Never treat the trace as the item.
        pass
    try:
        parsed = parse_model_json(resp["content"])
    except (json.JSONDecodeError, ValueError) as exc:
        rec = _record(seed, "rejected", "bad_json", None, None, resp["content"], None, _usage(resp), 1)
        rec["error"] = str(exc)
        return rec
    built = build_item(seed, parsed)
    usage = _usage(resp)
    generated_instructions = built.get("generated_instructions")
    if built["status"] != "ok":
        return _record(seed, "rejected", built["reason"], None, built["rationale"],
                       resp["content"], None, usage, 1, generated_instructions)
    try:
        answer, vresp = verify_view(client, built["view"], seed_int)
    except (OllamaError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        return _record(seed, "error", "verify_transport", built["item"], built["rationale"],
                       resp["content"], {"error": str(exc)}, usage, 1, generated_instructions)
    ok = agrees(built["item"], answer)
    verify = {
        "answer": answer if isinstance(answer, (str, bool, int, float)) else str(answer),
        "agree": bool(ok),
        "usage": _usage(vresp),
    }
    if not ok:
        return _record(seed, "rejected", "verify_disagree", None, built["rationale"],
                       resp["content"], verify, usage, 1, generated_instructions)
    return _record(seed, "accepted", None, built["item"], built["rationale"],
                   resp["content"], verify, usage, 1, generated_instructions)


def _usage(resp):
    if not resp:
        return None
    return {
        "eval_count": resp.get("eval_count"),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "total_s": resp.get("total_s"),
        "done_reason": resp.get("done_reason"),
    }


def process_seed(client, seed, temperature):
    """One generation. Retry only a broken JSON payload or a transport error, once."""
    rec = generate_once(client, seed, temperature)
    if rec["status"] == "error" or rec["reason"] in RETRYABLE:
        second = generate_once(client, seed, temperature)
        second["attempts"] = 2
        return second
    return rec


def counts_of(items):
    by_task = collections.Counter()
    by_lang = collections.Counter()
    by_task_lang = collections.Counter()
    by_type = collections.Counter()
    for item in items:
        src = item["src"].split("/")
        task = src[1] if len(src) > 1 else "?"
        lang = src[2] if len(src) > 2 else "?"
        by_task[task] += 1
        by_lang[lang] += 1
        by_task_lang["%s/%s" % (task, lang)] += 1
        by_type[item["q"]["type"]] += 1
    return {
        "by_task": dict(by_task),
        "by_language": dict(by_lang),
        "by_task_language": dict(by_task_lang),
        "by_type": dict(by_type),
    }


def write_outputs(items_path, manifest_path, provenance_rows, model_meta, args, cpu_snapshot):
    last = latest_by_id(provenance_rows)
    accepted = []
    for rec in last.values():
        if rec.get("status") == "accepted" and rec.get("item"):
            item = dict(rec["item"])
            item["_id"] = rec["id"]
            accepted.append(item)
    print("filter: %d accepted before overlap/dedup" % len(accepted), flush=True)
    strings = load_ban_strings()
    ban = BanIndex(strings)
    kept, dropped = apply_filter(
        accepted, ban, seed=args.seed, max_class_share=args.max_class_share, semantic=True)
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    final = [public_item(item) for item in kept]
    items_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(items_path, "wt", encoding="utf-8") as handle:
        for item in final:
            if set(item) != {"state", "q", "target", "src"} or has_key(item, "rationale"):
                raise SystemExit("refusing to write an item that is not the training schema")
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    reasons = collections.Counter(rec.get("reason") or rec.get("status") for rec in last.values())
    verify_rows = [rec for rec in last.values() if rec.get("verify") and "agree" in rec["verify"]]
    agreed = sum(1 for rec in verify_rows if rec["verify"]["agree"])
    rate = (agreed / len(verify_rows)) if verify_rows else None
    manifest = {
        "generator_model": args.model,
        "ollama": model_meta,
        "prompts_version": PROMPT_VERSION,
        "prompts_sha256": prompts_digest(),
        "n_per_task": args.n,
        "tasks": args.tasks,
        "seed": args.seed,
        "num_gpu": args.num_gpu,
        "workers": args.workers,
        "temperature": args.temperature,
        "cpu_check": cpu_snapshot,
        "provenance_records": len(provenance_rows),
        "ids": len(last),
        "verify_called": len(verify_rows),
        "verify_agreed": agreed,
        "verify_acceptance_rate": rate,
        "reasons": dict(reasons),
        "filter_dropped": dropped,
        "target_acceptance": acceptance_from_rows(list(last.values())),
        "target_weights": getattr(args, "target_weights", None),
        "max_class_share": args.max_class_share,
        "items": len(final),
        "counts": counts_of(final),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print("wrote %s (%d)" % (items_path, len(final)), flush=True)
    print("wrote %s" % manifest_path, flush=True)
    print("verify acceptance %s (%d/%d)" % (rate, agreed, len(verify_rows)), flush=True)
    return manifest, final


def main():
    ap = argparse.ArgumentParser(description="Generate, verify, and filter synthetic decisions")
    ap.add_argument("--n", type=int, required=True, help="attempts per task")
    ap.add_argument("--num-gpu", type=int, default=-1, help="Ollama options.num_gpu; -1 lets Ollama decide")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth-v1.jsonl.gz"))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--tasks", default="business,score,reading")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--stage", choices=("all", "generate", "filter"), default="all")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--balance-from", default=None,
                    help="manifest.json; target weights are 1/acceptance, clamped to [1, 4]")
    ap.add_argument("--max-class-share", type=float, default=0.45,
                    help="drop surplus so no class exceeds this share of a task")
    args = ap.parse_args()
    if args.n < 1 or args.workers < 1:
        raise SystemExit("--n and --workers must be >= 1")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in TASKS:
            raise SystemExit("unknown task %s" % task)
    args.tasks = tasks
    if not (0 < args.max_class_share <= 1):
        raise SystemExit("--max-class-share must be in (0, 1]")
    args.target_weights = load_target_weights(args.balance_from) if args.balance_from else None
    items_path, prov_path, manifest_path = derive_paths(Path(args.out))
    if args.no_resume and (prov_path.exists() or items_path.exists()):
        raise SystemExit("refusing --no-resume because %s already exists" % prov_path)

    existing = read_jsonl_gz(prov_path)

    client = OllamaHTTP(args.host, args.model, args.num_gpu)
    show = client.show()
    model_meta = digest_from_show(show)
    model_meta["requested_model"] = args.model
    print("model %s digest %s" % (args.model, model_meta.get("digest")), flush=True)

    if args.stage != "filter":
        seeds = []
        for task in tasks:
            weights = None if args.target_weights is None else args.target_weights.get(task)
            seeds.extend(plan(task, args.n, args.seed, weights=weights))
        done = set()
        for rec in latest_by_id(existing).values():
            # A new prompt version has to run again. The previous reject was for other wording.
            if rec.get("status") in {"accepted", "rejected"} and rec.get("prompt_version") == PROMPT_VERSION:
                done.add(rec["id"])
        pending = [seed for seed in seeds if seed["id"] not in done]
        print("planned %d pending %d resume-skip %d" % (len(seeds), len(pending), len(seeds) - len(pending)),
              flush=True)
        ban_pool = ThreadPoolExecutor(max_workers=1)
        ban_future = ban_pool.submit(load_ban_strings) if args.stage == "all" else None
        if pending:
            workers = min(args.workers, len(pending))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process_seed, client, seed, args.temperature) for seed in pending]
                for fut in as_completed(futures):
                    rec = fut.result()
                    append_jsonl_gz(prov_path, rec)
                    existing.append(rec)
                    print("%s %s %s %s" % (
                        rec["status"], rec["seed"]["task"], rec["seed"]["lang"], rec.get("reason") or ""),
                          flush=True)
        if ban_future is not None:
            # write_outputs loads the ban itself; starting it early overlaps the downloads.
            try:
                ban_future.result()
            except Exception:
                ban_pool.shutdown(wait=False, cancel_futures=True)
                raise
            ban_pool.shutdown(wait=True)
        else:
            ban_pool.shutdown(wait=False)
    if args.stage == "generate":
        print("generate stage done; provenance %s" % prov_path, flush=True)
        return
    manifest, _final = write_outputs(
        items_path, manifest_path, read_jsonl_gz(prov_path), model_meta, args, client.cpu_snapshot)
    print(json.dumps({"items": manifest["items"], "verify_acceptance_rate": manifest["verify_acceptance_rate"],
                      "counts": manifest["counts"], "reasons": manifest["reasons"],
                      "filter_dropped": manifest["filter_dropped"]}, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
