#!/usr/bin/env python3
"""Build the registry-driven Statim mixture v6."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing
import random
import resource
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.finetune.mixture_v6.eval_texts import load as load_eval_texts, norm
    from tools.finetune.mixture_v6.loaders import load_rows
    from tools.finetune.mixture_v6.registry import ADAPTERS, ENABLED_ENTRIES, adapt, metadata, source_key
else:
    from .eval_texts import load as load_eval_texts, norm
    from .loaders import load_rows
    from .registry import ADAPTERS, ENABLED_ENTRIES, adapt, metadata, source_key
from tools.finetune.source_policy import input_hashes

ROOT = Path(__file__).resolve().parents[3]
MAX_LOADED = 60_000
_BANNED = frozenset()  # evaluation texts; set by main() before any source worker is forked


class SourceFailed(RuntimeError):
    """A source failed inside its worker process: the worker's error line and traceback."""

    def __init__(self, error, trace):
        super().__init__(error)
        self.error, self.trace = error, trace


def _error_line(exc):
    return "%s: %s" % (type(exc).__name__, (str(exc).splitlines() or [""])[0])


def _worker(conn, fn, args):
    try:
        conn.send((True, fn(*args)))
    except BaseException as exc:  # the parent records the failure; the worker must not die silently
        conn.send((False, (_error_line(exc), traceback.format_exc(limit=3))))
    finally:
        conn.close()


def run_isolated(fn, *args):
    """Run fn(*args) in a forked worker and return its result. Hub streaming, Arrow buffers and
    the fragmented heap of one source die with its worker, so memory stays flat over 121 sources
    (in one process it reached 4.5 GB after 33 of 121 sources). In-process where fork is missing."""
    if "fork" not in multiprocessing.get_all_start_methods():
        return fn(*args)
    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(send, fn, args))
    proc.start()
    send.close()
    try:
        ok, payload = recv.recv()  # before join: a large result would otherwise block the worker
    except EOFError:
        proc.join()
        raise SourceFailed("WorkerDied: exit code %s" % proc.exitcode, "") from None
    finally:
        recv.close()
    proc.join()
    if not ok:
        raise SourceFailed(*payload)
    return payload


def _rss_mb():
    """(this process, largest finished worker) resident memory in MB, for the progress log."""
    try:
        with open("/proc/self/status") as fh:
            own = next(int(line.split()[1]) for line in fh if line.startswith("VmRSS:"))
    except (OSError, StopIteration):
        own = 0
    return own // 1024, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024


def load_source(entry, load_limit, cap, seed, number):
    """Load, adapt and clean one source (runs inside its worker)."""
    rows, warnings = load_rows(entry, load_limit)
    items, dropped = clean_items(entry, adapt(entry, rows, seed), _BANNED, cap, seed + number)
    return {"rows_loaded": len(rows), "items": items, "dropped": dict(dropped), "warnings": warnings}

def valid_item(item):
    if not isinstance(item, dict) or not isinstance(item.get("state"), str) or not item["state"].strip():
        return False
    q, target = item.get("q"), item.get("target")
    if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul") or not str(q.get("instructions", "")).strip():
        return False
    if q["type"] == "noul":
        n = 2
    else:
        criteria = q.get("criteria")
        n = len(criteria) if isinstance(criteria, (list, dict)) else 0
    return n >= 2 and isinstance(target, list) and len(target) == n and sum(x == 1.0 for x in target) == 1 and sum(target) == 1.0


def _trivial_fragment(text):
    """One- and two-word turns such as "thanks" collide with eval items and are not the decision text."""
    folded = norm(text)
    return len(folded) <= 12 or len(folded.split()) <= 2


def question_key(item):
    """Identity of the question, so the same state can carry two different tasks."""
    q = item.get("q") or {}
    crit = q.get("criteria")
    if isinstance(crit, dict):
        crit_key = tuple(crit.items())
    elif isinstance(crit, list):
        crit_key = tuple(crit)
    else:
        crit_key = ()
    return (q.get("type"), q.get("instructions"), crit_key)


def overlaps_eval(item, banned):
    state = item.get("state") or ""
    if isinstance(state, str) and state.strip() and norm(state) in banned:
        return True
    state_norm = norm(state) if isinstance(state, str) else ""
    for value in item.get("_texts") or []:
        if not isinstance(value, str) or not value.strip():
            continue
        folded = norm(value)
        if not folded or folded == state_norm or _trivial_fragment(value):
            continue
        if folded in banned:
            return True
    return False


def _balanced(items, cap, seed):
    buckets = collections.defaultdict(list)
    for item in items:
        label = item["target"].index(max(item["target"]))
        criteria = item["q"].get("criteria", ["false", "true"])
        key_name = list(criteria)[label] if label < len(criteria) else str(label)
        buckets[(item["q"]["type"], str(key_name))].append(item)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    out = []
    keys = sorted(buckets)
    while len(out) < cap:
        grew = False
        for key in keys:
            if buckets[key]:
                out.append(buckets[key].pop())
                grew = True
                if len(out) == cap:
                    break
        if not grew:
            break
    return out


def clean_items(entry, raw_items, banned, cap, seed):
    stats = collections.Counter()
    seen_exact, seen_norm, clean = set(), set(), []
    for item in raw_items:
        if not valid_item(item):
            stats["malformed"] += 1
            continue
        state, normal = item["state"], norm(item["state"])
        qkey = question_key(item)
        if (state, qkey) in seen_exact:
            stats["duplicate_exact"] += 1
            continue
        if (normal, qkey) in seen_norm:
            stats["duplicate_normalised"] += 1
            continue
        if overlaps_eval(item, banned):
            stats["evaluation_overlap"] += 1
            continue
        seen_exact.add((state, qkey))
        seen_norm.add((normal, qkey))
        clean.append(item)
    if len(clean) > cap:
        stats["over_cap"] += len(clean) - cap
    clean = _balanced(clean, cap, seed)
    for item in clean:
        item.pop("_texts", None)
        item.pop("_task", None)
    return clean, stats


def split_source(items, n_dev, seed, assigned=None):
    """Split one source's items into dev and train by state. ``assigned`` maps a normalised state
    to "dev" or "train" across ALL sources: a state that an earlier source already placed keeps
    that side, so the same text can never be in dev for one source and in train for another."""
    assigned = {} if assigned is None else assigned  # keys are 16-byte digests: ~700k states fit in ~60 MB
    by_state = collections.defaultdict(list)
    for item in items:
        by_state[norm(item["state"])].append(item)
    keys = sorted(by_state)
    random.Random(seed).shuffle(keys)
    dev, train = [], []
    for key in keys:
        slot = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
        side = assigned.get(slot)
        if side is None:
            side = "dev" if len(dev) < n_dev else "train"
            assigned[slot] = side
        (dev if side == "dev" else train).extend(by_state[key])
    return dev, train


def paths_for(out, smoke):
    out = Path(out)
    if smoke:
        out = out.with_name("mixture-v6-smoke.jsonl.gz")
    stem = str(out)
    if not stem.endswith(".jsonl.gz"):
        raise ValueError("--out must end in .jsonl.gz")
    return out, Path(stem[:-9] + ".manifest.json"), Path(stem[:-9] + ".report.md")


def _report(manifest):
    sources = manifest["sources"]
    ok = [s for s in sources if not s.get("error")]
    failed = [s for s in sources if s.get("error")]
    lines = ["# Mixture v6 build report", "", "Generated from enabled entries in `v6-keep.json` only.", "",
             "| Sources enabled | OK | Failed | Items | Train | Dev |", "|---:|---:|---:|---:|---:|---:|",
             "| %d | %d | %d | %d | %d | %d |" % (len(sources), len(ok), len(failed), manifest["items"],
                                                     manifest["train_items"], manifest["dev_items"]), "",
             "## Per source", "",
             "| Source | Loaded | Kept | Train | Dev | Languages | Types | Licence | Attribution | Drops |",
             "|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for s in sources:
        drops = ", ".join("%s: %s" % x for x in sorted(s.get("dropped", {}).items())) or "—"
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s["key"].replace("|", "/"), s.get("rows_loaded", 0), s.get("kept", 0), s.get("train", 0),
            s.get("dev", 0), ", ".join("%s:%s" % x for x in sorted(s.get("per_language", {}).items())) or "—",
            ", ".join("%s:%s" % x for x in sorted(s.get("per_question_type", {}).items())) or "—",
            s.get("licence", "").replace("|", "/"), s.get("attribution", "").replace("|", "/"), drops))
    lines += ["", "## Failed sources", ""]
    if failed:
        lines += ["| Source | Reason |", "|---|---|"]
        lines += ["| `%s` | %s |" % (s["key"].replace("|", "/"), s["error"].replace("|", "/")) for s in failed]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mixture-v6.jsonl.gz")
    ap.add_argument("--per-source", type=int, default=6000)
    ap.add_argument("--dev-per-source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--only", nargs="+", default=None, metavar="ID")
    ap.add_argument("--smoke", type=int, default=0, metavar="N")
    a = ap.parse_args(argv)
    if a.per_source < 1 or a.dev_per_source < 0 or a.smoke < 0:
        ap.error("caps must be non-negative and --per-source must be positive")
    selected = ENABLED_ENTRIES
    if a.only:
        wanted = set(a.only)
        selected = [e for e in selected if e["id"] in wanted or source_key(e) in wanted]
        missing = wanted - {e["id"] for e in selected} - {source_key(e) for e in selected}
        if missing:
            ap.error("unknown or disabled --only source(s): %s" % ", ".join(sorted(missing)))
    # This assertion is intentionally close to loading: disabled entries can
    # never reach load_rows, even if the JSON later gains extra metadata.
    assert all(e.get("use") is True and source_key(e) in ADAPTERS for e in selected)
    global _BANNED
    _BANNED = banned = load_eval_texts(ROOT / "data" / "eval-texts.pkl")
    out_path, manifest_path, report_path = paths_for(a.out, a.smoke)
    cap = a.smoke or (a.per_source + a.dev_per_source)
    all_dev, all_train, records = [], [], []  # items as JSON lines: a third of the memory of dicts
    assigned = {}  # normalised state -> "dev" / "train", shared by all sources
    for number, entry in enumerate(selected):
        key, meta = source_key(entry), metadata(entry)
        rec = {"id": entry["id"], "config": entry.get("config", "default"), "key": key, **meta,
               "rows_loaded": 0, "kept": 0, "train": 0, "dev": 0, "dropped": {}}
        try:
            load_limit = min(MAX_LOADED, max(cap * 8, 500))
            got = run_isolated(load_source, entry, load_limit, cap, a.seed, number)
            items = got["items"]
            rec["rows_loaded"] = got["rows_loaded"]
            dev_n = min(a.dev_per_source, max(0, len(items) // 5)) if a.smoke else min(a.dev_per_source, len(items))
            dev, train = split_source(items, dev_n, a.seed + number, assigned)
            all_dev.extend(json.dumps(x, ensure_ascii=False) for x in dev)
            all_train.extend(json.dumps(x, ensure_ascii=False) for x in train)
            rec.update(kept=len(items), dev=len(dev), train=len(train), dropped=got["dropped"])
            rec["per_language"] = dict(collections.Counter(x["lang"] for x in items))
            rec["per_question_type"] = dict(collections.Counter(x["q"]["type"] for x in items))
            if got["warnings"]:
                rec["load_warnings"] = got["warnings"]
            print("[%d/%d] %s: loaded %d, kept %d (RSS %d MB, largest worker %d MB)"
                  % ((number + 1, len(selected), key, got["rows_loaded"], len(items)) + _rss_mb()), flush=True)
        except Exception as exc:
            if isinstance(exc, SourceFailed):
                rec["error"], rec["traceback"] = exc.error, exc.trace
            else:
                rec["error"], rec["traceback"] = _error_line(exc), traceback.format_exc(limit=3)
            print("[%d/%d] %s: FAILED %s" % (number + 1, len(selected), key, rec["error"]), flush=True)
        records.append(rec)
    rng = random.Random(a.seed)
    rng.shuffle(all_dev)
    rng.shuffle(all_train)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for line in all_dev + all_train:  # consumer holds out the prefix
            fh.write(line + "\n")
    manifest = {"version": 6, "registry": str(Path("tools/finetune/sources/v6-keep.json")),
                "seed": a.seed, "per_source_cap": cap, "dev_per_source": a.dev_per_source,
                "smoke": a.smoke or None, "eval_texts": len(banned), "items": len(all_dev) + len(all_train),
                "train_items": len(all_train), "dev_items": len(all_dev), "sources": records,
                "sources_ok": sum("error" not in r for r in records),
                "sources_failed": sum("error" in r for r in records),
                "inputs": input_hashes(ROOT)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_report(manifest), encoding="utf-8")
    print("wrote %s: %d items; %d sources OK, %d failed" % (out_path, manifest["items"],
                                                            manifest["sources_ok"], manifest["sources_failed"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
