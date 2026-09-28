#!/usr/bin/env python3
"""Statim against baselines on identical held-out items.

The items are exactly the ones the promotion gate scores: the category suites (bench/eval_categories.py,
same pool, same stratified draw, texts that occur in the training mixture removed) and the first rows
of AG News, DAIR Emotion and Banking77 (bench/eval_accuracy.py, same questions and options). Every
system answers the same question with the same options; accuracy is the share of gold answers.

    python3 bench/baselines.py export --mixture data/mixture-v8.jsonl.gz --out data/baselines/items.jsonl
    python3 bench/baselines.py run statim --url http://127.0.0.1:8098 --items data/baselines/items.jsonl \
        --out data/baselines/statim.jsonl
    python3 bench/baselines.py run ollama --model qwen3:8b --items ... --out data/baselines/qwen3-8b.jsonl
    python3 bench/baselines.py run nli --model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli --items ... --out ...
    python3 bench/baselines.py report --items ... --preds data/baselines/*.jsonl --out docs/BASELINES.md

Baselines are evaluated, never trained on: no output of any system here enters training data.
"""
import argparse
import collections
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CLASSIC = ("ag_news", "emotion", "banking77")


# ------------------------------------------------------------------------------------------ items

def _labels(q):
    """Human-readable option labels in option order (what an LLM or NLI model is shown)."""
    if q["type"] == "noul":
        return ["no", "yes"]
    crit = q["criteria"]
    if q["type"] == "score":
        levels = list(crit.values()) if isinstance(crit, dict) else list(crit)
        return [str(x) for x in levels]
    return [k for k in crit]


def _descriptions(q):
    if q["type"] == "choice" and isinstance(q["criteria"], dict):
        return [v if isinstance(v, str) and v and v.casefold() != k.casefold() else "" for k, v in q["criteria"].items()]
    return [""] * len(_labels(q))


def export(mixture, out, classic_n):
    import eval_categories as ec
    pool, _failures = ec.load_pool(log=lambda m: print(m, file=sys.stderr, flush=True))
    if mixture:
        pool, _fp = ec.exclude_mixture(pool, mixture, log=lambda m: print(m, file=sys.stderr, flush=True))
    tasks, _skipped = ec.make_tasks(pool, list(ec.SUITES), None, ec.DEFAULT_N, ec.DEFAULT_SEED)
    rows = []
    for suite, lang, items in tasks:
        for i, item in enumerate(items):
            item = ec.sort_options(json.loads(json.dumps(item)))
            rows.append({"id": f"categories:{suite}/{lang}#{i}", "suite": f"categories:{suite}", "lang": lang,
                         "state": item["state"], "q": item["q"], "gold": ec.gold_index(item),
                         "gate": ec.gate_note(suite, lang) is None})
    if classic_n:
        import eval_accuracy as ea
        data = ea.suites(classic_n)
        for name in CLASSIC:
            states, questions, qid, keys, gold = data[name]
            q = dict(questions[qid])
            q["criteria"] = {k: (q["criteria"].get(k) if isinstance(q["criteria"], dict) else None) for k in keys}
            for i, (s, g) in enumerate(zip(states, gold)):
                rows.append({"id": f"test/{name}#{i}", "suite": f"test/{name}", "lang": "en", "state": s, "q": q,
                             "gold": g, "gate": True})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} items: {dict(collections.Counter(r['suite'] for r in rows))}", file=sys.stderr)


def load_items(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def render_state(state, limit=6000):
    if isinstance(state, dict):
        text = "\n".join(f"{k}: {v}" for k, v in state.items())
    else:
        text = str(state)
    return text[:limit]


# ------------------------------------------------------------------------------------------ systems

def run_statim(items, url, model, batch=16, head_max_len=512):
    """Batched over the HTTP API, one request per question (Statim's normal mode). head_max_len 512 is
    what the gate uses (eval_categories.py and eval_laya.py), so long option lists are not cut."""
    import eval_categories as ec
    groups = collections.OrderedDict()
    for i, it in enumerate(items):
        groups.setdefault(json.dumps(it["q"], sort_keys=True), []).append(i)
    out = [None] * len(items)
    for idxs in groups.values():
        q = items[idxs[0]]["q"]
        for start in range(0, len(idxs), batch):
            chunk = idxs[start:start + batch]
            states = [items[i]["state"][:ec.STATE_CHARS] if isinstance(items[i]["state"], str) else items[i]["state"]
                      for i in chunk]
            body = {"states": states, "questions": {"q": q}, "ensemble": 1, "model": model, "head_max_len": head_max_len}
            t0 = time.time()
            req = urllib.request.Request(url + "/v1/systemone/batch", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            res = json.load(urllib.request.urlopen(req, timeout=600))
            ms = (time.time() - t0) * 1000 / len(chunk)
            for i, r in zip(chunk, res["results"]):
                p = ec.probabilities(r["answers"]["q"], {"q": q})
                out[i] = {"id": items[i]["id"], "pred": max(range(len(p)), key=p.__getitem__), "ms": round(ms, 2)}
    return out


def _prompt(item):
    q = item["q"]
    labels, descs = _labels(q), _descriptions(q)
    opts = "\n".join(f"- {lab}" + (f": {d}" if d else "") for lab, d in zip(labels, descs))
    kind = {"noul": "Answer yes or no.", "score": "Pick the level that fits best.",
            "choice": "Pick the option that fits best."}[q["type"]]
    return (f"Text:\n{render_state(item['state'])}\n\nQuestion: {q['instructions']}\n{kind}\nOptions:\n{opts}\n\n"
            "Reply with JSON: {\"answer\": \"<option>\"}.")


def run_ollama(items, url, model, workers=4, done=None, sink=None):
    """sink: a file every new answer is appended to right away, so an interrupted run resumes."""
    done = done or {}
    lock = threading.Lock()

    def one(item):
        if item["id"] in done:
            return done[item["id"]]
        labels = _labels(item["q"])
        body = {"model": model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": "You are a precise classifier. Answer only with the requested JSON."},
                             {"role": "user", "content": _prompt(item)}],
                "format": {"type": "object", "properties": {"answer": {"type": "string", "enum": labels}},
                           "required": ["answer"]},
                "options": {"temperature": 0, "num_predict": 64, "seed": 0, "num_ctx": 4096}}
        t0 = time.time()
        req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            res = json.load(urllib.request.urlopen(req, timeout=300))
            ans = json.loads(res["message"]["content"]).get("answer")
            pred = labels.index(ans) if ans in labels else None
        except Exception as exc:  # a failed call counts as a wrong answer, and is reported
            pred, ans = None, "error: %s" % type(exc).__name__
        rec = {"id": item["id"], "pred": pred, "ms": round((time.time() - t0) * 1000, 1)}
        if pred is None:
            rec["raw"] = str(ans)[:120]
        if sink is not None:
            with lock:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
        return rec

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(one, items)):
            out.append(rec)
            if i % 250 == 0:
                with lock:
                    print(f"{i}/{len(items)}", file=sys.stderr, flush=True)
    return out


def run_nli(items, model, device=0, batch=16):
    """Zero-shot classification by entailment (the standard Hugging Face pipeline)."""
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=model, device=device)
    out = []
    for i, item in enumerate(items):
        labels = _labels(item["q"])
        template = item["q"]["instructions"].rstrip() + " The answer is {}."
        t0 = time.time()
        res = clf(render_state(item["state"], 2000), candidate_labels=labels, hypothesis_template=template,
                  multi_label=False)
        pred = labels.index(res["labels"][0])
        out.append({"id": item["id"], "pred": pred, "ms": round((time.time() - t0) * 1000, 1)})
        if i % 500 == 0:
            print(f"{i}/{len(items)}", file=sys.stderr, flush=True)
    return out


# ------------------------------------------------------------------------------------------ report

def report(items, pred_files, out):
    gold = {it["id"]: it for it in items}
    systems = []
    for path in pred_files:
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        meta = recs[0].get("meta", {}) if recs and "meta" in recs[0] else {}
        preds = {r["id"]: r for r in recs if "id" in r}
        systems.append((meta.get("name") or os.path.splitext(os.path.basename(path))[0], preds, meta))
    cells = collections.OrderedDict()
    for it in items:
        cells.setdefault((it["suite"], it["lang"]), []).append(it)

    def acc(preds, its):
        return sum(preds.get(i["id"], {}).get("pred") == i["gold"] for i in its) / len(its)

    lines = ["| Suite | " + " | ".join(s[0] for s in systems) + " |", "|---" * (len(systems) + 1) + "|"]
    by_cat = collections.defaultdict(lambda: collections.defaultdict(list))
    for (suite, lang), its in cells.items():
        accs = [acc(p, its) for _, p, _ in systems]
        best = max(accs)
        name = suite.replace("categories:", "") + (f" ({lang})" if suite.startswith("categories:") else "")
        lines.append(f"| {name} | " + " | ".join(f"**{a:.3f}**" if a == best else f"{a:.3f}" for a in accs) + " |")
        if suite.startswith("categories:") and all(i["gate"] for i in its):
            for (n, _, _), a in zip(systems, accs):
                by_cat[suite][n].append(a)
    macro = ["| Category (macro over languages) | " + " | ".join(s[0] for s in systems) + " |",
             "|---" * (len(systems) + 1) + "|"]
    totals = collections.defaultdict(list)
    for suite, per in by_cat.items():
        vals = [sum(per[n]) / len(per[n]) for n, _, _ in systems]
        best = max(vals)
        macro.append(f"| {suite.replace('categories:', '')} | " + " | ".join(
            f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in vals) + " |")
        for (n, _, _), v in zip(systems, vals):
            totals[n].append(v)
    macro.append("| **mean of categories** | " + " | ".join(
        f"**{sum(totals[n]) / len(totals[n]):.3f}**" for n, _, _ in systems) + " |")
    speed = ["| System | Parameters | Hardware | ms per decision | decisions per second |", "|---|---|---|---:|---:|"]
    for n, p, meta in systems:
        ms = [r["ms"] for r in p.values() if "ms" in r]
        median = sorted(ms)[len(ms) // 2]
        wall = meta.get("wall_seconds")
        if meta.get("resumed") or not wall:  # a resumed run's wall time covers only its last part
            dps = f"≈{(meta.get('workers') or 1) * 1000 / median:.1f}" if median else ""
        else:
            dps = f"{len(ms) / wall:.1f}"
        speed.append(f"| {n} | {meta.get('params', '')} | {meta.get('hardware', '')} | "
                     f"{median:.1f} | {dps} |")
    text = "\n".join(["## Decision categories", "", *macro, "", "## Every suite", "", *lines, "",
                      "## Speed on the same machine", "", *speed, ""])
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--mixture", default=None, help="training mixture whose texts are removed from the suites")
    e.add_argument("--classic-n", type=int, default=2000)
    e.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("system", choices=["statim", "ollama", "nli"])
    r.add_argument("--items", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--name", default=None, help="column name in the report")
    r.add_argument("--url", default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--params", default="", help="parameter count shown in the report")
    r.add_argument("--hardware", default="", help="hardware shown in the report")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--head-max-len", type=int, default=512, help="Statim only; the gate uses 512")
    rp = sub.add_parser("report")
    rp.add_argument("--items", required=True)
    rp.add_argument("--preds", nargs="+", required=True)
    rp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.mixture, a.out, a.classic_n)
    elif a.cmd == "report":
        report(load_items(a.items), a.preds, a.out)
    else:
        items = load_items(a.items)
        done, part = {}, a.out + ".part"
        if a.system == "ollama":  # resume: keep answered items from an interrupted or earlier run
            for path in (a.out, part):
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        done.update({r["id"]: r for r in map(json.loads, f) if "id" in r and r.get("pred") is not None})
            if done:
                print(f"resuming: {len(done)} answers kept", file=sys.stderr)
        t0 = time.time()
        if a.system == "statim":
            recs = run_statim(items, a.url or "http://127.0.0.1:8098", a.model or "multilingual", head_max_len=a.head_max_len)
        elif a.system == "ollama":
            with open(part, "a", encoding="utf-8") as sink:
                recs = run_ollama(items, a.url or "http://127.0.0.1:11434", a.model or "qwen3:8b", a.workers, done, sink)
        else:
            recs = run_nli(items, a.model or "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        meta = {"name": a.name or a.system, "model": a.model, "params": a.params, "hardware": a.hardware,
                "wall_seconds": round(time.time() - t0, 1), "items": len(items),
                "workers": a.workers if a.system == "ollama" else None, "resumed": len(done)}
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": meta}) + "\n")
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if os.path.exists(part):
            os.remove(part)  # the final file holds every answer now
        correct = sum(r["pred"] == it["gold"] for r, it in zip(recs, items))
        print(f"{meta['name']}: {correct}/{len(items)} correct in {meta['wall_seconds']} s", file=sys.stderr)


if __name__ == "__main__":
    main()
