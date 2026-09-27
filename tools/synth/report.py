#!/usr/bin/env python3
"""Quality and diversity report for a synthetic Statim generation run.

The core report uses only the Python standard library.  Semantic diversity is
enabled only when sentence-transformers and multilingual-e5-small are already
available locally; it is deliberately forced into offline, CPU-only mode.
"""

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
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path


SEED = 20260927
MIN_DISTRIBUTION = 20
SEMANTIC_MODEL = "intfloat/multilingual-e5-small"
SEED_ATTRIBUTES = ("domain", "industry", "register", "difficulty", "aspect", "topic", "form")


def read_jsonl_gz(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit("%s:%d: invalid JSON: %s" % (path, line_no, exc))
    return rows


def companion_paths(items_path):
    name = items_path.name
    if not name.endswith(".jsonl.gz"):
        raise SystemExit("input must end in .jsonl.gz: %s" % items_path)
    base = name[:-len(".jsonl.gz")]
    parent = items_path.parent
    return parent / (base + ".provenance.jsonl.gz"), parent / (base + ".manifest.json"), base


def task_lang(item):
    parts = str(item.get("src", "")).split("/")
    return (parts[1] if len(parts) > 1 else "?", parts[2] if len(parts) > 2 else "?")


def pct(value):
    return "n/a" if value is None else "%.1f%%" % (100.0 * value)


def rate(accepted, attempts):
    return accepted / attempts if attempts else None


def argmax(values):
    return max(range(len(values)), key=lambda i: values[i]) if values else None


def histogram_position(items):
    bins = [0] * 5
    for item in items:
        target = item.get("target") or []
        if not target:
            continue
        index = argmax(target)
        position = index / (len(target) - 1) if len(target) > 1 else 0.0
        bins[min(4, int(position * 5))] += 1
    return bins


def safe_name(value):
    return str(value) if value is not None else "(missing)"


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, section, name, warn, detail):
        # A check that could not run (too few items, missing dependency) is SKIP, never PASS:
        # silence from a check that saw no data must not read as a green light.
        skipped = not warn and str(detail).startswith(("too few", "skipped"))
        self.rows.append({"section": section, "name": name,
                          "status": "WARN" if warn else "SKIP" if skipped else "PASS", "detail": detail})

    @property
    def warnings(self):
        return [row for row in self.rows if row["status"] == "WARN"]


def breakdown(provenance, field):
    groups = collections.defaultdict(lambda: [0, 0])
    for rec in provenance:
        seed = rec.get("seed") or {}
        value = seed.get(field)
        if value is None:
            continue
        cell = groups[safe_name(value)]
        cell[0] += 1
        cell[1] += rec.get("status") == "accepted"
    return {key: {"attempts": val[0], "accepted": val[1],
                  "acceptance_rate": rate(val[1], val[0])}
            for key, val in sorted(groups.items())}


def yield_report(items, provenance, checks):
    accepted = sum(rec.get("status") == "accepted" for rec in provenance)
    attempts = len(provenance)
    overall_rate = rate(accepted, attempts)
    checks.add("yield", "acceptance rate", overall_rate is None or overall_rate < 0.30,
               "%s (%d/%d); scale threshold is 30%%" % (pct(overall_rate), accepted, attempts))
    attrs = {}
    for attr in SEED_ATTRIBUTES:
        if any(attr in (rec.get("seed") or {}) for rec in provenance):
            attrs[attr] = breakdown(provenance, attr)
    reasons = collections.Counter()
    statuses = collections.Counter()
    for rec in provenance:
        statuses[safe_name(rec.get("status"))] += 1
        if rec.get("status") != "accepted":
            reasons[safe_name(rec.get("reason") or rec.get("status"))] += 1
    return {
        "attempts": attempts,
        "accepted": accepted,
        "output_items": len(items),
        "acceptance_rate": overall_rate,
        "statuses": dict(sorted(statuses.items())),
        "by_task": breakdown(provenance, "task"),
        "by_language": breakdown(provenance, "lang"),
        "by_seed_attribute": attrs,
        "rejection_reasons": dict(reasons.most_common()),
    }


def gammaincc(a, x):
    """Regularized upper incomplete gamma Q(a, x), without scipy."""
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        total = term = 1.0 / a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1001):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - gln) * h))


def chi_square_uniform(position_counts):
    total = sum(position_counts)
    n_options = len(position_counts)
    expected = total / n_options if n_options else 0
    if not total or expected < 5:
        return {"n": total, "counts": position_counts, "chi_square": None,
                "p_value": None, "note": "too few items (expected count < 5 per position)"}
    statistic = sum((observed - expected) ** 2 / expected for observed in position_counts)
    p_value = gammaincc((n_options - 1) / 2.0, statistic / 2.0)
    return {"n": total, "counts": position_counts, "chi_square": statistic,
            "p_value": p_value, "note": None}


def label_balance(items, checks):
    choice = [x for x in items if x.get("q", {}).get("type") == "choice"]
    noul = [x for x in items if x.get("q", {}).get("type") == "noul"]
    score = [x for x in items if x.get("q", {}).get("type") == "score"]
    choice_bins = histogram_position(choice)
    score_bins = histogram_position(score)

    by_count = collections.defaultdict(list)
    for item in choice:
        by_count[len(item.get("target") or [])].append(item)
    chi = {}
    chi_warns = []
    for count, group in sorted(by_count.items()):
        positions = [0] * count
        for item in group:
            positions[argmax(item["target"])] += 1
        result = chi_square_uniform(positions)
        chi[str(count)] = result
        if result["p_value"] is not None and result["p_value"] < 0.01:
            chi_warns.append("%d options p=%.4g" % (count, result["p_value"]))
    checks.add("label_balance", "choice gold chi-square", bool(chi_warns),
               "; ".join(chi_warns) if chi_warns else
               ("too few items for valid tests" if not any(v["p_value"] is not None for v in chi.values())
                else "no option-count group has p < 0.01"))

    true_count = sum(bool(x.get("target", [0, 0])[1] == 1) for x in noul)
    true_share = rate(true_count, len(noul))
    noul_enough = len(noul) >= MIN_DISTRIBUTION
    noul_warn = noul_enough and not (0.40 <= true_share <= 0.60)
    checks.add("label_balance", "noul true share", noul_warn,
               ("%s (%d/%d)" % (pct(true_share), true_count, len(noul))) if noul_enough
               else "too few items (%d; need %d)" % (len(noul), MIN_DISTRIBUTION))

    score_total = sum(score_bins)
    score_shares = [x / score_total for x in score_bins] if score_total else [0.0] * 5
    score_enough = score_total >= MIN_DISTRIBUTION
    score_warn = score_enough and (any(x > 0.40 for x in score_shares) or score_shares[2] > 0.50)
    checks.add("label_balance", "score gold position", score_warn,
               ("bins " + ", ".join(pct(x) for x in score_shares)) if score_enough
               else "too few items (%d; need %d)" % (score_total, MIN_DISTRIBUTION))

    option_counts = collections.Counter(len(x.get("target") or []) for x in items)
    return {
        "choice": {"n": len(choice), "gold_fraction_bins": choice_bins,
                   "bin_edges": [0, .2, .4, .6, .8, 1.0], "chi_square_by_option_count": chi},
        "noul": {"n": len(noul), "true": true_count, "true_share": true_share},
        "score": {"n": len(score), "gold_fraction_bins": score_bins,
                  "gold_fraction_shares": score_shares, "bin_edges": [0, .2, .4, .6, .8, 1.0]},
        "option_count_histogram": {str(k): v for k, v in sorted(option_counts.items())},
    }


def tokens(text, lang):
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    if lang in {"ja", "zh"}:
        return [ch for ch in text if unicodedata.category(ch)[0] in {"L", "N"}]
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def distinct_n(texts, lang, n):
    unique = set()
    total = 0
    for text in texts:
        seq = tokens(text, lang)
        grams = [tuple(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1))]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else None


def distinct_n_items(items, field, n):
    unique = set()
    total = 0
    for item in items:
        lang = task_lang(item)[1]
        text = item.get("state", "") if field == "state" else item.get("q", {}).get("instructions", "")
        seq = tokens(text, lang)
        grams = [tuple(seq[i:i + n]) for i in range(max(0, len(seq) - n + 1))]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else None


def prefix_stats(texts, lang):
    counts = collections.Counter()
    for value in texts:
        seq = tokens(value, lang)
        if seq:
            prefix = "".join(seq[:3]) if lang in {"ja", "zh"} else " ".join(seq[:3])
            counts[prefix] += 1
    if not counts:
        return {"prefix": None, "count": 0, "share": None}
    prefix, count = counts.most_common(1)[0]
    return {"prefix": prefix, "count": count, "share": count / len(texts) if texts else None}


def prefix_stats_items(items, field):
    counts = collections.Counter()
    for item in items:
        lang = task_lang(item)[1]
        value = item.get("state", "") if field == "state" else item.get("q", {}).get("instructions", "")
        seq = tokens(value, lang)
        if seq:
            prefix = "".join(seq[:3]) if lang in {"ja", "zh"} else " ".join(seq[:3])
            counts[prefix] += 1
    if not counts:
        return {"prefix": None, "count": 0, "share": None}
    prefix, count = counts.most_common(1)[0]
    return {"prefix": prefix, "count": count, "share": count / len(items) if items else None}


def char_ngrams(text, n=5):
    value = " ".join(str(text or "").casefold().split())
    if not value:
        return set()
    if len(value) <= n:
        return {value}
    return {value[i:i + n] for i in range(len(value) - n + 1)}


def jaccard(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def sampled_pairwise(grams, seed_text, limit=2000):
    n = len(grams)
    total_pairs = n * (n - 1) // 2
    if not total_pairs:
        return None, 0
    if total_pairs <= limit:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        digest = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(SEED ^ digest)
        selected = set()
        while len(selected) < limit:
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            selected.add((min(i, j), max(i, j)))
        pairs = list(selected)
    return sum(jaccard(grams[i], grams[j]) for i, j in pairs) / len(pairs), len(pairs)


def repetition_group(items, lang=None, detect_near=True):
    states = [x.get("state", "") for x in items]
    instructions = [x.get("q", {}).get("instructions", "") for x in items]
    grams = [char_ngrams(x) for x in states]
    duplicate_count = 0
    duplicate_examples = []
    participating = set()
    if detect_near:
        for i in range(len(grams)):
            for j in range(i + 1, len(grams)):
                # Jaccard >= .6 is impossible when the set sizes differ by > 1/.6.
                small, large = sorted((len(grams[i]), len(grams[j])))
                if large and small / large < 0.6:
                    continue
                similarity = jaccard(grams[i], grams[j])
                if similarity >= 0.6:
                    duplicate_count += 1
                    if len(duplicate_examples) < 20:
                        duplicate_examples.append({"left": i, "right": j, "jaccard": similarity})
                    participating.update((i, j))
    mean_j, sampled = sampled_pairwise(grams, lang or "overall")
    if lang is None:
        state_distinct = {"distinct_%d" % n: distinct_n_items(items, "state", n) for n in (1, 2, 3)}
        instruction_distinct = {"distinct_%d" % n: distinct_n_items(items, "instructions", n) for n in (1, 2, 3)}
        state_prefix = prefix_stats_items(items, "state")
        instruction_prefix = prefix_stats_items(items, "instructions")
    else:
        state_distinct = {"distinct_%d" % n: distinct_n(states, lang, n) for n in (1, 2, 3)}
        instruction_distinct = {"distinct_%d" % n: distinct_n(instructions, lang, n) for n in (1, 2, 3)}
        state_prefix = prefix_stats(states, lang)
        instruction_prefix = prefix_stats(instructions, lang)
    return {
        "n": len(items),
        "states": dict(state_distinct, top_prefix=state_prefix),
        "instructions": dict(instruction_distinct, top_prefix=instruction_prefix),
        "near_duplicate_pairs": duplicate_count,
        "near_duplicate_examples": duplicate_examples,
        "near_duplicate_items": len(participating),
        "near_duplicate_item_share": len(participating) / len(items) if items else None,
        "mean_pairwise_jaccard": mean_j,
        "mean_pairwise_sample_pairs": sampled,
    }


def repetition_report(items, checks):
    by_lang = collections.defaultdict(list)
    for item in items:
        by_lang[task_lang(item)[1]].append(item)
    languages = {}
    prefix_warns = []
    duplicate_warns = []
    for lang, group in sorted(by_lang.items()):
        result = repetition_group(group, lang)
        languages[lang] = result
        if len(group) >= MIN_DISTRIBUTION:
            for field in ("states", "instructions"):
                top = result[field]["top_prefix"]
                if top["share"] is not None and top["share"] > 0.05:
                    prefix_warns.append("%s %s %r=%s" % (lang, field, top["prefix"], pct(top["share"])))
            if result["near_duplicate_item_share"] > 0.02:
                duplicate_warns.append("%s=%s" % (lang, pct(result["near_duplicate_item_share"])))
    overall = repetition_group(items, detect_near=False)
    # Near-duplicate detection is intentionally language-local.  The overall
    # figures aggregate those disjoint language groups rather than comparing
    # unrelated scripts and translations with each other.
    overall["near_duplicate_pairs"] = sum(x["near_duplicate_pairs"] for x in languages.values())
    overall["near_duplicate_items"] = sum(x["near_duplicate_items"] for x in languages.values())
    overall["near_duplicate_item_share"] = (overall["near_duplicate_items"] / len(items)) if items else None
    overall["near_duplicate_examples"] = []
    checks.add("repetition", "first-three-token prefixes", bool(prefix_warns),
               "; ".join(prefix_warns) if prefix_warns else "none exceeds 5% in a language with at least 20 items")
    checks.add("repetition", "near-duplicate states", bool(duplicate_warns),
               "; ".join(duplicate_warns) if duplicate_warns else "no language has >2% of items in a pair")

    key_counts = collections.Counter()
    choice_count = 0
    for item in items:
        q = item.get("q") or {}
        if q.get("type") == "choice" and isinstance(q.get("criteria"), dict):
            choice_count += 1
            key_counts.update(str(key) for key in q["criteria"])
    frequent = [{"key": key, "count": count, "share": count / choice_count if choice_count else None}
                for key, count in key_counts.most_common(20)]
    top_share = frequent[0]["share"] if frequent else None
    key_enough = choice_count >= MIN_DISTRIBUTION
    checks.add("repetition", "option-key reuse", bool(key_enough and top_share > 0.25),
               ("top key %r appears in %s of choice items" % (frequent[0]["key"], pct(top_share)))
               if key_enough and frequent else "too few choice items (%d; need %d)" % (choice_count, MIN_DISTRIBUTION))
    return {"overall": overall, "by_language": languages,
            "option_keys": {"choice_items": choice_count, "most_frequent": frequent}}


def normalize_vectors(raw):
    out = []
    for vector in raw:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(x * x for x in values))
        out.append([x / norm for x in values] if norm else values)
    return out


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


def jacobi_eigenvalues(matrix):
    """Eigenvalues of a real symmetric matrix via cyclic Jacobi rotations."""
    a = [row[:] for row in matrix]
    n = len(a)
    if n < 2:
        return [a[0][0]] if n else []
    max_sweeps = 50
    for _ in range(max_sweeps):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= 1e-12 * (1.0 + abs(a[p][p]) + abs(a[q][q])):
                    continue
                changed = True
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                app, aqq = a[p][p], a[q][q]
                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = a[q][p] = 0.0
                for k in range(n):
                    if k in (p, q):
                        continue
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = a[p][k] = c * akp - s * akq
                    a[k][q] = a[q][k] = s * akp + c * akq
        if not changed:
            break
    return [a[i][i] for i in range(n)]


def semantic_group(vectors):
    n = len(vectors)
    if n < 2:
        return {"n": n, "nearest_neighbor_ge_0_92": None, "nearest_neighbor_share": None,
                "vendi": float(n), "vendi_per_item": 1.0 if n else None, "note": "too few items"}
    kernel = [[0.0] * n for _ in range(n)]
    nearest = [-1.0] * n
    for i in range(n):
        kernel[i][i] = 1.0 / n
        for j in range(i + 1, n):
            value = cosine(vectors[i], vectors[j])
            nearest[i] = max(nearest[i], value)
            nearest[j] = max(nearest[j], value)
            kernel[i][j] = kernel[j][i] = value / n
    high = sum(value >= 0.92 for value in nearest)
    eigenvalues = [max(0.0, x) for x in jacobi_eigenvalues(kernel)]
    total = sum(eigenvalues)
    eigenvalues = [x / total for x in eigenvalues] if total else []
    entropy = -sum(x * math.log(x) for x in eigenvalues if x > 1e-15)
    vendi = math.exp(entropy)
    return {"n": n, "nearest_neighbor_ge_0_92": high,
            "nearest_neighbor_share": high / n, "vendi": vendi,
            "vendi_per_item": vendi / n, "note": None}


def _e5_mean_pooled(texts, batch_size=32):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SEMANTIC_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(SEMANTIC_MODEL, local_files_only=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.extend(((hidden * mask).sum(1) / mask.sum(1)).tolist())
    return out


def semantic_report(items, checks):
    # These settings must precede importing sentence_transformers.  Together
    # with local_files_only they make a cache miss a clean skip, never a download.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    texts = ["query: " + str(x.get("state", "")) for x in items]
    try:
        try:
            from sentence_transformers import SentenceTransformer
        except (ImportError, ModuleNotFoundError):
            raw = _e5_mean_pooled(texts)  # plain transformers: e5 embeddings are mean-pooled
        else:
            try:
                model = SentenceTransformer(SEMANTIC_MODEL, device="cpu", local_files_only=True)
            except TypeError:
                # Older releases do not expose local_files_only, but offline mode
                # above still forbids network access.
                model = SentenceTransformer(SEMANTIC_MODEL, device="cpu")
            raw = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=False)
        vectors = normalize_vectors(raw)
    except Exception as exc:  # cache misses vary by transformers/huggingface_hub version
        reason = "model not available locally: %s" % str(exc).splitlines()[0]
        checks.add("semantic", "semantic diversity", False, "skipped: " + reason)
        return {"available": False, "reason": reason, "model": SEMANTIC_MODEL, "by_language": {}}
    by_lang_indices = collections.defaultdict(list)
    for index, item in enumerate(items):
        by_lang_indices[task_lang(item)[1]].append(index)
    results = {}
    warns = []
    for lang, indices in sorted(by_lang_indices.items()):
        result = semantic_group([vectors[i] for i in indices])
        results[lang] = result
        if len(indices) >= MIN_DISTRIBUTION and result["nearest_neighbor_share"] > 0.03:
            warns.append("%s=%s" % (lang, pct(result["nearest_neighbor_share"])))
    checks.add("semantic", "semantic nearest neighbours", bool(warns),
               "; ".join(warns) if warns else "no language with at least 20 items exceeds 3%")
    return {"available": True, "reason": None, "model": SEMANTIC_MODEL, "by_language": results}


def statim_endpoint(base):
    value = base.rstrip("/")
    return value if value.endswith("/v1/systemone") else value + "/v1/systemone"


def post_json(url, payload, timeout=60):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def difficulty_report(items, url, checks):
    if not url:
        checks.add("difficulty", "Statim difficulty", False, "skipped: --statim not supplied")
        return {"available": False, "reason": "--statim not supplied", "by_task": {},
                "items": [], "confidently_wrong": []}
    endpoint = statim_endpoint(url)
    rows = []
    try:
        for index, item in enumerate(items):
            response = post_json(endpoint, {"state": item.get("state"), "questions": {"q": item.get("q")}})
            answer = response["answers"]["q"]
            kind = item["q"]["type"]
            target_index = argmax(item["target"])
            if kind == "choice":
                probabilities = answer.get("probabilities") or {}
                predicted = max(probabilities, key=probabilities.get) if probabilities else answer.get("choice")
                gold = list(item["q"]["criteria"])[target_index]
                correct = str(predicted) == str(gold)
            elif kind == "noul":
                predicted = float(answer["noul"]) >= 0.5
                gold = item["target"][1] == 1
                correct = predicted == gold
            else:
                probabilities = answer.get("probabilities") or {}
                predicted = int(max(probabilities, key=probabilities.get))
                gold = target_index
                correct = predicted == gold
            confidence = float(answer["answer_confidence"])
            task, lang = task_lang(item)
            rows.append({"index": index, "task": task, "language": lang, "type": kind,
                         "correct": correct, "answer_confidence": confidence,
                         "predicted": predicted, "gold": gold})
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
        reason = "%s: %s" % (type(exc).__name__, exc)
        checks.add("difficulty", "Statim difficulty", True, "request failed: " + reason)
        return {"available": False, "reason": reason, "endpoint": endpoint,
                "completed": len(rows), "items": rows, "by_task": {}, "confidently_wrong": []}

    by_task_rows = collections.defaultdict(list)
    for row in rows:
        by_task_rows[row["task"]].append(row)
    task_results = {}
    trivial_warns = []
    for task, group in sorted(by_task_rows.items()):
        trivial = sum(x["correct"] and x["answer_confidence"] >= 0.9 for x in group)
        result = {"n": len(group), "trivial": trivial, "trivial_share": trivial / len(group),
                  "learnable": len(group) - trivial, "learnable_share": (len(group) - trivial) / len(group)}
        task_results[task] = result
        if len(group) >= MIN_DISTRIBUTION and result["trivial_share"] > 0.80:
            trivial_warns.append("%s=%s" % (task, pct(result["trivial_share"])))
    checks.add("difficulty", "trivial share", bool(trivial_warns),
               "; ".join(trivial_warns) if trivial_warns else "no task with at least 20 items exceeds 80%")
    confident_wrong = [x for x in rows if not x["correct"] and x["answer_confidence"] >= 0.9]
    for row in confident_wrong:
        row["state"] = items[row["index"]].get("state", "")
    return {"available": True, "reason": None, "endpoint": endpoint,
            "items": rows, "by_task": task_results, "confidently_wrong": confident_wrong}


def item_key(item):
    return json.dumps({k: item.get(k) for k in ("state", "q", "target", "src")},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rationale_map(provenance):
    values = collections.defaultdict(list)
    for rec in provenance:
        item = rec.get("item")
        if rec.get("status") == "accepted" and isinstance(item, dict):
            values[item_key(item)].append(rec.get("rationale") or "(no rationale recorded)")
    return values


def render_item(item, rationale, ordinal):
    q = item.get("q") or {}
    target = item.get("target") or []
    gold = argmax(target)
    lines = ["### %d. `%s`" % (ordinal, item.get("src", "?")), "", "**State**", "",
             str(item.get("state", "")), "", "**Question**", "", str(q.get("instructions", "")), "",
             "**Options**", ""]
    criteria = q.get("criteria")
    if isinstance(criteria, dict):
        for index, (key, description) in enumerate(criteria.items()):
            mark = " **← GOLD**" if index == gold else ""
            lines.append("- `%s`: %s%s" % (key, description, mark))
    elif isinstance(criteria, list):
        for index, description in enumerate(criteria):
            mark = " **← GOLD**" if index == gold else ""
            lines.append("- %d: %s%s" % (index, description, mark))
    lines.extend(["", "**Provenance rationale:** %s" % rationale, "",
                  "[ ] correct  [ ] wrong gold  [ ] unclear  [ ] bad language", ""])
    return lines


def write_sample(path, items, provenance, n):
    rationales = rationale_map(provenance)
    groups = collections.defaultdict(list)
    for item in items:
        task, lang = task_lang(item)
        groups[(lang, task)].append(item)
    rng = random.Random(SEED)
    selected = []
    for group_key, group in sorted(groups.items()):
        picks = rng.sample(group, min(n, len(group)))
        selected.extend((group_key, item) for item in picks)
    lines = ["# Human spot check", "", "Seed: `%d`. Up to %d random items per language and task." % (SEED, n), ""]
    last = None
    for ordinal, ((lang, task), item) in enumerate(selected, 1):
        if (lang, task) != last:
            lines.extend(["## %s / %s" % (lang, task), ""])
            last = (lang, task)
        candidates = rationales.get(item_key(item), [])
        rationale = candidates.pop(0) if candidates else "(accepted provenance record not found)"
        lines.extend(render_item(item, rationale, ordinal))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return len(selected)


def md_table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(map(clean, headers)) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(clean(x) for x in row) + " |" for row in rows)
    return out


def yield_rows(mapping):
    return [[key, val["attempts"], val["accepted"], pct(val["acceptance_rate"])]
            for key, val in mapping.items()]


def write_markdown(path, report):
    y = report["yield"]
    lb = report["label_balance"]
    rep = report["repetition"]
    sem = report["semantic_diversity"]
    diff = report["difficulty"]
    lines = ["# Synthetic generation quality report", "", "Input: `%s`" % report["input"], "",
             "## Verdict: %s" % report["verdict"], ""]
    if report["verdict_reasons"]:
        lines.extend(["Reasons:", ""] + ["- %s" % x for x in report["verdict_reasons"]] + [""])
    lines.extend(["## Check summary", ""])
    lines.extend(md_table(["Section", "Check", "Status", "Detail"],
                          [[x["section"], x["name"], x["status"], x["detail"]] for x in report["checks"]]))
    lines.extend(["", "## 1. Yield", "",
                  "Attempts: **%d** · accepted: **%d** · output items: **%d** · acceptance: **%s**" %
                  (y["attempts"], y["accepted"], y["output_items"], pct(y["acceptance_rate"])), "",
                  "### By task", ""])
    lines.extend(md_table(["Task", "Attempts", "Accepted", "Rate"], yield_rows(y["by_task"])))
    lines.extend(["", "### By language", ""])
    lines.extend(md_table(["Language", "Attempts", "Accepted", "Rate"], yield_rows(y["by_language"])))
    for attr, mapping in y["by_seed_attribute"].items():
        lines.extend(["", "### Seed attribute: %s" % attr, ""])
        lines.extend(md_table([attr, "Attempts", "Accepted", "Rate"], yield_rows(mapping)))
    lines.extend(["", "### Rejection reasons", ""])
    lines.extend(md_table(["Reason", "Count"], list(y["rejection_reasons"].items()) or [["(none)", 0]]))

    lines.extend(["", "## 2. Label balance", "",
                  "Position bins are `[0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1]` over `index/(n-1)`.", "",
                  "- Choice gold bins: `%s`" % lb["choice"]["gold_fraction_bins"],
                  "- Noul true share: %s (%d/%d)" % (pct(lb["noul"]["true_share"]), lb["noul"]["true"], lb["noul"]["n"]),
                  "- Score gold bins: `%s`" % lb["score"]["gold_fraction_bins"],
                  "- Option counts: `%s`" % json.dumps(lb["option_count_histogram"], sort_keys=True), "",
                  "### Choice chi-square tests", ""])
    chi_rows = []
    for count, value in lb["choice"]["chi_square_by_option_count"].items():
        chi_rows.append([count, value["n"], value["counts"],
                         "n/a" if value["chi_square"] is None else "%.4f" % value["chi_square"],
                         "n/a" if value["p_value"] is None else "%.6g" % value["p_value"], value["note"] or ""])
    lines.extend(md_table(["Options", "N", "Positions", "χ²", "p", "Note"], chi_rows or [["-", 0, "[]", "n/a", "n/a", "no choice items"]]))

    lines.extend(["", "## 3. Repetition and template collapse", ""])
    repetition_rows = []
    for lang, value in [("overall", rep["overall"])] + list(rep["by_language"].items()):
        repetition_rows.append([lang, value["n"],
                                "/".join("n/a" if value["states"]["distinct_%d" % n] is None else "%.3f" % value["states"]["distinct_%d" % n] for n in (1, 2, 3)),
                                "/".join("n/a" if value["instructions"]["distinct_%d" % n] is None else "%.3f" % value["instructions"]["distinct_%d" % n] for n in (1, 2, 3)),
                                "%r (%s)" % (value["states"]["top_prefix"]["prefix"], pct(value["states"]["top_prefix"]["share"])),
                                "%r (%s)" % (value["instructions"]["top_prefix"]["prefix"], pct(value["instructions"]["top_prefix"]["share"])),
                                "%d / %s" % (value["near_duplicate_pairs"], pct(value["near_duplicate_item_share"])),
                                "n/a" if value["mean_pairwise_jaccard"] is None else "%.4f" % value["mean_pairwise_jaccard"]])
    lines.extend(md_table(["Language", "N", "State d1/d2/d3", "Instruction d1/d2/d3",
                           "Top state prefix", "Top instruction prefix", "Near pairs / item share", "Mean Jaccard"], repetition_rows))
    lines.extend(["", "### Frequent choice option keys", ""])
    lines.extend(md_table(["Key", "Count", "Share of choice items"],
                          [[x["key"], x["count"], pct(x["share"])] for x in rep["option_keys"]["most_frequent"]] or [["(none)", 0, "n/a"]]))

    lines.extend(["", "## 4. Semantic diversity", ""])
    if not sem["available"]:
        lines.extend(["Skipped: %s" % sem["reason"], ""])
    else:
        lines.extend(md_table(["Language", "N", "NN ≥ .92", "Share", "Vendi", "Vendi / n"],
                              [[lang, x["n"], x["nearest_neighbor_ge_0_92"], pct(x["nearest_neighbor_share"]),
                                "%.3f" % x["vendi"], "%.3f" % x["vendi_per_item"]]
                               for lang, x in sem["by_language"].items()]))

    lines.extend(["", "## 5. Difficulty", ""])
    if not diff["available"]:
        lines.extend(["Skipped: %s" % diff["reason"], ""])
    else:
        lines.extend(md_table(["Task", "N", "Trivial", "Trivial share", "Learnable", "Learnable share"],
                              [[task, x["n"], x["trivial"], pct(x["trivial_share"]), x["learnable"], pct(x["learnable_share"])]
                               for task, x in diff["by_task"].items()]))
        lines.extend(["", "### Confidently wrong (manual review candidates)", ""])
        if diff["confidently_wrong"]:
            for row in diff["confidently_wrong"]:
                lines.append("- Item %d `%s/%s`, confidence %.3f, predicted `%s`, gold `%s`: %s" %
                             (row["index"], row["task"], row["language"], row["answer_confidence"],
                              row["predicted"], row["gold"], str(row["state"]).replace("\n", " ")[:240]))
        else:
            lines.append("None.")
    lines.extend(["", "## 6. Human spot check", "",
                  "See [`sample.md`](sample.md) for %d seeded samples." % report["sample_items"], "",
                  "## 7. Verdict", "", "**%s**" % report["verdict"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Report whether a synthetic generation run is ready to scale")
    parser.add_argument("input", type=Path, help="data/<name>.jsonl.gz")
    parser.add_argument("--statim", help="base URL of a running Statim server")
    parser.add_argument("--sample", type=int, default=5, help="items per language and task (default: 5)")
    parser.add_argument("--out", type=Path, help="report directory (default: <input-stem>.report)")
    args = parser.parse_args(argv)
    if args.sample < 0:
        parser.error("--sample must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit("input not found: %s" % input_path)
    provenance_path, manifest_path, base = companion_paths(input_path)
    if not provenance_path.is_file():
        raise SystemExit("provenance not found: %s" % provenance_path)
    out_dir = (args.out.resolve() if args.out else input_path.parent / (base + ".report"))
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl_gz(input_path)
    provenance = read_jsonl_gz(provenance_path)
    checks = Checks()
    y = yield_report(items, provenance, checks)
    labels = label_balance(items, checks)
    repetition = repetition_report(items, checks)
    semantic = semantic_report(items, checks)
    difficulty = difficulty_report(items, args.statim, checks)
    sample_items = write_sample(out_dir / "sample.md", items, provenance, args.sample)

    gating = {"yield", "label_balance", "repetition", "semantic"}
    decisive = [x for x in checks.warnings if x["section"] in gating]
    skipped = [x for x in checks.rows if x["status"] == "SKIP" and x["section"] in gating]
    verdict = "REVISE" if decisive else "INSUFFICIENT DATA" if skipped else "SCALE UP"
    verdict_reasons = ["%s: %s — %s" % (x["section"], x["name"], x["detail"]) for x in decisive + skipped]
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
    report = {
        "input": str(input_path),
        "provenance": str(provenance_path),
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_data": manifest,
        "yield": y,
        "label_balance": labels,
        "repetition": repetition,
        "semantic_diversity": semantic,
        "difficulty": difficulty,
        "sample_items": sample_items,
        "checks": checks.rows,
        "warnings": checks.warnings,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_dir / "report.md", report)

    for row in checks.rows:
        print("%s %-15s %-28s %s" % (row["status"], row["section"], row["name"], row["detail"]))
    print("VERDICT %s" % verdict)
    print("wrote %s" % (out_dir / "report.md"))
    print("wrote %s" % (out_dir / "report.json"))
    print("wrote %s" % (out_dir / "sample.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
