"""Canonical per-item prediction records used to prove paired evaluations."""
import hashlib
import json


def item_sha256(state, questions) -> str:
    """SHA-256 of exactly the input state and question/options presented for an item."""
    raw = json.dumps([state, questions], sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prediction_rows(suite, lang, states, questions, gold, probabilities):
    """Build JSON-ready outcome rows. ``questions`` may be shared or one mapping per item."""
    per_item = isinstance(questions, (list, tuple))
    if not (len(states) == len(gold) == len(probabilities)):
        raise ValueError("states, gold and probabilities must have equal lengths")
    if per_item and len(questions) != len(states):
        raise ValueError("per-item questions must have the same length as states")
    rows = []
    for i, (state, target, probs) in enumerate(zip(states, gold, probabilities)):
        q = questions[i] if per_item else questions
        rows.append({"suite": suite, "lang": lang, "i": i, "item": item_sha256(state, q),
                     "gold": int(target), "pred": max(range(len(probs)), key=probs.__getitem__)})
    return rows


def write_prediction_rows(fh, suite, lang, states, questions, gold, probabilities):
    """Write canonical records without changing an evaluator's normal report output."""
    rows = prediction_rows(suite, lang, states, questions, gold, probabilities)
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    fh.flush()
    return rows

