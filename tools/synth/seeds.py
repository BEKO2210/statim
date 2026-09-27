"""Deterministic seed table: domain x persona x language x task.

The first n seeds are a prefix of the first n+k seeds (same --seed and the
same target weights), so a resumed run does not reshuffle work already done.
Arabic and Hindi are listed twice for business decisions only.

Sentiment, emotion, complaint, urgency, and sarcasm use the same attribute
axes. Gap languages are repeated in the cycle (weight x2). Legacy business,
score, and reading seeds keep their previous ids.

Target weights are optional. A weight is 1/acceptance of that target, clamped
to [1, 4]. Under-accepted targets are requested more often. Verify stays exact.
"""
import json
import random
from pathlib import Path

from common import canonical_id

LANGS = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "tr", "ru", "ar", "hi", "ja", "zh"]
# Weight x2: each other language appears once in the cycle, ar and hi twice.
LANGS_BUSINESS = LANGS + ["ar", "hi"]


def _repeat(langs, weight):
    out = []
    for lang in langs:
        out.extend([lang] * weight)
    return out


# Sentiment: all 14, gap languages twice.
_SENTIMENT_X2 = ("de", "fr", "es", "it", "nl", "pl", "tr", "ar", "ja")
SENTIMENT_LANGS = _repeat([lang for lang in LANGS if lang in _SENTIMENT_X2], 2) + [
    lang for lang in LANGS if lang not in _SENTIMENT_X2]
# Emotion: these languages only, each twice.
EMOTION_LANGS = _repeat(["de", "es", "it", "pt", "nl", "pl", "tr", "ru", "ar", "ja"], 2)
# Complaint: these languages only, each twice.
COMPLAINT_LANGS = _repeat(["fr", "it", "nl", "pl", "tr", "ar", "hi", "ja", "zh"], 2)
# Urgency: every language except English, each twice.
URGENCY_LANGS = _repeat([lang for lang in LANGS if lang != "en"], 2)

SENTIMENT_LABELS = ("positive", "negative", "mixed", "neutral")
EMOTION_LABELS = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "neutral")
COMPLAINT_LABELS = (
    "delivery", "billing", "product_defect", "service_quality", "communication", "refund", "other",
)
EMOTION_GENRES = (
    "a first-person private message",
    "a first-person review",
    "a first-person chat line",
    "a first-person diary-style note",
)
COMPLAINT_TOPICS = (
    "a delivery", "a bill or a payment", "a product", "a service visit",
    "a message from the company", "a refund", "a subscription", "a booking", "a repair",
)
COMPLAINT_TOPIC = {
    "delivery": "a delivery",
    "billing": "a bill or a payment",
    "product_defect": "a product",
    "service_quality": "a service visit",
    "communication": "a message from the company",
    "refund": "a refund",
    "other": "a problem that is none of delivery, billing, a broken product, service quality, communication, or a refund",
}
# Meanings for the ordinal urgency scale. The model phrases these in the target language.
URGENCY_MEANINGS = {
    4: (
        "1 lowest, can wait: no deadline, no outage, no safety issue, and no money at risk",
        "2 this week: a deadline or a consequence exists and it is not today",
        "3 today: it has to be handled today, and nothing is being damaged right now",
        "4 immediately: an outage is ongoing, someone is unsafe, or money is about to be lost",
    ),
    3: (
        "1 lowest, can wait: no deadline, no outage, no safety issue, and no money at risk",
        "2 today: it has to be handled today, and nothing is being damaged right now",
        "3 immediately: an outage is ongoing, someone is unsafe, or money is about to be lost",
    ),
}
PRODUCT_ASPECTS = [
    "price", "delivery time", "product quality", "customer support", "packaging",
    "ease of use", "durability", "size or fit", "cleanliness", "waiting time",
    "staff behaviour", "the returns process", "battery life", "accuracy of the description",
]

DOMAINS = [
    ("support_routing", "Customer support: route an incoming request to the right queue."),
    ("compliance_policy", "Compliance: decide whether a described action is allowed by an internal policy."),
    ("document_triage", "Document triage: decide what a received document is and where it should go."),
    ("it_request", "IT service desk: classify a workplace technology request so it can be handled."),
    ("hr_request", "Human resources: decide how an employee request should be handled."),
    ("finance_request", "Finance: decide how a payment, invoice, or expense request should be handled."),
    ("product_feedback", "Product feedback: decide which product team should act on a piece of feedback. Not a sentiment or emotion label."),
]

PERSONAS = [
    "a front-line support agent writing up what a caller said",
    "a compliance officer reviewing a short case note",
    "a records clerk describing a document that just arrived",
    "an employee submitting a request in their own words",
    "a team lead escalating a case to another desk",
    "an external customer writing to the company",
    "an auditor sampling one transaction",
    "a new hire who does not know internal jargon",
    "a vendor emailing the company about an order",
    "a finance clerk summarising an invoice problem",
    "a product manager pasting a raw note from a user",
    "a duty manager during a busy shift",
]

ASPECTS = [
    ("urgency", "how soon somebody needs to act"),
    ("severity", "how serious the harm or the impact already is"),
    ("risk", "how likely a bad outcome is if nothing changes"),
    ("relevance", "how relevant the candidate text is to a stated need"),
    ("similarity", "how similar in meaning two texts are"),
]

TOPICS = [
    "an internal security advisory about a workplace tool",
    "a product release note for an internal app",
    "a travel-expense rule",
    "an incident timeline for an office outage",
    "a vendor clause rewritten in plain language",
    "an onboarding explanation for a new hire",
    "a short budget memo",
    "a facilities notice",
    "a maintenance-window announcement",
    "a case note that quotes a customer email",
]

# Eight opening rules. The cue is part of the rule so the first words are not one shared phrase.
OPENING_KINDS = (
    "quote", "datetime", "order", "name", "question", "mid", "place", "number",
)
_OPENING_TEXT = {
    "quote": "Start with a quoted sentence in quotation marks. The quoted sentence is about: {cue}. Write that sentence in the requested language.",
    "datetime": "Start with this date or time before any other word: {cue}.",
    "order": "Start with this product or order number before any other word, the word translated into the requested language and the number kept: {cue}.",
    "name": "Start with this person's name before any other word: {cue}.",
    "question": "Start with a question that mentions: {cue}. Write the question in the requested language.",
    "mid": "Start in the middle of the situation, with no greeting and no date. The first words must bring up: {cue}. Write them in the requested language.",
    "place": "Start with this place before any other word, the word translated into the requested language and the number kept: {cue}.",
    "number": "Start with this number or amount before any other word, the unit written in the requested language: {cue}.",
}
_OPENING_CUES = {
    "quote": (
        "a locked door", "a missing reply", "a cold meal", "a late bus",
        "a cracked screen", "a quiet office", "a full waiting room", "a wrong size",
    ),
    "datetime": (
        "09:40", "16:05", "14 March", "Tuesday", "2024-11-02", "07:15", "30 June", "Friday 18:20",
    ),
    "order": (
        "order 48219", "item A-3307", "ticket 90112", "parcel 77-14",
        "invoice 2208", "booking 6610", "SKU 14823", "ref 3055",
    ),
    "name": (
        "Lina", "Omar", "Kenji", "Sofia", "Jonas", "Priya", "Elena", "Mateo",
    ),
    "question": (
        "the refund window", "the second parcel", "the evening shift", "the broken latch",
        "the extra charge", "the gate change", "the missing key", "the short reply",
    ),
    "mid": (
        "the queue at the counter", "the third flight of stairs", "the unpaid balance", "the torn sleeve",
        "the dark hallway", "the last empty seat", "the spilled coffee", "the blinking error light",
    ),
    "place": (
        "room 312", "platform 4", "gate B7", "desk 12", "locker 19", "floor 3", "bay 6", "counter 2",
    ),
    "number": (
        "18 EUR", "6 boxes", "45 minutes", "2 tickets", "120 pages", "8 percent", "3 nights", "90 kg",
    ),
}
# Concrete facts, not sentences. The text must contain one of these.
DETAIL_KINDS = ("amount", "deadline", "product_type", "duration", "location", "quantity")
DETAIL_VALUES = {
    "amount": ("18 EUR", "240 USD", "75 GBP", "1200 JPY", "90 BRL", "40 AED"),
    "deadline": ("16:00", "09:30", "Friday", "2 hours", "31 March", "Monday"),
    "product_type": ("kettle", "phone case", "winter coat", "lamp", "headphones", "suitcase"),
    "duration": ("10 minutes", "3 days", "2 weeks", "45 minutes", "6 months", "1 hour"),
    "location": ("room 418", "platform 9", "desk 27", "gate C3", "floor 5", "locker 44"),
    "quantity": ("2", "6", "12", "4", "30", "8"),
}

# Pilot-2 acceptance, per task. Nested keys are strata (complaint form, urgency
# level count). Leaf dicts are attempts/accepted. weights_from_acceptance()
# turns each rate into a request weight. This table is the fallback when no
# manifest is passed; --balance-from recomputes the same shape from a run.
STATIC_TARGET_ACCEPTANCE = {
    "sentiment": {
        "positive": {"attempts": 50, "accepted": 32},
        "negative": {"attempts": 50, "accepted": 30},
        "mixed": {"attempts": 50, "accepted": 34},
        "neutral": {"attempts": 50, "accepted": 7},
    },
    "emotion": {
        "sadness": {"attempts": 25, "accepted": 24},
        "surprise": {"attempts": 25, "accepted": 22},
        "joy": {"attempts": 25, "accepted": 21},
        "anger": {"attempts": 25, "accepted": 16},
        "fear": {"attempts": 25, "accepted": 15},
        "trust": {"attempts": 25, "accepted": 15},
        "disgust": {"attempts": 25, "accepted": 12},
        "neutral": {"attempts": 25, "accepted": 11},
    },
    "sarcasm": {
        "false": {"attempts": 100, "accepted": 47},
        "true": {"attempts": 100, "accepted": 43},
    },
    "complaint": {
        "noul": {
            "true": {"attempts": 50, "accepted": 41},
            "false": {"attempts": 50, "accepted": 41},
        },
        "choice": {
            "delivery": {"attempts": 15, "accepted": 13},
            "billing": {"attempts": 15, "accepted": 13},
            "product_defect": {"attempts": 14, "accepted": 12},
            "service_quality": {"attempts": 14, "accepted": 10},
            "refund": {"attempts": 14, "accepted": 5},
            "communication": {"attempts": 14, "accepted": 3},
            "other": {"attempts": 14, "accepted": 1},
        },
    },
    "urgency": {
        "4": {
            "1": {"attempts": 25, "accepted": 21},
            "2": {"attempts": 25, "accepted": 5},
            "3": {"attempts": 25, "accepted": 4},
            "4": {"attempts": 25, "accepted": 21},
        },
        "3": {
            "1": {"attempts": 34, "accepted": 21},
            "2": {"attempts": 33, "accepted": 10},
            "3": {"attempts": 33, "accepted": 17},
        },
    },
}

# Attribute axes (AttrPrompt, Yu et al. 2023): independent attributes drawn per seed, so the prompt
# space has millions of combinations instead of a short cycle that repeats every few hundred seeds.
INDUSTRIES = [
    "a retail bank", "a hospital", "a logistics company", "an online shop", "a software company",
    "a city administration", "a car-parts manufacturer", "a university", "a hotel chain",
    "a telecom provider", "an insurance company", "an energy utility", "a law firm", "an airline",
    "a pharmacy chain", "a construction firm",
]
REGISTERS = [
    "a short chat message", "a formal email", "a note taken during a phone call", "a web form submission",
    "an excerpt from a forwarded email thread", "an excerpt from meeting minutes",
    "an internal ticket with terse notes", "a letter",
]
LENGTHS = ["short: 25 to 50 words", "medium: 60 to 110 words", "long: 120 to 180 words"]
SCORE_LENGTHS = ["short: 30 to 60 words", "medium: 60 to 110 words"]
DIFFICULTIES = [
    "straightforward: the situation plainly supports one option",
    "subtle: the deciding detail sits in the middle of the text, not in the first sentence",
    "near miss: one wrong option looks right at first glance, and one specific detail in the situation rules it out",
]
READING_KINDS = [
    "an internal security advisory about a workplace tool", "a release note for an internal app",
    "a travel-expense rule", "an incident timeline for an outage", "a supplier contract clause in plain language",
    "an onboarding guide for new staff", "a short budget memo", "a facilities notice",
    "a maintenance-window announcement", "a case note that quotes a customer email", "a meeting summary",
    "an FAQ entry for customers", "a workplace safety instruction", "a change to the shift schedule",
]
READING_DIFFICULTIES = [
    "direct: the answer is stated in a single sentence of the passage",
    "combine: the answer needs two facts from different sentences",
    "paraphrase: the correct option restates the passage in different words; wrong options reuse its words",
]

# Private-message personas and registers for sentiment, emotion, complaint, urgency, sarcasm.
MESSAGE_PERSONAS = [
    "a person writing a private message to a friend",
    "a person writing a diary-style note to themselves",
    "a customer writing a first-person review",
    "a person sending a short chat message",
    "a person texting a family member",
    "a person writing a private note after a purchase",
    "a traveller writing about a service they just used",
    "a patient describing a visit in their own words",
    "a tenant writing about where they live",
    "a parent writing about something they bought for the household",
    "a student writing about a service they use",
    "a neighbour writing a short personal note",
]
MESSAGE_REGISTERS = [
    "a short chat message",
    "a formal email",
    "a letter",
    "a web form submission",
    "a note taken during a phone call",
    "a diary-style note",
    "a short review",
    "a private messaging-app line",
]

TASKS = (
    "business", "score", "reading",
    "sentiment", "emotion", "complaint", "urgency", "sarcasm",
)
_SALT = {
    "business": 1, "score": 2, "reading": 3,
    "sentiment": 4, "emotion": 5, "complaint": 6, "urgency": 7, "sarcasm": 8,
}
_LANG_CYCLE = {
    "business": LANGS_BUSINESS,
    "score": LANGS,
    "reading": LANGS,
    "sentiment": SENTIMENT_LANGS,
    "emotion": EMOTION_LANGS,
    "complaint": COMPLAINT_LANGS,
    "urgency": URGENCY_LANGS,
    "sarcasm": LANGS,
}


def _shuffled(items, rng):
    out = list(items)
    rng.shuffle(out)
    return out


def _plan_legacy(task, n, seed):
    """Business, score, reading. Rng order is frozen so existing ids stay valid."""
    rng = random.Random(seed + _SALT[task])
    langs = _shuffled(LANGS_BUSINESS if task == "business" else LANGS, rng)
    domains = _shuffled(DOMAINS, rng)
    personas = _shuffled(PERSONAS, rng)
    aspects = _shuffled(ASPECTS, rng)
    rows = []
    for i in range(n):
        # Languages cycle so every language gets the same share; every other attribute is drawn.
        if task == "business":
            domain_id, domain_brief = rng.choice(domains)
            form = "noul" if i % 3 == 0 else "choice"
            spec = {
                "task": task,
                "lang": langs[i % len(langs)],
                "domain": domain_id,
                "domain_brief": domain_brief,
                "industry": rng.choice(INDUSTRIES),
                "register": rng.choice(REGISTERS),
                "length": rng.choice(LENGTHS),
                "difficulty": rng.choice(DIFFICULTIES),
                "persona": rng.choice(personas),
                "form": form,
                "n_options": rng.randint(3, 12),
                "i": i,
            }
            if form == "noul":
                # Half the yes/no items must be answered "false": generators otherwise favour "true".
                spec["target"] = "true" if (i // 3) % 2 == 0 else "false"
        elif task == "score":
            aspect, aspect_help = aspects[i % len(aspects)]
            n_levels = rng.randint(3, 5)
            spec = {
                "task": task,
                "lang": langs[i % len(langs)],
                "aspect": aspect,
                "aspect_help": aspect_help,
                "industry": rng.choice(INDUSTRIES),
                "register": rng.choice(REGISTERS),
                "length": rng.choice(SCORE_LENGTHS),
                "persona": rng.choice(personas),
                "n_levels": n_levels,
                # Every level equally often as the right one, not mostly the middle.
                "target": rng.randint(1, n_levels),
                "i": i,
            }
        else:
            spec = {
                "task": task,
                "lang": langs[i % len(langs)],
                "topic": "%s at %s" % (rng.choice(READING_KINDS), rng.choice(INDUSTRIES)),
                "difficulty": rng.choice(READING_DIFFICULTIES),
                "persona": rng.choice(personas),
                "n_options": rng.choice((3, 4)),
                "i": i,
            }
        spec["id"] = canonical_id({k: v for k, v in spec.items() if k != "id"})
        rows.append(spec)
    return rows


def _opening_cue(kind, rng):
    """Numbers, places and times are drawn per seed: a fixed list of eight cues repeated the same
    "parcel 77-14" or "90 kg" at the start of many texts (pilot 3)."""
    if kind == "order":
        return "%s %d" % (rng.choice(("order", "item", "ticket", "parcel", "invoice", "booking", "reference", "case")), rng.randint(1000, 99999))
    if kind == "place":
        return "%s %d" % (rng.choice(("room", "platform", "gate", "desk", "locker", "floor", "bay", "counter", "office", "ward")), rng.randint(1, 80))
    if kind == "datetime":
        if rng.random() < 0.5:
            return "%02d:%02d" % (rng.randint(6, 22), rng.randrange(0, 60, 5))
        return "%d.%d." % (rng.randint(1, 28), rng.randint(1, 12))
    if kind == "number":
        return "%d %s" % (rng.randint(2, 480), rng.choice(("EUR", "USD", "boxes", "minutes", "tickets", "pages", "percent", "nights", "kg", "hours", "items", "calls")))
    return rng.choice(_OPENING_CUES[kind])


def _detail_value(kind, rng):
    if kind == "amount":
        return "%d %s" % (rng.randint(5, 2500), rng.choice(("EUR", "USD", "GBP", "JPY", "BRL", "AED", "INR", "TRY", "PLN")))
    if kind == "deadline":
        return rng.choice(("%02d:%02d" % (rng.randint(7, 20), rng.randrange(0, 60, 15)), "in %d hours" % rng.randint(1, 72), "day %d of the month" % rng.randint(1, 28)))
    if kind == "duration":
        return "%d %s" % (rng.randint(2, 90), rng.choice(("minutes", "hours", "days", "weeks", "months")))
    if kind == "location":
        return "%s %d" % (rng.choice(("room", "platform", "gate", "desk", "locker", "floor", "bay", "counter")), rng.randint(1, 80))
    if kind == "quantity":
        return str(rng.randint(2, 60))
    return rng.choice(DETAIL_VALUES[kind])


def _message_attrs(rng):
    kind = rng.choice(OPENING_KINDS)
    cue = _opening_cue(kind, rng)
    detail_kind = rng.choice(DETAIL_KINDS)
    detail_value = _detail_value(detail_kind, rng)
    return {
        "industry": rng.choice(INDUSTRIES),
        "register": rng.choice(MESSAGE_REGISTERS),
        "length": rng.choice(LENGTHS),
        "difficulty": rng.choice(DIFFICULTIES),
        "persona": rng.choice(MESSAGE_PERSONAS),
        "opening": _OPENING_TEXT[kind].format(cue=cue),
        "detail": "%s: %s" % (detail_kind, detail_value),
    }


def _finish_spec(spec):
    spec["id"] = canonical_id({k: v for k, v in spec.items() if k != "id"})
    return spec


def acceptance_to_weight(rate):
    """1/acceptance, clamped to [1, 4]. A zero rate takes the top of the clamp."""
    if rate <= 0:
        return 4.0
    return min(4.0, max(1.0, 1.0 / float(rate)))


def _is_count_leaf(node):
    return isinstance(node, dict) and "attempts" in node and "accepted" in node


def weights_from_acceptance(node):
    """Map an acceptance tree onto request weights. Strata stay nested."""
    if _is_count_leaf(node):
        attempts = node["attempts"]
        rate = (node["accepted"] / attempts) if attempts else 0.0
        return acceptance_to_weight(rate)
    if isinstance(node, dict):
        return {key: weights_from_acceptance(value) for key, value in node.items()}
    return acceptance_to_weight(float(node))


def iter_weight_strata(weights):
    """Yield each flat label->weight map. Nested strata are yielded separately."""
    if not isinstance(weights, dict) or not weights:
        return
    if all(isinstance(value, (int, float)) for value in weights.values()):
        yield weights
        return
    for value in weights.values():
        if isinstance(value, dict):
            yield from iter_weight_strata(value)


def normalized_weight_sum(weights):
    total = float(sum(weights.values()))
    if total <= 0:
        return 0.0
    return sum(float(value) / total for value in weights.values())


def _weight_list(weights, labels, stratum=None):
    """Aligned weights, or None when the caller asked for a uniform round robin."""
    if weights is None:
        return None
    node = weights
    if stratum is not None:
        node = weights.get(stratum) if isinstance(weights, dict) else None
        if not isinstance(node, dict):
            node = {}
    if not isinstance(node, dict):
        return None
    out = []
    for label in labels:
        raw = node.get(str(label), node.get(label, 1.0))
        if isinstance(raw, dict):
            raw = 1.0
        out.append(float(raw))
    return out


def _weighted_label(index, labels, weights):
    """Round robin when weights is None. Otherwise a golden-ratio scan.

    The scan only depends on `index`, so plan(n) stays a prefix of plan(n+k).
    """
    labels = tuple(labels)
    if not labels:
        raise SystemExit("no labels to draw")
    if weights is None:
        return labels[index % len(labels)]
    total = float(sum(weights))
    if total <= 0:
        return labels[index % len(labels)]
    point = (index * 0.6180339887498949) % 1.0
    acc = 0.0
    for label, weight in zip(labels, weights):
        acc += float(weight) / total
        if point < acc:
            return label
    return labels[-1]


def _stratum_of(seed):
    """(stratum or None, target string) for acceptance accounting."""
    task = seed.get("task")
    target = str(seed.get("target"))
    if task == "complaint":
        return seed.get("form") or "noul", target
    if task == "urgency":
        return str(seed.get("n_levels")), target
    return None, target


def acceptance_from_rows(rows):
    """Per-task acceptance tree. Latest row per id wins."""
    last = {}
    for row in rows:
        last[row.get("id")] = row
    counts = {}
    for rec in last.values():
        seed = rec.get("seed") or {}
        task = seed.get("task")
        if not task or "target" not in seed:
            continue
        stratum, target = _stratum_of(seed)
        key = (task, stratum, target)
        cell = counts.setdefault(key, [0, 0])
        cell[0] += 1
        cell[1] += rec.get("status") == "accepted"
    tree = {}
    for (task, stratum, target), (attempts, accepted) in counts.items():
        leaf = {"attempts": attempts, "accepted": accepted,
                "rate": (accepted / attempts) if attempts else 0.0}
        if stratum is None:
            tree.setdefault(task, {})[target] = leaf
        else:
            tree.setdefault(task, {}).setdefault(stratum, {})[target] = leaf
    return tree


def load_target_weights(path):
    """Weights from a manifest's target_acceptance, else from its provenance."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    acceptance = manifest.get("target_acceptance")
    if not acceptance:
        prov = path.with_name(path.name.replace(".manifest.json", ".provenance.jsonl.gz"))
        if not prov.is_file():
            raise SystemExit("no target_acceptance in %s and no provenance at %s" % (path, prov))
        import gzip
        rows = []
        with gzip.open(prov, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        acceptance = acceptance_from_rows(rows)
    return weights_from_acceptance(acceptance)


def _plan_message(task, n, seed, weights=None):
    rng = random.Random(seed + _SALT[task])
    langs = _shuffled(_LANG_CYCLE[task], rng)
    rows = []
    for i in range(n):
        spec = _message_attrs(rng)
        spec["task"] = task
        spec["lang"] = langs[i % len(langs)]
        spec["i"] = i
        if task == "sentiment":
            spec["target"] = _weighted_label(
                i, SENTIMENT_LABELS, _weight_list(weights, SENTIMENT_LABELS))
            # i % 4 would lock a class to one variant. Mix the two cycles.
            spec["variant"] = "aspect" if ((i // 4) + (i % 4)) % 2 else "overall"
            spec["aspects"] = []
            spec["focus"] = ""
            if spec["variant"] == "aspect":
                picked = rng.sample(PRODUCT_ASPECTS, 3)
                spec["aspects"] = picked
                spec["focus"] = picked[i % 3]
        elif task == "emotion":
            spec["target"] = _weighted_label(
                i, EMOTION_LABELS, _weight_list(weights, EMOTION_LABELS))
            spec["genre"] = EMOTION_GENRES[
                ((i // len(EMOTION_LABELS)) + (i % len(EMOTION_LABELS))) % len(EMOTION_GENRES)]
        elif task == "complaint":
            drawn = rng.choice(COMPLAINT_TOPICS)
            if i % 2 == 0:
                spec["form"] = "noul"
                spec["target"] = _weighted_label(
                    i // 2, ("true", "false"),
                    _weight_list(weights, ("true", "false"), "noul"))
                spec["topic"] = drawn
            else:
                spec["form"] = "choice"
                spec["target"] = _weighted_label(
                    i // 2, COMPLAINT_LABELS,
                    _weight_list(weights, COMPLAINT_LABELS, "choice"))
                spec["topic"] = COMPLAINT_TOPIC[spec["target"]]
        elif task == "urgency":
            if i % 2 == 0:
                spec["n_levels"] = 4
            else:
                spec["n_levels"] = 3
            labels = tuple(range(1, spec["n_levels"] + 1))
            spec["target"] = _weighted_label(
                i // 2, labels, _weight_list(weights, labels, str(spec["n_levels"])))
            spec["levels_brief"] = list(URGENCY_MEANINGS[spec["n_levels"]])
        else:
            spec["target"] = _weighted_label(
                i, ("true", "false"), _weight_list(weights, ("true", "false")))
        rows.append(_finish_spec(spec))
    return rows


def plan(task, n, seed, weights=None):
    """Plan n seeds. `weights` is an optional per-task acceptance-weight tree."""
    if task not in TASKS:
        raise SystemExit("unknown task %s" % task)
    if n < 1:
        raise SystemExit("--n must be >= 1")
    if task in {"business", "score", "reading"}:
        return _plan_legacy(task, n, seed)
    return _plan_message(task, n, seed, weights)
