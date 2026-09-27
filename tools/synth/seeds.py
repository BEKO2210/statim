"""Deterministic seed table: domain x persona x language x task.

The first n seeds are a prefix of the first n+k seeds (same --seed), so a
resumed run does not reshuffle work that was already done.
Arabic and Hindi are listed twice for business decisions only.

Sentiment, emotion, complaint, urgency, and sarcasm use the same attribute
axes. Gap languages are repeated in the cycle (weight x2). Legacy business,
score, and reading seeds keep their previous ids.
"""
import random

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


def _message_attrs(rng):
    return {
        "industry": rng.choice(INDUSTRIES),
        "register": rng.choice(MESSAGE_REGISTERS),
        "length": rng.choice(LENGTHS),
        "difficulty": rng.choice(DIFFICULTIES),
        "persona": rng.choice(MESSAGE_PERSONAS),
    }


def _finish_spec(spec):
    spec["id"] = canonical_id({k: v for k, v in spec.items() if k != "id"})
    return spec


def _plan_message(task, n, seed):
    rng = random.Random(seed + _SALT[task])
    langs = _shuffled(_LANG_CYCLE[task], rng)
    rows = []
    for i in range(n):
        spec = _message_attrs(rng)
        spec["task"] = task
        spec["lang"] = langs[i % len(langs)]
        spec["i"] = i
        if task == "sentiment":
            spec["target"] = SENTIMENT_LABELS[i % len(SENTIMENT_LABELS)]
            # i % 4 would lock a class to one variant. Mix the two cycles.
            spec["variant"] = "aspect" if ((i // 4) + (i % 4)) % 2 else "overall"
            spec["aspects"] = []
            spec["focus"] = ""
            if spec["variant"] == "aspect":
                picked = rng.sample(PRODUCT_ASPECTS, 3)
                spec["aspects"] = picked
                spec["focus"] = picked[i % 3]
        elif task == "emotion":
            spec["target"] = EMOTION_LABELS[i % len(EMOTION_LABELS)]
            spec["genre"] = EMOTION_GENRES[
                ((i // len(EMOTION_LABELS)) + (i % len(EMOTION_LABELS))) % len(EMOTION_GENRES)]
        elif task == "complaint":
            drawn = rng.choice(COMPLAINT_TOPICS)
            if i % 2 == 0:
                spec["form"] = "noul"
                spec["target"] = "true" if (i // 2) % 2 == 0 else "false"
                spec["topic"] = drawn
            else:
                spec["form"] = "choice"
                spec["target"] = COMPLAINT_LABELS[(i // 2) % len(COMPLAINT_LABELS)]
                spec["topic"] = COMPLAINT_TOPIC[spec["target"]]
        elif task == "urgency":
            if i % 2 == 0:
                spec["n_levels"] = 4
                spec["target"] = (i // 2) % 4 + 1
            else:
                spec["n_levels"] = 3
                spec["target"] = (i // 2) % 3 + 1
            spec["levels_brief"] = list(URGENCY_MEANINGS[spec["n_levels"]])
        else:
            spec["target"] = "true" if i % 2 == 0 else "false"
        rows.append(_finish_spec(spec))
    return rows


def plan(task, n, seed):
    if task not in TASKS:
        raise SystemExit("unknown task %s" % task)
    if n < 1:
        raise SystemExit("--n must be >= 1")
    if task in {"business", "score", "reading"}:
        return _plan_legacy(task, n, seed)
    return _plan_message(task, n, seed)
