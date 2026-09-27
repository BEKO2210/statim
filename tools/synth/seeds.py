"""Deterministic seed table: domain x persona x language x task.

The first n seeds are a prefix of the first n+k seeds (same --seed), so a
resumed run does not reshuffle work that was already done.
Arabic and Hindi are listed twice for business decisions only.
"""
import random

from common import canonical_id

LANGS = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "tr", "ru", "ar", "hi", "ja", "zh"]
# Weight x2: each other language appears once in the cycle, ar and hi twice.
LANGS_BUSINESS = LANGS + ["ar", "hi"]

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

TASKS = ("business", "score", "reading")


def _shuffled(items, rng):
    out = list(items)
    rng.shuffle(out)
    return out


def plan(task, n, seed):
    if task not in TASKS:
        raise SystemExit("unknown task %s" % task)
    if n < 1:
        raise SystemExit("--n must be >= 1")
    rng = random.Random(seed + {"business": 1, "score": 2, "reading": 3}[task])
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
