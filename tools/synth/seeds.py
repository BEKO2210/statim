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
    topics = _shuffled(TOPICS, rng)
    rows = []
    for i in range(n):
        if task == "business":
            domain_id, domain_brief = domains[i % len(domains)]
            form = "noul" if i % 3 == 0 else "choice"
            spec = {
                "task": task,
                "lang": langs[i % len(langs)],
                "domain": domain_id,
                "domain_brief": domain_brief,
                "persona": personas[(i * 5) % len(personas)],
                "form": form,
                "n_options": 3 + (i % 10),
                "i": i,
            }
        elif task == "score":
            aspect, aspect_help = aspects[i % len(aspects)]
            spec = {
                "task": task,
                "lang": langs[i % len(langs)],
                "aspect": aspect,
                "aspect_help": aspect_help,
                "persona": personas[(i * 5) % len(personas)],
                "n_levels": 3 + (i % 3),
                "i": i,
            }
        else:
            spec = {
                "task": task,
                "lang": langs[i % len(langs)],
                "topic": topics[i % len(topics)],
                "persona": personas[(i * 5) % len(personas)],
                "n_options": 4 if i % 2 else 3,
                "i": i,
            }
        spec["id"] = canonical_id({k: v for k, v in spec.items() if k != "id"})
        rows.append(spec)
    return rows
