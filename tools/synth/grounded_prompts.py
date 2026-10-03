"""Prompts for the grounded urgency/NLI pilot. No benchmark examples."""
import hashlib
import inspect
import json

PROMPT_VERSION = "grounded-pilot-2"
URGENCY_LABELS = ("not urgent", "soon", "critical")
NLI_LABELS = ("entailment", "contradiction", "neutral")

_LANG = {"en": "English", "de": "German", "fr": "French", "es": "Spanish",
         "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish"}

_URGENCY_SCHEMA = {
    "type": "object", "properties": {
        "request": {"type": "string"}, "label": {"type": "string", "enum": list(URGENCY_LABELS)}},
    "required": ["request", "label"]}
_NLI_SCHEMA = {
    "type": "object", "properties": {"items": {"type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "object", "properties": {
            "hypothesis": {"type": "string"}, "label": {"type": "string", "enum": list(NLI_LABELS)}},
            "required": ["hypothesis", "label"]}}}, "required": ["items"]}
_ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"]}


def generation_request(task, passage, lang, target=None):
    language = _LANG[lang]
    system = ("Create grounded classifier data from the supplied source passage. Use only its topic and facts. "
              "Do not copy a sentence of eight or more words. Return one JSON object and no commentary.")
    if task == "urgency":
        user = f"""Source passage ({language}):
{passage}

Write a short support or service request in {language} by a citizen or company affected by this topic.
Requested urgency label: {target}
Meanings: not urgent = a routine question, suggestion or information request; nothing has gone wrong, there is no
deadline and the writer can wait weeks; soon = something needs action within days, e.g. an upcoming deadline or a
problem that will grow, but nothing is failing yet; critical = immediate action is required because serious harm,
outage, an expiring deadline or loss is happening now or within hours.
The situation, not urgency words or the label itself, must establish the label. Return request and label."""
        return ([{"role": "system", "content": system}, {"role": "user", "content": user}],
                _URGENCY_SCHEMA, 300)
    user = f"""Premise passage ({language}):
{passage}

Write exactly three short hypotheses in {language}: one entailed by the premise, one contradicted by it,
and one neither supported nor contradicted. Use each canonical label exactly once: entailment, contradiction,
neutral. Do not add facts to the entailment. Return items with hypothesis and label."""
    return ([{"role": "system", "content": system}, {"role": "user", "content": user}], _NLI_SCHEMA, 500)


def verification_request(task, item, lang):
    language = _LANG[lang]
    system = ("Independently label the classifier item from its text. You are not shown another model's label. "
              "Return one JSON object and no explanation.")
    if task == "urgency":
        user = (f"Request ({language}):\n{item['request']}\n\nChoose exactly one: not urgent, soon, critical. "
                "Use urgency of required action, not general seriousness.")
        schema = dict(_ANSWER_SCHEMA)
        schema["properties"] = {"answer": {"type": "string", "enum": list(URGENCY_LABELS)}}
    else:
        user = (f"Premise ({language}):\n{item['premise']}\n\nHypothesis:\n{item['hypothesis']}\n\n"
                "Choose exactly one: entailment, contradiction, neutral.")
        schema = dict(_ANSWER_SCHEMA)
        schema["properties"] = {"answer": {"type": "string", "enum": list(NLI_LABELS)}}
    return ([{"role": "system", "content": system}, {"role": "user", "content": user}], schema, 32)


def prompts_digest():
    blob = "\n".join((PROMPT_VERSION, inspect.getsource(generation_request),
                       inspect.getsource(verification_request), json.dumps(_LANG, sort_keys=True)))
    return hashlib.sha256(blob.encode()).hexdigest()
