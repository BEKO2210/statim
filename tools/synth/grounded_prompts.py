"""Prompts for grounded synthetic classifier data. No benchmark examples."""
import hashlib
import inspect
import json

from grounded_tasks import LANG_NAMES, NLI_LABELS, TASKS, URGENCY_LABELS, _SYSTEM, _VERIFY_SYSTEM

PROMPT_VERSION = "grounded-pilot-3"

_ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"]}


def _single_schema(field, labels):
    return {"type": "object", "properties": {
        field: {"type": "string"}, "label": {"type": "string", "enum": list(labels)}},
        "required": [field, "label"]}


def _pair_schema(field, labels):
    return {"type": "object", "properties": {"items": {"type": "array",
        "minItems": len(labels), "maxItems": len(labels),
        "items": {"type": "object", "properties": {
            field: {"type": "string"}, "label": {"type": "string", "enum": list(labels)}},
            "required": [field, "label"]}}}, "required": ["items"]}


def _label_list(labels):
    return ", ".join(labels)


_URGENCY_MEANINGS = """Meanings: not urgent = a routine question, suggestion or information request; nothing has gone wrong, there is no
deadline and the writer can wait weeks; soon = something needs action within days, e.g. an upcoming deadline or a
problem that will grow, but nothing is failing yet; critical = immediate action is required because serious harm,
outage, an expiring deadline or loss is happening now or within hours."""


def generation_request(task, passage, lang, target=None):
    language = LANG_NAMES[lang]
    spec = TASKS[task]
    system = _SYSTEM
    if task == "urgency":
        user = f"""Source passage ({language}):
{passage}

Write a short support or service request in {language} by a citizen or company affected by this topic.
Requested urgency label: {target}
{_URGENCY_MEANINGS}
The situation, not urgency words or the label itself, must establish the label: never say how urgent
it is (no 'urgent', 'not urgent', 'no hurry', 'can wait'). Return request and label."""
        return ([{"role": "system", "content": system}, {"role": "user", "content": user}],
                _single_schema("request", URGENCY_LABELS), spec["gen_predict"])
    if task == "nli":
        user = f"""Premise passage ({language}):
{passage}

Write exactly three short hypotheses in {language}: one entailed by the premise, one contradicted by it,
and one neither supported nor contradicted. Use each canonical label exactly once: entailment, contradiction,
neutral. Do not add facts to the entailment. Return items with hypothesis and label."""
        return ([{"role": "system", "content": system}, {"role": "user", "content": user}],
                _pair_schema("hypothesis", NLI_LABELS), spec["gen_predict"])
    labels = spec["labels"]
    if spec["generation"] == "single":
        field = spec["text_field"]
        loc = spec["locales"][lang]
        meanings = "\n".join("%s = %s" % (labels[i], loc["criteria"][i]) for i in range(len(labels)))
        situation = spec.get("label_situation", {}).get(target, spec.get("situation", ""))
        user = f"""Source passage ({language}):
{passage}

Write one short {field} in {language} related to this topic. {situation}
Vary who writes, the tone and the length; sound like a real person, not a template.
Requested label: {target}
Meanings:
{meanings}
The wording, not the label itself, must establish the label. Return {field} and label."""
        return ([{"role": "system", "content": system}, {"role": "user", "content": user}],
                _single_schema(field, labels), spec["gen_predict"])
    field = spec["pair_json_field"]
    n = len(labels)
    if task == "stance":
        user = f"""Measure or report passage ({language}):
{passage}

Write exactly {n} short citizen comments in {language} on this topic: one {labels[0]}, one {labels[1]},
and one {labels[2]}. Use each canonical label exactly once: {_label_list(labels)}.
Return items with comment and label."""
    elif task == "reading":
        user = f"""Source passage ({language}):
{passage}

Write exactly {n} short yes/no questions in {language} about this topic: one answerable yes from the passage alone,
one answerable no from the passage alone, and one on topic but not answerable from the passage.
Use each canonical label exactly once: {_label_list(labels)}. Return items with question and label."""
    else:
        raise ValueError(task)
    return ([{"role": "system", "content": system}, {"role": "user", "content": user}],
            _pair_schema(field, labels), spec["gen_predict"])


def verification_request(task, item, lang):
    language = LANG_NAMES[lang]
    spec = TASKS[task]
    labels = spec["labels"]
    system = _VERIFY_SYSTEM
    if task == "urgency":
        user = (f"Request ({language}):\n{item['request']}\n\nChoose exactly one: not urgent, soon, critical. "
                "Use urgency of required action, not general seriousness.\n" + _URGENCY_MEANINGS)
    elif task == "nli":
        user = (f"Premise ({language}):\n{item['premise']}\n\nHypothesis:\n{item['hypothesis']}\n\n"
                "Choose exactly one: entailment, contradiction, neutral.\n"
                "Meanings: entailment = the premise alone makes the hypothesis true; contradiction = the "
                "premise makes it false; neutral = the premise neither confirms nor rules it out.")
    elif spec["shape"] == "single":
        field = spec["verify_field"]
        loc = spec["locales"][lang]
        choices = ", ".join(labels)
        meanings = "\n".join("%s = %s" % (labels[i], loc["criteria"][i]) for i in range(len(labels)))
        user = (f"{field.capitalize()} ({language}):\n{item[field]}\n\n"
                f"Choose exactly one: {choices}.\n{loc['question']}\nMeanings:\n{meanings}")
    else:
        pfield, gfield = spec["verify_fields"]
        loc = spec["locales"][lang]
        disp_p, disp_g = loc["pair_fields"]
        choices = ", ".join(labels)
        meanings = "\n".join("%s = %s" % (labels[i], loc["criteria"][i]) for i in range(len(labels)))
        user = (f"{disp_p} ({language}):\n{item[pfield]}\n\n{disp_g}:\n{item[gfield]}\n\n"
                f"Choose exactly one: {choices}.\n{loc['question']}\nMeanings:\n{meanings}")
    schema = dict(_ANSWER_SCHEMA)
    schema["properties"] = {"answer": {"type": "string", "enum": list(labels)}}
    return ([{"role": "system", "content": system}, {"role": "user", "content": user}], schema,
            spec["verify_predict"])


def prompts_digest():
    blob = "\n".join((PROMPT_VERSION, inspect.getsource(generation_request),
                       inspect.getsource(verification_request),
                       json.dumps({k: v["prompt_version"] for k, v in TASKS.items()}, sort_keys=True),
                       json.dumps(LANG_NAMES, sort_keys=True)))
    return hashlib.sha256(blob.encode()).hexdigest()
