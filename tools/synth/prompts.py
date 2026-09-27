"""Prompt templates. Slots are filled from the seed table. No filled-in examples.

PROMPT_VERSION is recorded in the manifest. Bump it when a template changes.
"""
import hashlib

PROMPT_VERSION = "synth-prompts-2"

GEN_SYSTEM = (
    "You write one original workplace decision as JSON for a classifier. "
    "Follow the user message for the language, the domain, and the answer form. "
    "Every string a trainee would read must be in the requested language. "
    "The situation must be concrete and fictional. Do not copy a published dataset, "
    "a news article, or a benchmark passage. "
    "Do not write an emotion, sentiment, or feeling classification. "
    "Do not name the correct option inside the question. "
    "The rationale is private metadata: it must not appear in the situation or the question. "
    "Output one JSON object and nothing else."
)

VER_SYSTEM = (
    "You answer one decision question from the material below and from nothing else. "
    "You are not shown an intended answer. "
    "Pick the single option the situation supports. "
    "Output one JSON object and nothing else."
)

_GEN_CHOICE = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Domain: {domain}
Perspective: {persona}
Form: one choice question with exactly {n_options} options.
Each option needs a short label (key) and a one-sentence description of what that label means. Keys must be distinct, in the requested language, and must not be single letters.
The question asks for a handling, routing, compliance, triage, or ownership decision in this domain. It must be answerable from the situation alone.
Situation length: 60 to 140 words. For Chinese and Japanese, count characters as words.
JSON fields: state (the situation), instructions (the question), options (array of {{key, description}}), gold (the key of the one correct option, copied exactly), rationale (one short sentence, not repeated elsewhere)."""

_GEN_NOUL = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Domain: {domain}
Perspective: {persona}
Form: one yes/no question (true means the statement holds, false means it does not).
The question asks for a handling, compliance, or applicability decision in this domain, answerable from the situation alone.
Also give a one-sentence description of what false means here and what true means here, in the requested language.
Situation length: 60 to 140 words. For Chinese and Japanese, count characters as words.
JSON fields: state, instructions, false_description, true_description, gold (boolean), rationale (one short sentence, not repeated elsewhere)."""

_GEN_SCORE = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Perspective: {persona}
Scale: {aspect} — {aspect_help}.
Use exactly {n_levels} ordinal levels, ordered from low to high, each a short phrase in the requested language. Levels must be mutually exclusive and cover the scale.
The situation must make exactly one level clearly right.
Situation length: 40 to 120 words. For Chinese and Japanese, count characters as words.
If the scale is similarity, the situation contains two separate texts divided by a line that is exactly ---.
If the scale is relevance, the situation states the need and then the candidate text.
JSON fields: state, instructions (the rating question, which must not contain the level phrases), levels (array of strings, low to high), gold (one level string copied exactly from levels), rationale (one short sentence, not repeated elsewhere)."""

_GEN_READING = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Write an original passage of 150 to 220 words about: {topic}. Perspective: {persona}.
For Chinese and Japanese, count characters as words. Do not use a real news story or a published benchmark passage.
Then write one question that can be answered only from the passage, with exactly {n_options} options.
Each option is an object with key "A", "B", "C", or "D" in order, and description equal to the answer text in the requested language.
Exactly one option is supported by the passage. The others contradict it or are not stated. The question must not contain the correct answer text.
JSON fields: passage, instructions (the question), options, gold (the key, copied exactly), rationale (one short sentence citing a fact in the passage, not repeated in the question)."""

_VER_CHOICE = """Type: choice.
Situation:
{state}
Question:
{instructions}
Options (reply with the key only):
{options}
JSON: {{"answer": "<one key copied exactly>"}}"""

_VER_NOUL = """Type: yes/no. true means the statement holds, false means it does not.
false: {false_description}
true: {true_description}
Situation:
{state}
Question:
{instructions}
JSON: {{"answer": "true"}} or {{"answer": "false"}}."""

_VER_SCORE = """Type: ordinal score. Reply with one level copied exactly from the list. Do not invent a nearby level.
Situation:
{state}
Question:
{instructions}
Levels, low to high:
{levels}
JSON: {{"answer": "<one level copied exactly>"}}"""

SCHEMA_CHOICE = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "instructions": {"type": "string"},
        "options": {"type": "array", "items": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "description": {"type": "string"}},
            "required": ["key", "description"],
        }},
        "gold": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["state", "instructions", "options", "gold", "rationale"],
}
SCHEMA_NOUL = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "instructions": {"type": "string"},
        "false_description": {"type": "string"},
        "true_description": {"type": "string"},
        "gold": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["state", "instructions", "false_description", "true_description", "gold", "rationale"],
}
SCHEMA_SCORE = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "instructions": {"type": "string"},
        "levels": {"type": "array", "items": {"type": "string"}},
        "gold": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["state", "instructions", "levels", "gold", "rationale"],
}
SCHEMA_READING = {
    "type": "object",
    "properties": {
        "passage": {"type": "string"},
        "instructions": {"type": "string"},
        "options": SCHEMA_CHOICE["properties"]["options"],
        "gold": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["passage", "instructions", "options", "gold", "rationale"],
}
SCHEMA_VERIFY = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

LANG_NAME = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ru": "Russian",
    "ar": "Arabic", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese (Simplified)",
}


def prompts_digest():
    blob = "\n".join([
        PROMPT_VERSION, GEN_SYSTEM, VER_SYSTEM, _GEN_CHOICE, _GEN_NOUL, _GEN_SCORE,
        _GEN_READING, _VER_CHOICE, _VER_NOUL, _VER_SCORE,
    ])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def generate_request(seed):
    """Return (messages, schema, num_predict) for one seed. No gold is known yet."""
    lang = LANG_NAME[seed["lang"]]
    task = seed["task"]
    if task == "business" and seed["form"] == "choice":
        user = _GEN_CHOICE.format(
            language=lang, code=seed["lang"], domain=seed["domain_brief"],
            persona=seed["persona"], n_options=seed["n_options"])
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user}], SCHEMA_CHOICE, 900
    if task == "business":
        user = _GEN_NOUL.format(
            language=lang, code=seed["lang"], domain=seed["domain_brief"],
            persona=seed["persona"])
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user}], SCHEMA_NOUL, 900
    if task == "score":
        user = _GEN_SCORE.format(
            language=lang, code=seed["lang"], persona=seed["persona"],
            aspect=seed["aspect"], aspect_help=seed["aspect_help"],
            n_levels=seed["n_levels"])
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user}], SCHEMA_SCORE, 800
    user = _GEN_READING.format(
        language=lang, code=seed["lang"], topic=seed["topic"],
        persona=seed["persona"], n_options=seed["n_options"])
    return [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": user}], SCHEMA_READING, 1400


def _fmt_options(options):
    lines = []
    for opt in options:
        lines.append("- %s: %s" % (opt["key"], opt["description"]))
    return "\n".join(lines)


def verify_request(view):
    """Second-pass prompt. `view` has state, instructions, type, options. No gold, no rationale."""
    kind = view["type"]
    if kind == "choice":
        user = _VER_CHOICE.format(
            state=view["state"], instructions=view["instructions"],
            options=_fmt_options(view["options"]))
    elif kind == "noul":
        by = {opt["key"]: opt["description"] for opt in view["options"]}
        user = _VER_NOUL.format(
            state=view["state"], instructions=view["instructions"],
            false_description=by.get("false", ""), true_description=by.get("true", ""))
    else:
        user = _VER_SCORE.format(
            state=view["state"], instructions=view["instructions"],
            levels="\n".join("- %s" % lv for lv in view["levels"]))
    return ([{"role": "system", "content": VER_SYSTEM},
             {"role": "user", "content": user}], SCHEMA_VERIFY, 64)
