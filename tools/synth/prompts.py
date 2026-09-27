"""Prompt templates. Slots are filled from the seed table. No filled-in examples.

PROMPT_VERSION is recorded in the manifest. Bump it when a template changes.
"""
import hashlib

from seeds import COMPLAINT_LABELS, EMOTION_LABELS, SENTIMENT_LABELS

PROMPT_VERSION = "synth-prompts-5"

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

GEN_SYSTEM_CLASS = (
    "You write one original classification item as JSON for a classifier. "
    "Follow the user message for the language, the label set, and the requested label. "
    "Every string a trainee would read must be in the requested language. "
    "Option keys that the user message lists in English must be copied exactly and must not be translated. "
    "Descriptions of those keys are in the requested language. "
    "The situation must be concrete and fictional. Do not copy a published dataset, "
    "a news article, or a benchmark passage. "
    "Do not name the correct option inside the question. "
    "The JSON field named state is the full text to classify, in the requested language. "
    "It is not a status, not a label, and not the word completed. "
    "It must meet the situation length in the user message. One word is never enough. "
    "Do not copy these instructions, the class rule, or an English gloss into the state or the question. "
    "The rationale is private metadata: it must not appear in the situation or the question. "
    "Output one JSON object and nothing else."
)

_SENTIMENT_HELP = {
    "positive": "a favourable opinion and no complaint",
    "negative": "an unfavourable opinion and no praise",
    "mixed": "the same text contains explicit praise and an explicit complaint",
    "neutral": "factual statements and no opinion",
}
_SENTIMENT_RULE = {
    "positive": "Write a favourable opinion and do not include a complaint.",
    "negative": "Write an unfavourable opinion and do not include praise.",
    "mixed": "The same text must contain explicit praise and an explicit complaint.",
    "neutral": "Write factual statements and no opinion.",
}
_EMOTION_HELP = {
    "joy": "happiness or delight",
    "sadness": "sorrow or grief",
    "anger": "irritation or rage",
    "fear": "anxiety or being afraid",
    "surprise": "astonishment at something unexpected",
    "disgust": "revulsion",
    "trust": "confidence that someone or something is reliable",
    "neutral": "no clear emotion",
}
_COMPLAINT_HELP = {
    "delivery": "shipping, arrival time, or where an order was left",
    "billing": "a charge, an invoice, or a payment",
    "product_defect": "the item itself is damaged or does not work",
    "service_quality": "how a service was performed",
    "communication": "a reply, a notice, or how someone was spoken to",
    "refund": "money to be returned",
    "other": "a complaint that fits none of the categories above",
}
_COMPLAINT_RULE = {
    "true": "The message is a complaint about the topic.",
    "false": "The message is not a complaint. It is a question, praise, or a neutral request that still mentions the topic.",
}
_SARCASM_RULE = {
    "true": (
        "The statement is sarcastic through context contrast: praise wording about a situation "
        "the same text shows is bad. Do not use an emoji. Do not mark it with a slash followed by the letter s. "
        "Do not write that the text is sarcastic."
    ),
    "false": (
        "The statement is literal, not sarcastic. It may be genuine praise of a good situation, "
        "a plain complaint with no ironic praise, or a neutral remark. "
        "Do not use an emoji. Do not mark it with a slash followed by the letter s."
    ),
}

_GEN_CHOICE = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Domain: {domain}
Setting: {industry}. The situation is written as {register}.
Perspective: {persona}
Form: one choice question with exactly {n_options} options.
Difficulty: {difficulty}.
Each option needs a short label (key) and a one-sentence description of what that label means. Keys must be distinct, in the requested language, and must not be single letters.
The question asks for a handling, routing, compliance, triage, or ownership decision in this domain. It must be answerable from the situation alone.
Situation length: {length}. For Chinese and Japanese, count characters as words.
JSON fields: state (the situation), instructions (the question), options (array of {{key, description}}), gold (the key of the one correct option, copied exactly), rationale (one short sentence, not repeated elsewhere)."""

_GEN_NOUL = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Domain: {domain}
Setting: {industry}. The situation is written as {register}.
Perspective: {persona}
Form: one yes/no question (true means the statement holds, false means it does not).
Difficulty: {difficulty}.
The correct answer must be {target}: write the situation so that it clearly supports {target}, without giving the answer away in the question.
The question asks for a handling, compliance, or applicability decision in this domain, answerable from the situation alone.
Also give a one-sentence description of what false means here and what true means here, in the requested language.
Situation length: {length}. For Chinese and Japanese, count characters as words.
JSON fields: state, instructions, false_description, true_description, gold (boolean), rationale (one short sentence, not repeated elsewhere)."""

_GEN_SCORE = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Setting: {industry}. The situation is written as {register}.
Perspective: {persona}
Scale: {aspect} — {aspect_help}.
Use exactly {n_levels} ordinal levels, ordered from low to high, each a short phrase in the requested language. Levels must be mutually exclusive and cover the scale.
The situation must make exactly one level clearly right: level {target} of {n_levels}, counting from 1 = lowest.
Situation length: {length}. For Chinese and Japanese, count characters as words.
If the scale is similarity, the situation contains two separate texts divided by a line that is exactly ---.
If the scale is relevance, the situation states the need and then the candidate text.
JSON fields: state, instructions (the rating question, which must not contain the level phrases), levels (array of strings, low to high), gold (one level string copied exactly from levels), rationale (one short sentence, not repeated elsewhere)."""

_GEN_READING = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Write an original passage of 150 to 220 words: {topic}. Perspective: {persona}.
Question difficulty: {difficulty}.
For Chinese and Japanese, count characters as words. Do not use a real news story or a published benchmark passage.
Then write one question that can be answered only from the passage, with exactly {n_options} options.
Each option is an object with key "A", "B", "C", or "D" in order, and description equal to the answer text in the requested language.
Exactly one option is supported by the passage. The others contradict it or are not stated. The question must not contain the correct answer text.
JSON fields: passage, instructions (the question), options, gold (the key, copied exactly), rationale (one short sentence citing a fact in the passage, not repeated in the question)."""

_GEN_SENTIMENT = """Language: {language} ({code}). Write every trainee-facing string in this language only. Copy the option keys exactly; do not translate them.
Setting: {industry}. The text is written as {register}.
Perspective: {persona}
Difficulty: {difficulty}.
Situation length: {length}. For Chinese and Japanese, count characters as words.
Form: one choice question. Use exactly these keys, each with a one-sentence description in the requested language. Do not copy the English gloss into the description.
{labels}
The requested class is {target}. gold must be exactly {target}.
Class rule: {class_rule}
The question must not contain any of the keys and must not contain the description of the correct class.
{variant_block}
JSON fields: state (the full text to classify, in the requested language), instructions (the question), options (array of {{key, description}} with those keys), gold (exactly {target}), rationale (one short sentence, not repeated elsewhere)."""

_GEN_EMOTION = """Language: {language} ({code}). Write every trainee-facing string in this language only. Copy the option keys exactly; do not translate them.
Setting: {industry}. Genre: {genre}. The text is written as {register}.
Perspective: {persona}. Write in the first person.
Difficulty: {difficulty}.
Situation length: {length}. For Chinese and Japanese, count characters as words.
Form: one choice question. Use exactly these keys, each with a one-sentence description in the requested language. Do not copy the English gloss into the description.
{labels}
The requested emotion is {target}. gold must be exactly {target}. The text must make that emotion clearly right.
The question asks which emotion the text expresses. It must not contain any of the keys.
JSON fields: state (the full text to classify, in the requested language), instructions (the question), options (array of {{key, description}} with those keys), gold (exactly {target}), rationale (one short sentence, not repeated elsewhere)."""

_GEN_COMPLAINT_NOUL = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Setting: {industry}. Topic of the message: {topic}. The text is written as {register}.
Perspective: {persona}
Difficulty: {difficulty}.
The correct answer must be {target}. gold must be the boolean {target}.
Class rule: {class_rule}
The question asks whether the message is a complaint, and it must not state the answer.
Situation length: {length}. For Chinese and Japanese, count characters as words.
Also give a one-sentence description of what false means here and what true means here, in the requested language.
JSON fields: state (the full text to classify, in the requested language), instructions (the question), false_description, true_description, gold (boolean {target}), rationale (one short sentence, not repeated elsewhere)."""

_GEN_COMPLAINT_CHOICE = """Language: {language} ({code}). Write every trainee-facing string in this language only. Copy the option keys exactly; do not translate them.
Setting: {industry}. Topic of the complaint: {topic}. The text is written as {register}.
Perspective: {persona}
Difficulty: {difficulty}.
The message is a complaint. The complaint category is {target}. gold must be exactly {target}.
Form: one choice question. Use exactly these keys, each with a one-sentence description in the requested language. Do not copy the English gloss into the description.
{labels}
The question asks which complaint category the message belongs to. It must not contain any of the keys.
Situation length: {length}. For Chinese and Japanese, count characters as words.
JSON fields: state (the full text to classify, in the requested language), instructions (the question), options (array of {{key, description}}), gold (exactly {target}), rationale (one short sentence, not repeated elsewhere)."""

_GEN_URGENCY = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Setting: {industry}. The text is written as {register}.
Perspective: {persona}
Difficulty: {difficulty}.
Situation length: {length}. For Chinese and Japanese, count characters as words.
Scale: how soon somebody needs to act.
Use exactly {n_levels} ordinal levels, ordered from low to high. Each level is a short phrase in the requested language. Do not copy these English meanings verbatim when the language is not English. Levels must be mutually exclusive and cover the scale.
Meanings, from 1 = lowest to {n_levels} = highest:
{level_meanings}
The correct level is number {target} of {n_levels}, counting from 1. Do not reorder the levels. gold must be the phrase at that position, copied exactly from levels.
The situation must contain the deciding detail that makes this level right: a deadline, an outage, a safety problem, or money at risk. For the lowest level, the deciding detail is the absence of all four.
The question must not quote any level phrase.
JSON fields: state (the full text to classify, in the requested language), instructions (the question), levels (array of {n_levels} strings, low to high), gold (one level string copied exactly from levels), rationale (one short sentence, not repeated elsewhere)."""

_GEN_SARCASM = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Setting: {industry}. The text is written as {register}.
Perspective: {persona}
Difficulty: {difficulty}.
The correct answer must be {target}. gold must be the boolean {target}.
Class rule: {class_rule}
The sarcasm, when the answer is true, must be context contrast (praise wording about a clearly bad situation), not an emoji and not a slash followed by the letter s.
The question asks whether the statement is sarcastic, and it must not state the answer.
Situation length: {length}. For Chinese and Japanese, count characters as words.
Also give a one-sentence description of what false means here and what true means here, in the requested language.
JSON fields: state (the full text to classify, in the requested language), instructions (the question), false_description, true_description, gold (boolean {target}), rationale (one short sentence, not repeated elsewhere)."""

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


def _label_block(labels, gloss):
    return "\n".join("- %s: %s" % (key, gloss[key]) for key in labels)


def prompts_digest():
    blob = "\n".join([
        PROMPT_VERSION, GEN_SYSTEM, GEN_SYSTEM_CLASS, VER_SYSTEM,
        _GEN_CHOICE, _GEN_NOUL, _GEN_SCORE, _GEN_READING,
        _GEN_SENTIMENT, _GEN_EMOTION, _GEN_COMPLAINT_NOUL, _GEN_COMPLAINT_CHOICE,
        _GEN_URGENCY, _GEN_SARCASM,
        _VER_CHOICE, _VER_NOUL, _VER_SCORE,
    ])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _class_messages(user, schema, npredict):
    return [{"role": "system", "content": GEN_SYSTEM_CLASS},
            {"role": "user", "content": user}], schema, npredict


def generate_request(seed):
    """Return (messages, schema, num_predict) for one seed. No gold is known yet."""
    lang = LANG_NAME[seed["lang"]]
    task = seed["task"]
    if task == "business" and seed["form"] == "choice":
        user = _GEN_CHOICE.format(
            language=lang, code=seed["lang"], domain=seed["domain_brief"],
            persona=seed["persona"], n_options=seed["n_options"], industry=seed["industry"],
            register=seed["register"], length=seed["length"], difficulty=seed["difficulty"])
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user}], SCHEMA_CHOICE, 900
    if task == "business":
        user = _GEN_NOUL.format(
            language=lang, code=seed["lang"], domain=seed["domain_brief"],
            persona=seed["persona"], industry=seed["industry"], register=seed["register"],
            length=seed["length"], difficulty=seed["difficulty"], target=seed["target"])
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user}], SCHEMA_NOUL, 900
    if task == "score":
        user = _GEN_SCORE.format(
            language=lang, code=seed["lang"], persona=seed["persona"],
            aspect=seed["aspect"], aspect_help=seed["aspect_help"],
            n_levels=seed["n_levels"], industry=seed["industry"], register=seed["register"],
            length=seed["length"], target=seed["target"])
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user}], SCHEMA_SCORE, 800
    if task == "reading":
        user = _GEN_READING.format(
            language=lang, code=seed["lang"], topic=seed["topic"],
            persona=seed["persona"], n_options=seed["n_options"], difficulty=seed["difficulty"])
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": user}], SCHEMA_READING, 1400
    if task == "sentiment":
        if seed.get("variant") == "aspect":
            block = (
                "Scope: aspect sentiment. The situation must mention every aspect in this list, "
                "expressed in the requested language: %s. "
                "The question asks for the sentiment toward only this aspect, expressed in the requested language: %s. "
                "The sentiment toward that aspect is %s. "
                "At least one other aspect must clearly have a different sentiment, so the tone of the whole text is not enough to answer."
                % (", ".join(seed.get("aspects") or []), seed.get("focus") or "", seed["target"])
            )
        else:
            block = "Scope: the question asks for the sentiment of the whole text, not of one aspect."
        user = _GEN_SENTIMENT.format(
            language=lang, code=seed["lang"], industry=seed["industry"], register=seed["register"],
            persona=seed["persona"], difficulty=seed["difficulty"], length=seed["length"],
            labels=_label_block(SENTIMENT_LABELS, _SENTIMENT_HELP), target=seed["target"],
            class_rule=_SENTIMENT_RULE[seed["target"]], variant_block=block)
        return _class_messages(user, SCHEMA_CHOICE, 1400)
    if task == "emotion":
        user = _GEN_EMOTION.format(
            language=lang, code=seed["lang"], industry=seed["industry"], genre=seed["genre"],
            register=seed["register"], persona=seed["persona"], difficulty=seed["difficulty"],
            length=seed["length"], labels=_label_block(EMOTION_LABELS, _EMOTION_HELP),
            target=seed["target"])
        return _class_messages(user, SCHEMA_CHOICE, 1400)
    if task == "complaint" and seed.get("form") == "choice":
        user = _GEN_COMPLAINT_CHOICE.format(
            language=lang, code=seed["lang"], industry=seed["industry"], topic=seed["topic"],
            register=seed["register"], persona=seed["persona"], difficulty=seed["difficulty"],
            target=seed["target"], length=seed["length"],
            labels=_label_block(COMPLAINT_LABELS, _COMPLAINT_HELP))
        return _class_messages(user, SCHEMA_CHOICE, 1400)
    if task == "complaint":
        user = _GEN_COMPLAINT_NOUL.format(
            language=lang, code=seed["lang"], industry=seed["industry"], topic=seed["topic"],
            register=seed["register"], persona=seed["persona"], difficulty=seed["difficulty"],
            target=seed["target"], class_rule=_COMPLAINT_RULE[seed["target"]], length=seed["length"])
        return _class_messages(user, SCHEMA_NOUL, 1000)
    if task == "urgency":
        user = _GEN_URGENCY.format(
            language=lang, code=seed["lang"], industry=seed["industry"], register=seed["register"],
            persona=seed["persona"], difficulty=seed["difficulty"], length=seed["length"],
            n_levels=seed["n_levels"], target=seed["target"],
            level_meanings="\n".join("- %s" % line for line in seed["levels_brief"]))
        return _class_messages(user, SCHEMA_SCORE, 1000)
    if task == "sarcasm":
        user = _GEN_SARCASM.format(
            language=lang, code=seed["lang"], industry=seed["industry"], register=seed["register"],
            persona=seed["persona"], difficulty=seed["difficulty"], target=seed["target"],
            class_rule=_SARCASM_RULE[seed["target"]], length=seed["length"])
        return _class_messages(user, SCHEMA_NOUL, 1000)
    raise SystemExit("unknown task %s" % task)


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
