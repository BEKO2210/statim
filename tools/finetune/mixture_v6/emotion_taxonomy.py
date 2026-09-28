"""One emotion taxonomy for the v6 emotion sources that opt in with ``"emotion_taxonomy": "basic8"``.

The eight classes are Ekman's six basic emotions (anger, disgust, fear, joy, sadness, surprise; Ekman 1992,
"An argument for basic emotions"), plus love, a basic-level category in Shaver et al. (1987, "Emotion
knowledge", J. Pers. Soc. Psychol. 52(6)), plus neutral for the absence of a strong emotion. These are exactly
the labels templates.GLOSSES["emotion"] describes in all 14 languages.

A source label maps to a class when it is the class itself, a synonym, or a subordinate emotion that Shaver
et al. place under that class (pride and relief under joy, disappointment under sadness, anxiety and
nervousness under fear, irritation under anger). Any other label (trust, gratitude, anticipation, guilt, ...)
has no class: the row is dropped, not forced into the nearest class. Labels are matched case-insensitively
after the registry's human() normalisation.

tools/finetune/sources/v6-research.md ("Emotion taxonomy") lists the mapping per source.
"""

from __future__ import annotations

NAME = "basic8"
CLASSES = ("anger", "disgust", "fear", "joy", "love", "sadness", "surprise", "neutral")

_MAP = {
    # the classes themselves and plain synonyms
    "anger": "anger", "angry": "anger", "rage": "anger", "irritation": "anger", "annoyance": "anger",
    "disgust": "disgust", "revulsion": "disgust",
    "fear": "fear", "afraid": "fear", "anxiety": "fear", "nervousness": "fear", "worry": "fear",
    "joy": "joy", "happiness": "joy", "happy": "joy", "pride": "joy", "relief": "joy", "contentment": "joy",
    "love": "love", "affection": "love",
    "sadness": "sadness", "sad": "sadness", "sorrow": "sadness", "grief": "sadness", "disappointment": "sadness",
    "surprise": "surprise", "surprised": "surprise", "amazement": "surprise", "astonishment": "surprise",
    "neutral": "neutral", "no emotion": "neutral", "none": "neutral",
}
# Labels seen in an opted-in source that deliberately have no class (documentation and tests).
UNMAPPED = frozenset({"trust", "gratitude", "anticipation", "guilt", "hope", "jealousy", "confusion"})


def to_class(label):
    """The taxonomy class of a source label, or None when the label has no class."""
    key = " ".join(str(label).replace("_", " ").split()).lower()
    return _MAP.get(key)


def map_labels(labels):
    """Mapped labels of one row, or None when any label has no class (the row is dropped: a partly
    mapped multi-label row would make a yes/no probe on a missing emotion wrong)."""
    out = []
    for label in labels:
        mapped = to_class(label)
        if mapped is None:
            return None
        if mapped not in out:
            out.append(mapped)
    return out
