"""Prompt templates. Slots are filled from the seed table. No filled-in examples.

PROMPT_VERSION is recorded in the manifest. Bump it when a template changes.
"""
import hashlib

from seeds import COMPLAINT_LABELS, EMOTION_LABELS, LANGS, PRODUCT_ASPECTS, SENTIMENT_LABELS, TASKS

PROMPT_VERSION = "synth-prompts-7"

# The situation must obey a seed opening and must contain one concrete fact.
_OPENING_DETAIL = (
    "Opening rule: {opening}\n"
    "The text must also contain this concrete fact somewhere after the first sentence, its words translated into the requested language and its numbers kept: {detail}.\n"
    "Do not open the text like a diary entry. The opening rule decides the first words.\n"
)

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
""" + _OPENING_DETAIL + """Form: one choice question. Use exactly these keys, each with a one-sentence description in the requested language. Do not copy the English gloss into the description.
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
""" + _OPENING_DETAIL + """Form: one choice question. Use exactly these keys, each with a one-sentence description in the requested language. Do not copy the English gloss into the description.
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
""" + _OPENING_DETAIL + """
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
""" + _OPENING_DETAIL + """
JSON fields: state (the full text to classify, in the requested language), instructions (the question), options (array of {{key, description}}), gold (exactly {target}), rationale (one short sentence, not repeated elsewhere)."""

_GEN_URGENCY = """Language: {language} ({code}). Write every trainee-facing string in this language only.
Setting: {industry}. The text is written as {register}.
Perspective: {persona}
Difficulty: {difficulty}.
Situation length: {length}. For Chinese and Japanese, count characters as words.
""" + _OPENING_DETAIL + """Scale: how soon somebody needs to act.
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
""" + _OPENING_DETAIL + """
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


# Human-written question paraphrases. The situation is the creative part; the
# question is picked from this list at build time. {focus} is filled with a
# bare noun from FOCUS_PHRASE. Six openings per list, first words differ.

FOCUS_PHRASE = {
    "price": {
        "en": "price", "de": "Preis", "fr": "prix", "es": "precio", "it": "prezzo",
        "pt": "preço", "nl": "prijs", "pl": "cena", "tr": "fiyat", "ru": "цена",
        "ar": "السعر", "hi": "कीमत", "ja": "価格", "zh": "价格",
    },
    "delivery time": {
        "en": "delivery time", "de": "Lieferzeit", "fr": "délai de livraison", "es": "plazo de entrega",
        "it": "tempi di consegna", "pt": "prazo de entrega", "nl": "levertijd", "pl": "czas dostawy",
        "tr": "teslim süresi", "ru": "срок доставки", "ar": "مدة التسليم", "hi": "डिलीवरी समय",
        "ja": "配送時間", "zh": "送达时间",
    },
    "product quality": {
        "en": "product quality", "de": "Produktqualität", "fr": "qualité du produit", "es": "calidad del producto",
        "it": "qualità del prodotto", "pt": "qualidade do produto", "nl": "productkwaliteit", "pl": "jakość produktu",
        "tr": "ürün kalitesi", "ru": "качество товара", "ar": "جودة المنتج", "hi": "उत्पाद गुणवत्ता",
        "ja": "製品の品質", "zh": "产品质量",
    },
    "customer support": {
        "en": "customer support", "de": "Kundendienst", "fr": "service client", "es": "atención al cliente",
        "it": "assistenza clienti", "pt": "apoio ao cliente", "nl": "klantenservice", "pl": "obsługa klienta",
        "tr": "müşteri desteği", "ru": "поддержка клиентов", "ar": "دعم العملاء", "hi": "ग्राहक सहायता",
        "ja": "顧客対応", "zh": "客户支持",
    },
    "packaging": {
        "en": "packaging", "de": "Verpackung", "fr": "emballage", "es": "embalaje", "it": "imballaggio",
        "pt": "embalagem", "nl": "verpakking", "pl": "opakowanie", "tr": "ambalaj", "ru": "упаковка",
        "ar": "التغليف", "hi": "पैकेजिंग", "ja": "包装", "zh": "包装",
    },
    "ease of use": {
        "en": "ease of use", "de": "Bedienung", "fr": "facilité d'usage", "es": "facilidad de uso",
        "it": "facilità d'uso", "pt": "facilidade de uso", "nl": "gebruiksgemak", "pl": "łatwość użycia",
        "tr": "kullanım kolaylığı", "ru": "удобство", "ar": "سهولة الاستخدام", "hi": "उपयोग में आसानी",
        "ja": "使いやすさ", "zh": "易用性",
    },
    "durability": {
        "en": "durability", "de": "Haltbarkeit", "fr": "durabilité", "es": "durabilidad", "it": "durata",
        "pt": "durabilidade", "nl": "duurzaamheid", "pl": "trwałość", "tr": "dayanıklılık", "ru": "прочность",
        "ar": "المتانة", "hi": "टिकाऊपन", "ja": "耐久性", "zh": "耐用性",
    },
    "size or fit": {
        "en": "size or fit", "de": "Größe oder Passform", "fr": "taille ou coupe", "es": "talla o ajuste",
        "it": "taglia o vestibilità", "pt": "tamanho ou caimento", "nl": "maat of pasvorm", "pl": "rozmiar lub krój",
        "tr": "beden veya kalıp", "ru": "размер или посадка", "ar": "المقاس أو الملاءمة", "hi": "आकार या फिट",
        "ja": "サイズや着心地", "zh": "尺寸或合身",
    },
    "cleanliness": {
        "en": "cleanliness", "de": "Sauberkeit", "fr": "propreté", "es": "limpieza", "it": "pulizia",
        "pt": "limpeza", "nl": "netheid", "pl": "czystość", "tr": "temizlik", "ru": "чистота",
        "ar": "النظافة", "hi": "सफाई", "ja": "清潔さ", "zh": "清洁程度",
    },
    "waiting time": {
        "en": "waiting time", "de": "Wartezeit", "fr": "temps d'attente", "es": "tiempo de espera",
        "it": "tempo di attesa", "pt": "tempo de espera", "nl": "wachttijd", "pl": "czas oczekiwania",
        "tr": "bekleme süresi", "ru": "время ожидания", "ar": "مدة الانتظار", "hi": "प्रतीक्षा समय",
        "ja": "待ち時間", "zh": "等待时间",
    },
    "staff behaviour": {
        "en": "staff behaviour", "de": "Verhalten des Personals", "fr": "comportement du personnel",
        "es": "trato del personal", "it": "comportamento del personale", "pt": "conduta da equipe",
        "nl": "gedrag van het personeel", "pl": "zachowanie personelu", "tr": "personel davranışı",
        "ru": "поведение персонала", "ar": "سلوك الموظفين", "hi": "कर्मचारी व्यवहार",
        "ja": "スタッフの対応", "zh": "员工态度",
    },
    "the returns process": {
        "en": "the returns process", "de": "Rückgabeablauf", "fr": "procédure de retour", "es": "proceso de devolución",
        "it": "procedura di reso", "pt": "processo de devolução", "nl": "retourproces", "pl": "proces zwrotu",
        "tr": "iade süreci", "ru": "процедура возврата", "ar": "إجراءات الإرجاع", "hi": "वापसी प्रक्रिया",
        "ja": "返品の手順", "zh": "退货流程",
    },
    "battery life": {
        "en": "battery life", "de": "Akkulaufzeit", "fr": "autonomie de la batterie", "es": "duración de la batería",
        "it": "durata della batteria", "pt": "duração da bateria", "nl": "accuduur", "pl": "czas pracy baterii",
        "tr": "pil ömrü", "ru": "время работы батареи", "ar": "عمر البطارية", "hi": "बैटरी जीवन",
        "ja": "電池の持ち", "zh": "电池续航",
    },
    "accuracy of the description": {
        "en": "accuracy of the description", "de": "Genauigkeit der Beschreibung", "fr": "exactitude de la description",
        "es": "exactitud de la descripción", "it": "accuratezza della descrizione", "pt": "exatidão da descrição",
        "nl": "nauwkeurigheid van de beschrijving", "pl": "trafność opisu", "tr": "açıklamanın doğruluğu",
        "ru": "точность описания", "ar": "دقة الوصف", "hi": "विवरण की सटीकता",
        "ja": "説明の正確さ", "zh": "描述的准确性",
    },
}

_SENTIMENT_OVERALL = {
    "en": (
        "Which sentiment fits this text as a whole?",
        "How would you describe the writer's overall opinion?",
        "What view of the subject does the full text show?",
        "Classify the tone of this entire message.",
        "The whole text supports which description of opinion?",
        "Read the full text and pick the matching opinion.",
    ),
    "de": (
        "Welche Stimmung passt zu diesem Text insgesamt?",
        "Wie lässt sich die Meinung der schreibenden Person zusammenfassen?",
        "Welche Sicht auf das Thema zeigt der ganze Text?",
        "Ordne den Ton dieser gesamten Nachricht ein.",
        "Der vollständige Text stützt welche Beschreibung der Meinung?",
        "Lies den ganzen Text und wähle die passende Meinung.",
    ),
    "fr": (
        "Quel sentiment correspond à ce texte dans son ensemble ?",
        "Comment décrire l'opinion globale de la personne qui écrit ?",
        "Quel regard sur le sujet le texte complet montre-t-il ?",
        "Classe le ton de ce message entier.",
        "Le texte complet soutient quelle description de l'opinion ?",
        "Lis le texte entier et choisis l'opinion qui correspond.",
    ),
    "es": (
        "¿Qué sentimiento encaja con este texto en conjunto?",
        "¿Cómo describirías la opinión global de quien escribe?",
        "¿Qué visión del tema muestra el texto completo?",
        "Clasifica el tono de este mensaje entero.",
        "El texto completo respalda cuál descripción de la opinión?",
        "Lee el texto entero y elige la opinión que corresponde.",
    ),
    "it": (
        "Quale sentimento corrisponde a questo testo nel suo insieme?",
        "Come descriveresti l'opinione complessiva di chi scrive?",
        "Quale visione dell'argomento mostra il testo intero?",
        "Classifica il tono di questo messaggio per intero.",
        "Il testo completo sostiene quale descrizione dell'opinione?",
        "Leggi il testo intero e scegli l'opinione corrispondente.",
    ),
    "pt": (
        "Que sentimento corresponde a este texto no conjunto?",
        "Como descrever a opinião global de quem escreve?",
        "Que visão do assunto o texto completo mostra?",
        "Classifique o tom desta mensagem inteira.",
        "O texto completo sustenta qual descrição da opinião?",
        "Leia o texto inteiro e escolha a opinião correspondente.",
    ),
    "nl": (
        "Welk sentiment past bij deze tekst als geheel?",
        "Hoe zou je de algemene mening van de schrijver omschrijven?",
        "Welke kijk op het onderwerp toont de volledige tekst?",
        "Rubriceer de toon van dit hele bericht.",
        "De volledige tekst ondersteunt welke omschrijving van de mening?",
        "Lees de hele tekst en kies de passende mening.",
    ),
    "pl": (
        "Jaki wydźwięk pasuje do tego tekstu jako całości?",
        "Jak opisać ogólną opinię osoby, która pisze?",
        "Jakie spojrzenie na temat pokazuje cały tekst?",
        "Sklasyfikuj ton tej całej wiadomości.",
        "Pełny tekst wspiera który opis opinii?",
        "Przeczytaj cały tekst i wybierz pasującą opinię.",
    ),
    "tr": (
        "Bu metnin bütününe hangi duygu durumu uyar?",
        "Yazan kişinin genel görüşünü nasıl tanımlarsınız?",
        "Konuya dair hangi bakışı tam metin gösteriyor?",
        "Bu iletinin tümünün tonunu sınıflandırın.",
        "Metnin tamamı görüşün hangi tarifini destekliyor?",
        "Metnin tamamını okuyup uyan görüşü seçin.",
    ),
    "ru": (
        "Какое настроение соответствует этому тексту в целом?",
        "Как описать общее мнение того, кто пишет?",
        "Какой взгляд на тему показывает весь текст?",
        "Определи тон этого сообщения целиком.",
        "Полный текст поддерживает какое описание мнения?",
        "Прочитай весь текст и выбери подходящее мнение.",
    ),
    "ar": (
        "ما الشعور الذي يناسب هذا النص بأكمله؟",
        "كيف تصف الرأي العام لمن كتب هذا النص؟",
        "أي نظرة إلى الموضوع يظهرها النص الكامل؟",
        "صنف نبرة هذه الرسالة كلها.",
        "أي وصف للرأي يدعمه النص الكامل؟",
        "اقرأ النص كله واختر الرأي المطابق.",
    ),
    "hi": (
        "इस पूरे पाठ के लिए कौन सा भाव उपयुक्त है?",
        "लिखने वाले की समग्र राय को आप कैसे बताएँगे?",
        "विषय पर कौन सा दृष्टिकोण पूरा पाठ दिखाता है?",
        "इस पूरे संदेश का लहजा वर्गीकृत कीजिए।",
        "पूरा पाठ राय के किस वर्णन का समर्थन करता है?",
        "पूरा पाठ पढ़कर मेल खाती राय चुनिए।",
    ),
    "ja": (
        "文章全体にはどの印象が当てはまりますか。",
        "書き手の総合的な見方をどう表しますか。",
        "内容の全体が示す意見はどれですか。",
        "全文の論調を分類してください。",
        "メッセージ全体はどの意見の説明に合いますか。",
        "通して読んだとき合う評価を選んでください。",
    ),
    "zh": (
        "整段文字的评价属于哪一种？",
        "作者对事情的总体看法该如何描述？",
        "通篇内容表现出的态度该怎么归类？",
        "请为全文的论调选择匹配的一项。",
        "从完整内容看，意见属于哪一类？",
        "把全文读完后，选出相符的看法。",
    ),
}

_SENTIMENT_ASPECT = {
    "en": (
        "What sentiment applies only to the topic {focus}?",
        "How is the topic {focus} judged in this text?",
        "Ignore every other point and classify the topic {focus}.",
        "The opinion on the topic {focus} matches which description?",
        "Pick the view that applies to the topic {focus} alone.",
        "Which description fits the remarks about the topic {focus}?",
    ),
    "de": (
        "Welche Stimmung gilt nur für das Thema {focus}?",
        "Wie wird das Thema {focus} in diesem Text beurteilt?",
        "Lass alle anderen Punkte weg und ordne das Thema {focus} ein.",
        "Die Meinung zum Thema {focus} passt zu welcher Beschreibung?",
        "Wähle die Sicht, die allein das Thema {focus} betrifft.",
        "Welche Beschreibung trifft auf die Bemerkungen zum Thema {focus} zu?",
    ),
    "fr": (
        "Quel sentiment vaut seulement pour le thème {focus} ?",
        "Comment le thème {focus} est-il jugé dans ce texte ?",
        "Laisse de côté les autres points et classe le thème {focus}.",
        "L'opinion sur le thème {focus} correspond à quelle description ?",
        "Choisis la vue qui concerne seulement le thème {focus}.",
        "Quelle description convient aux remarques sur le thème {focus} ?",
    ),
    "es": (
        "¿Qué sentimiento vale solo para el tema {focus}?",
        "¿Cómo se juzga el tema {focus} en este texto?",
        "Deja de lado los demás puntos y clasifica el tema {focus}.",
        "La opinión sobre el tema {focus} coincide con cuál descripción?",
        "Elige la visión que afecta solo al tema {focus}.",
        "¿Cuál descripción encaja con los comentarios sobre el tema {focus}?",
    ),
    "it": (
        "Quale sentimento vale solo per il tema {focus}?",
        "Come viene giudicato il tema {focus} in questo testo?",
        "Lascia da parte gli altri punti e classifica il tema {focus}.",
        "L'opinione sul tema {focus} corrisponde a quale descrizione?",
        "Scegli la visione che riguarda solo il tema {focus}.",
        "Quale descrizione si adatta ai commenti sul tema {focus}?",
    ),
    "pt": (
        "Que sentimento vale apenas para o tema {focus}?",
        "Como o tema {focus} é julgado neste texto?",
        "Deixe os outros pontos de lado e classifique o tema {focus}.",
        "A opinião sobre o tema {focus} combina com qual descrição?",
        "Escolha a visão que trata só do tema {focus}.",
        "Qual descrição cabe aos comentários sobre o tema {focus}?",
    ),
    "nl": (
        "Welk sentiment geldt alleen voor het thema {focus}?",
        "Hoe wordt het thema {focus} in deze tekst beoordeeld?",
        "Laat de andere punten weg en rubriceer het thema {focus}.",
        "De mening over het thema {focus} past bij welke omschrijving?",
        "Kies de kijk die alleen het thema {focus} raakt.",
        "Welke omschrijving past bij de opmerkingen over het thema {focus}?",
    ),
    "pl": (
        "Jaki wydźwięk dotyczy tylko tematu {focus}?",
        "Jak temat {focus} jest oceniony w tym tekście?",
        "Pomiń pozostałe punkty i sklasyfikuj temat {focus}.",
        "Opinia o temacie {focus} pasuje do którego opisu?",
        "Wybierz spojrzenie, które dotyczy samego tematu {focus}.",
        "Który opis pasuje do uwag o temacie {focus}?",
    ),
    "tr": (
        "Yalnızca {focus} konusu için hangi duygu durumu geçerlidir?",
        "{focus} konusu bu metinde nasıl değerlendiriliyor?",
        "Diğer noktaları bırakıp {focus} konusunu sınıflandırın.",
        "{focus} konusundaki görüş hangi tarife uyar?",
        "Yalnızca {focus} konusunu ilgilendiren bakışı seçin.",
        "{focus} konusu hakkındaki sözlere hangi tarif uyar?",
    ),
    "ru": (
        "Какое настроение относится только к теме {focus}?",
        "Как в этом тексте оценивается тема {focus}?",
        "Отложи остальные пункты и классифицируй тему {focus}.",
        "Мнение о теме {focus} подходит к какому описанию?",
        "Выбери взгляд, который касается только темы {focus}.",
        "Какое описание подходит к замечаниям о теме {focus}?",
    ),
    "ar": (
        "أي شعور ينطبق فقط على موضوع {focus}؟",
        "كيف يُقيَّم موضوع {focus} في هذا النص؟",
        "تجاهل النقاط الأخرى وصنف موضوع {focus}.",
        "الرأي حول موضوع {focus} يطابق أي وصف؟",
        "اختر النظرة التي تخص موضوع {focus} وحده.",
        "أي وصف يناسب الملاحظات عن موضوع {focus}؟",
    ),
    "hi": (
        "केवल विषय {focus} पर कौन सा भाव लागू होता है?",
        "इस पाठ में विषय {focus} का आकलन कैसे हुआ है?",
        "बाकी बातें छोड़कर विषय {focus} को वर्गीकृत कीजिए।",
        "विषय {focus} पर राय किस वर्णन से मेल खाती है?",
        "वह दृष्टि चुनिए जो सिर्फ विषय {focus} पर है।",
        "विषय {focus} की टिप्पणियों पर कौन सा वर्णन बैठता है?",
    ),
    "ja": (
        "話題の{focus}だけに当てはまる印象はどれですか。",
        "本文では{focus}がどのように評価されていますか。",
        "他の点は置いて{focus}だけを分類してください。",
        "{focus}についての意見はどの説明に合いますか。",
        "{focus}だけに関する見方を選んでください。",
        "{focus}への言及に合う説明はどれですか。",
    ),
    "zh": (
        "只看话题{focus}时，评价属于哪一种？",
        "文中对{focus}的判断是怎样的？",
        "先不管其他点，只给{focus}归类。",
        "关于{focus}的意见符合哪一项说明？",
        "请选出只针对{focus}的那种看法。",
        "谈到{focus}的那些话适合哪一种说明？",
    ),
}
_EMOTION = {
    "en": (
        "Which emotion does this text express?",
        "What feeling comes through most clearly here?",
        "The writer shows which emotional state?",
        "Name the emotion that fits the person writing.",
        "This message conveys which emotion?",
        "Select the emotion expressed by the writer.",
    ),
    "de": (
        "Welche Emotion drückt dieser Text aus?",
        "Welches Gefühl kommt hier am deutlichsten durch?",
        "Die schreibende Person zeigt welchen Gefühlszustand?",
        "Benenne die Emotion, die zur Person passt.",
        "Diese Nachricht vermittelt welche Emotion?",
        "Wähle die Emotion, die der Text ausdrückt.",
    ),
    "fr": (
        "Quelle émotion ce texte exprime-t-il ?",
        "Quel ressenti ressort le plus clairement ici ?",
        "La personne qui écrit montre quel état émotionnel ?",
        "Nomme l'émotion qui correspond à la personne.",
        "Ce message transmet quelle émotion ?",
        "Choisis l'émotion exprimée par le texte.",
    ),
    "es": (
        "¿Qué emoción expresa este texto?",
        "¿Qué sentir se nota con más claridad aquí?",
        "Quien escribe muestra cuál estado emocional?",
        "Nombra la emoción que encaja con la persona.",
        "Este mensaje transmite cuál emoción?",
        "Elige la emoción que el texto expresa.",
    ),
    "it": (
        "Quale emozione esprime questo testo?",
        "Quale sentire emerge con più chiarezza qui?",
        "Chi scrive mostra quale stato emotivo?",
        "Indica l'emozione che corrisponde alla persona.",
        "Questo messaggio trasmette quale emozione?",
        "Scegli l'emozione espressa dal testo.",
    ),
    "pt": (
        "Que emoção este texto exprime?",
        "Que sentir aparece com mais clareza aqui?",
        "Quem escreve mostra qual estado emocional?",
        "Indique a emoção que corresponde à pessoa.",
        "Esta mensagem transmite qual emoção?",
        "Escolha a emoção expressa pelo texto.",
    ),
    "nl": (
        "Welke emotie drukt deze tekst uit?",
        "Welk gevoel komt hier het duidelijkst naar voren?",
        "De schrijver toont welke gemoedstoestand?",
        "Noem de emotie die bij de persoon past.",
        "Dit bericht brengt welke emotie over?",
        "Kies de emotie die de tekst uitdrukt.",
    ),
    "pl": (
        "Jaką emocję wyraża ten tekst?",
        "Jakie uczucie przebija tu najmocniej?",
        "Osoba pisząca pokazuje jaki stan uczuciowy?",
        "Nazwij emocję, która pasuje do osoby.",
        "Ta wiadomość przekazuje jaką emocję?",
        "Wybierz emocję wyrażoną w tekście.",
    ),
    "tr": (
        "Bu metin hangi duyguyu ifade ediyor?",
        "Burada en açık hangi his öne çıkıyor?",
        "Yazan kişi hangi duygu durumunu gösteriyor?",
        "Kişiye uyan duyguyu adlandırın.",
        "Bu ileti hangi duyguyu aktarıyor?",
        "Metnin ifade ettiği duyguyu seçin.",
    ),
    "ru": (
        "Какую эмоцию выражает этот текст?",
        "Какое чувство здесь заметнее всего?",
        "Пишущий человек показывает какое душевное состояние?",
        "Назови эмоцию, которая подходит человеку.",
        "Это сообщение передаёт какую эмоцию?",
        "Выбери эмоцию, выраженную в тексте.",
    ),
    "ar": (
        "أي شعور يعبّر عنه هذا النص؟",
        "أي إحساس يظهر هنا بوضوح أكبر؟",
        "ما الحالة الوجدانية التي يُظهرها من كتب؟",
        "سمّ الشعور الذي يلائم الشخص.",
        "أي شعور تنقله هذه الرسالة؟",
        "اختر الشعور الذي يعبّر عنه النص.",
    ),
    "hi": (
        "यह पाठ कौन सी भावना व्यक्त करता है?",
        "यहाँ सबसे स्पष्ट कौन सा अहसास उभरता है?",
        "लिखने वाला कौन सी भाव-स्थिति दिखाता है?",
        "व्यक्ति पर बैठने वाली भावना का नाम लिखिए।",
        "यह संदेश कौन सी भावना पहुँचाता है?",
        "पाठ में व्यक्त भावना चुनिए।",
    ),
    "ja": (
        "この文章はどの感情を表していますか。",
        "ここで最もはっきり出る気持ちは何ですか。",
        "書き手が示す心の状態はどれですか。",
        "その人に合う感情を挙げてください。",
        "この伝言はどの感情を伝えていますか。",
        "本文が表す感情を選んでください。",
    ),
    "zh": (
        "这段文字表达了哪种情绪？",
        "这里最清楚的感受是什么？",
        "写的人呈现出哪一种心情？",
        "请指出适合这个人的情绪。",
        "这条消息传递了哪种情绪？",
        "选出文字所表达的情绪。",
    ),
}

_COMPLAINT_NOUL = {
    "en": (
        "Is this message a complaint?",
        "Does the writer complain in this message?",
        "Is the writer raising a complaint here?",
        "Does this message bring up a complaint?",
        "Should this text be read as a complaint?",
        "Is there a complaint in what the writer says?",
    ),
    "de": (
        "Ist diese Nachricht eine Beschwerde?",
        "Beschwert sich die schreibende Person in dieser Nachricht?",
        "Bringt die Person hier eine Beschwerde vor?",
        "Kommt in dieser Nachricht eine Beschwerde vor?",
        "Soll dieser Text als Beschwerde gelesen werden?",
        "Steht in den Worten der Person eine Beschwerde?",
    ),
    "fr": (
        "Ce message est-il une plainte ?",
        "La personne qui écrit se plaint-elle dans ce message ?",
        "La personne formule-t-elle une plainte ici ?",
        "Ce message contient-il une plainte ?",
        "Faut-il lire ce texte comme une plainte ?",
        "Y a-t-il une plainte dans ce que dit la personne ?",
    ),
    "es": (
        "¿Este mensaje es una queja?",
        "¿Quien escribe se queja en este mensaje?",
        "¿La persona plantea una queja aquí?",
        "¿Este mensaje trae una queja?",
        "¿Hay que leer este texto como una queja?",
        "¿Hay una queja en lo que dice la persona?",
    ),
    "it": (
        "Questo messaggio è un reclamo?",
        "Chi scrive si lamenta in questo messaggio?",
        "La persona presenta un reclamo qui?",
        "Questo messaggio contiene un reclamo?",
        "Questo testo va letto come un reclamo?",
        "C'è un reclamo in ciò che dice la persona?",
    ),
    "pt": (
        "Esta mensagem é uma reclamação?",
        "Quem escreve reclama nesta mensagem?",
        "A pessoa apresenta uma reclamação aqui?",
        "Esta mensagem traz uma reclamação?",
        "Este texto deve ser lido como uma reclamação?",
        "Há uma reclamação no que a pessoa diz?",
    ),
    "nl": (
        "Is dit bericht een klacht?",
        "Klaagt de schrijver in dit bericht?",
        "Brengt de persoon hier een klacht naar voren?",
        "Komt er in dit bericht een klacht voor?",
        "Moet deze tekst als een klacht gelezen worden?",
        "Staat er een klacht in wat de persoon zegt?",
    ),
    "pl": (
        "Czy ta wiadomość jest skargą?",
        "Czy osoba pisząca skarży się w tej wiadomości?",
        "Czy osoba zgłasza tu skargę?",
        "Czy w tej wiadomości jest skarga?",
        "Czy ten tekst należy czytać jako skargę?",
        "Czy w słowach osoby jest skarga?",
    ),
    "tr": (
        "Bu ileti bir şikâyet mi?",
        "Yazan kişi bu iletide şikâyet ediyor mu?",
        "Kişi burada bir şikâyet mi getiriyor?",
        "Bu iletide bir şikâyet var mı?",
        "Bu metin şikâyet olarak mı okunmalı?",
        "Kişinin sözlerinde bir şikâyet var mı?",
    ),
    "ru": (
        "Это сообщение является жалобой?",
        "Жалуется ли пишущий человек в этом сообщении?",
        "Человек высказывает здесь жалобу?",
        "Есть ли в этом сообщении жалоба?",
        "Нужно ли читать этот текст как жалобу?",
        "Есть ли жалоба в словах человека?",
    ),
    "ar": (
        "هل هذه الرسالة شكوى؟",
        "هل يشتكي من كتب في هذه الرسالة؟",
        "هل ي Raise الشخص شكوى هنا؟",
        "هل تتضمن هذه الرسالة شكوى؟",
        "هل يُقرأ هذا النص على أنه شكوى؟",
        "هل توجد شكوى في كلام الشخص؟",
    ),
    "hi": (
        "क्या यह संदेश एक शिकायत है?",
        "क्या लिखने वाला इस संदेश में शिकायत करता है?",
        "क्या व्यक्ति यहाँ शिकायत उठा रहा है?",
        "क्या इस संदेश में शिकायत आती है?",
        "क्या इस पाठ को शिकायत की तरह पढ़ना चाहिए?",
        "क्या व्यक्ति की बात में शिकायत है?",
    ),
    "ja": (
        "この伝言は苦情ですか。",
        "書き手はこの文章で不満を述べていますか。",
        "ここで苦情を出していますか。",
        "本文の中に苦情はありますか。",
        "この文面は苦情として読めますか。",
        "相手の言葉に苦情は含まれますか。",
    ),
    "zh": (
        "这条消息是不是投诉？",
        "写的人是不是在抱怨？",
        "这里有没有提出不满？",
        "这段话里包含投诉吗？",
        "这段文字应当当成投诉来读吗？",
        "对方的话里有没有抱怨？",
    ),
}

_COMPLAINT_CHOICE = {
    "en": (
        "Which complaint category fits this message?",
        "What kind of complaint is this?",
        "The complaint belongs to which category?",
        "How should this complaint be categorised?",
        "Pick the category of this complaint.",
        "Which type of complaint matches this message?",
    ),
    "de": (
        "Welche Beschwerdekategorie passt zu dieser Nachricht?",
        "Was für eine Beschwerde ist das?",
        "Die Beschwerde gehört in welche Kategorie?",
        "Wie soll diese Beschwerde eingeordnet werden?",
        "Wähle die Kategorie dieser Beschwerde.",
        "Welcher Typ von Beschwerde trifft auf diese Nachricht zu?",
    ),
    "fr": (
        "Quelle catégorie de plainte convient à ce message ?",
        "De quelle sorte de plainte s'agit-il ?",
        "La plainte appartient à quelle catégorie ?",
        "Comment faut-il classer cette plainte ?",
        "Choisis la catégorie de cette plainte.",
        "Quel type de plainte correspond à ce message ?",
    ),
    "es": (
        "¿Qué categoría de queja encaja con este mensaje?",
        "¿Qué clase de queja es esta?",
        "La queja pertenece a cuál categoría?",
        "¿Cómo hay que clasificar esta queja?",
        "Elige la categoría de esta queja.",
        "¿Qué tipo de queja corresponde a este mensaje?",
    ),
    "it": (
        "Quale categoria di reclamo si adatta a questo messaggio?",
        "Che tipo di reclamo è questo?",
        "Il reclamo appartiene a quale categoria?",
        "Come va classificato questo reclamo?",
        "Scegli la categoria di questo reclamo.",
        "Quale tipo di reclamo corrisponde a questo messaggio?",
    ),
    "pt": (
        "Que categoria de reclamação cabe a esta mensagem?",
        "Que tipo de reclamação é esta?",
        "A reclamação pertence a qual categoria?",
        "Como se deve classificar esta reclamação?",
        "Escolha a categoria desta reclamação.",
        "Qual tipo de reclamação corresponde a esta mensagem?",
    ),
    "nl": (
        "Welke klachtcategorie past bij dit bericht?",
        "Wat voor klacht is dit?",
        "De klacht hoort in welke categorie?",
        "Hoe moet deze klacht worden ingedeeld?",
        "Kies de categorie van deze klacht.",
        "Welk type klacht komt overeen met dit bericht?",
    ),
    "pl": (
        "Która kategoria skargi pasuje do tej wiadomości?",
        "Jakiego rodzaju jest ta skarga?",
        "Skarga należy do której kategorii?",
        "Jak należy zaklasyfikować tę skargę?",
        "Wybierz kategorię tej skargi.",
        "Który typ skargi odpowiada tej wiadomości?",
    ),
    "tr": (
        "Bu iletiye hangi şikâyet kategorisi uyar?",
        "Bu ne tür bir şikâyettir?",
        "Şikâyet hangi kategoriye girer?",
        "Bu şikâyet nasıl sınıflandırılmalı?",
        "Bu şikâyetin kategorisini seçin.",
        "Bu iletiye hangi şikâyet türü uyar?",
    ),
    "ru": (
        "Какая категория жалобы подходит к этому сообщению?",
        "Что это за жалоба?",
        "Жалоба относится к какой категории?",
        "Как следует классифицировать эту жалобу?",
        "Выбери категорию этой жалобы.",
        "Какой тип жалобы соответствует этому сообщению?",
    ),
    "ar": (
        "أي فئة من الشكاوى تناسب هذه الرسالة؟",
        "ما نوع هذه الشكوى؟",
        "الشكوى تنتمي إلى أي فئة؟",
        "كيف ينبغي تصنيف هذه الشكوى؟",
        "اختر فئة هذه الشكوى.",
        "أي نوع من الشكاوى يطابق هذه الرسالة؟",
    ),
    "hi": (
        "इस संदेश पर कौन सी शिकायत श्रेणी बैठती है?",
        "यह किस तरह की शिकायत है?",
        "शिकायत किस श्रेणी में आती है?",
        "इस शिकायत को कैसे वर्गीकृत करें?",
        "इस शिकायत की श्रेणी चुनिए।",
        "इस संदेश से कौन सा शिकायत प्रकार मेल खाता है?",
    ),
    "ja": (
        "この伝言はどの苦情の種類に当たりますか。",
        "これはどんな苦情ですか。",
        "苦情はどの区分に入りますか。",
        "この苦情をどう分類しますか。",
        "この苦情の区分を選んでください。",
        "本文に合う苦情の型はどれですか。",
    ),
    "zh": (
        "这条消息属于哪一类投诉？",
        "这是什么样的投诉？",
        "这项不满归入哪个类别？",
        "这段投诉应当如何分类？",
        "请选出这段投诉的类别。",
        "哪种投诉类型与这段话相符？",
    ),
}

_URGENCY = {
    "en": (
        "How soon does someone need to act on this?",
        "What time pressure does this message carry?",
        "How quickly should this be handled?",
        "Which time frame fits the need to act?",
        "How pressing is the need to respond?",
        "Which description matches the need to act?",
    ),
    "de": (
        "Wie bald muss jemand hier tätig werden?",
        "Welchen Zeitdruck trägt diese Nachricht?",
        "Wie rasch soll das bearbeitet werden?",
        "Welcher Zeitrahmen passt zum Handlungsbedarf?",
        "Wie dringend ist die nötige Reaktion?",
        "Welche Beschreibung trifft den Handlungsbedarf?",
    ),
    "fr": (
        "Dans quel délai faut-il agir ici ?",
        "Quelle pression de temps porte ce message ?",
        "À quelle vitesse faut-il traiter cela ?",
        "Quel cadre de temps correspond au besoin d'agir ?",
        "Le besoin de répondre est-il pressant, et à quel point ?",
        "Quelle description correspond au besoin d'agir ?",
    ),
    "es": (
        "¿Con qué prontitud hay que actuar aquí?",
        "¿Qué presión de tiempo trae este mensaje?",
        "¿Con qué rapidez hay que atender esto?",
        "¿Qué marco de tiempo encaja con la necesidad de actuar?",
        "¿Qué tan apremiante es la necesidad de responder?",
        "¿Qué descripción corresponde a la necesidad de actuar?",
    ),
    "it": (
        "Entro quanto tempo qualcuno deve agire?",
        "Quale pressione di tempo porta questo messaggio?",
        "Con quanta rapidità va gestito questo caso?",
        "Quale arco di tempo si adatta al bisogno di agire?",
        "Quanto è pressante il bisogno di rispondere?",
        "Quale descrizione corrisponde al bisogno di agire?",
    ),
    "pt": (
        "Em quanto tempo alguém precisa agir?",
        "Que pressão de tempo esta mensagem traz?",
        "Com que rapidez isto deve ser tratado?",
        "Qual intervalo de tempo cabe à necessidade de agir?",
        "Quão premente é a necessidade de responder?",
        "Qual descrição corresponde à necessidade de agir?",
    ),
    "nl": (
        "Hoe snel moet iemand hier handelen?",
        "Welke tijdsdruk draagt dit bericht?",
        "Hoe vlot moet dit worden afgehandeld?",
        "Welk tijdsbestek past bij de nood om te handelen?",
        "Hoe dringend is de nood om te reageren?",
        "Welke omschrijving past bij de nood om te handelen?",
    ),
    "pl": (
        "Jak szybko ktoś musi tu działać?",
        "Jaką presję czasu niesie ta wiadomość?",
        "Jak prędko należy to załatwić?",
        "Jaki przedział czasu pasuje do potrzeby działania?",
        "Jak nagląca jest potrzeba odpowiedzi?",
        "Który opis odpowiada potrzebie działania?",
    ),
    "tr": (
        "Birinin ne kadar çabuk harekete geçmesi gerekir?",
        "Bu ileti ne kadar zaman baskısı taşıyor?",
        "Bu ne kadar hızlı ele alınmalı?",
        "Harekete geçme ihtiyacına hangi süre uyar?",
        "Yanıt verme ihtiyacı ne kadar sıkı?",
        "Harekete geçme ihtiyacına hangi tarif uyar?",
    ),
    "ru": (
        "Как скоро здесь нужно действовать?",
        "Какое давление времени несёт это сообщение?",
        "Как быстро это следует разобрать?",
        "Какой срок соответствует необходимости действовать?",
        "Насколько остро нужно ответить?",
        "Какое описание подходит к необходимости действовать?",
    ),
    "ar": (
        "ما مدى سرعة الحاجة إلى التصرف هنا؟",
        "أي ضغط زمني تحمله هذه الرسالة؟",
        "بأي سرعة ينبغي معالجة هذا الأمر؟",
        "أي إطار زمني يناسب الحاجة إلى التصرف؟",
        "ما مدى إلحاح الحاجة إلى الرد؟",
        "أي وصف يطابق الحاجة إلى التصرف؟",
    ),
    "hi": (
        "यहाँ कितनी जल्दी कदम उठाना होगा?",
        "इस संदेश में कितना समय का दबाव है?",
        "इसे कितनी तेज़ी से संभालना चाहिए?",
        "कार्रवाई की ज़रूरत पर कौन सी समय-सीमा बैठती है?",
        "जवाब की ज़रूरत कितनी तीव्र है?",
        "कार्रवाई की ज़रूरत से कौन सा वर्णन मेल खाता है?",
    ),
    "ja": (
        "ここではどれほど早く動く必要がありますか。",
        "この伝言にはどの程度の時間的な圧がありますか。",
        "これはどれほど迅速に扱うべきですか。",
        "動く必要に合う時間の幅はどれですか。",
        "返事の必要はどれほど差し迫っていますか。",
        "動く必要に合う説明はどれですか。",
    ),
    "zh": (
        "这里需要多快采取行动？",
        "这段话带有多大的时间压力？",
        "这件事应当多快处理？",
        "哪种时间范围符合行动的需要？",
        "作出回应的需要有多紧？",
        "哪种说明符合必须行动的程度？",
    ),
}

_SARCASM = {
    "en": (
        "Is this statement sarcastic?",
        "Does the writer mean the opposite of the praise?",
        "Is the wording ironic rather than literal?",
        "Should this statement be read as sarcasm?",
        "Is the writer being sarcastic here?",
        "Does this text rely on sarcasm?",
    ),
    "de": (
        "Ist diese Aussage sarkastisch?",
        "Meint die Person das Gegenteil des Lobes?",
        "Ist die Formulierung ironisch statt wörtlich?",
        "Soll diese Aussage als Sarkasmus gelesen werden?",
        "Ist die schreibende Person hier sarkastisch?",
        "Beruht dieser Text auf Sarkasmus?",
    ),
    "fr": (
        "Cet énoncé est-il sarcastique ?",
        "La personne veut-elle le contraire de l'éloge ?",
        "La formulation est-elle ironique plutôt que littérale ?",
        "Faut-il lire cet énoncé comme du sarcasme ?",
        "La personne est-elle sarcastique ici ?",
        "Ce texte repose-t-il sur le sarcasme ?",
    ),
    "es": (
        "¿Esta frase es sarcástica?",
        "¿La persona quiere decir lo contrario del elogio?",
        "¿La redacción es irónica en vez de literal?",
        "¿Hay que leer esta frase como sarcasmo?",
        "¿La persona está siendo sarcástica aquí?",
        "¿Este texto se apoya en el sarcasmo?",
    ),
    "it": (
        "Questa frase è sarcastica?",
        "La persona intende il contrario dell'elogio?",
        "La formulazione è ironica invece che letterale?",
        "Questa frase va letta come sarcasmo?",
        "La persona è sarcastica qui?",
        "Questo testo si basa sul sarcasmo?",
    ),
    "pt": (
        "Esta frase é sarcástica?",
        "A pessoa quer dizer o contrário do elogio?",
        "A formulação é irónica em vez de literal?",
        "Esta frase deve ser lida como sarcasmo?",
        "A pessoa está a ser sarcástica aqui?",
        "Este texto assenta no sarcasmo?",
    ),
    "nl": (
        "Is deze uitspraak sarcastisch?",
        "Bedoelt de persoon het tegendeel van de lof?",
        "Is de formulering ironisch in plaats van letterlijk?",
        "Moet deze uitspraak als sarcasme gelezen worden?",
        "Is de persoon hier sarcastisch?",
        "Berust deze tekst op sarcasme?",
    ),
    "pl": (
        "Czy ta wypowiedź jest sarkastyczna?",
        "Czy osoba ma na myśli przeciwieństwo pochwały?",
        "Czy sformułowanie jest ironiczne, a nie dosłowne?",
        "Czy tę wypowiedź należy czytać jako sarkazm?",
        "Czy osoba jest tu sarkastyczna?",
        "Czy ten tekst opiera się na sarkazmie?",
    ),
    "tr": (
        "Bu ifade alaycı mı?",
        "Kişi övgünün tersini mi kastediyor?",
        "Söz kalıbı düz değil de ironik mi?",
        "Bu ifade alay olarak mı okunmalı?",
        "Kişi burada alaycı mı?",
        "Bu metin alaya mı dayanıyor?",
    ),
    "ru": (
        "Это высказывание саркастическое?",
        "Человек имеет в виду противоположность похвале?",
        "Формулировка ироническая, а не буквальная?",
        "Нужно ли читать это высказывание как сарказм?",
        "Человек здесь саркастичен?",
        "Этот текст держится на сарказме?",
    ),
    "ar": (
        "هل هذه العبارة ساخرة؟",
        "هل يقصد الشخص عكس المديح؟",
        "هل الصياغة تهكمية وليست حرفية؟",
        "هل تُقرأ هذه العبارة على أنها سخرية؟",
        "هل الشخص ساخر هنا؟",
        "هل يعتمد هذا النص على السخرية؟",
    ),
    "hi": (
        "क्या यह कथन व्यंग्य है?",
        "क्या व्यक्ति प्रशंसा का उलटा मतलब रखता है?",
        "क्या शब्दावली शाब्दिक नहीं बल्कि व्यंग्यात्मक है?",
        "क्या इस कथन को व्यंग्य की तरह पढ़ना चाहिए?",
        "क्या व्यक्ति यहाँ व्यंग्य कर रहा है?",
        "क्या यह पाठ व्यंग्य पर टिका है?",
    ),
    "ja": (
        "この発言は皮肉ですか。",
        "書き手は褒め言葉と逆のことを意図していますか。",
        "言い回しは文字通りではなく皮肉ですか。",
        "この発言は皮肉として読むべきですか。",
        "ここでは皮肉を使っていますか。",
        "本文は皮肉に依存していますか。",
    ),
    "zh": (
        "这句话是不是讽刺？",
        "写的人是不是意在称赞的反面？",
        "措辞是反话而不是字面意思吗？",
        "这句话应当当成讽刺来读吗？",
        "这里的人是不是在讽刺？",
        "这段文字是不是靠讽刺成立？",
    ),
}
_BUSINESS_CHOICE = {
    "en": (
        "Which option is the right decision here?",
        "Which choice fits this situation?",
        "What should be done in this case?",
        "Which option matches the right handling?",
        "Which decision does this situation support?",
        "Which option should be selected?",
    ),
    "de": (
        "Welche Option ist hier die richtige Entscheidung?",
        "Welche Wahl passt zu dieser Situation?",
        "Was soll in diesem Fall getan werden?",
        "Welche Option entspricht der richtigen Bearbeitung?",
        "Welche Entscheidung stützt diese Situation?",
        "Welche Option soll ausgewählt werden?",
    ),
    "fr": (
        "Quelle option est la bonne décision ici ?",
        "Quel choix convient à cette situation ?",
        "Que faut-il faire dans ce cas ?",
        "Quelle option correspond au bon traitement ?",
        "Quelle décision cette situation soutient-elle ?",
        "Quelle option faut-il sélectionner ?",
    ),
    "es": (
        "¿Qué opción es la decisión correcta aquí?",
        "¿Qué elección encaja con esta situación?",
        "¿Qué hay que hacer en este caso?",
        "¿Qué opción corresponde al manejo correcto?",
        "¿Qué decisión respalda esta situación?",
        "¿Qué opción hay que seleccionar?",
    ),
    "it": (
        "Quale opzione è la decisione giusta qui?",
        "Quale scelta si adatta a questa situazione?",
        "Che cosa va fatto in questo caso?",
        "Quale opzione corrisponde alla gestione giusta?",
        "Quale decisione sostiene questa situazione?",
        "Quale opzione va selezionata?",
    ),
    "pt": (
        "Qual opção é a decisão certa aqui?",
        "Qual escolha cabe a esta situação?",
        "O que se deve fazer neste caso?",
        "Qual opção corresponde ao tratamento certo?",
        "Qual decisão esta situação sustenta?",
        "Qual opção deve ser selecionada?",
    ),
    "nl": (
        "Welke optie is hier de juiste beslissing?",
        "Welke keuze past bij deze situatie?",
        "Wat moet er in dit geval gedaan worden?",
        "Welke optie komt overeen met de juiste afhandeling?",
        "Welke beslissing ondersteunt deze situatie?",
        "Welke optie moet worden gekozen?",
    ),
    "pl": (
        "Która opcja jest tu właściwą decyzją?",
        "Który wybór pasuje do tej sytuacji?",
        "Co należy zrobić w tym przypadku?",
        "Która opcja odpowiada właściwemu załatwieniu?",
        "Którą decyzję wspiera ta sytuacja?",
        "Którą opcję należy wybrać?",
    ),
    "tr": (
        "Burada doğru karar hangi seçenektir?",
        "Bu duruma hangi seçim uyar?",
        "Bu durumda ne yapılmalıdır?",
        "Doğru işleme hangi seçenek uyar?",
        "Bu durum hangi kararı destekler?",
        "Hangi seçenek seçilmelidir?",
    ),
    "ru": (
        "Какой вариант является здесь верным решением?",
        "Какой выбор подходит к этой ситуации?",
        "Что следует сделать в этом случае?",
        "Какой вариант соответствует верной обработке?",
        "Какое решение поддерживает эта ситуация?",
        "Какой вариант следует выбрать?",
    ),
    "ar": (
        "أي خيار هو القرار الصحيح هنا؟",
        "أي اختيار يناسب هذا الموقف؟",
        "ماذا ينبغي أن يُفعل في هذه الحالة؟",
        "أي خيار يطابق المعالجة الصحيحة؟",
        "أي قرار يدعمه هذا الموقف؟",
        "أي خيار ينبغي اختياره؟",
    ),
    "hi": (
        "यहाँ सही निर्णय कौन सा विकल्प है?",
        "इस स्थिति पर कौन सा चुनाव बैठता है?",
        "इस मामले में क्या किया जाना चाहिए?",
        "सही निपटान से कौन सा विकल्प मेल खाता है?",
        "यह स्थिति किस निर्णय का समर्थन करती है?",
        "कौन सा विकल्प चुना जाना चाहिए?",
    ),
    "ja": (
        "ここではどの選択肢が正しい判断ですか。",
        "この状況に合う選び方はどれですか。",
        "この場合は何をすべきですか。",
        "正しい扱いに合う選択肢はどれですか。",
        "この状況はどの判断を支えますか。",
        "選ぶべき選択肢はどれですか。",
    ),
    "zh": (
        "这里哪个选项才是正确决定？",
        "哪种选择适合这个情况？",
        "这个情形下应当做什么？",
        "哪个选项符合正确的处理？",
        "这个情况支持哪一项决定？",
        "应当挑选哪个选项？",
    ),
}

_BUSINESS_NOUL = {
    "en": (
        "Does the statement hold in this situation?",
        "Is the answer yes for this situation?",
        "Does this situation support a yes?",
        "Is the claim applicable based on this situation?",
        "Does the described condition apply here?",
        "Is the statement supported by this situation?",
    ),
    "de": (
        "Trifft die Aussage in dieser Situation zu?",
        "Ist die Antwort für diese Situation ja?",
        "Stützt diese Situation ein Ja?",
        "Greift die Behauptung anhand dieser Situation?",
        "Gilt die beschriebene Bedingung hier?",
        "Wird die Aussage von dieser Situation gestützt?",
    ),
    "fr": (
        "L'énoncé tient-il dans cette situation ?",
        "La réponse est-elle oui pour cette situation ?",
        "Cette situation soutient-elle un oui ?",
        "L'affirmation s'applique-t-elle d'après cette situation ?",
        "La condition décrite vaut-elle ici ?",
        "L'énoncé est-il soutenu par cette situation ?",
    ),
    "es": (
        "¿La afirmación se sostiene en esta situación?",
        "¿La respuesta es sí para esta situación?",
        "¿Esta situación respalda un sí?",
        "¿La afirmación aplica según esta situación?",
        "¿La condición descrita vale aquí?",
        "¿La afirmación queda respaldada por esta situación?",
    ),
    "it": (
        "L'affermazione regge in questa situazione?",
        "La risposta è sì per questa situazione?",
        "Questa situazione sostiene un sì?",
        "L'affermazione si applica in base a questa situazione?",
        "La condizione descritta vale qui?",
        "L'affermazione è sostenuta da questa situazione?",
    ),
    "pt": (
        "A afirmação vale nesta situação?",
        "A resposta é sim para esta situação?",
        "Esta situação sustenta um sim?",
        "A afirmação aplica-se com base nesta situação?",
        "A condição descrita vale aqui?",
        "A afirmação é sustentada por esta situação?",
    ),
    "nl": (
        "Gaat de stelling op in deze situatie?",
        "Is het antwoord ja voor deze situatie?",
        "Ondersteunt deze situatie een ja?",
        "Is de bewering van toepassing op grond van deze situatie?",
        "Geldt de beschreven voorwaarde hier?",
        "Wordt de stelling door deze situatie gesteund?",
    ),
    "pl": (
        "Czy stwierdzenie obowiązuje w tej sytuacji?",
        "Czy odpowiedź dla tej sytuacji brzmi tak?",
        "Czy ta sytuacja wspiera odpowiedź tak?",
        "Czy twierdzenie ma zastosowanie na podstawie tej sytuacji?",
        "Czy opisany warunek obowiązuje tutaj?",
        "Czy stwierdzenie jest wsparte przez tę sytuację?",
    ),
    "tr": (
        "İfade bu durumda geçerli mi?",
        "Bu durum için yanıt evet mi?",
        "Bu durum bir eveti destekliyor mu?",
        "İddia bu duruma göre uygulanır mı?",
        "Anlatılan koşul burada geçerli mi?",
        "İfade bu durum tarafından destekleniyor mu?",
    ),
    "ru": (
        "Утверждение верно в этой ситуации?",
        "Ответ для этой ситуации — да?",
        "Эта ситуация поддерживает ответ да?",
        "Утверждение применимо на основании этой ситуации?",
        "Описанное условие действует здесь?",
        "Это утверждение поддержано этой ситуацией?",
    ),
    "ar": (
        "هل تصح العبارة في هذا الموقف؟",
        "هل الإجابة نعم في هذا الموقف؟",
        "هل يدعم هذا الموقف الإجابة بنعم؟",
        "هل يسري الادعاء بناء على هذا الموقف؟",
        "هل ينطبق الشرط الموصوف هنا؟",
        "هل تدعم هذه الحالة العبارة؟",
    ),
    "hi": (
        "क्या यह कथन इस स्थिति में टिकता है?",
        "क्या इस स्थिति का उत्तर हाँ है?",
        "क्या यह स्थिति हाँ का समर्थन करती है?",
        "क्या दावा इस स्थिति के आधार पर लागू होता है?",
        "क्या वर्णित शर्त यहाँ लागू होती है?",
        "क्या इस स्थिति से कथन को सहारा मिलता है?",
    ),
    "ja": (
        "この状況でその文は成り立ちますか。",
        "この状況への答えははいですか。",
        "この状況ははいを支えますか。",
        "この状況に基づけば主張は当てはまりますか。",
        "述べられた条件はここで当てはまりますか。",
        "その文はこの状況に支えられていますか。",
    ),
    "zh": (
        "这个情况里该说法成立吗？",
        "针对这个情况，答案是肯定的吗？",
        "这个情况支持肯定的答案吗？",
        "根据这个情况，主张适用吗？",
        "所描述的条件在这里适用吗？",
        "这个情况支撑那个说法吗？",
    ),
}

_SCORE = {
    "en": (
        "Which level on the scale fits this situation?",
        "Which point on the scale is right here?",
        "Where does this situation fall on the scale?",
        "Which scale point matches this situation?",
        "Which listed level is the right rating?",
        "What rating on the scale fits best?",
    ),
    "de": (
        "Welche Stufe der Skala passt zu dieser Situation?",
        "Welcher Punkt der Skala stimmt hier?",
        "Wo liegt diese Situation auf der Skala?",
        "Welcher Skalenpunkt entspricht dieser Situation?",
        "Welche genannte Stufe ist die richtige Bewertung?",
        "Welche Bewertung auf der Skala passt am besten?",
    ),
    "fr": (
        "Quel niveau de l'échelle convient à cette situation ?",
        "Quel point de l'échelle est juste ici ?",
        "Où se place cette situation sur l'échelle ?",
        "Quel point de l'échelle correspond à cette situation ?",
        "Quel niveau listé est la bonne appréciation ?",
        "Quelle appréciation sur l'échelle convient le mieux ?",
    ),
    "es": (
        "¿Qué nivel de la escala encaja con esta situación?",
        "¿Qué punto de la escala es el correcto aquí?",
        "¿Dónde cae esta situación en la escala?",
        "¿Qué punto de la escala corresponde a esta situación?",
        "¿Qué nivel de la lista es la valoración correcta?",
        "¿Qué valoración de la escala encaja mejor?",
    ),
    "it": (
        "Quale livello della scala si adatta a questa situazione?",
        "Quale punto della scala è giusto qui?",
        "Dove si colloca questa situazione sulla scala?",
        "Quale punto della scala corrisponde a questa situazione?",
        "Quale livello elencato è la valutazione giusta?",
        "Quale valutazione sulla scala si adatta meglio?",
    ),
    "pt": (
        "Qual nível da escala cabe a esta situação?",
        "Qual ponto da escala está certo aqui?",
        "Onde se situa esta situação na escala?",
        "Qual ponto da escala corresponde a esta situação?",
        "Qual nível da lista é a avaliação certa?",
        "Qual avaliação na escala encaixa melhor?",
    ),
    "nl": (
        "Welk niveau op de schaal past bij deze situatie?",
        "Welk punt op de schaal klopt hier?",
        "Waar valt deze situatie op de schaal?",
        "Welk schaalpunt komt overeen met deze situatie?",
        "Welk genoemd niveau is de juiste beoordeling?",
        "Welke beoordeling op de schaal past het best?",
    ),
    "pl": (
        "Który poziom skali pasuje do tej sytuacji?",
        "Który punkt skali jest tu właściwy?",
        "Gdzie ta sytuacja mieści się na skali?",
        "Który punkt skali odpowiada tej sytuacji?",
        "Który wymieniony poziom jest właściwą oceną?",
        "Która ocena na skali pasuje najlepiej?",
    ),
    "tr": (
        "Ölçekteki hangi basamak bu duruma uyar?",
        "Ölçekte hangi nokta burada doğrudur?",
        "Bu durum ölçekte nereye düşer?",
        "Ölçekteki hangi nokta bu duruma karşılık gelir?",
        "Listelenen hangi basamak doğru değerlendirmedir?",
        "Ölçekteki hangi değerlendirme en iyi uyar?",
    ),
    "ru": (
        "Какая ступень шкалы подходит к этой ситуации?",
        "Какая точка шкалы верна здесь?",
        "Где эта ситуация находится на шкале?",
        "Какая точка шкалы соответствует этой ситуации?",
        "Какая названная ступень является верной оценкой?",
        "Какая оценка по шкале подходит лучше всего?",
    ),
    "ar": (
        "أي درجة على المقياس تناسب هذا الموقف؟",
        "أي نقطة على المقياس صحيحة هنا؟",
        "أين يقع هذا الموقف على المقياس؟",
        "أي نقطة من المقياس تطابق هذا الموقف؟",
        "أي درجة مذكورة هي التقييم الصحيح؟",
        "أي تقييم على المقياس هو الأنسب؟",
    ),
    "hi": (
        "पैमाने का कौन सा स्तर इस स्थिति पर बैठता है?",
        "पैमाने पर कौन सा बिंदु यहाँ सही है?",
        "यह स्थिति पैमाने पर कहाँ आती है?",
        "पैमाने का कौन सा बिंदु इस स्थिति से मेल खाता है?",
        "सूची का कौन सा स्तर सही आकलन है?",
        "पैमाने पर कौन सा आकलन सबसे ठीक बैठता है?",
    ),
    "ja": (
        "尺度のどの段階がこの状況に合いますか。",
        "尺度のどの位置がここで正しいですか。",
        "この状況は尺度のどこに落ちますか。",
        "この状況に合う尺度の位置はどれですか。",
        "挙げられた段階のうち正しい評価はどれですか。",
        "尺度の上で最も合う評価はどれですか。",
    ),
    "zh": (
        "量表上的哪一级适合这个情况？",
        "量表上的哪一点在这里是对的？",
        "这个情况落在量表的什么位置？",
        "量表上的哪一点与这个情况相符？",
        "列出的哪一级才是正确评价？",
        "量表上哪种评价最合适？",
    ),
}

_READING = {
    "en": (
        "Which option does the passage support?",
        "Which answer follows from the passage?",
        "What does the passage indicate is correct?",
        "Which option is backed by the passage?",
        "Which choice agrees with the passage?",
        "Which option matches what the passage says?",
    ),
    "de": (
        "Welche Option stützt der Text?",
        "Welche Antwort folgt aus dem Text?",
        "Was weist der Text als richtig aus?",
        "Welche Option wird vom Text getragen?",
        "Welche Wahl stimmt mit dem Text überein?",
        "Welche Option entspricht dem, was der Text sagt?",
    ),
    "fr": (
        "Quelle option le passage soutient-il ?",
        "Quelle réponse découle du passage ?",
        "Qu'est-ce que le passage indique comme correct ?",
        "Quelle option est appuyée par le passage ?",
        "Quel choix s'accorde avec le passage ?",
        "Quelle option correspond à ce que dit le passage ?",
    ),
    "es": (
        "¿Qué opción respalda el pasaje?",
        "¿Qué respuesta se sigue del pasaje?",
        "¿Qué indica el pasaje como correcto?",
        "¿Qué opción está apoyada por el pasaje?",
        "¿Qué elección concuerda con el pasaje?",
        "¿Qué opción coincide con lo que dice el pasaje?",
    ),
    "it": (
        "Quale opzione sostiene il brano?",
        "Quale risposta segue dal brano?",
        "Che cosa il brano indica come corretto?",
        "Quale opzione è sostenuta dal brano?",
        "Quale scelta concorda con il brano?",
        "Quale opzione corrisponde a ciò che dice il brano?",
    ),
    "pt": (
        "Qual opção o trecho sustenta?",
        "Qual resposta decorre do trecho?",
        "O que o trecho indica como certo?",
        "Qual opção é apoiada pelo trecho?",
        "Qual escolha concorda com o trecho?",
        "Qual opção corresponde ao que o trecho diz?",
    ),
    "nl": (
        "Welke optie ondersteunt de passage?",
        "Welk antwoord volgt uit de passage?",
        "Wat geeft de passage als juist aan?",
        "Welke optie wordt door de passage gedragen?",
        "Welke keuze stemt overeen met de passage?",
        "Welke optie komt overeen met wat de passage zegt?",
    ),
    "pl": (
        "Którą opcję wspiera fragment?",
        "Która odpowiedź wynika z fragmentu?",
        "Co fragment wskazuje jako poprawne?",
        "Która opcja jest poparta przez fragment?",
        "Który wybór zgadza się z fragmentem?",
        "Która opcja odpowiada temu, co mówi fragment?",
    ),
    "tr": (
        "Parça hangi seçeneği destekliyor?",
        "Parçadan hangi yanıt çıkar?",
        "Parça neyi doğru olarak gösteriyor?",
        "Hangi seçenek parça tarafından destekleniyor?",
        "Hangi seçim parçayla uyuşuyor?",
        "Parçanın söylediğiyle hangi seçenek örtüşüyor?",
    ),
    "ru": (
        "Какой вариант поддерживает отрывок?",
        "Какой ответ следует из отрывка?",
        "Что отрывок указывает как верное?",
        "Какой вариант подкреплён отрывком?",
        "Какой выбор согласуется с отрывком?",
        "Какой вариант соответствует тому, что говорит отрывок?",
    ),
    "ar": (
        "أي خيار يدعمه النص؟",
        "أي إجابة تترتب على النص؟",
        "ماذا يشير النص على أنه صحيح؟",
        "أي خيار تسنده الفقرة؟",
        "أي اختيار يتفق مع النص؟",
        "أي خيار يطابق ما يقوله النص؟",
    ),
    "hi": (
        "अनुच्छेद किस विकल्प का समर्थन करता है?",
        "अनुच्छेद से कौन सा उत्तर निकलता है?",
        "अनुच्छेद किसे सही बताता है?",
        "कौन सा विकल्प अनुच्छेद पर टिका है?",
        "कौन सा चुनाव अनुच्छेद से मेल खाता है?",
        "अनुच्छेद जो कहता है, उससे कौन सा विकल्प मेल खाता है?",
    ),
    "ja": (
        "文章はどの選択肢を支えていますか。",
        "文章から導かれる答えはどれですか。",
        "文章は何を正しいと示していますか。",
        "文章に裏付けられた選択肢はどれですか。",
        "文章と一致する選び方はどれですか。",
        "文章の内容に合う選択肢はどれですか。",
    ),
    "zh": (
        "短文支持哪个选项？",
        "从短文能推出哪个答案？",
        "短文表明什么是对的？",
        "哪个选项有短文作为依据？",
        "哪种选择与短文一致？",
        "哪个选项符合短文所说的内容？",
    ),
}

# Fix the accidental English word in the Arabic complaint question.
_COMPLAINT_NOUL["ar"] = (
    "هل هذه الرسالة شكوى؟",
    "هل يشتكي من كتب في هذه الرسالة؟",
    "هل يطرح الشخص شكوى هنا؟",
    "هل تتضمن هذه الرسالة شكوى؟",
    "هل يُقرأ هذا النص على أنه شكوى؟",
    "هل توجد شكوى في كلام الشخص؟",
)

PARAPHRASES = {
    "sentiment": {"overall": _SENTIMENT_OVERALL, "aspect": _SENTIMENT_ASPECT},
    "emotion": _EMOTION,
    "complaint": {"noul": _COMPLAINT_NOUL, "choice": _COMPLAINT_CHOICE},
    "urgency": _URGENCY,
    "sarcasm": _SARCASM,
    "business": {"choice": _BUSINESS_CHOICE, "noul": _BUSINESS_NOUL},
    "score": _SCORE,
    "reading": _READING,
}
def _is_lang_table(node):
    return isinstance(node, dict) and bool(node) and all(
        isinstance(value, (list, tuple)) for value in node.values())


def iter_paraphrase_tables():
    """Yield (task, slot, lang->sentences). Slot is empty when the task has one list."""
    for task, node in PARAPHRASES.items():
        if _is_lang_table(node):
            yield task, "", node
        else:
            for slot, table in node.items():
                yield task, slot, table


def paraphrase_slot(seed):
    task = seed.get("task")
    if task == "sentiment":
        return "aspect" if seed.get("variant") == "aspect" else "overall"
    if task == "complaint":
        return "choice" if seed.get("form") == "choice" else "noul"
    if task == "business":
        return "noul" if seed.get("form") == "noul" else "choice"
    return ""


def paraphrase_candidates(seed):
    """Question paraphrases for this seed. Aspect slots already contain the focus noun."""
    slot = paraphrase_slot(seed)
    table = None
    for task, found, node in iter_paraphrase_tables():
        if task == seed.get("task") and found == slot:
            table = node
            break
    if not table:
        return []
    lines = [str(line) for line in (table.get(seed.get("lang")) or [])]
    if slot == "aspect":
        aspect = seed.get("focus") or (PRODUCT_ASPECTS[0] if PRODUCT_ASPECTS else "")
        phrase = (FOCUS_PHRASE.get(aspect) or {}).get(seed.get("lang")) or aspect
        lines = [line.replace("{focus}", phrase) for line in lines]
    return lines


def paraphrase_coverage_gaps():
    """Empty when every task and every language has at least six paraphrases."""
    gaps = []
    seen = set()
    for task, slot, table in iter_paraphrase_tables():
        seen.add(task)
        for lang in LANGS:
            lines = list(table.get(lang) or [])
            if len(lines) < 6:
                gaps.append("%s/%s/%s has %d" % (task, slot or "-", lang, len(lines)))
            if slot == "aspect":
                for line in lines:
                    if "{focus}" not in line:
                        gaps.append("%s/aspect/%s missing focus slot" % (task, lang))
                        break
    for task in TASKS:
        if task not in seen:
            gaps.append("task %s has no paraphrases" % task)
    for aspect in PRODUCT_ASPECTS:
        phrases = FOCUS_PHRASE.get(aspect) or {}
        for lang in LANGS:
            if not phrases.get(lang):
                gaps.append("focus %s/%s missing" % (aspect, lang))
    return gaps

def _label_block(labels, gloss):
    return "\n".join("- %s: %s" % (key, gloss[key]) for key in labels)


def _flatten(obj):
    if isinstance(obj, dict):
        return "\n".join("%s\n%s" % (key, _flatten(obj[key])) for key in sorted(obj))
    if isinstance(obj, (list, tuple)):
        return "\n".join(_flatten(value) for value in obj)
    return str(obj)


def prompts_digest():
    blob = "\n".join([
        PROMPT_VERSION, GEN_SYSTEM, GEN_SYSTEM_CLASS, VER_SYSTEM,
        _GEN_CHOICE, _GEN_NOUL, _GEN_SCORE, _GEN_READING,
        _GEN_SENTIMENT, _GEN_EMOTION, _GEN_COMPLAINT_NOUL, _GEN_COMPLAINT_CHOICE,
        _GEN_URGENCY, _GEN_SARCASM,
        _VER_CHOICE, _VER_NOUL, _VER_SCORE,
        _flatten(PARAPHRASES), _flatten(FOCUS_PHRASE),
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
            class_rule=_SENTIMENT_RULE[seed["target"]], variant_block=block,
            opening=seed["opening"], detail=seed["detail"])
        return _class_messages(user, SCHEMA_CHOICE, 1400)
    if task == "emotion":
        user = _GEN_EMOTION.format(
            language=lang, code=seed["lang"], industry=seed["industry"], genre=seed["genre"],
            register=seed["register"], persona=seed["persona"], difficulty=seed["difficulty"],
            length=seed["length"], labels=_label_block(EMOTION_LABELS, _EMOTION_HELP),
            target=seed["target"], opening=seed["opening"], detail=seed["detail"])
        return _class_messages(user, SCHEMA_CHOICE, 1400)
    if task == "complaint" and seed.get("form") == "choice":
        user = _GEN_COMPLAINT_CHOICE.format(
            language=lang, code=seed["lang"], industry=seed["industry"], topic=seed["topic"],
            register=seed["register"], persona=seed["persona"], difficulty=seed["difficulty"],
            target=seed["target"], length=seed["length"],
            labels=_label_block(COMPLAINT_LABELS, _COMPLAINT_HELP),
            opening=seed["opening"], detail=seed["detail"])
        return _class_messages(user, SCHEMA_CHOICE, 1400)
    if task == "complaint":
        user = _GEN_COMPLAINT_NOUL.format(
            language=lang, code=seed["lang"], industry=seed["industry"], topic=seed["topic"],
            register=seed["register"], persona=seed["persona"], difficulty=seed["difficulty"],
            target=seed["target"], class_rule=_COMPLAINT_RULE[seed["target"]], length=seed["length"],
            opening=seed["opening"], detail=seed["detail"])
        return _class_messages(user, SCHEMA_NOUL, 1000)
    if task == "urgency":
        user = _GEN_URGENCY.format(
            language=lang, code=seed["lang"], industry=seed["industry"], register=seed["register"],
            persona=seed["persona"], difficulty=seed["difficulty"], length=seed["length"],
            n_levels=seed["n_levels"], target=seed["target"],
            level_meanings="\n".join("- %s" % line for line in seed["levels_brief"]),
            opening=seed["opening"], detail=seed["detail"])
        return _class_messages(user, SCHEMA_SCORE, 1000)
    if task == "sarcasm":
        user = _GEN_SARCASM.format(
            language=lang, code=seed["lang"], industry=seed["industry"], register=seed["register"],
            persona=seed["persona"], difficulty=seed["difficulty"], target=seed["target"],
            class_rule=_SARCASM_RULE[seed["target"]], length=seed["length"],
            opening=seed["opening"], detail=seed["detail"])
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
