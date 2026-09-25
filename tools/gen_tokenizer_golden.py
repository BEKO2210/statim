#!/usr/bin/env python3
"""Builds the tokenizer parity corpus and HF golden ids.

Writes
  tests/data/tokenizer_corpus.jsonl             {"text": ...}
  tests/data/tokenizer_golden_multilingual.jsonl {"text", "ids", "ids48"}
  tests/data/tokenizer_golden_english.jsonl

ids   = Tokenizer.encode(text, add_special_tokens=False).ids
ids48 = same with truncation max_length=48
Every line is also cross-checked against transformers
(tok(text, add_special_tokens=False) and truncation=True, max_length=48).

Run with the reference env: .venv-ref/bin/python tools/gen_tokenizer_golden.py
"""
import json
import os
import random
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
TOKENIZERS = {
    "multilingual": os.path.join(ROOT, "models", "laya-multilingual", "tokenizer"),
    "english": os.path.join(ROOT, "models", "laya", "tokenizer"),
}
TRUNC = 48

SENTENCES = {
    "de": [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Straßenbahnhaltestelle, Größenordnung und Übermäßigkeit sind typisch deutsche Wörter.",
        "Könnten Sie mir bitte sagen, wo der nächste Bahnhof ist?",
        "Die Rechnung über 1.234,56 € ist am 03.10.2026 fällig.",
    ],
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "I'm sure they'll say it's fine, but we've seen what they'd done.",
        "Please reply ASAP: the Q3 report (v2.1) is due by 5:00 PM EST.",
        "Rock 'n' roll isn't dead; it's just resting.",
    ],
    "fr": [
        "L'été dernier, nous sommes allés à Besançon près de la forêt.",
        "Ça coûte 12,50 € — c'est trop cher, n'est-ce pas ?",
        "Où est la bibliothèque ? « Juste à côté », répondit-elle.",
    ],
    "es": [
        "¿Dónde está la estación de tren? ¡No lo sé!",
        "El niño comió piña y jalapeños en la mañana.",
        "Mañana llegaré a las 8:30 con Ñoño y Begoña.",
    ],
    "pt": [
        "A tripulação do navio não conseguiu ver a ilha à distância.",
        "Você já comeu pão de queijo em São Paulo?",
        "Informações adicionais estão disponíveis no endereço eletrônico.",
    ],
    "it": [
        "Perché non andiamo a mangiare una pizza più tardi?",
        "L'università è chiusa finché non arriverà la città.",
        "Qual è la differenza tra però e perciò?",
    ],
    "ru": [
        "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        "Москва — столица России; население превышает 12 млн человек.",
        "Ёжик в тумане искал свою лошадку.",
    ],
    "uk": [
        "Чуєш їх, доцю, га? Кумедна ж ти, прощайся без ґольфів!",
        "Київ є столицею України з давніх часів.",
        "Щастя — це коли тебе розуміють.",
    ],
    "ar": [
        "مرحبا بالعالم! كيف حالك اليوم؟",
        "اللغة العربية جميلة جداً وتُكتب من اليمين إلى اليسار.",
        "سعر الكتاب ٢٥ ريالاً فقط، أي 25 SAR.",
    ],
    "he": [
        "שלום עולם, מה שלומך היום?",
        "עברית נכתבת מימין לשמאל, עם ניקוד: בְּרֵאשִׁית בָּרָא אֱלֹהִים.",
        "הפגישה נקבעה ל-10:30 בבוקר.",
    ],
    "fa": [
        "سلام دنیا! امروز هوا خیلی خوب است.",
        "زبان فارسی با الفبای عربی نوشته می‌شود و نیم‌فاصله دارد.",
        "قیمت ۱۲۳۴ تومان است.",
    ],
    "hi": [
        "नमस्ते दुनिया! आज मौसम बहुत अच्छा है।",
        "हिन्दी देवनागरी लिपि में लिखी जाती है, जैसे क्षत्रिय और ज्ञान।",
        "मेरा फ़ोन नंबर ९८७६५४३२१० है।",
    ],
    "bn": [
        "আমি বাংলায় গান গাই।",
        "বাংলাদেশের রাজধানী ঢাকা, জনসংখ্যা প্রায় ২ কোটি।",
        "স্বাধীনতা দিবস ২৬শে মার্চ।",
    ],
    "ta": [
        "வணக்கம் உலகம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "தமிழ் மொழி மிகவும் பழமையானது.",
        "சென்னை தமிழ்நாட்டின் தலைநகரம்.",
    ],
    "th": [
        "สวัสดีชาวโลก วันนี้อากาศดีมาก",
        "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ",
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
    ],
    "zh": [
        "你好，世界！今天天气很好。",
        "中华人民共和国成立于一九四九年十月一日。",
        "我喜欢吃饺子，尤其是韭菜猪肉馅的。",
        "機器學習與自然語言處理（NLP）是人工智慧的重要分支。",
    ],
    "ja": [
        "こんにちは世界！今日はいい天気ですね。",
        "東京タワーは高さ３３３メートルです。",
        "ｶﾀｶﾅの半角文字とカタカナの全角文字。",
        "吾輩は猫である。名前はまだ無い。",
    ],
    "ko": [
        "안녕하세요 세계! 오늘 날씨가 좋네요.",
        "한국어는 한글로 쓰여지며 세종대왕이 창제했습니다.",
        "서울특별시의 인구는 약 950만 명입니다.",
    ],
    "vi": [
        "Xin chào thế giới! Hôm nay trời đẹp quá.",
        "Tiếng Việt có nhiều dấu thanh: à, á, ả, ã, ạ.",
        "Thành phố Hồ Chí Minh là thành phố lớn nhất Việt Nam.",
    ],
    "tr": [
        "Merhaba dünya! Bugün hava çok güzel.",
        "İstanbul'da ılık bir öğleden sonra; ĞÜŞİÖÇ ğüşıöç.",
        "Çekoslovakyalılaştıramadıklarımızdan mısınız?",
    ],
    "pl": [
        "Zażółć gęślą jaźń.",
        "Pchnąć w tę łódź jeża lub ośm skrzyń fig.",
        "W Szczebrzeszynie chrząszcz brzmi w trzcinie.",
    ],
    "el": [
        "Γειά σου Κόσμε! Πώς είσαι σήμερα;",
        "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία.",
        "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
    ],
    "am": [
        "ሰላም ልዑል! እንዴት ነህ?",
        "አማርኛ በግዕዝ ፊደል ይጻፋል።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
    ],
    "km": [
        "សួស្តីពិភពលោក!",
        "ភាសាខ្មែរជាភាសាផ្លូវការរបស់ប្រទេសកម្ពុជា។",
        "ភ្នំពេញជារាជធានីនៃកម្ពុជា។",
    ],
    "misc": [
        "Ελληνικά, русский, 中文, العربية, עברית and English in one line.",
        "Ünïcödé täst wïth ümläuts ånd åccénts — naïve café résumé coöperate.",
        "Mixed: abc١٢٣def ٤٥٦ ghi 七八九 jkl ⅣⅤⅥ ①②③ ½ ¾ ² ³",
        "Math: ∀x∈ℝ, ∃y: x² + y² ≥ 0 ⇒ √(x²) = |x| ∑∏∫∂∇ ≈ ≠ ≤ ≥ ∞",
        "Currency: $100, €200, £300, ¥400, ₹500, ₽600, ₩700, ₿0.01",
    ],
}

EMOJI = [
    "😀", "😂", "🥲", "❤️", "👍🏽", "👩‍💻", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇩🇪", "🇯🇵", "🇺🇸", "1️⃣", "#️⃣",
    "🧑🏿‍🚀", "👁️‍🗨️", "🐈‍⬛", "🫠", "🦀", "☕", "✈️", "™", "©", "®", "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "🤷‍♂️", "🙋🏻‍♀️",
]

WS_CASES = [
    "", " ", "  ", "   ", "\t", "\n", "\r\n", "\r", "\n\n\n", " \n ", "\t\t\t", "a", " a", "a ",
    "  a", "a  ", " a ", "a  b", "a   b", "a    b", "a\tb", "a\t\tb", "a\nb", "a\n\nb", "a\r\nb",
    "a \n b", "a\u00a0b", "\u00a0", "a\u00a0\u00a0b", "a\u2003b", "a\u3000b", "a\u2028b",
    "a\u0085b", "a\u200bb", "a\u200db", "\ufeffbom", "x\x00y", "\x07\x08\x1b[0m", "line1\nline2\n",
    "trailing   ", "   leading", "  both  ", "tab\tseparated\tvalues\n1\t2\t3",
    "windows\r\nline\r\nendings\r\n", " " * 30, " " * 64, "\n" * 40, "\t" * 40, "▁", "▁▁x", "x▁y",
    "Ġ", "ĠĠ", "Ċ", " \t \n \r\n\u00a0 ", "a" + " " * 25 + "b", ".", "..", "...", "!!!???",
    "'", "''", "'s", "'S", "'''s", "don't", "DON'T", "we'll've", "y'all're", "rock'n'roll",
    "‘quoted’ “quoted”", "it’s", "l'homme", "0", "00", "007", "3.14159", "1,000,000", "1e-9",
    "12345678901234567890", "٠١٢٣٤٥٦٧٨٩", "०१२३४५६७८९", "¹²³", "Ⅻ", "⑩", "𝟘𝟙𝟚",
]

CODE = [
    "def foo(x: int) -> int:\n    return x ** 2  # square\n",
    "for (int i = 0; i < n; ++i) {\n\tsum += a[i] * b[i];\n}",
    "const f = async (a, b) => { await fetch(`/api/${a}?q=${b}`); };",
    "SELECT id, name FROM users WHERE email LIKE '%@example.com' ORDER BY id DESC LIMIT 10;",
    "#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do echo \"$f\"; done",
    "<div class=\"card\"><p>Hello &amp; welcome</p><br/></div>",
    "fn main() { let v: Vec<u8> = vec![1, 2, 3]; println!(\"{:?}\", v); }",
    "if __name__ == '__main__':\n    main()\n",
    "x = {'a': [1, 2, {'b': None}], \"c\": True}",
    "#include <iostream>\nint main() { std::cout << \"hi\\n\"; }",
    "    indented four\n        indented eight\n\tindented tab\n",
    "λx.λy.x  // lambda calculus; α-β-η",
    "regex: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
    "<mask>.forward(<unused3>) // not really tokens? [MASK] [CLS] [SEP]",
]

URLS = [
    "https://www.example.com/path/to/page?query=1&lang=de#section-2",
    "http://localhost:8080/api/v1/users/42",
    "ftp://files.example.org/pub/file.tar.gz",
    "https://de.wikipedia.org/wiki/Stra%C3%9Fe",
    "https://例え.jp/パス?キー=値",
    "mailto:someone@example.com",
    "user.name+tag@sub.domain.co.uk",
    "belkis@example.de, info@firma-gmbh.de; kontakt@bücher.de",
    "/home/user/projects/statim/src/tokenizer.cpp:123:45",
    "C:\\Users\\Name\\Documents\\file (1).docx",
    "git@github.com:org/repo.git",
    "192.168.2.221:8443",
    "2001:db8::ff00:42:8329",
    "+49 (0) 711 123 456-78",
    "2026-09-26T01:23:45.678Z",
    "SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
]

BLOCKS = [
    (0x0020, 0x007E), (0x00A0, 0x024F), (0x0300, 0x036F), (0x0370, 0x03FF), (0x0400, 0x04FF),
    (0x0530, 0x058F), (0x0590, 0x05FF), (0x0600, 0x06FF), (0x0900, 0x097F), (0x0980, 0x09FF),
    (0x0B80, 0x0BFF), (0x0E00, 0x0E7F), (0x10A0, 0x10FF), (0x1100, 0x11FF), (0x1200, 0x137F),
    (0x1780, 0x17FF), (0x1E00, 0x1EFF), (0x2000, 0x206F), (0x2070, 0x209F), (0x20A0, 0x20CF),
    (0x2100, 0x214F), (0x2150, 0x218F), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2460, 0x24FF),
    (0x2500, 0x257F), (0x2600, 0x26FF), (0x2700, 0x27BF), (0x3000, 0x303F), (0x3040, 0x309F),
    (0x30A0, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0xE000, 0xE0FF),
    (0xF900, 0xFAFF), (0xFB00, 0xFB4F), (0xFE00, 0xFE0F), (0xFE70, 0xFEFF), (0xFF00, 0xFFEF),
    (0x1D400, 0x1D7FF), (0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F900, 0x1F9FF),
    (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x30000, 0x3134F), (0xE0000, 0xE007F),
]


def assigned(cp):
    return unicodedata.category(chr(cp)) not in ("Cn", "Cs")


def rand_char(rng):
    lo, hi = rng.choice(BLOCKS)
    while True:
        cp = rng.randint(lo, hi)
        if assigned(cp):
            return chr(cp)


def all_sentences():
    return [s for v in SENTENCES.values() for s in v]


def rand_json(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.3:
        choice = rng.randint(0, 5)
        if choice == 0:
            return rng.randint(-10**6, 10**6)
        if choice == 1:
            return round(rng.uniform(-1000, 1000), rng.randint(0, 6))
        if choice == 2:
            return rng.choice([True, False, None])
        return rng.choice(all_sentences() + EMOJI + URLS)
    if r < 0.65:
        return [rand_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    keys = ["id", "name", "Straße", "名前", "описание", "tags", "value", "créé_le", "données", "url"]
    return {rng.choice(keys) + str(i): rand_json(rng, depth + 1) for i in range(rng.randint(0, 5))}


def load_added_tokens():
    toks = []
    for d in TOKENIZERS.values():
        with open(os.path.join(d, "tokenizer.json"), encoding="utf-8") as f:
            toks += [a["content"] for a in json.load(f)["added_tokens"]]
    return toks


def mine_reference(rng, limit):
    """Real-world English/code/multilingual text from the vendored Laya sources."""
    base = os.path.join(ROOT, "reference", "laya-src")
    exts = (".md", ".py", ".ts", ".json", ".toml", ".yaml", ".yml", ".sh")
    chunks = []
    for dirpath, dirnames, files in os.walk(base):
        dirnames.sort()
        for fn in sorted(files):
            if not fn.endswith(exts):
                continue
            try:
                with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                    txt = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            if len(txt) > 2_000_000:
                continue
            paras = [p for p in txt.split("\n\n") if p.strip()]
            chunks += [p[:3000] for p in paras]
    rng.shuffle(chunks)
    return chunks[:limit]


def build_corpus(seed=20260926):
    rng = random.Random(seed)
    out = []
    sents = all_sentences()

    out += sents
    out += WS_CASES + CODE + URLS + EMOJI
    out += ["".join(EMOJI), " ".join(EMOJI), "Hello 👋🏼 world 🌍!", "ok👍🏽ok", "🇩🇪🇫🇷🇯🇵"]

    # Normalization variants.
    for s in sents:
        for form in ("NFD", "NFKC", "NFKD"):
            v = unicodedata.normalize(form, s)
            if v != s:
                out.append(v)
    out += ["e\u0301", "\u00e9", "A\u030a", "\u212b", "\u1e9b\u0323", "q\u0307\u0323", "\u0344",
            "\u0958", "\uac00\u11a8", "\u1100\u1161\u11a8", "\u0f73", "a\u0300\u0301\u0302\u0303",
            "\u0041\u030a\u0301", "\u2126", "\u00c5ngstr\u00f6m", "Zalgo: Z\u0351\u0352a\u0353l\u0354g\u0355o\u0356",
            # HF NFC uses old Unicode data: U+09FE has ccc 0 there (230 today).
            "a\u0302\u09fe\u031f", "\U0001f936\ufe01\u031f\u0302\u09fe", "\u0995\u09fe\u0301\u0316",
            "\u0915\u093c", "\u0958\u0301", "\u1e9b\u0323\u0301"]

    # Added tokens in context.
    added = load_added_tokens()
    templates = ["{t}", "a{t}b", "a {t} b", "  {t}  ", "{t}{t}", "\n{t}\n", "x {t}", "{t}hello",
                 "hello{t}", "hello {t}", "{t} hello", "The {t} is here.", "{t}\t{t}", "Die {t}Stadt",
                 "\u00a0{t}\u00a0", "日本{t}語", "{t}.", "({t})", "{t}   {t}", " \n {t}", "{t}{t2}",
                 "{t} {t2} {t}", "x{t}{t2}y"]
    special = ["<mask>", "<eos>", "<bos>", "<pad>", "<unk>", "<unused3>", "<unused0>", "[@BOS@]",
               "<2mass>", "<|padding|>", "[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[unused7]",
               "|||IP_ADDRESS|||", "|||EMAIL_ADDRESS|||", "|||PHONE_NUMBER|||", "<start_of_turn>",
               "<end_of_turn>", "\t\t", "\t\t\t\t", "  ", "    "]
    pool = special + [t for t in added if t.strip()]
    for tpl in templates:
        for t in special:
            out.append(tpl.format(t=t, t2=rng.choice(pool)))
    for _ in range(400):
        out.append(rng.choice(templates).format(t=rng.choice(pool), t2=rng.choice(pool)))
    out += ["<mask", "mask>", "<unused", "<unused999>", "<<mask>>", "<mask><mask>", "[MASK][MASK]",
            "[MAS K]", "< mask>", "<MASK>", "<Mask>", "hello<mask>", "hello <mask>", "hello  <mask>",
            "hello\n<mask>", "hello\t \u00a0<mask>world", "<mask> ", " <mask>", "  <mask>  ",
            "hello [MASK] world", "hello  [MASK]", "\n[MASK]\n", "|||IP_ADDRESS", "||||IP_ADDRESS|||",
            "IP: |||IP_ADDRESS|||:8080"]

    # Word salad across scripts with varied separators.
    words = [w for s in sents for w in s.split()]
    seps = [" ", "  ", "\t", "\n", "", "\u00a0", " \n", "\r\n", "   ", "\u3000", "-", "_", "/"]
    for _ in range(700):
        n = rng.randint(1, 25)
        pieces = []
        for _ in range(n):
            pieces.append(rng.choice(words))
            pieces.append(rng.choice(seps))
        out.append("".join(pieces[: rng.randint(1, len(pieces))]))

    # Random character soups (incl. rare CJK needing byte fallback, astral, combining marks).
    for _ in range(600):
        n = rng.randint(1, 40)
        out.append("".join(rand_char(rng) if rng.random() < 0.8 else rng.choice(" \n\t") for _ in range(n)))
    for _ in range(100):
        lo, hi = rng.choice([(0x20000, 0x2A6DF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3)])
        out.append("".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 30))))

    # JSON documents.
    for _ in range(250):
        obj = rand_json(rng)
        indent = rng.choice([None, None, 2, 4, "\t"])
        out.append(json.dumps(obj, ensure_ascii=False, indent=indent))
    for _ in range(30):
        out.append(json.dumps(rand_json(rng), ensure_ascii=True))

    # Numbers.
    for _ in range(150):
        k = rng.randint(0, 4)
        if k == 0:
            out.append(str(rng.randint(0, 10**rng.randint(1, 30))))
        elif k == 1:
            out.append(f"{rng.uniform(-1e6, 1e6):.{rng.randint(0, 8)}f}")
        elif k == 2:
            out.append(f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1900, 2100)}")
        elif k == 3:
            out.append(f"#{rng.randint(0, 2**24):06x} 0x{rng.getrandbits(64):X}")
        else:
            out.append(" ".join(str(rng.randint(0, 999)) for _ in range(rng.randint(2, 12))))

    # Real text mined from the reference sources.
    out += mine_reference(rng, 900)

    # Very long strings.
    out.append(" ".join(sents) * 3)
    out.append("\n".join(mine_reference(random.Random(1), 40)))
    out.append("a" * 5000)
    out.append("ab" * 3000)
    out.append(" " * 1000 + "x")
    out.append("x" + "\n" * 600 + "y")
    out.append("9" * 3000)
    out.append("中" * 2000)
    out.append("".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range(4000)))
    out.append("𠀋" * 500)
    out.append("😀" * 800)
    out.append("é" * 1000 + "e\u0301" * 1000)
    out.append("\t" * 700 + "tab")
    out.append("-" * 3000)
    out.append("=" * 100 + "\n" + "#" * 100)
    out.append("<mask>" * 300)
    out.append(" <mask>" * 300)
    out.append(("Laya multilingual test. " * 200).strip())

    # Dedupe, keep order, enforce valid UTF-8.
    seen, res = set(), []
    for s in out:
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            continue
        if s not in seen:
            seen.add(s)
            res.append(s)
    return res


def main():
    os.makedirs(DATA, exist_ok=True)
    corpus = build_corpus()
    with open(os.path.join(DATA, "tokenizer_corpus.jsonl"), "w", encoding="utf-8") as f:
        for s in corpus:
            f.write(json.dumps({"text": s}) + "\n")
    print(f"corpus: {len(corpus)} strings, {sum(len(s.encode()) for s in corpus)/1e6:.2f} MB")

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    ok = True
    for name, d in TOKENIZERS.items():
        path = os.path.join(d, "tokenizer.json")
        tok = Tokenizer.from_file(path)
        full = [e.ids for e in tok.encode_batch(corpus, add_special_tokens=False)]
        tok.enable_truncation(TRUNC)
        trunc = [e.ids for e in tok.encode_batch(corpus, add_special_tokens=False)]

        hf = AutoTokenizer.from_pretrained(d)
        bad = 0
        for i, s in enumerate(corpus):
            a = hf(s, add_special_tokens=False)["input_ids"]
            b = hf(s, add_special_tokens=False, truncation=True, max_length=TRUNC)["input_ids"]
            if a != full[i] or b != trunc[i] or trunc[i] != full[i][:TRUNC]:
                bad += 1
                if bad <= 5:
                    print(f"  [{name}] tokenizers/transformers disagree on {s[:60]!r}")
        print(f"{name}: tokenizers vs transformers disagreements: {bad}")
        ok &= bad == 0

        out_path = os.path.join(DATA, f"tokenizer_golden_{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s, a, b in zip(corpus, full, trunc):
                f.write(json.dumps({"text": s, "ids": a, "ids48": b}) + "\n")
        print(f"wrote {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
