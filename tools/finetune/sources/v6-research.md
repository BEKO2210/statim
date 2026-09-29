# Statim training data: licence-clean candidates per decision category

Date: 2026-09-27. Host: pop-os. Scope: metadata, cards, upstream licences, first rows (datasets-server). Nothing large was downloaded, the GPU was not used, and nothing in the repo was changed.

## Policy applied

- **KEEP** only if the licence permits commercial use and puts no ShareAlike/copyleft on a trained model: Apache-2.0, MIT, BSD, CC0, CC-BY (any version), ODC-By, ODC-PDDL, AFL-3.0, Unlicense, public domain.
- **EXCLUDE**: NC, SA, ND, GPL/AGPL/LGPL/MPL, ODbL, CDLA-Sharing, "research only", "other"/custom terms, a missing licence, and any card whose licence conflicts with the licence or terms of the original text. This includes text from X/Twitter, Reddit, YouTube, Gab, and reviews from Amazon, Yelp, IMDb, TripAdvisor or app stores, as well as scraped news articles, Wikipedia, FLORES (CC-BY-SA), Quora and Stack Exchange. The only exception is text the rights holder released under a permissive licence itself. This is the same standard as `tools/finetune/licence_audit.json` and `build_extra.py`.
- **LLM-generated or template-generated sets** are kept on the creator's permissive licence. The repo audit already does this (WANLI, agent_action_safety). Model-provider terms on outputs are not treated as a data licence. Each such case is marked "LLM-generated".
- **Translations** inherit the licence of the text they translate.
- **Evidence**: every licence string is quoted from the HF card YAML (`HfApi().dataset_info(...).card_data['license']`), the card tags, or the README, and, for mirrors, from the upstream project: the GitHub API `license.spdx_id`, the upstream README, or the paper or project page. The helper script is `scratchpad/hfcheck.py`.

## Evaluation suites and how overlap is handled

| Suite | Source | Rule for training data |
|---|---|---|
| typed-decisions | LocalLLaMA/typed-decisions test | train already used; test texts deduped by `build_mixture.py` |
| Banking77 | PolyAI/banking77 test | train already used; NLU++ / Multi3NLU++ (same company, different collection) must be deduped against the test |
| MASSIVE (12 langs) | mteb/amazon_massive_intent test | train already used; SLURP = the English source of MASSIVE (excluded); HWU64 is its ancestor |
| AG News | fancyzhx/ag_news test | never trained; no AG News derivative is kept |
| DAIR Emotion | dair-ai/emotion test | today a *zero-shot concept*: `build_mixture.py` (EXCLUDE regex) and `build_extra.py` (FORBIDDEN) drop every emotion source |
| GoEmotions | go_emotions simplified test | zero-shot suite in `bench/eval_zeroshot.py`; all GoEmotions translations excluded |
| multilingual sentiment | tyqiangz test CSVs (12 langs) | its sources (Amazon Reviews Multi, tweet_sentiment_multilingual, IMDb, Yelp) are excluded anyway |
| multi_hatecheck | mteb/multi-hatecheck test | zero-shot; every HateCheck/MHC derivative excluded, HatemojiCheck held out as a precaution |
| SIB-200, Belebele | FLORES-200 sentences (drawn from Wikinews/Wikibooks/Wikivoyage) | zero-shot; FLORES-based sets excluded; **Wikinews-multilingual (cat. 8) must be filtered against Belebele links and FLORES sentences** |
| IndoNLI, FarsTail | test files | zero-shot NLI in id/fa; GlobalNLI (contains IndoNLI) and an id NLI repack excluded; no fa NLI kept |
| SemRel2024 | SemRel/SemRel2024 test | zero-shot relatedness; SemRel train not used |
| HWU64 | DeepPavlov/hwu64 test | sibling transfer; NLU-Evaluation-Data derivatives (deutsche-telekom, RuNLU) excluded |

**Every new source still needs the exact-match dedup of `build_mixture.py` (`norm()` over all eval test texts) before training.** The `test_overlap_note` field in the JSON names the suites a train split must additionally be checked against. Two additions to the banned set are recommended: the `mteb/multi-hatecheck` test texts (all 11 languages) for every toxicity source, and the FLORES dev/devtest sentences plus Belebele `link` URLs for Wikinews.

## Counts per category (new KEEP items; already-used sets listed in the tables but not counted)

| # | Category | KEEP | of which real / human-labelled | Priority languages covered by clean data | Verdict on coverage |
|---|---|---:|---|---|---|
| 1 | Sentiment (incl. neutral/mixed, aspect) | 12 (+poem_sentiment already used) | IndicSentiment (en+hi, annotator-written reviews, CC0 per AI4Bharat README), ReDial (en, CC-BY-4.0) | en, hi, zh (synthetic), pt (synthetic), ru (synthetic) | **Gap** for de fr es it nl pl tr ar ja; aspect-level and "mixed" almost only synthetic |
| 2 | Emotion | 9 | BRIGHTER hin+mar (annotator-written, CC-BY-4.0), empathic_reactions (en, CC-BY-4.0) | hi, mr, en, fr (synthetic), zh (AI text, human labels); ar/de/es/hi test-only (Cultural-Emo) | **Gap** for de es it pt nl pl tr ru ar ja; policy decision needed (zero-shot status of DAIR/GoEmotions) |
| 3 | Complaint / customer issue | 15 (+it-support-tickets, minds14 already used) | ABCD (en, MIT), help-desk-tickets (CC-BY-4.0, 357 rows), students-complaints (CC-BY-4.0); CFPB and NHTSA conditional | en; de/pt/es (it-support-tickets); ru (synthetic); pt (synthetic) | **Gap**: binary complaint-vs-not; fr it nl pl tr ar hi ja zh |
| 4 | NLI / entailment | 10 (+WANLI already used) | MNLI non-fiction (OANC, en), XNLI-2.0 MT of MNLI (de fr es ru tr ar hi zh …), NLI-TR (tr), WANLI-ja | en de fr es ru tr ar hi zh ja (all translated except en) | **Gap**: it pt nl pl; no native non-English NLI |
| 5 | Toxicity / moderation / spam / injection | 35 | DynaHate (CC-BY-4.0 upstream), GAHD (de, filtered), oasst2 votes (12 langs), Aegis 1.0/2.0, Aya red-teaming (8 langs), XSTest, JBB, HarmBench … | en de; prompts in fr es ru ar hi zh; oasst2 votes es ru zh de fr pt | **Gap**: comment-style toxicity outside en/de; spam/phishing outside en/zh; injection outside en/tr/zh |
| 6 | Reading comprehension | 4 | CUAD-QA, MAUD, QuALITY, FairytaleQA (all en, expert-written) | en only | **Gap**: every other language; en outside legal/fiction |
| 7 | Similarity / paraphrase / relevance | 10 (+esci, patent-phrase already used) | TaPaCo + Tatoeba (all 14 languages, CC-BY-2.0 FR), HEADLINES (en, public-domain papers), WANDS, Job-Title-Similarity (11 langs), MELO (21 langs), JaGovFaqs (ja) | all 14 for paraphrase/translation equivalence; en es ja product search | **Gap**: graded STS scores; general query–passage relevance; duplicate questions |
| 8 | Topic / document classification | 8 | Wikinews-multilingual (33 langs, CC-BY-2.5), arXiv abstracts (CC0), BIGPATENT, EUR-Lex (24 langs), COVID legislation (7 langs), HAL (fr), ITA-CaseHold (it), CUAD clauses | all EU priority languages via EUR-Lex; ja zh ar hi tr ru only a few hundred Wikinews articles | **Gap**: news-style taxonomy with train data; ja zh ar hi tr ru volume |
| 9 | Intent / dialogue act | 15 (+2 HOLD under CDLA-Permissive) | NLU++, Multi3NLU++ (es tr), Taskmaster, MultiWOZ 2.2, CaSiNo, CrossWOZ (zh), BiToD (en zh), Home-Assistant templates (60 langs), VaccinChatNL (nl) | en zh es tr nl; de fr it pt pl ru ar smart-home only; all 14 for yes/no polarity (LLM) | **Gap**: general-domain intents in ja hi ar ru de fr it pt pl; dialogue acts outside en/zh |
| 10 | Other decision tasks | 29 | stance es (ALIA, city open data CC-BY-4.0); ClaimBuster; TruthfulQA (+X in 7 langs); help-desk priority; German argument mining; BLiMP; Tatoeba/Aya language ID; 6 PII sets; CEFR (10 langs); Decima typed decisions (en fa ar ru) | varies (see tables) | **Gaps**: sarcasm (all langs), stance outside es/en, formality/politeness outside tr/ja, urgency outside en, subjectivity, humor negatives, clickbait |

Total: **145 distinct KEEP items** in the JSON (an item cross-listed in two categories is counted once). Roughly 250 candidates were examined and excluded; every exclusion is listed with its reason in the part tables.

## Decisions the owner has to make (they change the counts above)

1. **Emotion policy.** Clean emotion data exists (cat. 2), but `build_mixture.py`, `build_extra.py` and the docstring of `bench/eval_zeroshot.py` deliberately keep every emotion source out so that DAIR Emotion and GoEmotions stay zero-shot. Training on cat. 2 turns those two numbers into supervised transfer. Alternative: keep DAIR/GoEmotions zero-shot and add `llm-for-emotion/Cultural-Emo` (400 author-crafted items in ar/de/en/es/hi) as a new clean evaluation set.
2. **DynaHate.** tasksource tags it `gpl`, so the permissive filter drops it. The upstream README (bvidgen/Dynamically-Generated-Hate-Speech-Dataset) says "The dataset is licenced under CC-BY 4.0." and "All content is synthetic" (annotator-written). Recommendation: load the upstream v0.2.3 CSV directly and record it in DATA_LICENSES.md. Same pattern for `oasst2` (tasksource "unspecified", upstream card Apache-2.0).
3. **CDLA licences.** CDLA-Permissive-1.0/2.0 (MultiDoGo, DSTC11 intent induction) impose no obligation on "Results" (a trained model); CDLA-Sharing-1.0 (all Bitext sets) is share-alike on data but its §3.5 also exempts Results. Neither is on the allowlist. Not counted as KEEP; listed as HOLD.
4. **US-government-published consumer narratives** (CFPB Consumer Complaint Database, NHTSA vehicle complaints): structured fields are US federal works, the narratives are consumer-written and published with opt-in consent, no explicit licence. Counted as KEEP (conditional). CFPB stopped publishing new narratives on 14 Aug 2026, so a dated snapshot is needed.
5. **EUR-Lex.** `ddrg/super_eurlex` card says MIT but its body says "[More Information Needed]"; the actual basis is the EUR-Lex legal notice ("Reuse … for commercial or non-commercial purposes is authorised provided the source is acknowledged", Decision 2011/833/EU). A permissive rights-holder grant, but not an SPDX licence. Counted as KEEP with attribution "© European Union, https://eur-lex.europa.eu".
6. **Generator-model output clauses.** The repo audit already keeps LLM-generated data on the creator's licence (WANLI, agent_action_safety). Items where the generator's licence has an explicit output/derivative clause are marked in the notes: dhruv0808 (Llama 3.x/Gemma), OrSabbach (Qwen-research), AdvBench and BothBosu (Llama), ToxicCommons labels (Llama 3.1), SPML (GPT-4), Decima (Gemma-4, Apache-2.0 model card so no issue).
7. **TruthfulQA** is a common LLM benchmark; if any downstream comparison relies on it, hold it out instead of training.
8. **ToxiGen** (data CDLA-Permissive-2.0, but README "research purposes only") and **Nemotron-3.5-Content-Safety** (README CC-BY-4.0, no card licence, mixed provenance) are excluded for now; both could be revisited.

## Gaps (synthetic data needed, by priority)

1. **Sentiment, human-labelled, for de fr es it nl pl tr ar ja** — nothing clean exists; every candidate is tweets, scraped reviews, NC or SA. Also: aspect-level sentiment in any language (only IndicSentiment `ASPECT COMBO`, ReDial entity-level and 687 templated rows), and a real **mixed/neutral** class (only poem_sentiment, 892 rows).
2. **Emotion for de es it pt nl pl tr ru ar ja** (clean: en, fr synthetic, zh AI-text, hi/mr BRIGHTER); multi-label emotion outside hi/mr.
3. **Complaint vs non-complaint** (binary) in every language; complaint categorisation for fr it nl pl tr ar hi ja zh; urgency/priority outside English and non-urgent negatives (only 357 real priority rows in total).
4. **NLI for it pt nl pl** and native (non-translated) NLI in every non-English language — fix by translating MNLI non-fiction / WANLI / SynCSE with a permissively licensed MT model, then filtering as wanli-ja did.
5. **Toxicity/hate in comment style for fr es it pt nl pl tr ru ar hi ja zh**; harassment/cyberbullying as its own label; a benign-comment negative pool; spam/phishing outside en/zh; prompt injection outside en/tr/zh; response-harm labels outside English.
6. **Reading comprehension in every non-English language** (all multilingual sets use Wikipedia, FLORES or exam text) and English RC outside legal/fiction domains.
7. **Graded STS (0–5) in any priority language**; general query–passage relevance (MS MARCO is NC, MIRACL/NQ are Wikipedia); Quora-style duplicate questions; graded cross-lingual similarity.
8. **News-style topic taxonomy with a train split** (AG-News-like) and topic volume for ja zh ar hi tr ru. Buildable clean sources: Wikinews dumps per language (CC-BY-2.5, own categories), Global Voices (CC-BY-3.0), VOA (US public domain, minus wire items).
9. **General-domain intents for ja hi ar ru de fr it pt pl** (only smart-home templates) and dialogue acts outside en/zh.
10. **Sarcasm/irony in every language** (1.5k synthetic en items only), stance outside es/en, formality/politeness outside tr/ja, subjectivity, humor negatives, clickbait, multilingual claim verdicts with evidence.

## Corrections and follow-ups for the repo

- `licence_audit.json` / `build_mixture.py`: DynaHate is dropped by the "gpl" tag although upstream is CC-BY-4.0 (see decision 2). `oasst2/*` is dropped as "unspecified" although the upstream card is Apache-2.0.
- `bench/eval_zeroshot.py` docstring: Wikinews is the source of many FLORES sentences; if Wikinews-multilingual is trained on, the SIB-200/Belebele filter described above must be documented there.
- `tasksource/esci` is capped at 2,000 rows in the mixture; its es and ja rows (Apache-2.0, Amazon's own release) are the cleanest non-English relevance data found and deserve a higher cap.
- HF repos that are loading scripts (datasets 5 cannot run them): CUAD-QA, Taskmaster, MultiWOZ, QuALITY, Tatoeba, TruthfulQA-X; their data files must be fetched from upstream, with the revision pinned.
- IndicSentiment has no licence field on its HF card; the CC0 grant is only in the AI4Bharat/IndicBERT README. Pin the revision and cite that README.

# Detailed tables per category

Category index: 1 → Part A · 2 → Part B · 3 → Part A · 4 → Part C · 5 → Part D · 6 → Part E · 7 → Part C · 8 → Part E · 9 → Part E · 10 → Part B (+ Decima below). Each table quotes the licence string and where it was seen.


## Part A: categories 1 (sentiment) and 3 (complaint / customer issue)


Checked 2026-09-27 on pop-os. Metadata only: `HfApi().dataset_info` (card YAML `license` + `license:` tags), README text,
datasets-server `/size` and `/first-rows`. Original sources were checked with the GitHub API (`gh api repos/...`), Zenodo API,
Mendeley public API, the SPDX licence text, and project pages. "card" means HF card YAML licence.

Policy as in the repo audit (`tools/finetune/licence_audit.json`). A permissive tag does not count when the text comes from
tweets, Reddit, YouTube, app stores, review sites (Amazon/Yelp/IMDb/TripAdvisor/Booking/Ctrip, etc.), scraped news or
Wikipedia/FLORES (CC-BY-SA).

**Rule for LLM-generated sets (repo precedent: WANLI, I2D2, agent_action_safety):** the dataset creator's licence counts,
and the generator model's terms are not treated as a data licence. Items where the generator's own licence has an explicit
outputs/derivatives clause are marked **flag** so the owner can decide:

- Llama 3.1: "include 'Llama' in the model name"
- Gemma: model derivatives
- Qwen2.5-3B: `qwen-research`

**Evaluation overlap:** none of the KEEP items below comes from an evaluation-suite source. That includes the tyqiangz
test sets (Amazon Reviews Multi, Yelp, IMDb, SemEval tweets and so on), Banking77, MASSIVE, HWU64, GoEmotions, DAIR
Emotion and typed-decisions. Exact-match dedup against every eval test split (build_mixture `test_texts()`) is still
required, because these are cheap to run and some synthetic sets imitate review text.

### Category 1: sentiment (incl. neutral/mixed, aspect-level)

#### KEEP

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---:|---|---|---|
| google-research-datasets/poem_sentiment (Google, Gutenberg verses) | card `cc-by-4.0`; repo audit: "Google CC-BY-4.0, Project Gutenberg verses (public domain)" | en | 892 (+105 val, 104 test) | negative / positive / no_impact / **mixed** | none | **KEEP**. Already in mixture v5. It is the only real "mixed" class found. |
| ai4bharat/IndicSentiment (AI4Bharat IndicXTREME; reviews written by annotators, then translated) | HF card has **no** licence field. Official repo AI4Bharat/IndicBERT README: "All the datasets created as part of this work will be released under a CC-0 license" (GitHub, `gh api repos/AI4Bharat/IndicBERT/readme`). Code licence MIT. | en (ENGLISH REVIEW) + hi, bn, mr, ta, te, ur, gu, kn, ml, or, pa, as, bd | no train. validation 156 + test 1000 per language (checked hi: 1000 rows, 506 Positive / 492 Negative) | Positive / Negative (Neutral almost absent). Also CATEGORY, PRODUCT, ASPECTS, **ASPECT COMBO** (aspect-level) | none. Not one of our suites, so both splits can be trained on. | **KEEP**. Human-written product reviews, n-way parallel, with an aspect signal. Pin the revision, because the HF card lacks the licence. |
| community-datasets/re_dial (ReDial, Li et al. 2018; MTurk crowd dialogues) | card `cc-by-4.0`. Project page redialdata.github.io: "The dataset is published under the CC BY 4.0 License" | en | 10,006 dialogues (test 1,342) | per movie mention: seeker/recommender `liked` 0/1/2 (no / yes / did not say), `seen`, `suggested` | none | **KEEP**. Entity-targeted sentiment in dialogue (aspect-like), with a "did not say" neutral class. |
| tanaos/synthetic-sentiment-analysis-dataset-v1 (Tanaos, Artifex generator) | card `mit`. README: "created synthetically by Tanaos with the Artifex Python library" | en | 11,497 | very_negative / negative / neutral / positive / very_positive | none | **KEEP** (synthetic). The generator LLM is not named. Gives a 5-level score. |
| Novora/Tri-Class-Sentiment-Synthetic (phi3.5-mini, MIT model) | card `cc0-1.0`. README: "License: CC0-1.0 … generated by phi3.5-mini-instruct" | en | 33,000 | POSITIVE / NEUTRAL / NEGATIVE | none | **KEEP** (synthetic, uncurated, repetitive restaurant reviews). Sample lightly. |
| sutro/synthetic-product-reviews-20k (Sutro docs example; LLM reviews for catalogue products) | card `mit` | en | 20,000 | `rating_out_of_5` (1–5) | none | **KEEP** (synthetic). Use only `review_title` + `review_text` and the rating. Drop the `inputs`/`product_description` columns, which are catalogue text of unknown origin. Generator not stated. |
| YiMeng-SYSU/chinese-logic-sentiment-dataset (generated with the Doubao API, then manually cleaned) | card `apache-2.0`. README: "由豆包API生成并经过人工清洗/筛选" | **zh** | 2,176 (+545 val, 960 test) | 0 negative / 1 positive, plus `type` (irony, double negation, contrast, simple) | none | **KEEP**. The only clean zh sentiment set found. Hard cases: irony and negation. |
| dhruv0808/indic_sentiment_analyzer (English LLM-generated via Groq, translated into 11 Indic languages, plus about 1k rows per language from IndicSentiment) | card `cc-by-4.0`. README lists generators "Gemma 2 9B IT, LLaMA 3.1 70B/8B, LLaMA 3.2 1B/3B, Mixtral 8x7B" | en, **hi**, te, ta, kn, or, bn, gu, pa, ml, mr, as | 131,044 (all languages) | Positive / Neutral / Negative | none. It contains IndicSentiment rows, so dedup against it if both are used. | **KEEP (flag)**. Llama 3.x and Gemma output/derivative clauses apply to the generator. Neutral class present. |
| Kenshiii/synthetic-product-reviews (template generator) | card `cc-by-4.0`. README: "synthetic dataset generated using controlled templates … free from copyright restrictions" | en | 687 (+82 / 231) | overall sentiment + per-attribute (aspect) polarity (battery_life, durability, …) + `has_contradiction` | none | **KEEP** (small, templated). Aspect-level. Low weight. |
| KhiredNetworks/synthetic-product-reviews (Python templates, seed 42) | card `mit`. README: "Synthetically generated using Python … MIT License" | en | 9,000 | sentiment pos/neg/neutral + rating 1–5 + category | none | **KEEP (low value)**. Template text, low diversity. Cap small. |
| RichardSakaguchiMS/brazilian-customer-service-conversations | card `apache-2.0`. README: "Dataset sintetico (gerado por LLM)" | **pt** | 755 (+94 / 95) | `metadata.sentiment` (3 classes), `metadata.intent` | none | **KEEP** (also listed in cat 3). Sentiment in dialogue. Generator not named. |
| Adilbai/kz-gov-complaints-data-kz-ru | card `apache-2.0`. README: "generated using Gemini 2.5 Pro" | **ru**, kk | 1,200 | `sentiment`, `category`, `urgency`, `urgency_level` | none | **KEEP** (also listed in cat 3). |
| leonvanbokhorst/synthetic-complaints-v2 | card `mit` (the README has no body) | en | 34,229 (+8,680 test) | `sentiment` (score), `subjectivity`, `style`, `topic` | none | **KEEP** (also listed in cat 3). Every text is a complaint, so it gives no positive class. Generator not stated. |

#### EXCLUDE (tempting but not clean)

| HF id | Licence seen (card) | Reason |
|---|---|---|
| fancyzhx/amazon_polarity | `apache-2.0` | Amazon reviews (SNAP scrape). The tag is wrong; already excluded by the repo audit. |
| mteb/amazon_reviews_multi (MARC) | none | Amazon research-only terms. It is also the source of the tyqiangz eval suite. |
| stanfordnlp/sst2, SetFit/sst5 | `unknown` / none | Rotten Tomatoes review snippets (scraped). |
| stanfordnlp/imdb | `other` | IMDb reviews. |
| Yelp/yelp_review_full | `other` | Yelp dataset terms. |
| cardiffnlp/tweet_eval, cardiffnlp/tweet_sentiment_multilingual, mteb/tweet_sentiment_multilingual | `unknown` / none / `cc-by-3.0` | Tweets. tweet_sentiment_multilingual is also the eval source (tyqiangz). |
| takala/financial_phrasebank | `cc-by-nc-sa-3.0` | NC + SA. |
| tblard/allocine | `mit` | AlloCiné reviews (scraped). The tag conflicts with the text's origin. |
| clarin-pl/polemo2, allegro/klej-polemo2-in, clarin-pl/aspectemo | `bsd-3-clause`, `cc-by-sa-4.0`, `mit` | PolEmo 2.0 online customer reviews (scraped). klej copy is SA. |
| ruanchaves/b2w-reviews01 | `cc-by-4.0` | **Card conflicts with source.** americanas-tech/b2w-reviews01 README: "under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license". |
| shunk031/JGLUE (MARC-ja) | `cc-by-4.0` | Upstream yahoojapan/JGLUE is `CC-BY-SA-4.0` (GitHub API). MARC-ja is Amazon reviews. |
| dynabench/dynasent | none | Yelp-derived. Already excluded by the audit. |
| jakartaresearch/semeval-absa, yangheng/ABSADatasets, HiTZ/Multilingual-Opinion-Target-Extraction, srinivasbilla/semeval-2016-absa-* | `cc-by-4.0` / none / `apache-2.0` / `mit` | SemEval ABSA review texts. The task site says "All Right Reserved", and the SemEval-2016 French annotations are CC BY-NC-ND 4.0 (alt.qcri.org/semeval2016/task5). HiTZ is a DeepL translation of SemEval-2014. |
| Multilingual-NLP/M-ABSA | `apache-2.0` | GitHub swaggy66/M-ABSA has no licence. Domains coursera/hotel/laptop/restaurant/phone/sight/food are scraped reviews, LLM-translated into 21 languages. |
| jordiclive/OATS-ABSA, bhavnicksm/sentihood | none / `cc-by-4.0` | Coursera/hotel/Amazon reviews; Yahoo Answers. |
| Booking-com/absa-dataset | none | README §10: "released for non-commercial research use, cc-by-sa-4.0 license". |
| ThiagoFNJ/esci-llm-aspect-mining | `cc-by-4.0` | Card says "No Amazon review text is included". Text needs the unofficial ESCI-S scrape. |
| Brand24/mms, clapAI/MultiLingualSentiment, mteb/multilingual-sentiment-classification, Youseff1987/multilingual-sentiment-dataset, hungnm/multilingual-amazon-review-sentiment-processed | `other`, `apache-2.0`, `unknown`, `mit`, `mit` | Aggregates of tweets and review corpora (Youseff: "stanfordnlp/imdb, Sentiment140Twitter, …amazon…"). |
| Sp1786/multiclass-sentiment-analysis-dataset | `apache-2.0` | Tweet-sentiment-extraction texts; no provenance. |
| aisingapore/NLU-Sentiment-Analysis | `cc-by-sa-4.0`+`cc-by-4.0`+`cc0-1.0` | Mixes NusaX (CC BY-SA) and social-media sources. Gated. |
| Kenpache/multilingual-financial-sentiment | `apache-2.0` | README: "released for academic and non-commercial research only … Copyright of original texts remains with their respective publishers". |
| numind/NuSentiment | `mit` | C4 web-crawl sentences with GPT-3.5 labels. Same web-text rule as the audited `discovery`. |
| GillesJacobs/sentivent | `cc-by-4.0` | Business-news articles (publisher copyright). |
| JanosAudran/financial-reports-sec | `apache-2.0` | EDGAR text would be acceptable (LEDGAR precedent), but the label is filing-level stock-return movement, not sentence sentiment. Rejected on label validity. |
| TheFinAI/fiqa-sentiment-classification, zeroshot/twitter-financial-news-sentiment, FinGPT/fingpt-sentiment-train, nickmuchi/financial-classification | `mit` / `mit` / none / none | StockTwits/Reddit/tweets, FPB (NC-SA). |
| ai-forever/kinopoisk-sentiment-classification, ai-forever/ru-reviews-classification, irfan-ahmad/AURA-Sentiment, left0ver/sentiment-classification, dralsarrani/AraReview | `mit` / `apache-2.0` / `mit` / `mit` / `mit` | Scraped movie, e-commerce, app-store and hotel reviews (ChnSentiCorp style). |
| DmitrySharonov/ru_sentiment_neg_pos_neutral, aieng-lab/en-sentiment-nrc-neutral, winvoker/turkish-sentiment-analysis-dataset, pythainlp/wisesight_sentiment | `apache-2.0` / `cc-by-3.0` / `cc-by-sa-4.0` / `cc0-1.0` | Tweets (aieng-lab card: "Twitter / X ToS still apply"); SA; social-media posts. |
| argilla/banking_sentiment_setfit, argilla/sentiment-banking | none | Missing licence **and** these are Banking77 texts, which overlap the Banking77 test. |
| AiresPucrs/sentiment-analysis-pt, PORTULAN/extraglue (SST-2 part) | `apache-2.0` / `mit` | Machine translations of IMDb / SST-2. |
| asapp/slue (VoxCeleb sentiment) | `cc0-1.0`,`cc-by-4.0` | Transcripts of YouTube interviews (platform text). |
| leduckhai/Sentiment-Reasoning | `mit` | Provenance of the VietMed audio is unverified. Mainly vi; en/de/zh/fr are machine translations. |
| SahmBenchmark/Sentiment_Analysis_MCQ_train, iam-tsr/hindi-sentiments, hanerdem/turkish-sentiment-dataset, Suzana/synthetic_product_reviews_sentiment, shimaa22/arabic_students_comments | various | Scraped Arabic news; tweet-like text with no provenance; malformed file; no card/provenance (<1k rows); "research and educational purposes only" contradicts MIT. |

### Category 3: complaint / customer issue (tickets, product issues, escalation, priority)

#### KEEP

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---:|---|---|---|
| github:asappresearch/abcd (ABCD, ASAPP 2021; human-to-human "Expert Live Chat" crowd dialogues) | GitHub API `license.spdx_id` = `MIT`; LICENSE file "MIT License, Copyright (c) 2021 ASAPP Research". Not on HF: data is `data/abcd_v1.1.json.gz` in the repo. | en | ~8k train conversations (README: "over 10K human-to-human dialogues") | `scenario.flow` (10 issue families: product_defect, shipping_issue, account_access, purchase_dispute, …) and `scenario.subflow` (55 intents) | none (not Banking77, HWU64 or MASSIVE) | **KEEP**. Best real customer-issue set found. |
| tasksource/it-support-tickets (Zenodo 10.5281/zenodo.7648117, real IT tickets, Brazil) | card `cc-by-4.0`; Zenodo API `license.id` = `cc-by-4.0` | en, de, pt, es | 1,572 (test 657) | 7 categories (Active Directory, O365, Software, …) | none | **KEEP**. Already in mixture v5. |
| tasksource/help-desk-tickets (Mendeley 10.17632/btm76zndnt.3, real tickets from a software company) | card `cc-by-4.0`; Mendeley public API `data_licence.short_name` = "CC BY 4.0" | en + others (mixed) | 357 text issues (`issues` config 66,691 metadata-only rows) | `issue_priority`, `issue_type`, `issue_resolution`, `issue_status` | none | **KEEP (small)**. Real **priority/urgency** labels. |
| PolyAI/minds14 | card `cc-by-4.0` (repo DATA_LICENSES, pinned revision) | cs, de, en, es, fr, it, ko, nl, pl, pt, ru, zh | ~8k transcripts | 14 e-banking issues (card_issues, app_error, abroad, freeze, fraud…) | none | **KEEP**. Already used in extra-v1. |
| BEE-spoke-data/consumer-finance-complaints, config `has-text` (CFPB Consumer Complaint Database) | card `cc0-1.0`. Source: US federal agency. CFPB 2015 policy statement (files.consumerfinance.gov/f/201503_cfpb_disclosure-of-consumer-complaint-narrative-data.pdf): "Only those narratives for which opt-in consumer consent is obtained and a robust personal information scrubbing standard … will be eligible for disclosure". No explicit copyright licence for consumer-written narratives. | en | 1,689,573 with narrative | Product (~18), Issue, Sub-issue, Company response (closed with monetary / non-monetary relief / explanation), Timely response, Consumer disputed | none | **KEEP (conditional)**. Structured fields are a US federal work; narratives are consumer-authored and published with opt-in consent that consumers may withdraw. Same risk class as the civil_comments precedent. Needs the owner's OK. Note: as of 14 Aug 2026 CFPB stopped discretionary narrative publication (CFPB newsroom), so use a dated snapshot. |
| liri-uzh/cfpb-complaints-mini (CFPB subset, 2022–23) | card `cc0-1.0`. README: "a U.S. federal work in the public domain. This derived subset is released as CC0-1.0" | en | 6,469 (test 1,635) | product (balanced, capped at 1,200 per class) | none | **KEEP (conditional, same caveat)**. Clean parquet alternative to the full dump. |
| vic35get/nhtsa_complaints_dataset (NHTSA ODI vehicle complaints) | card `apache-2.0`. Source is US federal (NHTSA), consumer-written summaries. No licence statement verified at nhtsa.gov (data.gov API did not respond). | en | 8,356 (+2,089 eval / 2,089 test) | `components` (AIR BAGS, STEERING, ELECTRICAL SYSTEM, …) | none | **KEEP (conditional)**. Product-defect complaint categorisation. Same caveat as CFPB. |
| alaminxpro/university-students-complaints | card `cc-by-4.0`. README: "332 real-world complaints collected from university students … CC BY 4.0" | en | 273 (test 59) | Category, severity, responsible departments (multi-label), aspects | none | **KEEP (small, real)**. |
| hblim/customer-complaints | card `mit`. README: "generated with ChatGPT 4o" | en | 1,261 (+210 / 211) | billing / delivery / product | none | **KEEP** (synthetic). |
| cngchis/Support-Ticket-Router-12K-Cleaned | card `apache-2.0`. README: "synthetically generated using GPT-4-class models … intended for research, benchmarking, and model training only" | en | 9,354 (+1,169 / 1,170) | ticket intent incl. **complaint**, billing, upgrade, … | none | **KEEP** (synthetic). The phrase "intended for … model training only" is a usage note, not a licence restriction. |
| Adilbai/kz-gov-complaints-data-kz-ru | card `apache-2.0`; Gemini 2.5 Pro generated | **ru**, kk | 1,200 | category, urgency, urgency_level, sentiment, status | none | **KEEP** (synthetic). Urgency labels. |
| RichardSakaguchiMS/brazilian-customer-service-conversations | card `apache-2.0`; "gerado por LLM" | **pt** | 755 (+94 / 95) | intent (e.g. duvida_servico, cancelamento), sentiment | none | **KEEP** (synthetic). |
| leonvanbokhorst/synthetic-complaints-v2 | card `mit` | en | 34,229 (+8,680) | topic, style, sentiment, subjectivity (every row is a complaint) | none | **KEEP** (synthetic). Complaint topic/intensity only. |
| OrSabbach/food-delivery-support-tickets | card `mit`. README: generated with `Qwen/Qwen2.5-3B-Instruct` (that model's licence is `qwen-research`) | en | 10,153 (in `data/dataset.parquet`; the viewer shows only a metrics file) | 8 issue categories (reliable). Sentiment/urgency are **unreliable**: README says "~84% of rows labeled positive read as negative" | none | **KEEP (flag)**. Use `category` only. The generator's research licence needs an owner decision. |
| Console-AI/IT-helpdesk-synthetic-tickets | card `mit` (no README) | en | 500 | priority (Low/Medium/High), category (Network, Software, Account, …) | none | **KEEP (minor)**. Synthetic; priority labels. |
| s2pidape/support-ticket-dataset | card `cc-by-4.0`. README: "1.5M synthetic support tickets … I generated it with a simulator … CC BY 4.0 … including commercially" | en (field `language` is metadata only) | 1,500,000 × 2 configs | category, issue_type_id | none | **KEEP (low value)**. README: "issue_description is template-generated and one-to-one with issue_type_id". Cap at a few hundred. |
| adiprog14/lingrow-support-tickets | card `mit`. Synthetic: 116 base messages paraphrased with humarin/chatgpt_paraphraser_on_T5_base | en | 11,008 | error_code / category | none | **KEEP (low value)**. Low diversity. |

#### EXCLUDE

| HF id | Licence seen (card) | Reason |
|---|---|---|
| bitext/Bitext-customer-support-llm-chatbot-training-dataset and all 11 other bitext/* verticals; Faramir/…-spanish; abhi23457 copy | `cdla-sharing-1.0` | ShareAlike data licence, not on the allowed list. **Policy note:** CDLA-Sharing-1.0 §3.5 (SPDX text) says "This Agreement imposes no obligations or restrictions on Your Use or Publication of Results", so a trained model may count as "Results". If the owner whitelists CDLA-Sharing, Bitext (27 intents incl. `complaint`, 11 verticals) becomes the richest synthetic customer-support source. |
| jonathansuru/customer_service_intent_detection, matefh/bitext-customer-support-intent-classification | `apache-2.0` / none | Bitext-derived texts (CDLA-Sharing), relabelled. |
| Tobi-Bueck/customer-support-tickets (+ vasu1111, Prady06, alibinfaizan, miku1211, yadavmana, manojroyal23 copies); ale-dp/german-english-email-ticket-classification, ale-dp/bilingual-ticket-classification; mindweave/help-desk-tickets; w1z4rd3k/it-support-l1-ticket-classification | `cc-by-nc-4.0` | Non-commercial. Tobi-Bueck was already rejected in extra-v1. |
| TNE-AI/customer-support-on-twitter-conversation, gorkemsevinc/Customer_Support_on_Twitter, MohammadOthman/mo-customer-support-tweets-945k, venetis/customer_support_sentiment_on_twitter | none / none / `cc-by-nc-sa-4.0` / `afl-3.0` | Tweets (TWCS). |
| milesbutler/consumer_complaints, davidheineman/consumer-finance-complaints-large | `mit` / `apache-2.0` | CFPB data relabelled with a licence the source does not grant. Use the CC0 mirrors above instead. |
| claritystorm/nhtsa-vehicle-complaints, claritystorm/cfpb-consumer-complaints | `other` (license_name `public-domain`) | 1k-row preview of a "$79 purchase" snapshot "subject to its documented limitations and license terms". |
| FinanceMTEB/Complaints | none | No licence; 16 rows. |
| rjac/e-commerce-customer-support-qa | `mit` (README: "License: [More Information Needed]") | Unknown provenance. |
| Roy229/customer-support-intents-v1 | `mit` | README: "Source: Acme Customer Experience Team (proprietary corpus)". Conflicting. |
| rootfs/user-satisfaction-dataset | `apache-2.0` | README: "Derived from … MIMICS (Microsoft), SGD (Google), INSCIT". SGD is CC-BY-SA-4.0; MIMICS has its own terms. |
| irongateprd/support-ticket-intents, karanverma19/Multilingual_Customer_Support_Intent_Dataset_for_Indian_Contexts | `cc-by-4.0` / `apache-2.0` | Too small (20 and 41 train rows). |
| Porameht/customer-support-th-26.9k | `cc-by-sa-3.0` | ShareAlike. |
| Aleph-Alpha/ticket-classification, synthdataq9x260918/synthetic-ecommerce-customer-service-dialogues-100k | `apache-2.0` / `cc-by-4.0` | No readable data/README (viewer empty). Not verifiable. |
| electricsheepafrica/*support-ticket* | `other` | Custom terms. |

### Gaps (category 1 + 3)

1. **Multilingual real sentiment:** no licence-clean human-labelled sentiment data exists for **de, fr, es, it, nl, pl, tr, ar, ja**. Every candidate is tweets, scraped reviews, NC or SA. Clean coverage outside en:
   - hi: IndicSentiment, dhruv0808 (flag)
   - zh: YiMeng, 2k synthetic
   - pt: Sakaguchi, 755 synthetic dialogues
   - ru: Adilbai, 1.2k synthetic

   This needs synthetic generation, ideally translate-and-verify of IndicSentiment-style annotator-written reviews.
2. **Aspect-based sentiment:** no clean human ABSA corpus in any priority language. The only aspect signals are:
   - IndicSentiment `ASPECT COMBO` (en/hi)
   - ReDial entity-level `liked`
   - Kenshiii templated attributes (687 rows)
   - alaminxpro complaint aspects

   SemEval/M-ABSA/MAMS/OATS/Booking are all excluded, so ABSA should be synthetic.
3. **Mixed sentiment:** only poem_sentiment has a real `mixed` class (892 train rows, verse domain). Neutral exists only in synthetic sets and ReDial "did not say".
4. **Complaint vs non-complaint (binary) detection:** no clean real set; the Preotiuc-Pietro complaints corpus is tweets. The positives available are all "everything is a complaint" sets (CFPB, NHTSA, leonvanbokhorst). Negatives must come from other support intents (ABCD, minds14) or be synthetic.
5. **Multilingual complaint/issue:** only it-support-tickets (en/de/pt/es), minds14 (12 languages, already used), Sakaguchi (pt) and Adilbai (ru). Nothing clean for fr, it, nl, pl, tr, ar, hi, ja, zh complaint categorisation.
6. **Urgency/priority/escalation:** real labels only in help-desk-tickets (357 rows). The rest is synthetic (Console-AI 500, Adilbai 1.2k). OrSabbach's urgency labels are unusable by its own audit. Needs synthetic data.
7. **Owner decisions that would change the counts:**
   - CDLA-Sharing-1.0 whitelisting (Bitext)
   - accepting consumer-authored US-government-published narratives (CFPB, NHTSA)
   - accepting sets whose generator licences carry output clauses (dhruv0808: Llama 3.x/Gemma; OrSabbach: Qwen research)

#### Part F: weak categories (fact-check, emotion, topic), 2026-09-28

Why: on the held-out category suites of 0.7.0 the weakest cells are fact-check 0.313 (ClaimBuster
crowdsourced.csv), emotion 0.586 (French-emotion, BRIGHTER test splits) and topic 0.607 (big_patent test).
Targets: claim/fact sources with both classes, emotion data in languages that had none (pt ru it nl ja ar tr),
topic sources with single-label gold.

Rules applied on top of the policy above: commercial use and no ShareAlike, verified on the card **and** at the
upstream source (the stricter wins); no benchmark or benchmark translation; no text generated by hosted OpenAI,
Anthropic, xAI or Google models (Gemma treated as Google to be safe); no split that `bench/eval_categories.py`
HELD_OUT reads (checked by `test_no_new_source_is_a_held_out_source`). Roughly 60 candidates were examined; the
ones not kept are listed below with the reason. All five kept sources pass `audit.py` with 0 FAIL.

### KEEP (source_part F in v6-keep.json)

| Source | Licence and evidence | Provenance | Languages | Rows after filter (items at full size) | Category |
|---|---|---|---|---|---|
| `Horizon-Labs/multilingual-zeroshot-synthetic`, task `emotion`, kind `tax_short`, commit 11ed607c | Card YAML `odc-by`; README "ODC-BY 1.0 (… generated content is from an Apache-2.0 model)"; code repo [horizon-ai-labs/agent-io-guards](https://github.com/horizon-ai-labs/agent-io-guards) LICENSE Apache-2.0; generator [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) card `apache-2.0` | README: "All labels and generated texts come from Qwen3.8-27B"; `zeroshot/gen_zeroshot_v2.py` `MODEL = "Qwen/Qwen3.8-27B-FP8"`. Open-weight model; only `text_origin == qwen-generated` rows (the FineWeb passages are not read) | 14 model languages: en 2,758, es 462, it 382, zh 343, nl 341, hi 326, ru 324, pl 301, fr 299, ar 296, pt 279, tr 278, ja 267, de 257 | 7,568 rows (13,826 items: choice + balanced yes/no) | 2-emotion |
| NAIST LIFE STORY (`sociocom:naist-life-story`), 4 quarterly xlsx files 2023-Q4 to 2024-Q3, SHA-256 pinned | [Data page](https://sociocom.naist.jp/life-story-data/): "本成果物は クリエイティブ・コモンズ 表示 4.0 国際 ライセンス の下に提供されています。" (CC BY 4.0) | Human, crowd-written: "1,000 crowdsourced participants per quarter have written short texts describing personal experiences associated with seven emotions" (LREC 2026, 2026.lrec-1.522). The label is the survey column the text was written for | ja | 14,637 rows (29,274 items); 2,299 to 2,658 per class | 2-emotion |
| `agentlans/fact-or-opinion`, train, **`source == "DeepSeek"` only** | Card YAML `odc-by`, tag `synthetic`; no upstream repo or paper (weakest evidence of the five) | The generator of every row is in the `source` column: ChatGPT, Claude Sonnet 4, Gemini 2.5 Flash, Microsoft Copilot, Perplexity, Le Chat, DeepSeek. Only DeepSeek rows are read (no hosted OpenAI/Anthropic/xAI/Google output; Copilot and Perplexity route to those, Le Chat is a hosted Mistral service and was left out as well) | en de fr es it pt ru ar ja zh hi (about 300 each) | 3,305 rows (6,610 items); fact 571, opinion 835, fact and opinion 976, neither 923; "states a fact" yes 1,547 / no 1,758 | 10-claim-detection |
| `hheiden/us-congress-bill-policy-115_117`, commit a46b7d34 | Card YAML `mit` ("License: MIT"); upstream GovInfo BILLSTATUS: "Pursuant to Title 17 Section 105 of the United States Code, this file is not subject to copyright protection and is in the public domain." ([117hr1](https://www.govinfo.gov/bulkdata/BILLSTATUS/117/hr/BILLSTATUS-117hr1.xml)) | Titles and Congressional Research Service summaries from congress.gov; "One policy area term, which best describes an entire measure, is assigned to every public bill" ([congress.gov](https://www.congress.gov/help/find-bills-by-subject)). Full bill text not read | en | 47,771 bills, 31 policy areas (Health largest, Animals 250 smallest); "Private Legislation" and "Social Sciences and History" dropped | 8-topic |
| `nlp-waseda/e_gov`, train, commit 08e57e36 | Card YAML `cc-by-4.0`; statutes are not copyrightable: Copyright Act art. 13 "次の各号のいずれかに該当する著作物は、この章の規定による権利の目的となることができない。一 憲法その他の法令" | "Japanese law dataset obtained from e-Gov"; label = the official e-Gov law field (法令分野), one per law | ja | 9,075 laws, 50 fields (行政組織 largest, 河川 14 smallest); first 1,600 characters (name, table of contents, first articles) | 8-topic |

Honest limits:
- **Fact-check** stays thin. No clean, human-labelled check-worthiness data exists outside ClaimBuster (whose other derivatives would leak into the eval suite). fact-or-opinion teaches the factual / non-factual axis of the ClaimBuster labels, not check-worthiness itself, and its labels come from the generator. It is asked as a new task `claim_detection` (templates and glosses in all 14 languages), so the fact-check verdict question keeps its meaning.
- **Emotion** for pt ru it nl ar tr is only the Horizon synthetic subset (≈280–380 rows per language, one generator, the card estimates ≈7 % label noise). Japanese gets real, human-written data. No clean human-written emotion corpus was found for the other six languages.
- **Topic**: two large single-label public-sector sets (en, ja). Nothing clean for ar hi tr ru pl nl.

### Emotion taxonomy (`tools/finetune/mixture_v6/emotion_taxonomy.py`, name `basic8`)

Eight classes: Ekman's six basic emotions (anger, disgust, fear, joy, sadness, surprise), love (a basic-level
category in Shaver et al. 1987), and neutral. They are the labels `templates.GLOSSES["emotion"]` already describes
in 14 languages. A source label maps to a class if it is the class, a synonym, or a subordinate emotion that Shaver
et al. place under the class. Any other label has no class, and the row is dropped rather than forced into a class.
A registry entry opts in with `"emotion_taxonomy": "basic8"`.

| Source label | Class | Horizon rows (14 langs) | LIFE STORY column |
|---|---|---|---|
| joy / Joy (楽しい) | joy | yes | yes |
| pride, relief | joy | yes | – |
| sadness / Sadness (悲しい), disappointment | sadness | yes | yes |
| anger / Anger (怒り) | anger | yes | yes |
| fear, anxiety / Anxiety (不安) | fear | yes | yes |
| disgust / Disgust (嫌悪感) | disgust | yes | yes |
| surprise / Surprise (驚き) | surprise | yes | yes |
| love | love | yes | – |
| neutral | neutral | yes | – |
| gratitude | none: row dropped | yes | – |
| Trust (信頼感) | none: column not read | – | yes |

The existing emotion sources keep their own label sets, because they do not opt in. JusteLeo/French-emotion and
BRIGHTER feed the held-out emotion suite through the same adapter, and remapping them would change that suite. Their
labels map without loss in any case: BRIGHTER already uses the six basic labels, and French-emotion's
colere/degout/joie/neutre/peur/surprise/tristesse are the seven basic classes in French. Moving those two onto `basic8`
is a follow-up to decide with the owners of `bench/eval_categories.py`.

### EXCLUDED (examined for this part)

| Candidate | Reason |
|---|---|
| BRIGHTER train configs ptbr / ptmz / rus / arq / ary | Card `cc-by-4.0`, but the text fails the provenance rule (arXiv 2502.11926): ptbr "mainly from subreddits", ptmz news headlines, rus "Russian Twitter corpus", arq a translation of a copyrighted novel, ary tweets and BBC headlines. ptbr/rus test are also held-out suites |
| NAIST LIFE STORY other quarters | Same licence; not added in this round (four quarters already give 14.6k texts); can be added by pinning further files |
| `NoeFlandre/EmoTalk-7` | Generator `mistral-medium-2505` (hosted, not open-weight); joy-skewed open-vocabulary labels |
| GoEmotions / DAIR-emotion translations (`43ntropy/synthlangs-goemotions`, `AnasAlokla/multilingual_go_emotions`, `seara/ru_go_emotions`, `log101/emotion-turkish`, `Adilmar/caramelo-emotions-v2`, …) | Evaluation suites and their translations |
| CEDR (`sagteam/cedr_v1` and copies), ArPanEmo, ExaAEC, emotone_ar, MultiEmotions-It, FEEL-IT, EmotioNL, EXALT | Twitter / Reddit / YouTube / Facebook / imageboard text, or NC/SA, or a WASSA task |
| WRIME | CC BY-NC-ND (already rejected) |
| TREMO (tr) | Kaggle licence "Unknown", described as non-commercial |
| `VBadazhkov/therapy_emotions_ru`, `psytechlab/EmpatheticIntents-ru` | Claude- / GPT-4o-generated |
| `GiliGold/Knesset_check_worthiness` | Card cc-by-4.0 but its text source KnessetCorpus card says cc-by-sa-4.0 (stricter wins); labels are model predictions; Hebrew only; 12.9 GB |
| `Intel/misinformation-guard` | CDLA-Permissive-2.0 is not on the allowlist (HOLD, as MultiDoGo/DSTC11) |
| ClaimHack (Zenodo 15110129) | 4,019 of 5,217 labelled rows are in ClaimBuster crowdsourced.csv, the held-out suite |
| CheckThat! 2021–2025 sets, `Nithiwat/*`, `OpenFact/*`, ClaimBuster re-uploads | Benchmarks, ClaimBuster-derived (eval leakage), tweets, or no licence |
| AFaCTA / PoliClaim | No licence; GPT-3.5/4 labels; contains CLEF debate data |
| `climatebert/environmental_claims` | cc-by-nc-sa-4.0 |
| `gtfintechlab/Numclaim` | Proprietary analyst reports / earnings calls; label is forward-looking vs factual |
| Fact-check-site and news sets (ClaimReview2024plus, FactChecksbr, `tomn24/czech-politician-statements`, X-Fact, MultiFC, LIAR, FEVER family, SciFact, AVeriTeC) | Copyrighted fact-check/news text, Wikipedia SA, NC, or benchmarks |
| `rcds/swiss_law_area_prediction`, `rcds/swiss_judgment_prediction` | cc-by-sa-4.0 |
| `tanaos/synthetic-topic-classification-dataset-v1` | Generator not disclosed (hosted service) |
| `GoktugD/turkish-topic-classification-1.5m` | Template-generated with the label cued by keywords: no topic signal |
| `agentlans/en-document-topic-classification`, `agentlans/multilingual-document-classification` | Web scrape with classifier (silver) labels; the second has no licence |
| OpenAlex / arXiv-style topic sets with publisher abstracts | Publisher abstracts; model-assigned topics |
| `AI-team-UoA/greek_legal_code` | Clean (cc-by-4.0, Greek statutes) but Greek is not a model language and it is an MTEB/LEXTREME task; not added |
| Horizon `news headline section` / `question topic` subsets | Label sets mirror AG News and Yahoo Answers; AG News is a zero-shot suite whose label space must stay unseen |
| Taiwan law database, Federal Register API, CORDIS projects | Clean or likely clean, but need a new raw-file builder: the Taiwan files change daily, the Federal Register label is the issuing agency (a topic proxy), and CORDIS objectives are consortium-written text under a notice that excludes non-EU-owned content. Follow-ups |
| Bundestag DIP API (`sachgebiet`, de) | Terms allow commercial reuse with attribution but add a non-standard use restriction (§5); rows not inspected. Follow-up |


#### Part G: categories where 0.7.0 trails Qwen3-8B zero-shot, 2026-09-29

Why: in docs/BASELINES.md Statim 0.7.0 trails Qwen3-8B (zero-shot) on emotion (0.586 vs 0.726), fact-check
(0.313 vs 0.493), sentiment (0.800 vs 0.873), safety (0.727 vs 0.753) and PII (0.856 vs 0.878). Targets per
category came from the suites: fact-check asks the ClaimBuster three-way verdict on debate sentences; emotion
asks the six Ekman classes on short first-person posts; PII asks "contains personal data?" and English
per-type probes in 11 languages (Qwen leads most in de, ar, ja, nl, ru, it).

Rules applied on top of Part F: licence checked on the card **and** in the repository's LICENSE/NOTICE or the
paper; commit pinned; no gated dataset (the CI audit runs without HF_TOKEN); no text or label from hosted
OpenAI, Anthropic, Google (Gemma included), xAI, Microsoft Copilot, Perplexity or hosted Mistral models, from
Llama- or Qwen-research-licensed models, or from an undisclosed generator; no benchmark; no HELD_OUT source or
derivative. Candidates were also checked for exact text overlap against the 29,767 pooled items of the five
suites, and 20+ gold labels per source were read by hand.

Coverage: fact-check was searched exhaustively (38 candidates: Hub, Zenodo, OSF, figshare, arXiv, ACL
Anthology, GitHub). Emotion, sentiment, safety and PII searches were cut short for budget, and a second search
on 2026-09-29 continued them: 60 more candidates checked at the source (the handoff's open candidates first,
then emotion in ru/es/hi/de, Simplified-Chinese sentiment with irony and contrast, English safety hard
negatives, PII in ar/ja/ru/zh, and one new fact-check set). It kept one source (`Powpowpow23/ru-pii-ner-data`);
the 59 rejected ones are in the EXCLUDED table with their reasons.

### KEEP (source_part G in v6-keep.json)

| Source | Licence and evidence | Provenance | Languages | Rows after filter (items at full size) | Category |
|---|---|---|---|---|---|
| `naeyn/nobody-pii-synth-de`, train split, commit f78fcbca | Card YAML `apache-2.0`; repository [LICENSE](https://huggingface.co/datasets/naeyn/nobody-pii-synth-de/blob/f78fcbcad638f5e7f69bfe55a8ebd508e32c3731/LICENSE) "Apache License, Version 2.0"; [NOTICE](https://huggingface.co/datasets/naeyn/nobody-pii-synth-de/blob/f78fcbcad638f5e7f69bfe55a8ebd508e32c3731/NOTICE) "Apache-2.0 (see LICENSE). This repository contains procedurally generated rows and does not contain records copied from ai4privacy/…" | README: "generated from templates and Faker locale providers. No source records containing real people's personal data were used"; no language model. Not gated. 0 exact overlap with the pooled suite items | de 4,795, en 2,379, nl 1,792 (8,966 of 9,450 rows; PII-free rows whose language the text does not confirm are dropped) | 9,133 items: pii_type probes de 4,710, en 2,151, nl 1,746; pii choice de 368, en 74, nl 84. The pii yes/no task is dropped (97.6 % yes) | 10-pii |
| `Powpowpow23/ru-pii-ner-data`, train split, commit 55acbec0 | Card YAML `apache-2.0`; pinned [LICENSE](https://huggingface.co/datasets/Powpowpow23/ru-pii-ner-data/blob/55acbec07fb04a18455111a34337cb515a7f821e/LICENSE) "Apache License, Version 2.0"; README confirms the author's distribution right. DeepSeek Terms (2026-03-27) §4.2(3) allow outputs for "training other models (such as model distillation)", the same evidence rule as `agentlans/fact-or-opinion` | README: DeepSeek-family teacher wrote templates; code filled them with fictitious Faker/custom-generator values and tracked character spans. External corpora are excluded; not gated. 0 exact matches against the 29,767 pooled sentiment/emotion/safety/fact_check/pii items | ru; 104,111 train rows (validation was the development split), 25 nested-span types | Type names are mapped to the existing PII vocabulary where meanings match (`FULL_NAME`→`person`, `DATE_OF_BIRTH`→`date of birth`, `PHONE`→`phone number`, `INN`→`tax id`, etc.). A negative type probe must be in `supervised_types`; only the 2,400 `negative_examples` rows can answer the broad PII question "no". The normal per-source item cap applies | 10-pii |

Honest limits:
- **PII** gets one source for de and nl, plus one Russian source with finer document, payment and address types.
  The de/nl source has too few PII-free texts for the "contains personal data?" question, so it trains the
  per-type probes only. The Russian source is automatically labelled and not fully hand-checked; its partial
  `supervised_types` coverage must be retained, and only its explicit negative family is safe as broad negatives.
  ar, ja and zh still rely on the nym train split alone.
- **Fact-check**: no source passed. Outside ClaimBuster, check-worthiness data is either CheckThat!-derived,
  fact-check-site or social-media text, unlicensed, restricted, or labelled by a hosted model. The one clean
  human-labelled set (op-fed) labels opinion, not the three-way verdict.
- **Emotion, sentiment, safety**: nothing kept, after both searches (the second one checked 39 more candidates
  for these three). What exists is NonCommercial, a benchmark, platform or scraped text, or written by a
  hosted model the policy excludes. These categories need synthetic data from a model whose licence allows it
  (tools/synth with a local Apache-2.0 model).

### EXCLUDED (examined for Part G)

| Candidate | Category | Reason |
|---|---|---|
| `kakeith406/op-fed` | fact-check | CC BY 4.0, human labels on FOMC transcripts (public domain), not gated; but the label is opinion yes/no, which cannot give the ClaimBuster check-worthy / unimportant factual / non-factual gold without inventing the factual split |
| `Aniemore/resd_annotated` | emotion | MIT, not gated, 0 overlap; but the emotion was acted in the voice and transcripts often carry none of it (a polite sales greeting is labelled anger): wrong gold for a text model |
| `DataikuNLP/kiji-pii-training-data` | pii | Apache-2.0, but 'synthetically generated using LLMs' with no generator named |
| `menamerai/cheer-ekman` | emotion | Apache-2.0 card with no provenance, annotation or generator documented |
| `github:lejafar/FactRank` | fact-check | Text origin is not clean: about 15% politicians' tweets, about 16% newspaper interviews (copyrighted news) and about 8% fact-check-site claims. The CSV has only id,statement,label, so the roughly 61% parliamentary rows cannot be f… |
| `zenodo:14748539` | fact-check | YouTube text, which the policy rejects. The text is not distributed and would have to be re-transcribed from YouTube. Zenodo access is restricted, which counts as gating. |
| `zenodo:17482958` | fact-check | No licence and restricted access. Built from rejected and held-out families (CheckThat!, ClaimBuster, which would leak the held-out crowdsourced.csv) and Wikipedia (SA). Presented as a benchmark. |
| `figshare:10.6084/m9.figshare.28797572 (DebatES)` | fact-check | Claim labels were generated by Google Gemini (banned generator). Manual validation only removed wrong annotations and does not make the labels human. Broadcaster transcript rights are not addressed. |
| `github:LIAAD/ClaimPT` | fact-check | Full data sits behind a Data Use Agreement (gated or custom terms). The text is copyrighted news with no allowed-licence grant from LUSA. Only a 20-article sample is public. |
| `zenodo:10802196` | fact-check | The file has no text, only URLs and sentence ids. Reconstructing the text means scraping fact-check sites and news, which the policy rejects. |
| `zenodo:15449758` | fact-check | Fact-check-site text that the rights holder has not released. The labels are veracity verdicts, not check-worthiness, and every item is by selection check-worthy (positives only). |
| `michiel/hints_of_truth` | fact-check | NC-SA licence. LLaVA (Llama-derived) generator. Fact-check-site and news text. Multimodal benchmark. |
| `zenodo:4890950` | fact-check | Social-media and fact-check-site text. Not political speech. |
| `github:HaifaCLG/Factuality` | fact-check | No licence, and the human-labelled file is not published. The only data is the previously rejected silver HF set (KnessetCorpus card cc-by-sa-4.0; stricter wins). Hebrew only. Israeli Copyright Act s.6 puts Knesset protocols in th… |
| `arxiv:1809.08193 (Full Fact claim detection, Konstantinovskiy et al.)` | fact-check | The dataset is not publicly released under any licence. The text is broadcaster closed captions (copyrighted). |
| `github:pgencheva/claim-rank` | fact-check | Held-out family (US presidential debates 1960-2016). No licence. Labels derived from fact-check sites. |
| `github:petar-iv/audio-checkworthiness-detection` | fact-check | CheckThat! derivative (all years rejected). No licence. |
| `github:germeval2021toxic/SharedTask` | fact-check | Facebook text. No licence. Shared-task data. |
| `DFKI-SLT/cdcp` | fact-check | Missing licence. The text is user comments whose authors hold copyright; they were not released CC BY. Only 731 comments (about 4.9k propositions), public-comment domain rather than political speech. |
| `osf:z6utw (Congressional Record evidence vs intuition)` | fact-check | The deposit has no licence. Only 592 human-rated items. The evidence/intuition axis is only loosely related to factual/non-factual. |
| `zenodo:3710507` | fact-check | No labelled sentences in the record. Frame schema, not check-worthiness. ClaimBuster-lab data linked to debates and fact-check sites. |
| `zenodo:3836810` | fact-check | Another version of ClaimBuster (ClaimBuster_Datasets.zip). It contains the held-out crowdsourced.csv, and groundtruth is already in the mixture. |
| `zenodo:13957177` | fact-check | ChatGPT-generated annotations (banned generator). No factuality or check-worthiness label. |
| `microsoft/claimify-dataset` | fact-check | CDLA-Permissive-2.0 is not an allowed licence, and the text was generated by Microsoft Copilot / Bing Chat, a forbidden generator. |
| `chaewanC/MAD2` | fact-check | Non-commercial custom licence; generator undocumented; the card says it is 'not a record of public-release approval'. |
| `toni5rovic/bcms-claim-sentences` | fact-check | Scraped news text with no provenance; opaque 0/1 labels with no definition (audit rejects opaque options); the card licence cannot cover third-party news. |
| `Lots-of-LoRAs/task375_classify_type_of_sentence_in_debate` | fact-check | Scraped debate.org user text with no upstream licence (the NI Apache licence covers only the task wrapper); tiny. |
| `LeTG/congress-psyop-dataset` | fact-check | Labels come from Anthropic Claude (and GPT-5-mini for the multilingual sibling), both forbidden. Propaganda-technique labels do not map to check-worthiness. |
| `franciellevargas/FactNews` | fact-check | Scraped news text: the card licence conflicts with the publishers' copyright on the underlying text (the stricter wins). |
| `BenjaminOcampo/wsf_arg_plus` | fact-check | CC BY-SA (ShareAlike); forum text. |
| `rashmikamath01/claimbuster2Cfrom3C` | fact-check | A ClaimBuster re-upload: likely contains the held-out crowdsourced.csv sentences (US presidential debates 1960-2016 family). The CC0 relicensing of CC BY data is unsupported. |
| `arxiv:1809.08193 (Full Fact / Konstantinovskiy et al. claim detection)` | fact-check | Not publicly released; BBC broadcast transcripts are copyrighted. |
| `infinite-dataset-hub/TextClaimsDataset` | fact-check | Wrong semantics ('claim' means an insurance claim), tiny, low quality. The siblings FactualFinder and PreciseClaimsExtraction are also off-task. |
| `ComplexDataLab/Misinfo_Datasets` | fact-check | Aggregate of rejected sources (LIAR, X-Fact, FEVEROUS, tweets, fact-check sites). The blanket licence conflicts with the upstream licences. Veracity labels, not check-worthiness. |
| `abhiram4572/VeriSpeak` | fact-check | Veracity, not check-worthiness. Test-only evaluation benchmark. |
| `gtfintechlab/SubjECTive-QA` | fact-check | Gated (auto); earnings-call transcripts have unclear rights; labels do not map. |
| `ibm-research/debate_speeches` | fact-check | CDLA-Permissive-2.0 is not allowed; labels unrelated (speech quality). |
| `tanmayvasvani/hindi-misinfo-taxonomy-800` | fact-check | Social-media platform text (X/Facebook); the licence covers only the annotations. |
| `redmadrobot-rnd/pii_train` | pii | Text origin fails: 9,940 rows are real user and log text, not released by the rights holders, and they cannot be separated from the templates. The MIT line is a YAML tag with no grant. Machine-generated annotations name no model. |
| `mabahboh/sitr-arabic-pii` | pii | Text origin is undocumented. The Apache sentence in the README does not identify who wrote the text. |
| `alrosait/pii-synthetic-ru` | pii | Generator is Claude. Anthropic outputs are excluded. The count mismatch (card 4,500 vs repo ~3,000) does not name an allowed generator for the extra rows. |
| `FouratAI/arabic-pii-dataset` | pii | YAML licence tag with no grant text, no provenance and no label definition. Same bar as the cheer-ekman rejection. |
| `C-Ilyas/arabic-pii-dataset` | pii | No licence and no origin. |
| `Bisher/synthetic-arabic-pii` | pii | No licence, and the generator is unnamed. The English slice is "source-derived" from an unnamed public source. |
| `wolframko/russian-pii-66k` | pii | No licence and no origin statement. |
| `gorkem371/pii-intent-detection-multilingual` | pii | The gold is sharing intent. Mapping it onto presence of personal data would be a guess. The card's own counterexample is a phone number labelled not-PII. |
| `KhalidAlharbi377/pii-detection-multisource-en-saudi-arabic` | pii | Re-ships rejected, already-used and held-out sources. A CC-BY compilation licence does not clean that. |
| `isotonic/pii-masking-200k` | pii | CC-BY-NC-4.0 forbids commercial use. English-centric as well, so it does not fill ar/ja/ru/zh. |
| `raayraay/privacyleak-pii` | pii | Published machine-unlearning benchmark. Faker and the ja/zh locales would otherwise fit. Do not train on retain or forget. |
| `auren-research/pii-shield` | pii | Real Enron, legal and SEC text, with model detections treated as labels, then machine-translated without human checks. |
| `subhash-holla/pii-anon` | pii | It is a published benchmark, the synthetic majority names no model, and the non-synthetic slice is not identified. |
| `mapo80/aliasit-pii-dataset-v3` | pii | Aggregate of sources already in the mixture, already rejected (ai4privacy), or held out (gretel test). Italian-first, and the partial E3-JSI rows are not a clean slice. |
| `wan9yu/pii-bench-zh` | pii | Named a benchmark, and the card limits use to research and evaluation. That restriction is stricter than Apache-2.0. |
| `lianghsun/tw-PII-bench` | pii | Evaluation benchmark for Traditional Chinese, designed against OpenAI's privacy filter. |
| `Meddies/meddies-pii` | pii | CC-BY-NC-4.0. ja, zh and ru are in the language list, and that does not cure the NonCommercial clause. |
| `ScienceSoft/piibench` | pii | Conflicting licences including ShareAlike, a published benchmark, real court text, and a re-pack of Gretel and Nemotron. |
| `guneeshv/REDACT-PII-Benchmark` | pii | Gated (auto) and licence "other". Also framed as a benchmark. |
| `alex-shvets/EmoPillars` | emotion | Utterances are grounded in English Wikipedia plot synopses (CC-BY-SA). English only, so it also misses ru/es/hi/de. Many of the 28 labels do not map onto basic8. |
| `laion/emotional-roleplay-finetuning-dataset` | emotion | Speech clips, not short first-person text. The text label is a Gemini-3.5-flash caption. |
| `Fischerboot/german-emotions-alpaca-gemini` | emotion | Gemini outputs, no label schema. |
| `codaco/german-emotional-speech` | emotion | Audio for speech-emotion classification, no transcripts, label set not specified. |
| `langswap/dialogs-ru-emotional-conversations` | emotion | OpenRAIL is not an allowed licence. Even under a permissive licence the labels are on acted speech, and five of the twelve classes do not map. |
| `Djacon/ru-izard-emotions` | emotion | Russian translation of GoEmotions Reddit comments. Already excluded. The card's MIT tag and the GitHub Apache pointer also disagree. |
| `mrm8488/go_emotions-es-mt` | emotion | Spanish machine translation of the GoEmotions eval suite. |
| `seara/ru_go_emotions` | emotion | Russian (and parallel English) GoEmotions. Already excluded. |
| `SkyWater21/ru_twitter_emotions` | emotion | Russian Twitter emotions. Platform text is excluded. |
| `BrunoGR/HRECPW` | emotion | Not loadable without a Hugging Face token. A BrunoGR/HRECPW Plutchik release was already excluded as tweets, DAIR and GPT-3.5. |
| `BrunoGR/HEAR` | emotion | HTTP 401 without a token, so the CI audit cannot load it. The card was not read. |
| `ma2za/many_emotions` | emotion | Licence is "other". de and es are in the language list, which does not cure the licence. |
| `TIX007/chinese-sentiment` | sentiment | DAIR-emotion derivative built by template expansion. It has no irony, double-negation or contrast labels. The MIT line is a tag, not a grant. |
| `wangbulehouhouhou/chinese-sentiment` | sentiment | Reupload of TIX007/chinese-sentiment. Same DAIR-template origin, same six emotion labels, no irony. |
| `lumynex11/chinese-sentiment` | sentiment | Reupload of TIX007/chinese-sentiment. Same DAIR-template origin, same six emotion labels, no irony. |
| `wickedlin/chinese-sentiment` | sentiment | Reupload of TIX007/chinese-sentiment. Same DAIR-template origin, same six emotion labels, no irony. |
| `speedxd/nbd-sentiment-dataset` | sentiment | Scraped news headlines, labels from Copilot, no irony or contrast annotation. |
| `OpenModels/Chinese-Herbal-Medicine-Sentiment` | sentiment | Scraped product reviews. No irony, double-negation or contrast labels, and the MIT tag does not cover the reviewers' text. |
| `kenhktsui/chinese_sentiment_syn` | sentiment | No licence, unnamed generator, 22 rows, integer labels with no class names. |
| `sepidmnorozy/Chinese_sentiment` | sentiment | No licence and no dataset card. |
| `lushan0621/chinese-sentiment-dataset` | sentiment | No licence, no origin, ten rows. |
| `tyqiangz/multilingual-sentiments` | sentiment | Aggregation of Amazon, Yelp, IMDb, tweets, GoEmotions and DAIR. It is itself an excluded eval source. The Chinese part is product reviews, with no irony gold. |
| `Youseff1987/multilingual-sentiment-dataset` | sentiment | Amazon-review aggregate. The YAML says MIT and the card body says the licence is unknown. |
| `leduckhai/Sentiment-Reasoning` | sentiment | Text origin and the rationale generator are unnamed. It is clinical speech and rationales, not Simplified-Chinese irony or contrast. |
| `left0ver/sentiment-classification` | sentiment | Upstream rights are unknown. No irony, double-negation or contrast labels. Integer labels have no names on the card. |
| `microsoft/llmail-inject-challenge` | safety | Public prompt-injection benchmark. Labels are attack success, not English hard-negative content safety. Part of the annotation is an unnamed LLM judge, and the victim stack includes GPT-4o mini. |
| `stindardlogic/refusal-overrefusal-50k` | safety | Generator is unnamed local models. The gold is four assistant responses for DPO, not a safe/unsafe label on English text that looks unsafe. |
| `Ybakman/synthetic-data-safety-benign` | safety | No licence, no origin, no safe/unsafe label. |
| `Sakonii/task-over-refusal-dataset` | safety | No licence and no origin. |
| `fevziegeyurtsevenler/turkish-over-refusal-set` | safety | XSTest-style and OR-Bench-style evaluation set. 120 pairs is an eval, not training data, and the language priority here is English. |
| `hirundo-io/bloom-over-refusal-free-text` | safety | No licence, no origin, and the label values are undefined. 400 rows. |
| `longphann/wildchat_over_refusal` | safety | No licence, and the text is WildChat user conversations with OpenAI moderation labels. |
| `AmberYifan/safetyQA_DPO` | safety | MIT is a YAML tag with no grant and no origin. The columns are DPO responses, not a content-safety label. |
| `AmazonScience/FalseReject` | safety | CC-BY-NC-4.0. This is the right shape (benign prompts that look unsafe) and the wrong licence. |
| `jang1563/bio-overrefusal-v0.1` | safety | CC-BY-NC-SA-4.0: no commercial use, and ShareAlike. |
| `jkminder/xstest-overrefusal` | safety | It is the XSTest safe-prompt test split, with a manual correction of 36 items. A CC-BY tag does not make a benchmark trainable. |
| `jkminder/or-bench-1k-overrefusal` | safety | OR-Bench derivative. Excluded benchmark. |
| `MarkrAI/k-overrefusal` | safety | Licence is "other". |
| `CounterSteer/overrefusal-data` | safety | Licence is "other". |
| `nikchar/claim_detection_training_set` | fact-check | No licence, no origin, and the label is an unnamed integer. Not one of the 38 previously examined fact-check sets; it is new and it fails. |

# Machine-readable KEEP list

See `part-A.json` (same scratchpad directory).


## Part B: categories 2 (emotion) and 10 (stance, sarcasm, formality, urgency, fact-check, other)


Checked 2026-09-27 on pop-os. Metadata came from `HfApi().dataset_info` card data and tags, the README body, and datasets-server `/size`, `/first-rows` and `/statistics`. Upstream licences came from the GitHub API (`license.spdx_id` and the README), Zenodo and Mendeley record metadata, and paper text on arXiv. No data files were downloaded.

The policy follows the repo audit: text from tweets/X, Reddit, YouTube, Weibo, review sites, scraped news or fact-check sites, Wikipedia, FLORES or subtitles is excluded even when the card tag is permissive. LLM-generated and crowd-written sets are kept on the creator's permissive licence; a model's terms on its outputs are not treated as a data licence term, as in the audit.

**Eval-suite overlap.** None of the candidates below is, or contains, one of our evaluation suites (typed-decisions, Banking77, MASSIVE, AG News, DAIR Emotion, GoEmotions, tyqiangz, multi_hatecheck, SIB-200/FLORES, Belebele, IndoNLI, FarsTail, SemRel, HWU64), except where a row says so. Every excluded emotion set that contains GoEmotions or DAIR/CARER texts is flagged.

**Emotion caveat.** `build_mixture.py` (the EXCLUDE regex) and `build_extra.py` (FORBIDDEN "any emotion dataset") deliberately keep all emotion data out, so that DAIR Emotion and GoEmotions stay zero-shot. Training on any KEEP below turns those two numbers into supervised transfer. Both scripts, and the held-out docstring of `bench/eval_zeroshot.py`, would need a deliberate policy change.

Verdict key:
- **KEEP**: licence and provenance are clean.
- **KEEP (uncertain)**: the licence is clean but one provenance point is undocumented; the note says which. These are in the JSON with an `uncertain:` note.
- **EXCLUDE**: not usable, for the reason given.

---

### Category 2: emotion (multi-class, multi-label, score)

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| `brighter-dataset/BRIGHTER-emotion-categories`, configs **`hin`, `mar` only** (Muhammad et al. 2025, arXiv 2502.11926) | card YAML `cc-by-4.0`; README "This dataset is licensed under CC-BY 4.0." Paper App. B: hin/mar created "from scratch … annotators generate emotive sentences on a given topic … augment both datasets with a few hundred quality-approved instances generated by ChatGPT" | hi, mr | hin 2,556 (+dev 200, test 2,020); mar 2,415 (+200, 2,000) | multi-label anger, disgust, fear, joy, sadness, surprise (`emotions` list + 0/1 columns) | none (SemEval-2025 T11 = same data, not our suite) | **KEEP**: human-written plus ChatGPT, CC-BY-4.0. Emotion caveat applies. |
| same dataset, all other configs (eng, deu, esp, ptbr, rus, chn, arq, ary, afr, …) | CC-BY-4.0 on card, but paper Table 1/App. B gives sources: eng/deu/ptbr = Reddit; rus/ukr = Twitter; esp/ind/jav/sun = YouTube comments; chn = Weibo; ary/hau/ibo/… = AfriSenti tweets + BBC news; arq = translation of Mohammed Dib's novel *La Grande Maison*; afr = speeches (Barnard 2014, unverified) | 26 | – | same | none | **EXCLUDE**: platform or copyrighted source text. |
| `brighter-dataset/BRIGHTER-emotion-intensities`, `mteb/EmotionAnalysis` (SemEval-2025 T11 mirror) | cc-by-4.0 tags | many | – | intensities / multi-label | none | **EXCLUDE**: same per-language sources as above; the intensity track has no hin/mar. |
| `tanaos/synthetic-emotion-detection-dataset-v1` | card `mit`; README "created synthetically by Tanaos with the Artifex Python library" | en | 11,384 | joy, anger, fear, sadness, surprise, disgust, excitement, neutral (`labels` int 0–7) | none | **KEEP**: synthetic, MIT. Emotion caveat applies. |
| `JusteLeo/French-emotion` | card `mit`; README "entirely synthetically generated … Google Gemini 2.5 Pro"; "distributed under the MIT License" | fr | 9,800 (+val 2,100, test 2,100) | colère, dégoût, joie, neutre, peur, surprise, tristesse | none | **KEEP**: LLM-generated. Emotion caveat applies. |
| `shreyaspullehf/emotion-dataset-20-emotions` | card `mit`; README "synthetically generated using … DeepInfra API with various LLMs"; "released under the MIT License" | en | 79,595 | 20 emotions (`emotion`) | none | **KEEP**: synthetic, short and repetitive sentences, so cap it. Emotion caveat applies. |
| `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset` (+ `zzhdbw/Simplified_…`, apache-2.0 conversion) | card `mit`; README "AI-Generated Dialogues … manually annotated for emotion categories by a team of experts" | zh (Traditional; zzhdbw = Simplified) | 4,159 | 8 tones: 平淡, 開心, 悲傷, 憤怒, 驚奇, 關切, 疑問, 厭惡 (`emotion`) | none | **KEEP**: AI-generated text with human labels. Emotion caveat applies. |
| `jmccardle/pulse-sofroniew-emotion-concept-texts` | card `cc-by-4.0`; README: stories by "Qwen3-32B (GGUF, llama.cpp)", graded by Claude Sonnet | en | ≈8,550 stories (JSON files) | 171 emotion concepts (`emotion`); implicit, with the emotion word banned | none | **KEEP**: LLM-written stories. Long texts (150–300 words). Emotion caveat applies. |
| `llm-for-emotion/Cultural-Emo` (CuLEmo, arXiv 2503.10688) | card `mit`; abstract: "400 crafted questions per language" | ar, de, en, es, hi (+am) | **test only**, 400 per language | emotion_eng (joy, anger, fear, sadness, …), sentiment_eng | none | **KEEP (small)**: author-crafted. Better used as a new clean multilingual emotion eval than for training. |
| GitHub `wwbp/empathic_reactions` (Buechel et al. 2018; not on HF) | GitHub README "Our dataset is available under CC BY 4.0" (spdx null) | en | 1,860 | empathy and distress ratings (score) | none | **KEEP**: crowd-written reactions; the news articles themselves are not included. |
| `google-research-datasets/go_emotions` + translations (`seara/ru_go_emotions`, `Djacon/ru-izard-emotions`, `mrm8488/go_emotions-es-mt`, `antoniomenezes/go_emotions_ptbr`, `AiLab-IMCS-UL/go_emotions-*`, `SkyWater21/*`, `AnasAlokla/multilingual_go_emotions`, …) | apache-2.0 / mit tags | en + MT | – | 28 | **GoEmotions = our zero-shot suite** | **EXCLUDE**: Reddit comments, and it is an eval suite. Contains GoEmotions texts. |
| `dair-ai/emotion` (CARER) + translations `TPM-28/emotion-FR` (apache), `Adilmar/caramelo-emotions-v2` (cc-by-4.0, pt), `log101/emotion-turkish` | no or other licence; the translations carry tags | en, fr, pt, tr | – | 6 | **DAIR = our eval suite** | **EXCLUDE**: tweets; eval suite. Contains DAIR texts. |
| `BrunoGR/HRECPW-…Plutchik_Wheel` | card `mit`; README: sources "TweetEval", "Emotion (Saravia et al., 2018)" + GPT-3.5 | es | 121,708 | Plutchik | contains DAIR texts | **EXCLUDE**: tweets and DAIR (translated). |
| `Urnisha/Enhanced_Emotion_Classification_Dataset`, `boltuix/emotions-dataset` | mit | en | 240k / 131k | Ekman / 13 | contain GoEmotions (+ DailyDialog, tweets) | **EXCLUDE**: aggregates of Reddit, tweets and DailyDialog (NC-SA). |
| `Helsinki-NLP/xed_en_fi`, `SEACrowd/xed` | card `cc-by-4.0`; source_datasets `extended|other-OpenSubtitles2016` | en, fi (+ projections) | 17.5k | Plutchik 8, multi-label | none | **EXCLUDE**: OpenSubtitles movie subtitles, a copyrighted source with no rights-holder grant. |
| `savalera/isear-from-original` (ISEAR) | card apache-2.0, but README: "original ISEAR data was collected … for research purposes"; "should not be used for commercial emotion categorization systems" | en | 5,874 | 7 | none | **EXCLUDE**: research-only origin plus a use restriction. |
| EmoBank (GitHub `JULIELab/EmoBank`) | README "licensed under CC-BY-SA 4.0" | en | 10k | VAD scores | none | **EXCLUDE**: ShareAlike. |
| MELD (`declare-lab/MELD`) | GitHub spdx `GPL-3.0` | en | 10k | 7 | none | **EXCLUDE**: GPL, and Friends TV transcripts. |
| WRIME (`ids-cv/wrime`) | README "CC BY-NC-ND 4.0" | ja | 40k | Plutchik 8 intensities | none | **EXCLUDE**: NC-ND. |
| `facebook/empathetic_dialogues`, `li2017dailydialog/daily_dialog`, `ConvLab/dailydialog` | `cc-by-nc-4.0` / `cc-by-nc-sa-4.0` | en | – | 32 / 7 | none | **EXCLUDE**: NC. |
| `ConvLab/emowoz` vs `hhu-dsml/emowoz` | apache-2.0 vs `cc-by-nc-4.0` on the original card | en | – | 7 | none | **EXCLUDE**: the mirror's licence conflicts with the original. |
| crowd-enVent / enISEAR / deISEAR (romanklinger.de) | the data page states no licence; GitHub `sarnthil/crowd-enVent-modeling` is MIT (code only) | en, de | 4,320 | 13 emotions + appraisals | none | **EXCLUDE**: data licence missing. Worth asking the authors; this would be excellent crowd-written data. |
| `knoveleng/emotion-datasets` | cc-by-4.0, plus README terms: "Intended use: research on language-model interpretability … Do not use it to infer or attribute emotions to real people" | en | ≈4k | emotion | none | **EXCLUDE**: extra use terms (custom) that contradict an emotion classifier. |
| `sdeakin/LLM-BIO-Emotions` | cc-by-4.0; README "Reddit-based text" | en | – | – | none | **EXCLUDE**: Reddit. |
| `asas-ai/Arabic-Poem-Emotion` | cc-by-4.0; empty README, no provenance | ar | 9,449 | emotion | none | **EXCLUDE**: poem source undocumented (likely scraped poetry sites). |
| `ukr-detect/ukr-emotions-*`, `hajili/azerbaijani_tweet_emotion_classification`, `AiLab-IMCS-UL/twitter_emotions-*`, `Houdna-khilouf/Dz-Emotion`, `Lots-of-LoRAs/task512/517/518/875` | various permissive tags | uk, az, lv/lt/ru, ar-DZ, en | – | – | none | **EXCLUDE**: tweets or social media; Natural-Instructions tasks inherit DailyDialog (NC-SA) and tweet sources. |

**Emotion: 9 KEEP** (BRIGHTER hin+mar counted once), about 25 excluded. Multilingual coverage: hi, mr, fr, zh, en, plus the test-only Cultural-Emo for ar/de/es/hi. **Gap:** licence-clean human emotion data for de, es, it, pt, nl, pl, tr, ru, ar and ja does not exist on the Hub. All real multilingual emotion corpora come from Reddit, X, YouTube, Weibo or subtitles.

---

### Category 10a: stance

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| `SINAI/ALIA-es-discriminative-stance-detection` | card `cc-by-4.0`; README "Original data source: Decide Madrid — Portal de Datos Abiertos del Ayuntamiento de Madrid … License of original data: CC BY 4.0"; Prolific annotators | es | 3,000 (150 without majority) | favor / against / neutral (`majority_label`) over (target + description, comment) | none | **KEEP**: the city published the comments as open data under CC BY 4.0. The portal page itself was not fetched (404). |
| `pacoreyes/StanceSentences` | card `apache-2.0`; README empty; row metadata cites presidency.ucsb.edu (US presidential documents, public domain) | en | 972 (+val 108, test 200) | support / oppose (balanced) | none | **KEEP (uncertain)**: the text is public domain, but the labelling method is undocumented. |
| `ibm-research/claim_stance` | card YAML `cc-by-3.0`, but README "(c) Copyright Wikipedia … Released under CC-BY-SA 3.0" | en | 2,394 | PRO / CON | none | **EXCLUDE**: the card metadata conflicts with the body (SA), and the text is from Wikipedia. |
| `ibm-research/argument_quality_ranking_30k` (IBM ArgQ-30k; also stance) | card YAML `cc-by-3.0`; README "Released under CC-BY-SA 3.0" | en | 20,974 | quality score + stance ±1 | none | **EXCLUDE**: SA; the card metadata conflicts with the body. |
| `strombergnlp/x-stance` | mit tag; upstream data CC BY-NC 4.0 (already in the repo audit) | de, fr, it | – | favor / against | none | **EXCLUDE**: NC (audit). |
| `strombergnlp/ans-stance` (Khouja 2020) | card "Apache License, Version 2.0" | ar | ≈3.8k | agree / disagree / other | none | **EXCLUDE**: premises are Arabic news headlines (publisher text), consistent with the audit's headline_cause/hlgd rule. |
| `strombergnlp/nlpcc-stance`, `strombergnlp/ara-stance`, `strombergnlp/zulu_stance`, `projecte-aina/CaSET-*`, `mlburnham/PoliStance_Affect`, `tweet_eval/stance_*`, SemEval-2016 T6, P-Stance | cc-by-4.0 / mit tags | zh, ar, zu, ca, en | – | – | none | **EXCLUDE**: Weibo, tweets or news articles. |
| `DebateLabKIT/arguments-and-debates` | card `odc-by` "collection" | en | – | – | none | **EXCLUDE**: the text is ProCon.org / debate-portal content (publisher copyright). |
| Perspectrum (`CogComp/perspectrum`), UKP sentential AM, args.me, Kialo, VAST | "Creative Commons license" (unspecified) / not redistributable / debate portals / NYT comments | en | – | – | none | **EXCLUDE**: platform text or unspecified licences. |
| `giuseppe-aiello/stance-detection-it-dataset` | mit, no README | it | ? | ? | none | **EXCLUDE**: no provenance. |

**Stance: 2 KEEP** (1 uncertain). **Gap:** multilingual stance beyond es is a gap for synthetic data.

### Category 10b: sarcasm / irony

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| `sweatSmile/sarcastic-dataset` | card `mit`; README "720 sentences with sarcastic … versions generated using OpenAI GPT models" (base sentences are the public-domain Harvard sentences) | en | 720 pairs, so 1,440 binary items | sarcastic vs neutral (use `sentence` / `translation`) | none | **KEEP (small)**. |
| `elvanalabs/sarcasm-statements-90` | card `mit`; README "manually labeled" | en | 90 | True / False | none | **KEEP (tiny)**. |
| `Arrebol-yzq/Metaphor_Driven_Sarcasm_dataset_source` | card `mit`; README "converted from the BSD-MM-V3 dataset"; sources "AI-generated, WeChat, news, forums"; ai_generated = 124,722 (52.9%) | zh (+en gloss) | 235,909 | sarcasm 0/1 + hierarchy | none | **EXCLUDE**: upstream BSD-MM-V3 licence unverified; 47% scraped WeChat, news and forums. The `data_source == ai_generated` subset could be reconsidered if BSD-MM-V3 turns out to be permissive. |
| `iabufarha/ar_sarcasm`, `CreativeLang/SARC_Sarcasm`, `SarcasmNet/sarcasm`, `Dalilame/Multi-Sarcasm`, `winduald/figlang2020-sarcasm`, `w11wo/*_indonesia_sarcastic`, `tweet_eval/irony`, SemEval-2018 T3, iSarcasm | mit / cc-by / apache tags | ar, en, id | – | – | none | **EXCLUDE**: tweets or Reddit (SARC). |
| `processvenue/SARCASM_VS_NON_SARCASM` | apache-2.0, no README; texts are Internet Argument Corpus forum posts ("emoticonxrolleyes") | en | ≈4.7k | sarcastic / not | none | **EXCLUDE**: forum text, no provenance or licence at source. |
| `clips/mteb-nl-sarcastic-headlines`, `Sarcasm_News_Headline` (tasksource) | cc0 claimed by the collector | nl, en | – | – | none | **EXCLUDE**: scraped headlines (De Speld / nu.nl, The Onion / HuffPost); the collector is not the rights holder (audit rule). |
| `ramondomiingos/ptbr-irony-idioms-regionalism` | cc-by-4.0 | pt | <1k | open-ended gold explanations | none | **EXCLUDE**: not a decision dataset (free-text gold). |

**Sarcasm: 2 tiny KEEP** (≈1.5k items). **Gap.** No licence-clean real sarcasm data exists in any language, so it has to be generated.

### Category 10c: formality / register / politeness

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| `GoktugD/turkish-formality-rewrite-500k` | card `cc0-1.0`; README: rows "deterministik olarak oluşturulur" by the repo's generator code | tr | 490,000 pairs (+5k val, 5k test) | informal vs formal (from `informal` / `formal`) | none | **KEEP**: procedural, so low diversity; cap it. |
| `NagaYu/deference-keigo-corpus` | card `cc-by-4.0`; README: sentences generated "mechanically from the norm"; the norm is quoted from MEXT under the Public Data License 1.0, "compatible with CC BY … including commercial use" | ja | 3,280 (+410, 411) | keigo error present yes/no (`n_errors` > 0) and error types | none | **KEEP**: rule-generated. |
| `ronantakizawa/japanese-honorifics` | card `cc-by-4.0`; README "generated using GPT-4o" | ja (+en) | 115 rows × 4 registers | plain / teineigo / sonkeigo / kenjogo | none | **KEEP (tiny)**. |
| `osyvokon/pavlick-formality-scores` (Pavlick & Tetreault 2016) | card `cc-by-3.0`; README: sentences from Yahoo! Answers, 20 news sites (CNN, NYT, …), top blogs, Jeb Bush emails | en | ≈11k | formality score −3…3 | none | **EXCLUDE**: Yahoo Answers ToS and publisher news text; CC-BY covers annotations only (tasksource `pragmeval/squinky-formality` is already excluded). |
| GYAFC, XFORMAL, `ukr-detect/ukr-formality-dataset-translated-gyafc` | Yahoo L6 licence / openrail++ | en, fr, it, pt-BR, uk | – | – | none | **EXCLUDE**: Yahoo Answers, research-only terms. |
| CoCoA-MT / IWSLT formality (`amazon-science/contrastive-controlled-mt`) | GitHub README "licensed under the CDLA-Sharing-1.0 License" | de, es, fr, hi, it, ja, pt, ru | – | formal / informal | none | **EXCLUDE**: CDLA-Sharing is copyleft-like on data. |
| `PhillipsLab/formal_informal_contrast` | cc-by-4.0, empty README | en | ≈? | formal / informal answer pairs | none | **EXCLUDE (strict)**: no provenance (appears LLM-generated). Could be kept if the author confirms. |
| `Cleanlab/stanford-politeness`, `Genius1237/TyDiP` (multilingual politeness), `BrachioLab/multilingual_politeness` | mit / cc-by-4.0 / "cc" | en, hi, ko, es, ta, fr, vi, ru | – | polite / impolite | none | **EXCLUDE**: Wikipedia talk pages and StackExchange (CC BY-SA). |
| `IWAN/adab-arabic-politeness` | cc-by-4.0 (gated) | ar | ≈5k | polite / neutral / impolite | none | **EXCLUDE**: README sources are Tweet, YouTube, Shein reviews. |
| `ralipanah/email-politeness-corpus` | `cc-by-nc-sa-4.0` | en | – | – | none | **EXCLUDE**: NC-SA. |

**Formality: 3 KEEP** (tr, ja, ja). **Gap:** formality for en, de, fr, es, it, pt, nl, pl and ru, and politeness in every language.

### Category 10d: urgency / priority

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| `weijianzhg/email-triage-action-seed` | card `apache-2.0`; README "All 3,905 rows were generated synthetically using Claude Opus 4.6 … No human emails" | en | 3,905 | priority urgent / high / medium; category (6, incl. phishing); action (reply / flag / forward / ignore) | none | **KEEP**: LLM-generated. The priority classes lack "low". |
| `AshenFdo/synthetic_blood_request_urgency_dataset` | card `mit`; README "All samples were AI-generated" | en | 2,500 | emergency / not_emergency (balanced) | none | **KEEP (small, narrow domain)**. |
| `IDinsight/urgency_detection_maternal_health_synthetic` | card `mit`; README: messages generated by "Gemini-1.5-Flash", verified by Gemini-1.5-Pro; personas derived from analysing MomConnect messages (no copied text) | en | 11,529 (+1,263 val) | `matching_rule`: 35 urgent maternal warning-sign rules | none | **KEEP**: urgency-rule matching. Negatives (non-urgent) must be added. |
| `tasksource/help-desk-tickets` (Mendeley Data, Abdellatif, DOI 10.17632/btm76zndnt.3) | HF card `cc-by-4.0`; Mendeley record: "License: CC BY 4.0"; real PII-masked tickets from "an international software company" | en (+ other, unfiltered) | 357 text-bearing issues (default config) | `issue_priority`: Blocker / Highest / High / Medium / Low | none | **KEEP (small, real)**. The `issues` config (66,691) has no text. |
| `Console-AI/IT-helpdesk-synthetic-tickets` | card `mit`; empty README; the name says synthetic | en | 500 | priority Urgent / High / Medium / Low (skewed); 11 categories | none | **KEEP (uncertain)**: generator undocumented. |
| `KameronB/synthetic-it-callcenter-tickets` | card `apache-2.0`, empty README | en | 27,602 | type Request / Incident, category, subcategory; **no priority** | none | Not urgency. Relevant to category 3 ticket routing (fork A). |
| `Tobi-Bueck/customer-support-tickets` | `cc-by-nc-4.0` (already rejected in build_extra) | en, de | – | priority | none | **EXCLUDE**: NC. |
| CrisisNLP / HumAID / TREC-IS | – | multi | – | urgency / humanitarian | none | **EXCLUDE**: tweets. |
| `wongqihan/triagebench`, `TimotheeB/triage-medical-dataset` | mit / cc-by-4.0 | multi / fr, en | 14 test / mixed | – | none | **EXCLUDE**: triagebench is 14 test probes; TimotheeB aggregates medical QA sources of mixed licence (HealthCareMagic etc.). |

**Urgency / priority: 5 KEEP** (1 uncertain). **Gap:** multilingual urgency, and low-priority / non-urgent negatives.

### Category 10e: fact-check verdicts, claim verification, check-worthiness

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| `truthfulqa/truthful_qa` (Lin et al. 2022; GitHub `sylinrl/TruthfulQA` spdx `Apache-2.0`) | card `apache-2.0` | en | 817 questions, validation only; ≈3 correct + 4 incorrect answers each, so ≈5.9k true/false items | correct_answers vs incorrect_answers (true/false); `mc1_targets` / `mc2_targets` | none (not our suite; widely used as an LLM benchmark, so consider holding out) | **KEEP**: author-written questions and answers. |
| `Eurolingua/truthfulqax` (openGPT-X translation of TruthfulQA) | card `apache-2.0`; files `truthfulqa_{gen,mc}_{BG,CS,DA,DE,EL,ES,ET,FI,FR,HU,IT,LT,LV,NL,PL,PT,RO,SK,SL,SV}_validation.jsonl` | de, fr, es, it, pt, nl, pl (+13 EU) | ≈817 per language | same | none | **KEEP**: machine translation of an Apache source. The HF viewer is blocked by the loading script, so read the JSONL directly. |
| `seyled/Phantom_Hallucination_Detection` (Ji et al., NeurIPS 2025 D&B) | card `apache-2.0` (license_link apache.org) | en | 30,886 (many context-length variants of the same seeds) | `ground_truth_label` (hallucinated vs faithful answer) given SEC 10-K / DEF 14A context | none | **KEEP**: SEC filings are public (same basis as the audit's LEDGAR); dedup on seed question and answer. |
| ClaimBuster (Zenodo record 3609356; not on HF with a clean licence) | Zenodo API `license.id = cc-by-4.0`; "statements extracted from all U.S. general election presidential debates (1960-2016)" | en | 23,533 | non-factual / unimportant factual / check-worthy | none | **KEEP (uncertain)**: debate transcripts are public political speech with no explicit transcript licence. Load from Zenodo; the HF mirror `Nithiwat/claimbuster` says cc-by-sa-4.0, which is a conflict. |
| `lytang/C2D-and-D2C-MiniCheck` | card `mit`; paper: C2D uses "a set of human-written claim statements", D2C "human-written documents" (origin not named); GPT-3.5/GPT-4 | en | ≈14k | supported 0/1 | none | **EXCLUDE**: seed claim and document provenance undocumented (D2C docs likely news). |
| FEVER, `tals/vitaminc`, `fever/feverous`, `tasksource/wice`, climate-fever, HoVer, fool-me-twice | cc-by-sa-3.0 / gpl-3.0 / cc-by-sa-4.0 | en | – | – | none | **EXCLUDE**: Wikipedia CC BY-SA. |
| `allenai/scifact`, `dwadden/covidfact_entailment` | `cc-by-nc-2.0` | en | – | – | none | **EXCLUDE**: NC. |
| `utahnlp/x-fact` (X-Fact, 25 languages), `chengxuphd/liar2`, `ucsbai/liar`, `pszemraj/multi_fc`, `NaughtyConstrictor/fact-check-bureau`, `MultiMind-SemEval2025/*MultiClaim*`, PUBHEALTH, AVeriTeC | mit / apache / other / cc-by-4.0 | multi | – | verdict labels | none | **EXCLUDE**: claims and verdicts copied from fact-check websites (PolitiFact, Snopes, AFP, …), the same rule the audit applied to health_fact; MultiFC "other". |
| `iai-group/clef2024_checkthat_task1_*`, `AIWizards/clef2025_checkthat_task1_subjectivity` | cc-by-sa-4.0 / cc-by-nc-sa-4.0 | en, es, ar, de, it | – | check-worthy / subjective | none | **EXCLUDE**: SA or NC; tweets and news. |
| `llm-semantic-router/fact-check-classification-dataset`, `FactCheck-AI/FactCheck`, `notrichardren/truthfulness_legacy` | apache / mit | en | – | – | none | **EXCLUDE**: aggregates of TriviaQA, Dolly (CC BY-SA) and WritingPrompts (Reddit); web-scraped evidence; mixed origins (use TruthfulQA directly). |
| fake-news sets (`genesisqu/fake-real-news`, FakeNewsNet, GossipCop, `mariagrandury/fake_news_corpus_spanish`, Turkish / Arabic / PT sets) | bsd / apache / cc-by / mit tags | en, es, tr, ar, pt | – | fake / real | none | **EXCLUDE**: scraped news articles. |

**Fact-check: 4 KEEP** (1 uncertain; TruthfulQA and TruthfulQA-X share the same questions). **Gap:** real multilingual claim-verdict data with evidence. Everything real comes from fact-check sites or Wikipedia.

### Category 10f: other decision tasks

| HF id (original source) | Licence + where verified | Langs | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| **PII** `nvidia/Nemotron-PII` | card `cc-by-4.0`; README "This dataset is ready for commercial use"; NeMo Data Designer synthetic | en | 100,000 (+100,000 test) | 55+ PII/PHI span types, giving "contains PII of type X?" | none | **KEEP**. |
| **PII** `gretelai/synthetic_pii_finance_multilingual` | card `apache-2.0`; README "generated using Gretel Navigator and released under Apache 2.0" | en, fr, de, nl, es, it, sv | 50,346 (+5,594) | 29 PII types; document_type | none | **KEEP**. |
| **PII** `gretelai/gretel-pii-masking-en-v1` | card `apache-2.0`; README "entirely synthetically generated using Gretel Navigator" | en | 50,000 (+5k, 5k) | PII/PHI entity types; domain | none | **KEEP**. |
| **PII** `Wismut/nym-pii-multilingual-data` | card `mit`; README: template-built, Faker values, "exactly-labeled" | 22 languages incl. all 14 priority languages | 724,500 (+40,250, 40,250) | PII entity labels | none | **KEEP**: template-generated, so cap it. |
| **PII** `E3-JSI/synthetic-multi-pii-ner-v1` | card `mit`; README: LLM-generated texts with entities | en, fr, de, el, nl, it, sl | 2,971 | PII types | none | **KEEP**. |
| **PII** `urchade/synthetic-pii-ner-mistral-v1` | card `apache-2.0`; README "synthetic dataset used for training gliner_multi_pii-v1" (Mistral) | en, fr, it, de, es | data.json (size not listed) | PII types | none | **KEEP**. |
| PII `ai4privacy/pii-masking-health-phi-preview` | cc-by-4.0, but the source text is withheld ("Contact us for full access") | 23 | <1k | – | none | **EXCLUDE**: preview without text. |
| **Language ID** `Helsinki-NLP/tatoeba` (tatoeba.org) | card `cc-by-2.0` (Tatoeba sentences CC-BY 2.0 FR) | 300+ | millions of sentences | language code | none (not FLORES) | **KEEP**: crowd-written sentences. |
| **Language ID** `CohereLabs/aya_dataset` | card `apache-2.0`; README "brand new prompts and completions written by annotators" | 65 | 202,362 | `language` | none | **KEEP**: use `annotation_type == original-annotations` only (re-annotations edit templated upstream data). |
| Language ID `cis-lmu/udhr-lid` | cc0-1.0; README "UDHR should remain a test corpus … not a training corpus" | 400+ | – | language | none | Use for **eval only** (creators' request). |
| Language ID `papluca/language-identification`, `laurievb/open-lid-dataset`, `multilingual/wili_2018` | none / other / ODbL | – | – | – | none | **EXCLUDE**: Amazon reviews + XNLI; mixed "other"; ODbL. |
| **Argument mining** `joelniklaus/german_argument_mining` (Urchs et al., German court judgments) | card `cc-by-4.0`; German court decisions are official works (§5 UrhG) | de | 19,271 (+2,726, 3,078) | conclusion / definition / subsumption / other (`label`) | none | **KEEP**. |
| **Acceptability** `nyu-mll/blimp` (GitHub `alexwarstadt/blimp`) | card `cc-by-4.0`; GitHub README "BLiMP is distributed under a CC-BY license" | en | 67 configs × 1,000 pairs (≈134k sentences) | good vs bad sentence | none | **KEEP**: template-generated minimal pairs. |
| Acceptability `tasksource/mega-acceptability-v2` | tasksource apache-2.0; the upstream megaattitude.io page states no licence | en | – | rating | none | **EXCLUDE**: upstream licence not stated. |
| **Readability / CEFR** `pinialt/cefr-texts-10languages` | card `cc-by-4.0`; README "generated using OpenAI's GPT-4o-mini" | en, fr, es, de, it, pt, nl, ru, zh, ar | 2,400 (+600 test) | A1–C2 | none | **KEEP**. |
| Readability `agentlans/readability` | cc0-1.0 claimed; paragraphs from Wikipedia and arXiv etc. | en | 200k | formula grade | none | **EXCLUDE**: Wikipedia (SA), mixed arXiv licences, formula labels. |
| **Humor** `yoonholee/humor-greats-public-domain` | card `cc0-1.0`; README "extracted from public-domain humor collections on Project Gutenberg … pre-1929" | en | 19,354 | positives only | none | **KEEP (positive-only)**: needs negatives. |
| Humor: ColBERT humor, `fchaubard/funny_bench`, `SocialGrep/one-million-reddit-jokes`, `iammytoo/japanese-humor-evaluation*` (Bokete), Humicroedit, `Young25/Humor_Vote` (no README) | cc-by-2.0 / apache / cc-by / mit tags | en, ja, zh | – | – | none | **EXCLUDE**: Reddit, scraped news or Bokete; no provenance. |
| **Subjectivity** `tasksource/subjectivity`, CheckThat subjectivity, SUBJ (Pang & Lee) | mit / cc-by-nc-sa-4.0 | en + 5 | – | subj / obj | none | **EXCLUDE**: copyrighted news (audit), NC-SA, RT/IMDb snippets. |
| Bias `nishan-chatterjee/llm-bias-detection` | cc-by-4.0, but README: "does not relicense source text … IBM retains CC BY-SA 3.0" | 14 | – | – | none | **EXCLUDE**: components are SA. |
| Clickbait (`christinacdl/*`, `id_clickbait`, `detector-clickbait-br`) | apache / cc-by / mit tags | en, id, pt, multi | – | – | none | **EXCLUDE**: scraped headlines. |

**Other: 13 KEEP** (6 PII, 2 LID, 1 argument mining, 1 acceptability, 1 CEFR, 1 humor positive-only, and ClaimBuster counted under 10e). **Gaps:** subjectivity, politeness, humor (negatives, and anything non-English), clickbait.

---

### Tasksource sources already audited (not re-audited)

- Excluded in `licence_audit.json` / DATA_LICENSES.md: `multilingual/x-stance` (NC), `subjectivity`, `health_fact`, `fool-me-twice`, `hover`, `hover-3way/nli`.
- Excluded because tasksource marks them non-commercial or unspecified: `pragmeval/*` (incl. sarcasm, squinky-formality), `Sarcasm_News_Headline`, `liar`, `tweet_eval/*` (irony, stance, emotion), `Politeness_Disagreement`, `humicroedit/*`, `crowdflower/*`.
- In the commercial but not-permissive list: `rumoureval_2019` ("Custom (DPI)"; tweets/Reddit), `scruples` (Custom), `open_question_type` (Custom). `go_emotions/simplified` and `emo/emo2019` (MPL) are both excluded by the emotion rule; go_emotions is also an eval suite.

### Gaps (for synthetic data)

1. Emotion outside en/fr/zh/hi/mr: de, es, it, pt, nl, pl, tr, ru, ar, ja (Cultural-Emo covers ar/de/es/hi with 400 test items each only).
2. Sarcasm/irony in every language (only ≈1.5k English synthetic items are clean).
3. Stance outside es/en (and en has only ≈1.3k items).
4. Formality and politeness for en, de, fr, es, it, pt, nl, pl, ru, ar, hi, zh (only tr and ja are covered).
5. Urgency/priority outside English; non-urgent negatives.
6. Real claim-verdict data with evidence, multilingual (TruthfulQA-X covers only truthfulness of short answers).
7. Subjectivity, humor classification, clickbait: none.


## Part C: categories 4 (NLI) and 7 (similarity / paraphrase / relevance)


Checked on 2026-09-27 from pop-os. Only metadata, READMEs and first rows or single parquet columns were read. Nothing was trained and the repo was not modified.

How each licence was checked:
- HF: `HfApi().dataset_info()` card licence and `license:` tags, the README body, datasets-server `/size` and `/first-rows` (helper `scratchpad/hfcheck.py`).
- Upstream: `gh api repos/<o>/<r>` (`license.spdx_id`), the upstream README or LICENSE file, and project pages fetched with curl.

Policy applied, the same as `tools/finetune/licence_audit.json`:
- Allowed licences: Apache-2.0, MIT, BSD, CC0, CC-BY, ODC-By, AFL.
- Excluded licences: NC, SA, ND, GPL, "other", and missing licences.
- Excluded by origin: text from Wikipedia, SNLI or other CC-BY-SA sources; tweets, Reddit, scraped news or web crawl; Quora; Stack Exchange. This applies even when the card says MIT, CC-BY or Apache.
- Translations inherit the licence of their source.
- LLM-generated sets are kept on their creator's licence, as WANLI was. The generating model's terms of service are not treated as a data licence.

Eval-suite overlap check: none of the KEEP items contain IndoNLI (id), FarsTail (fa), SemRel2024, FLORES text (SIB-200 or Belebele), MASSIVE, Banking77, GoEmotions or tyqiangz. None of the KEEP NLI sets contain Indonesian or Persian.

### Category 4: NLI / entailment

#### KEEP

| HF id (original source) | Licence string + where verified | Languages | Train rows | Label set | Eval-suite overlap | Verdict |
|---|---|---|---:|---|---|---|
| `nyu-mll/multi_nli` (MultiNLI, Williams et al. 2018) | Card tags `cc-by-3.0, cc-by-sa-3.0, mit, other`. Card "Licensing Information" quotes the paper: "The majority of the corpus is released under the OANC's license … Seven Swords is available under a Creative Commons Share-Alike 3.0 Unported License … the remaining works of fiction are in the public domain in the United States". The OANC page anc.org/data/oanc says: "freely use all of our data and resources without restriction (including commercial use)". GitHub nyu-mll/multiNLI: MIT (code). | en | 392,702 in total. **315,354 after dropping fiction** (telephone 83,348, government 77,350, travel 77,350, slate 77,306; fiction 77,348 counted from the parquet `genre` column) | entailment / neutral / contradiction | None. IndoNLI's `translate_train` is MNLI, but we load only the IndoNLI expert test. | **KEEP with filter `genre != "fiction"`.** The filter removes the CC-BY-SA Seven Swords text, which is the reason the audit excluded `syntactic-augmentation-nli`. |
| `ankitkupadhyay/XNLI` (XNLI 2.0 = MNLI train machine-translated, Upadhyay & Upadhya 2023) | Card licence `apache-2.0` (API tag). The README is 28 bytes and says nothing about provenance. The content inherits the MNLI terms above. | **ar bg de el en es fr hi ru sw th tr ur vi zh** (15 files `<lang>_train.csv`) | 392,702 per language (5,890,530 in total). Checked: German row 0 "Konzeptionell hat die Rahmabschöpfung …" is MNLI row 0 (government) and the labels match. | entailment / neutral / contradiction (0 / 1 / 2) | None. No id or fa. Contains no XNLI dev/test rows (those are CC-BY-NC). | **KEEP, conditional:** join row index to MNLI train, drop fiction, assert that the labels agree. This is the only clean human-label NLI source for de, fr, es, ru, tr, ar, hi and zh. |
| `boun-tabi/nli_tr`, config `multinli_tr` only (NLI-TR, Budur et al. 2020) | Card tags `cc-by-3.0, cc-by-4.0, cc-by-sa-3.0, mit, other`. GitHub boun-tabi/NLI-TR README: "MultiNLI-TR is licensed under the same terms as MultiNLI"; "SNLI-TR licensed under … CC BY-SA 4.0". | tr (Amazon Translate) | 392,702. `idx` is aligned with MNLI train (idx 0, 2 and 3 checked). | entailment / neutral / contradiction | None | **KEEP, conditional:** join on `idx` to drop fiction. `snli_tr` is EXCLUDED (SA). Second Turkish MT next to XNLI 2.0; dedupe by idx. |
| `takehika/wanli-ja-nli` (from WANLI) | Card `cc-by-4.0`. README: "This dataset is licensed under CC BY 4.0." The source is WANLI, which is CC-BY-4.0 and already kept in the mixture. | ja (+ en) | 73,942 (test 3,505) | entailment / neutral / contradiction (`gold`) | None | **KEEP.** LLM translation plus LLM filtering; the model is not named. Neutral is under-represented, per the card. |
| `maximoss/mnli-nineeleven-fr` | Card `bsd-2-clause`. The README says this is the MNLI 9/11 slice (OANC 9/11 report), machine-translated with opus-mt-tc-big-en-fr, and does not overlap the XNLI 9/11 items. | fr (+ en originals) | 2,000 | entailment / neutral / contradiction | None | **KEEP** (small). |
| `hkust-nlp/SynCSE-scratch-NLI` | Card `mit`. GitHub SJTU-LIT/SynCSE: MIT. README: "generated by GPT-3.5-Turbo". | en | 275,579 triplets | `sent1` = entailment and `nli_hard` = contradiction, both relative to `sent0`. There is no neutral class. | None | **KEEP** (synthetic, from scratch; same treatment as WANLI). It also serves as paraphrase data (category 7). |
| `jhu-cogsci/hans` (HANS, McCoy et al. 2019) | Card `unknown`. **Upstream GitHub tommccoy1/hans: MIT.** README: "This repository is licensed under an MIT License". | en | 30,000 (validation 30,000; parquet under refs/convert) | entailment / non-entailment | None | **KEEP** (template-generated; the upstream licence resolves the blank card). Cap it low, since the lexical heuristics are artificial. |
| `sileod/attempto-nli` | Card `apache-2.0`. Controlled-English sentences; labels come from the RACE reasoner (`race_label`). | en | 29,086 | entailment / neutral / contradiction | None | **KEEP** (synthetic logic; small cap). |
| `MoritzLaurer/synthetic_zeroshot_mixtral_v0.1`, **only** the configs `mixtral_written_texts_for_tasks`, `_v2`, `_v3`, `_v4` | Card `apache-2.0`. Texts and hypotheses were written by Mixtral-8x7B (Apache-2.0), according to the config names, the `prompt_formatted` field and Laurer's paper. The card has no prose provenance. | en | 105,806 + 156,096 + 679,516 + 308,586 | `labels` (entailment vs not) for a text / hypothesis pair ("This text is about …") | None | **KEEP** (synthetic label-NLI, i.e. zero-shot classification). The three `mixtral_refinedweb_*` configs are **EXCLUDED** because their texts are Common Crawl (RefinedWeb) pages. |
| `GoktugD/turkish-nli-constructed-1.5m` | Card `cc0-1.0`. README: "Üretici ve ortaya çıkan sentetik veri CC0-1.0 olarak sunulur" (the generator and the resulting synthetic data are released under CC0-1.0). The data is deterministic template output and contains no human or web text. | tr | 1,470,000 | entailment / neutral / contradiction | None (contamination report included) | **KEEP, low priority** (template artefacts; cap hard). |
| `alisawuffles/WANLI` | CC-BY-4.0 | en | – | – | – | Already in the mixture (`WANLI`). Listed only for completeness. |

#### EXCLUDE

| HF id (original) | Licence seen + where | Languages | Train rows | Eval overlap | Reason |
|---|---|---|---:|---|---|
| `facebook/xnli` | Card: no licence. GitHub facebookresearch/XNLI `LICENSE`: "Attribution-NonCommercial 4.0 International". | 15 | 392,702 (MT) | – | NC |
| `MoritzLaurer/multilingual-NLI-26lang-2mil7` | Card: **no licence** field and no licence text in the README. Its sources include ANLI (NC) and FEVER (SA). | 26 incl. id, fa | 25k per split | Contains `id_*` / `fa_*` MNLI translations (not IndoNLI or FarsTail text, but the same languages) | Missing licence plus NC/SA components. Its clean parts (mnli, wanli) are recoverable from XNLI 2.0 or our own MT. |
| `facebook/anli` | Card `cc-by-nc-4.0`. GitHub README: "ANLI is licensed under Creative Commons-Non Commercial 4.0". | en | 162,865 | – | NC |
| `tasksource/lingnli`, `maximoss/lingnli-multi` | Card `unknown`. GitHub Alicia-Parrish/ling_in_loop: `license: null`. lingnli-multi is BSD-2, but it is MT of LingNLI. | en; el fr it es pt ko fi lt bg | 44,982 | – | Missing upstream licence; the derivative's licence conflicts with it. |
| `stanfordnlp/snli` and derivatives (`shibing624/snli-zh`, `KETI-NLP/kor_snli`, `boun-tabi/nli_tr:snli_tr`, `Omartificial-Intelligence-Space/Arabic-NLi-Pair-Class` (row 0 = SNLI horse premise), `sagnikrayc/snli-cf-kaushik`) | `cc-by-sa-4.0` (SNLI card) | en, zh, ko, tr, ar | 550k | – | SA; translations inherit it. |
| KorNLI (kakaobrain/kor-nlu-datasets) | GitHub `CC-BY-SA-4.0` | ko | – | – | SA |
| OCNLI (CLUEbenchmark/OCNLI) | README: "Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)" | zh | – | – | NC |
| `C-MTEB/CMNLI` | Card: no licence (MNLI+XNLI MT) | zh | – | – | Missing licence; contains XNLI (NC). |
| JGLUE JNLI (`shunk031/JGLUE`) | Card `cc-by-4.0`, **but** GitHub yahoojapan/JGLUE `CC-BY-SA-4.0` ("licensed under a Creative Commons Attribution-ShareAlike 4.0") | ja | – | – | The card conflicts with the source, which is SA. |
| JSICK (`verypluming/JSICK`, `mteb/JSICK`) | GitHub LICENSE `CC-BY-SA-4.0` while its README says CC-BY-4.0. It is a translation of SICK (`cc-by-nc-sa-3.0`, per tasksource sick/label and `mteb/sickr-sts`). | ja | – | – | SA / NC source |
| SICK / `SemEvalWorkshop/sem_eval_2014_task_1` | Card `cc-by-4.0`, but SICK is `cc-by-nc-sa-3.0` (mteb/sickr-sts card; tasksource) | en | – | – | The card conflicts with the source. |
| `RussianNLP/russian_super_glue` (TERRa, RCB) | Card `mit`; russiansuperglue.com TERRa page "MIT License". **Same page:** "All text examples were collected from open news sources and literary magazines". | ru | TERRa 2,616 | – | Text is scraped news and literary text (same rule as masakhanews / hyperpartisan in the audit). |
| `cointegrated/nli-rus-translated-v2021` | Card: no licence. It mixes copa, mnli, snli, anli and others. | ru | 1,756,548 | – | Missing licence plus SA/NC sources. |
| `allegro/klej-cdsc-e` | Card `cc-by-nc-sa-4.0` ("Attribution-NonCommercial-ShareAlike 4.0") | pl | 8,000 | – | NC-SA |
| `PORTULAN/extraglue` | Card `mit`. It is MT of GLUE/SuperGLUE: RTE (news), QNLI/BoolQ (Wikipedia), MRPC (news), CB (WSJ/BNC), SST-2 (RT reviews). There is no MNLI train config. | pt | rte 2,490, qnli 104,743, … | – | Translations inherit non-permissive source text. Only `copa_pt-*` (COPA, BSD-2) would be clean; that is out of scope here (commonsense). |
| `mteb/RTE3`, `maximoss/rte3-multi`, `nlpyeditepe/tr_rte` | CC-BY-4.0 / MIT on the translations. The RTE source (PASCAL; news and IE pairs) has no licence; GLUE card says `other`. | de en fr it / tr | ~800 / 2,490 | – | The source licence is unknown and the text is news. |
| `masakhane/afrixnli`, `mteb/AfriXNLI` | Apache-2.0 / CC-BY-4.0 cards; README: "translations of a subset of the XNLI dataset" | en fr + African languages | test/dev only | – | Derived from XNLI (NC). |
| `McGill-NLP/GlobalNLI` | Card `apache-2.0`; `source_datasets`: "XNLI, AfriXNLI, IndicXNLI, AmericasNLI, …, **IndoNLI**, JNLI, KLUE, …" | 59 | small | **Contains IndoNLI (our eval suite)** | NC/SA sources plus overlap with an eval suite. |
| `mteb/xnli2.0-multi-pair` | Card `cc-by-4.0`; it is XNLI test re-translated | 13 | test only | – | XNLI (NC) |
| `aisingapore/SEA-NLI` | Card `mit`, gated (manual); LLM-generated plus human-verified; concepts taken from Wikipedia | en id km ms my ta th tl vi | 1K–10K (gated) | Indonesian content (IndoNLI language) | Skip: a gated evaluation benchmark; the only priority language is en. |
| `NirantK/hda_nli_hindi`, `midas/bbc_hindi_nli` | Card `mit`. README: "Source Dataset … BBC Hindi Headlines Dataset … collected … from Hindi Websites". | hi | ~10–100k | – | Scraped news text |
| `Flaglab/ESNLIR-dataset` | Card `cc-by-4.0`. README lists genre `escomments__reddit`, and states: "Connector labels are not human labels." | es | 4,387,968 | – | Reddit text plus automatic labels |
| `venelin/inferes` | Card `cc-by-4.0`. README Source Data: "Wikipedia + text generated from 'sentence generators'". | es | 6,444 | – | Wikipedia premises (CC-BY-SA), same as CONDAQA in the audit |
| `ruanchaves/faquad-nli` | Card `cc-by-4.0`, `source_datasets: extended|wikipedia` | pt | 3,128 | – | Wikipedia text |
| `muhammadravi251001/indonesian-nli-and-qa` | Card `mit`, no README text. The premises look like IndoNLI (news, "dilansir aceshowbiz.com"). | id | ? | **Likely IndoNLI-derived (eval suite)** | Eval overlap plus news text |
| `ziaddddd/arabic-embedding-dataset-nli` | Card `cc-by-4.0`; empty README, no provenance | ar | 8,214 | – | Unverified provenance (strict policy, same as `prompt-injection-dataset` in the audit) |
| `feyzaakyurek/BBNLI` | Card `mit`; premises carry a news `reference` | en | test 3,642 only | – | News text; test only |
| `allenai/scitail` | Card: no licence; premises are web sentences | en | 23,596 | – | Missing licence plus web text |
| `tasksource/zero-shot-label-nli` | Card `other` | en | 1,090,333 | – | "other" |
| FEVER-NLI (`nli_fever`, `tommasobonomo/sem_augmented_fever_nli`), DocNLI, SciNLI, ContractNLI, ConTRoL, AmericasNLI | Existing records: doc-nli excluded by the audit (CNN/DM + SQuAD). tasksource: `scinli` "apache-2.0, CC BY-SA 4.0 (DPI)"; `contract-nli` "cc-by-nc-sa-4.0"; `ConTRoL-nli` "CC BY-NC-SA 4.0 (DPI)"; `americas_nli` "cc-by-sa-4.0". FEVER evidence comes from Wikipedia. | – | – | – | SA / NC (existing verdicts; not re-audited) |
| `xksteven/dialogue_nli` (DNLI on PersonaChat) | Card `mit`; no upstream licence found (tasksource records `dialogue_nli` as unspecified) | en | ~310k | – | Upstream licence unverified |
| `Turkish-NLI/legal_nli_TR_V1` | Card `apache-2.0`. Text is court rulings scraped from emsal.uyap.gov.tr (official decisions); labels are derived automatically from shared statute articles. | tr | 474,283 | – | The licence may be fine, but the labels are heuristic, not human. Not recommended. |
| `Lots-of-LoRAs/task*_anli_*`, `…_mnli_…`, `…_cb_…` | Apache-2.0 on Natural-Instructions repacks | en | – | – | Inherit the source licence (ANLI NC, CB news) |

### Category 7: semantic similarity, paraphrase and relevance ranking

#### KEEP

| HF id (original source) | Licence string + where verified | Languages | Train rows | Label set | Eval-suite overlap | Verdict |
|---|---|---|---:|---|---|---|
| `community-datasets/tapaco` (TaPaCo from Tatoeba, Scherrer 2020) | Card `cc-by-2.0`. tatoeba.org/en/terms_of_use §6.2: "Tatoeba … uses the default Creative Commons Attribution 2.0 France license (CC-BY 2.0 FR) for the use of textual sentences". | 73 languages, **all 14 priority**: en 158,053; de 125,091; fr 116,733; es 85,064; it 198,919; pt 78,430; nl 23,561; pl 22,391; tr 142,088; ru 251,263; ar 6,446; hi 1,913; ja 44,267; cmn 12,549 | 1,926,192 sentences in paraphrase sets | `paraphrase_set_id` gives same-set pairs = paraphrase. Negatives are built from other sets (hard negatives by lexical overlap). | None | **KEEP.** Attribution goes to "Tatoeba contributors"; `sentence_id` is kept for credit. |
| Tatoeba translation pairs (`Helsinki-NLP/tatoeba`; better, the raw tatoeba.org / OPUS export) | Card `cc-by-2.0`; the same Tatoeba terms apply | 300+ incl. all priority | millions of pairs | Cross-lingual equivalence. Positives are translation pairs; negatives are sampled. | None (not FLORES) | **KEEP.** The HF repo is a loading script (datasets 5 cannot run it), so load the OPUS/Tatoeba files directly. Do not use `sentence-transformers/parallel-sentences-tatoeba`: its card has no licence. |
| `dell-research-harvard/headlines-semantic-similarity` (HEADLINES, Silcock et al.) | Card `cc-by-2.0`. README: "HEADLINES is released under the Creative Commons CC-BY 2.0 license" … "off-copyright, local U.S. newspapers". | en (historic, OCR) | 34,867,488 headlines in `group_id` clusters | Same group means the same story (paraphrase); different group means not. | None | **KEEP** (sample it; there is OCR noise and dated language). The text is public domain, unlike the modern headline sets the audit excluded. |
| `tasksource/esci` (Amazon Shopping Queries, amazon-science/esci-data) | Card `apache-2.0`. GitHub amazon-science/esci-data: `Apache-2.0`, "This project is licensed under the Apache-2.0 License." | en, **es, ja** | 2,027,874 (test 652,490) | Exact / Substitute / Complement / Irrelevant | None | **KEEP.** It is already in the mixture as `esci`, capped at 2,000; draw more rows from es and ja. Only Amazon's own catalogue text is included, not reviews, as the audit already notes. |
| `napsternxg/wands` (Wayfair WANDS) | Mirror card `mit`. GitHub wayfair/WANDS: `MIT`, "Distributed under the MIT License". | en | 140,068 in the mirror's train split (233,448 labels in total) | Exact / Partial / Irrelevant | None | **KEEP.** Human product-search relevance labels; product text is Wayfair's own. |
| `Avature/Job-Title-Similarity` | Card `apache-2.0`. GitHub Avature/jobtitlesimilarity-dataset: `Apache-2.0`. | de en es fr it ja ko nl pl pt zh | about 104 queries × about 2.5k titles per language (benchmark, no train split) | Binary human relevance (86% inter-annotator agreement); human or machine translations | None | **KEEP** (small; covers 10 priority languages). |
| `federetyk/MELO-Benchmark` (Avature) | Card `mit`. GitHub Avature/melo-benchmark: `MIT`. National occupation terminologies are linked to ESCO. | 21 EU languages incl. de fr es it nl pl pt + en | 328–4,438 queries per config (48 configs) | Query term → set of relevant ESCO titles | None | **KEEP.** Assumption: the ESCO labels fall under the European Commission reuse decision (CC-BY-4.0-compatible); not verified separately. |
| `matsuxr/JaGovFaqs-22k` | Card `cc-by-4.0`. README: "ライセンスもCC-BY-4.0（国際）です" (the licence is also CC-BY-4.0 International), citing digital.go.jp/copyright-policy. | ja | 22,794 Q–A pairs | Question → answer relevance (negatives are other answers) | None. JMTEB uses it, but JMTEB is not our eval. | **KEEP.** |
| `stjiris/IRIS_sts` | Card `mit`. Portuguese Supreme Court decision sentences plus text-davinci-003 rewrites (README). | pt | 1,667 (validation and test 556 each) | `relatedness_score` 1–5, `entailment_judgment` | None | **KEEP, low priority** (small; the high scores partly come from generated pairs). |
| `washenkov/synthetic-chat-duplicates` | Card `cc-by-4.0`. README: "fully-synthetic … CC-BY-4.0 – free for commercial & research". | en | 8,000 | `dialog_type` (original / duplicate_L1 … / fact-flip) → duplicate yes/no | None | **KEEP** (small; dialogue-level duplicate detection). |
| `hkust-nlp/SynCSE-scratch-NLI` | MIT (see category 4) | en | 275,579 | `sent0`~`sent1` paraphrase; `nli_hard` is a hard non-paraphrase | None | **KEEP** (listed under category 4; usable here too). |
| `tasksource/patent-phrase-similarity` | CC-BY-4.0 (Google) | en | – | 0–1 score | – | Already in the mixture. |

#### EXCLUDE

| HF id (original) | Licence seen + where | Languages | Reason |
|---|---|---|---|
| PAWS / PAWS-X (`google-research-datasets/paws`, `paws-x`, `mteb/PawsXPairClassification`, `NbAiLab/norwegian-paws-x`) | Card `other`. GitHub README: "We cannot directly distribute the raw PAWS-QQP data due to the license". PAWS-Wiki sentences are from Wikipedia. | en de es fr ja ko zh | Wikipedia (SA) plus Quora text; "other" |
| GLUE MRPC / QQP / STS-B (`nyu-mll/glue`, `sentence-transformers/stsb`, `PhilipMay/stsb_multi_mt`, `mteb/stsbenchmark-sts`, `mteb/stsb_multi_mt`, `mteb/GermanSTSBenchmark`, `Omartificial-Intelligence-Space/Arabic-stsb`, `dumitrescustefan/ro_sts`) | `other` / none / `unknown` / `cc-by-sa-3.0` | en + MT | News (MRPC), Quora (QQP), mixed news/forum/caption sources (STS-B); tasksource `sts-companion` is already excluded by the audit |
| `silma-ai/silma-arabic-english-sts-dataset-v1.0` | Card `apache-2.0`, but row 1 "طائرة ستقلع" is STS-B "A plane is taking off." | ar en | Translation of STS-B; the card conflicts with the source |
| SICK-R / SemEval-2014 T1 / KorSTS / JSTS / JSICK | `cc-by-nc-sa-3.0` (mteb/sickr-sts), `cc-by-sa-4.0` (mteb/KorSTS, mteb/JSTS) | en ko ja | NC / SA |
| STS12–17, STS22 (`mteb/sts17-crosslingual-sts`, `mteb/sts22-crosslingual-sts`) | `unknown` | multi | News, forums and SNLI captions; unknown licence |
| **SemRel2024** (`SemRel/SemRel2024`, `mteb/SemRel24STS`) | `unknown`; the official repo has no licence file | en ar hi … | **Our eval suite**: evaluation only |
| Opusparcus (`GEM/opusparcus`, `mteb/OpusparcusPC`) | `cc-by-nc-4.0` | de en fi fr ru sv | NC (OpenSubtitles) |
| Quora duplicates (`sentence-transformers/quora-duplicates`, `embedding-data/QQP_triplets`, `aisuko/quora_duplicate_questions`, `Gabriel/quora_swe`, `Omartificial-Intelligence-Space/Arabic-Quora-Duplicates`) | none / MIT / Apache on repacks | en sv ar | Quora text under platform terms |
| Stack Exchange / Twitter duplicates (`mteb/askubuntudupquestions-reranking`, `stackoverflowdupquestions-reranking`, `twittersemeval2015-pairclassification`) | none / `unknown` | en | CC-BY-SA / tweets |
| `merionum/ru_paraphraser`, `mteb/RUParaPhraserSTS` | `mit` | ru | News headlines. Same rule as the audit's `headline_cause` / `hlgd`. |
| `cointegrated/ru-paraphrase-NMT-Leipzig` | Card `cc-by-4.0`. README: sentences come from the Leipzig `rus-ru_web-public_2019_1M` crawl, and paraphrases are automatic back-translations. | ru | Web-crawl text; labels are not human |
| `agentlans/sentence-paraphrases` → `humarin/chatgpt-paraphrases` | `mit` → `openrail`. humarin README: "based on the Quora paraphrase question, texts from the SQUAD 2.0 and the CNN news dataset". | en | Quora, Wikipedia and CNN source text; openrail |
| `redis/llm-paraphrases` | Card `apache-2.0`; LLM-generated from "original queries" whose origin is not documented. The first rows are verbatim SynCSE rows. | en | Seed provenance unverified (7,065,517 rows). Use SynCSE directly. |
| `pszemraj/synthetic-text-similarity` | Card `odc-by`. README: documents are sampled from web sources and cosmopedia; the score comes from embeddings. | en | Web text; labels are model-derived, not human |
| `jpwahle/machine-paraphrase-dataset` (and the autoencoder/autoregressive variants) | `cc-by-4.0` | en | Wikipedia / arXiv source text |
| `deutsche-telekom/ger-backtrans-paraphrase` | `cc-by-sa-4.0` | de | SA |
| `nilc-nlp/assin`, `nilc-nlp/assin2` | `unknown` | pt | Unknown licence; ASSIN is news, ASSIN2 is SICK-BR (NC-SA source) |
| `mteb/STSES` (PlanTL sts-es) | Card `cc-by-4.0`; test only (155); news and Wikipedia sentences | es | News / Wikipedia; test only |
| `mteb/indic_sts` | Card `cc0-1.0`; test only; `source` = IndicCorp / CatchNews | en + 12 Indic | News-crawl sentences; test only |
| `LivNLP/C-STS-Reannotated` | Card `cc-by-4.0` (LLM re-annotation). Sentences come from C-STS; GitHub princeton-nlp/c-sts: `license: null`; they are image captions. | en | Source licence unverified |
| `eleftheria/refresd`, `copenlu/spiced`, `Exr0n/wiki-entity-similarity` | `mit` | en fr / en | Wikipedia (WikiMatrix) / tweets and news / Wikipedia |
| `Antix5/Product_Similarity_Dataset` | Card `apache-2.0`, no README. Product names look like Open Food Facts (ODbL) and Amazon listings. | 12 | Undocumented, likely ODbL/scraped |
| `Ukhushn/home-depot` | Card `afl-3.0`, no README | en | Kaggle competition data (competition terms) |
| MS MARCO / mMARCO (`microsoft/ms_marco`, `unicamp-dl/mmarco`, `hotchpotch/mmarco-hard-negatives-reranker-score`, `C-MTEB/MMarco*`) | GitHub microsoft/msmarco README: "intended for non-commercial research purposes only". mMARCO repo Apache-2.0 covers the translation only. | en + 13 | NC terms; translations inherit them |
| MIRACL, Mr. TyDi (`miracl/miracl`, `castorini/mr-tydi`, `miracl/mmteb-miracl-reranking`) | Cards `apache-2.0`; GitHub Apache-2.0 | 18 incl. ar de es fr hi ja ru zh | Passages are Wikipedia (CC-BY-SA); Apache covers only queries and code |
| Natural Questions (`google-research-datasets/natural_questions`), WikiQA (`microsoft/wiki_qa`), GooAQ (`allenai/gooaq`) | `cc-by-sa-3.0` / `other` / `apache-2.0` (GooAQ answers are Google search snippets of web pages) | en | Wikipedia / web text |
| BEIR subsets: `BeIR/scifact` (`cc-by-sa-4.0`; SciFact itself is CC-BY-NC), `mteb/scidocs` (`cc-by-sa-4.0`), `mteb/fiqa` (`unknown`), `mteb/nfcorpus` / `trec-covid` (none), ArguAna / Touché (args.me; the audit excluded `args_me`), `mteb/mind_small` (none; MIND research licence) | as listed | en | SA / NC / missing |
| `THUIR/T2Ranking` | Card `apache-2.0`. README: passages "from real-world search engines" (web pages). | zh | Web-page passages. Borderline; excluded under the audit's web-crawl rule (`discovery`). |
| `lightblue/reranker_continuous_filt_max7_train` | Card `apache-2.0`; rows from `mlqa`, `hotpot_qa` and others; labels from Qwen2.5-32B | 50+ | Wikipedia sources; the card conflicts with them |
| Aggregates: `KaLM-Embedding/KaLM-reranker-training-data`, `abdoelsayed/reranking-datasets`, `codefuse-ai/F2LLM-v2`, `BidirLM/BidirLM-Contrastive`, `BEE-spoke-data/sbert-paraphrase-data`, `DMetaSoul/chinese-semantic-textual-similarity`, `shibing624/nli_zh` / `nli-zh-all` (ATEC/BQ/LCQMC/PAWS-X/STS-B), `shibing624/sts-sohu2021` | MIT / Apache / ODC-By / CC-BY on the repacks | multi | Mixed upstream (MS MARCO, Wikipedia, LCQMC/BQ research-only, Sohu news) |
| `PaDaS-Lab/webfaq` | Card `cc-by-4.0`. README: "Data was collected from the Common Crawl … Please ensure compliance with … the source website's terms." | 75 | Web-crawled FAQ text |
| `lyon-nlp/mteb-fr-reranking-alloprof-s2p` / `antoinelb7/alloprof` | Card `mit`, but the alloprof card's citation says `copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}` | fr | Conflicting licence; NC-SA |
| `sbintuitions/JMTEB` (JQaRA, JaCWIR, jaqket) | Card `cc-by-sa-4.0`; JQaRA CC-BY-SA; JaCWIR is web; jaqket train is "non-commercial research" | ja | SA / NC / web. JaGovFaqs is taken from its own CC-BY source instead (KEEP above). |
| `mteb/GerDaLIR`, `mteb/PublicHealthQA`, `sentence-transformers/parallel-sentences-tatoeba`, `C-MTEB/*` (LCQMC, BQ, ATEC, AFQMC, QBQTC, T2Reranking, CMedQAv2) | Cards: no licence | de / multi / zh | Missing licence (GerDaLIR comes from Open Legal Data dumps; LCQMC and BQ are research-only) |
| `MTEB-BR/juristcu-reranking`, `MTEB-BR/quati-reranking` | `cc-by-4.0`; test only; Quati passages come from ClueWeb22 | pt | Quati is web text. JurisTCU (Brazilian court-of-accounts texts) was not checked beyond the card and may be worth a later look. |
| `NeuML/historical-english-books-similarity` | `apache-2.0`; public-domain Victorian books plus LLM queries | en | Not verified further (the script repo has no size info). Possible later add; not needed. |

### Summary

- Category 4 NLI: **10 KEEP** (plus WANLI already used) and about 30 EXCLUDE rows.
  - Human-labelled: MNLI non-fiction (en), plus its machine translations in XNLI 2.0 (de fr es ru tr ar hi zh, and bg el sw th ur vi), NLI-TR (tr), WANLI-ja (ja) and MNLI-9/11-fr (fr).
  - Synthetic: SynCSE, HANS, attempto, Mixtral label-NLI and Turkish-constructed.
- Category 7 similarity / paraphrase / relevance: **11 KEEP** (plus esci and patent-phrase-similarity already in the mixture) and about 35 EXCLUDE rows.

### Gaps (to fill with synthetic data)

1. **NLI in it, pt, nl, pl, ja (native), ar, hi (native).** XNLI 2.0 has no it, pt, nl, pl or ja. No native (non-translated), human-labelled, permissively licensed NLI exists in any priority language other than en. Fix: machine-translate MNLI non-fiction, WANLI and SynCSE with a permissive MT model into it, pt, nl, pl and ja, then filter as wanli-ja did.
2. **Graded STS scores (0–5)**: none is clean in any priority language except the tiny pt IRIS_sts. STS-B, SICK, KorSTS, JSTS and STS22 are all SA, NC, news or unknown. Generate graded pairs synthetically (LLM-rated), as Decima does.
3. **Query–passage relevance for general web and QA**: none clean. MS MARCO is NC; MIRACL, Mr. TyDi, NQ and SQuAD are Wikipedia; T2Ranking and WebFAQ are web. Only product search (ESCI en/es/ja, WANDS en), job titles (11 languages) and JaGovFaqs (ja) remain. Synthetic query generation over licence-clean passages (public-domain books, government texts, own synthetic documents) is needed for de fr it nl pl ru ar hi zh.
4. **Duplicate-question detection** (the Quora / Stack Exchange style): none clean; only synthetic sets (SynCSE, synthetic-chat-duplicates).
5. **Cross-lingual STS**: only binary translation equivalence (Tatoeba); no graded cross-lingual scores.

### Out-of-category finds (for other forks / parent)

- `amyrmahdy/decima-synthetic-decisions`:
  - Licence `cc-by-4.0`. The teacher is Gemma-4-26B-A4B-it, whose licence is `apache-2.0` on the HF model card.
  - About 190k fully synthetic typed decisions (choose / verify / score / rank) with soft labels, in en, fa, ar and ru.
  - It matches Statim's schema directly, but it contains fa and ar decision text; FarsTail is fa NLI, and there is no text overlap.
  - Recommended for category 10 / general decisions.
- `PORTULAN/extraglue` `copa_pt-*`: COPA (BSD-2) translated to pt; clean for commonsense.
- Mintaka (`amazon-science/mintaka`, CC-BY-4.0 per JMTEB/README): multilingual QA (en ar de ja hi pt es it fr); candidate for reading comprehension / QA.


## Part D: category 5 (toxicity, moderation, spam, prompt injection)


Checked 2026-09-27 (pop-os) against the HF Hub API (`dataset_info().card_data` / tags / README via
`hfcheck.py`), datasets-server `/size`, `/first-rows` and `/statistics`, and upstream GitHub
(`gh api repos/...` licence + README) or the paper. Only metadata and first rows were read. Three small
upstream CSVs were fetched to count rows and sources: GAHD 1.5 MB, DynaHate 9.7 MB, Aya red-teaming 3.4 MB.

Policy is applied as in `DATA_LICENSES.md` / `licence_audit.json`. Text posted on a platform (tweets,
Reddit, YouTube, Gab, Telegram, Wikipedia talk pages under CC-BY-SA), scraped web or news text, and real
e-mail corpora are excluded even when the card tag is permissive. Text written by annotators, crowd
workers, templates or LLMs is kept under its creator's permissive licence. Following the audit,
model terms on LLM outputs are treated as binding the generator, not us; those rows are marked
"caution".

**Eval-suite overlap.** None of the KEEP sets below is one of our evaluation suites (typed-decisions,
Banking77, MASSIVE, AG News, DAIR Emotion, GoEmotions, tyqiangz, multi_hatecheck, SIB-200, Belebele,
IndoNLI, FarsTail, SemRel, HWU64). None is derived from HateCheck or Multilingual HateCheck:
- DynaHate, GAHD and HatemojiBuild come from the same research group but were written separately on Dynabench.
- GAHD's `translation` rows translate DynaHate, not HateCheck.
Two precautions still apply:
- Run the usual `build_mixture.test_texts()` exact-match dedup, and add the `mteb/multi-hatecheck`
  test texts (all 11 languages) to the banned set for every hate or toxicity source.
- Dedup across the HH-derived sources. Aegis 1.0, Aegis 2.0, Salad, AttaQ, MultiJail and
  hh-rlhf red-team reuse the same Anthropic red-team prompts. hh-rlhf and Nemotron Safety Guard v3
  are already in training.

### 5a. Toxicity / hate speech / harassment

| HF id (original source) | Licence + where verified | Languages | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| **DynaHate**: upstream GitHub `bvidgen/Dynamically-Generated-Hate-Speech-Dataset` v0.2.3.csv (HF mirror `tasksource/dynahate`) | Upstream README: "The dataset is licenced under CC-BY 4.0." and "All content is synthetic" (annotator-written on Dynabench). The mirror's tag is `gpl`, which contradicts upstream; do not load the mirror. | en | 41,144 total: train 32,924 / dev 4,100 / test 4,120 | `label` hate/nothate; `type` (animosity, derogation, dehumanization, threatening, support); `target` | Not a suite; not HateCheck-derived | **KEEP**. The repo audit currently excludes `dynahate` because of the tasksource "gpl" tag. Upstream says CC-BY-4.0, so that exclusion should be reversed. Load the upstream CSV. |
| **GAHD** `jagoldz/gahd` (GitHub `jagol/gahd`) | HF card `cc-by-4.0`; GitHub spdx `CC-BY-4.0` | de | 10,996 total. After dropping `news` rows: train 5,441 / dev 1,166 / test 1,162 | `label` 0/1 (hate) | Not a suite | **KEEP-FILTERED.** Keep `source` ∈ {dynabench, contrastive, translation}. The `translation` rows are DynaHate translations (CC-BY). Drop `news`: 3,227 Leipzig news-corpus sentences under publisher copyright. The `source` column exists only in GitHub `gahd_disaggregated.csv`, not on HF. |
| `google/civil_comments` (Civil Comments) | CC0 (already audited) | en | — | toxicity sub-scores | — | **already used** (tasksource `civil_comments/*`) |
| `mteb/toxic_conversations_50k` (Jigsaw Unintended Bias = Civil Comments) | HF card `cc-by-4.0`; README "comments from the Civil Comments platform"; Civil Comments text is CC0 | en | train 50,000 / test 50,000 | toxic / not toxic | Not a suite | **KEEP (duplicate).** Same text pool as Civil Comments. Only worth using as an extra balanced binary view; dedup against civil_comments. |
| `HannahRoseKirk/HatemojiBuild` | CC-BY-4.0 | en | — | hate/not (emoji) | — | **already used** |
| `OpenAssistant/oasst2` (moderation flags) | HF card `apache-2.0`. Human volunteer messages on open-assistant.io. | en 61k, es 27k, ru 13k, zh 8k, de 5.8k, fr 3.7k, pt-BR 2.6k, it 0.9k, ja 0.8k, pl 0.4k, ar/nl <0.1k (message counts) | 128,575 train / 6,599 val messages | `labels.{spam, pii, not_appropriate, hate_speech, sexual_content, toxicity, violence}`: mean of 2–5 crowd votes in [0,1] | Not a suite | **KEEP.** Multilingual human moderation votes (score or thresholded yes/no); positives are rare. Ignore the `detoxify` model scores and drop `synthetic: true` rows. oasst1 is a subset, so use oasst2 only. tasksource marks `oasst2/*` "unspecified", but the upstream card is Apache-2.0. |
| `PleIAs/ToxicCommons` (GitHub `Pleias/toxic-commons`) | HF `mit`; GitHub spdx `MIT`. Text from Common Corpus, which is public domain. | en fr es de pl nl pt it la | 1,290,000 | `scores` 0–3 on 5 axes (race/origin, gender/sex, religion, ability, violence) | Not a suite | **KEEP (caution).** Labels are LLM-generated (Llama 3.1 8B Instruct). The Llama output clause binds PleIAs, not us, following the audit rule. The text is noisy historical OCR and mostly clean; subsample the toxic tail. |
| `ToxicityPrompts/PolygloToxicityPrompts` | Tag `cc-by-4.0` but README: "made available under the AI2 ImpACT License - Low Risk Artifacts" | 17 | ~425k | Perspective scores | — | **EXCLUDE.** Card licence contradicts the README. Text is mC4/Pile web text; labels come from the Perspective API. |
| `adewynter/RTP-LX`, `ToxicityPrompts/RTP-LX` (GitHub `microsoft/RTP-LX`) | MIT (GitHub LICENSE); NOTICE: "Original English corpus from RTP" (Apache-2.0); mirror `odc-by` | 38 | test 30,253 | toxicity sub-scores | — | **EXCLUDE (strict).** Prompts are transcreations of RealToxicityPrompts snippets, i.e. OpenWebText (Reddit-linked web pages); the licence comes from the collector, not the rights holder. HF original is gated and password-zipped. |
| `allenai/real-toxicity-prompts` | `apache-2.0` (GitHub Apache-2.0) | en | 99k | Perspective | — | **EXCLUDE.** OpenWebText web text; labels from the Perspective API. |
| `toxigen/toxigen-data` (GitHub `microsoft/TOXIGEN`) | HF card: no licence. GitHub LICENSE.txt: MIT (code) + CDLA-Permissive-2.0 (data). README: "intended to be used for research purposes only". | en | train 250,951 (GPT-3), annotated 8,960 | toxicity_human 1–5, target group | — | **EXCLUDE.** Explicit research-only statement. The data licence itself (CDLA-P-2.0, "No Restrictions on Results") is permissive, so the owner could override. |
| `textdetox/multilingual_toxicity_dataset` (+ `_explained`, `toxic_spans`, `paradetox`) | `openrail++` | 15 | 5k/lang | toxic 0/1 | — | **EXCLUDE.** RAIL licence carries use restrictions (not on the list); sources are tweets, Jigsaw and forum comments. |
| `ucberkeley-dlab/measuring-hate-speech` | `cc-by-4.0` | en | 135,556 annotations | hate score + 10 ordinal items | — | **EXCLUDE.** Comments from Reddit, Twitter, YouTube and Gab (`platform` column). |
| `Hate-speech-CNERG/hatexplain` | `cc-by-4.0` | en | 20k | hate/offensive/normal | — | **EXCLUDE.** Twitter and Gab posts. |
| `OxAISH-AL-LLM/wiki_toxic`, `google/jigsaw_toxicity_pred`, Jigsaw Multilingual (Kaggle) | cc0 tag | en (+tr es it pt ru fr) | — | toxic | — | **EXCLUDE.** Wikipedia talk pages are CC-BY-SA 3.0 (as `jigsaw_toxicity` in the audit). |
| `FredZhang7/toxi-text-3M` | `apache-2.0` | 55 | 2.88M | is_toxic | — | **EXCLUDE.** README: "completely forgot the original source"; includes tweets fetched via the Twitter API. |
| `KoalaAI/Text-Moderation-Multilingual`, `enguard/multi-lingual-prompt-moderation`, `ifmain/text-moderation-02-multilingual` | `apache-2.0` | 17 | 1.46–1.55M | OpenAI moderation categories | — | **EXCLUDE.** Reddit comments (Kaggle), machine-translated; labels from the OpenAI moderation API; card: "research and safety purposes only". |
| Marco Guerini CONAN / Multitarget-CONAN / DIALOCONAN (`Rhma/Multitarget-CONAN`, `SINAI/CONAN-*`) | GitHub README: "can be used for research purposes and cannot be redistributed"; SINAI copies are NC-SA or SA | en fr it es | — | hate/counter-narrative | — | **EXCLUDE.** Research only. |
| `HannahRoseKirk/HatemojiCheck` | `cc-by-4.0` | en | test 3,930 | hateful/not | Same functional-test method and authors as HateCheck | **EXCLUDE (caution).** Keep held out so the multi_hatecheck zero-shot claim stays clean. |
| `Paul/hatecheck-*`, GPT-HateCheck, SGHateCheck, `mteb/multi-hatecheck` | — | — | — | — | **Is or derives from the multi_hatecheck suite** | **EXCLUDE.** Evaluation contamination. |
| Korean hate sets (`jeanlee/kmhas…`, `nayohan/korean-hate-speech`), `TUKE-KEMT/hate_speech_slovak`, `evalitahf/hatespeech_detection`, `tmu-nlp/thai_toxicity_tweet`, `JunyuLu/ToxiCN`, `AlexSham/Toxic_Russian_Comments`, `esclient/toxicity_multilanguage_dataset` | SA, NC, no licence or GPL (tags seen in the Hub listing) | various | — | — | — | **EXCLUDE.** Licence tag alone disqualifies; most are also platform text. |

### 5b. Content moderation / safety prompts (prompt and response harm)

| HF id (original source) | Licence + where verified | Languages | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| **`nvidia/Aegis-AI-Content-Safety-Dataset-1.0`** | HF `cc-by-4.0`; README: "open-source content safety dataset (CC-BY-4.0)". Sources: HH-RLHF prompts only (MIT) + Mistral-7B-v0.1 responses (Apache-2.0). | en | train 10,798 / test 1,199 | `labels_0..4` per annotator: Safe, Needs Caution + 13 categories | Not a suite | **KEEP.** Clean lineage: HH prompts, Apache model responses, NVIDIA human labels. |
| **`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`** | HF `cc-by-4.0` ("License: CC-BY-4.0"). Prompt sources: HH-RLHF, DAN jailbreaks (verazuo/jailbreak_llms: Reddit/Discord scrape), AART (Kaggle). | en | train 30,007 / val 1,445 / test 1,964 | `prompt_label` safe/unsafe (all human); `response_label` (human 4,682 / llm_jury 5,552 / refusal aug 5,000); `violated_categories` | Not a suite | **KEEP-FILTERED.** Apply the extra-v1 lineage join: keep only prompts traceable to HH harmless-base; drop `REDACTED`, DAN and AART rows. Prefer `response_label_source == human`. Already used as a join key. |
| `nvidia/Nemotron-Safety-Guard-Dataset-v3`, `l3cube-pune/IndicGuard` | CC-BY-4.0 | 12 / 11 | — | — | — | **already used** (extra-v1) |
| **`CohereForAI/aya_redteaming`** | HF `apache-2.0`; README: "can be used for any purpose, whether academic or commercial, under the terms of the Apache 2.0 License". Written by paid annotators. | en fr es ru ar hi sr tl | 7,419 (no split; eng 987, fra 813, spa 782, rus 1,007, arb 900, hin 915) | `harm_category` multi-label (9: hate speech, discrimination, violence, self-harm, profanity, …); `global_or_local` | Not a suite | **KEEP.** Human-written harmful prompts in 6 priority languages. All rows are harmful, so pair with benign prompts or use for harm-category decisions. |
| **`Anthropic/hh-rlhf`, data_dir `red-team-attempts`** | HF `mit`; GitHub `anthropics/hh-rlhf` spdx MIT. Card: "not meant for fine-tuning or preference modeling" (guidance, not licence). | en | 38,961 transcripts | `rating` 0–4 (attack success, by red-teamer); `min_harmlessness_score_transcript` (PM score); `task_description`; `tags` | Not a suite | **KEEP.** New config of an already-used MIT family. Use as score ("how harmful is this conversation"). |
| **`thu-coai/Safety-Prompts`** | HF `apache-2.0`; GitHub spdx Apache-2.0. Prompts augmented by a model, responses from ChatGPT. | zh | ~100k (7×10k typical + 6×5k instruction attack) | `type`: 13 scenarios (insult, unfairness, crime, physical harm, mental health, privacy, ethics, goal hijacking, prompt leaking, role-play instruction, unsafe instruction topic, unsafe-opinion inquiry, reverse exposure) | Not a suite | **KEEP.** The main clean Chinese safety source. Use `field=` per scenario from the JSON files; the dataset viewer is not available. |
| **`bench-llms/or-bench`** (GitHub `justincui03/or-bench`) | HF `cc-by-4.0`; GitHub spdx Apache-2.0. Prompts from Mixtral-8x7B, moderated by an LLM ensemble. | en | or-bench-80k 80,359 (seemingly toxic, benign), hard-1k 1,319, toxic 655 | benign vs toxic + 10 categories | Not a suite | **KEEP.** Over-refusal contrast: harmful-looking but benign prompts. |
| **XSTest**: `natolambert/xstest-v2-copy` (GitHub `paul-rottger/exaggerated-safety`) | HF card: "The test prompts are subject to Creative Commons Attribution 4.0"; GitHub spdx CC-BY-4.0 | en | 450 prompts (250 safe / 200 unsafe) | safe/unsafe + type | Not a suite; not HateCheck | **KEEP (prompts only).** Completions carry model licences, so drop them. |
| **`Bertievidgen/SimpleSafetyTests`** | HF `cc-by-2.0`; GitHub `bertiev/SimpleSafetyTests` CC-BY-4.0 | en | test 100 | harm_area (5), category | Not a suite | **KEEP (tiny).** |
| **`JailbreakBench/JBB-Behaviors`** | HF `mit`: "This dataset, like the code, is released under MIT License"; GitHub MIT | en | harmful 100 + benign 100 | harmful/benign; 10 categories | Not a suite | **KEEP (tiny).** |
| **HarmBench** `walledai/HarmBench` (GitHub `centerforaisafety/HarmBench`) | HF `mit`; GitHub spdx MIT | en | ~400 behaviors | semantic category | Not a suite | **KEEP-FILTERED (tiny).** Standard behaviors only; drop the copyright and contextual rows, whose context is third-party text. |
| **AdvBench** `walledai/AdvBench` (GitHub `llm-attacks/llm-attacks`) | HF `mit`; GitHub spdx MIT | en | 520 | all harmful | Not a suite | **KEEP (caution, tiny).** Generated with Wizard-Vicuna-30B-Uncensored, a LLaMA-1 derivative; audit rule applies. |
| **`declare-lab/HarmfulQA`** (GitHub `declare-lab/red-instruct` Apache-2.0) | HF `apache-2.0`; ChatGPT-generated | en | 1,960 questions; ~9.5k blue (harmless) + ~7.3k red (harmful) conversations | topic/subtopic; blue vs red | Not a suite | **KEEP (caution).** Response-harm pairs. |
| **`declare-lab/CategoricalHarmfulQA`** | HF `apache-2.0` | en zh vi | 550 per language | 11 categories × 5 subcategories | Not a suite | **KEEP (small).** |
| **`OpenSafetyLab/Salad-Data`** `base_set` (GitHub `OpenSafetyLab/SALAD-BENCH` Apache-2.0) | HF `apache-2.0`. README source table: GPT-Gen 15,433, HH-harmless 4,184, HH-red-team 659, Advbench 359, Multilingual 230, Do-Not-Answer 189, ToxicChat 129, DAN 93, GPTFuzzer 42 | en | 21,318 → ~20,635 after filter | 3-level taxonomy (6/16/66), auto-labelled | Not a suite | **KEEP-FILTERED.** Keep `source` ∈ {GPT-Gen, HH-harmless, HH-red-team, Advbench}. Drop Do-Not-Answer (NC-SA), ToxicChat (NC), DAN and GPTFuzzer (scraped jailbreaks) and Multilingual (unverified). Skip `attack_enhanced_set`, whose templates are scraped jailbreaks. |
| **`ibm-research/AttaQ`** | HF `mit` (source_datasets `extended|Anthropic/hh-rlhf`) | en | 1,402 | 7 harm labels | Not a suite | **KEEP (small).** |
| **`DAMO-NLP-SG/MultiJail`** (GitHub `DAMO-NLP-SG/multilingual-safety-for-LLMs`; mirror `ToxicityPrompts/DAMO-MultiJail` odc-by) | HF `mit`; GitHub spdx MIT. Human translations of HH red-team prompts. README: "our research is solely for academic purposes and ethical use". | en zh it vi ar ko th bn sw jv | 315 × 10 languages | `tags` (harm categories) | Not a suite | **KEEP (caution).** MIT governs. Owner should confirm the "academic purposes" sentence is an intent statement, like the research note on the hh-rlhf card. |
| **`Virtue-AI-HUB/PolyGuard`** (GitHub `AI-secure/PolyGuard`, NeurIPS 2025, arXiv 2506.19054) | HF `cc-by-4.0`; GitHub repo has no licence file (code); data LLM-generated (o4-mini/GPT-4) from 150+ public policies | en | ~136k rows across 8 domains (social_media, finance, law, hr, education, regulation, cyber, code) | safe/unsafe + policy category | Not a suite | **KEEP (caution).** Policy-grounded moderation. The card has no provenance section; the paper documents generation. Skip `cyber/cve` and `cyber/mitre`, which embed third-party text. |
| **`allenai/prosocial-dialog`** | HF `cc-by-4.0` | en | 120,236 train turns; keep `source == ethics_amt` (5,853) | `safety_label` 5 levels (casual … needs intervention) | Not a suite | **KEEP-FILTERED.** Drop socialchemistry (85k, CC-BY-SA, Reddit), sbic (Reddit/Twitter/Gab) and ethics_reddit, consistent with the audit's moral_stories exclusion. |
| **`nvidia/CantTalkAboutThis-Topic-Control-Dataset`** | HF `cc-by-4.0`; "commercially friendly version … generated using Mixtral-8x7B-Instruct" | en | mixtral train 1,073 dialogues / test 20 | on-topic vs distractor turn per system instruction | Not a suite | **KEEP.** Topic-control moderation ("Is this user turn in scope?"). |
| `nvidia/Nemotron-3.5-Content-Safety-Dataset` | Card metadata: no licence; README: "governed by … CC-BY-4.0" | 12 (en 55.7k; ar de es fr hi it ja ko nl th zh 3k each) | 88,688 | input/response safe/unsafe + categories | — | **EXCLUDE (for now).** No licence in the card metadata; `provenance` mixed (48.5k); built from aegis_v3 (DAN/AART lineage) and a reasoning set. Revisit with the extra-v1 lineage join. |
| `nvidia/Nemotron-Content-Safety-Reasoning-Dataset` | `cc-by-4.0` | en | — | Aegis 2.0 labels + reasoning | — | **EXCLUDE (redundant).** Same rows as Aegis 2.0 and CantTalkAboutThis. |
| `ToxicityPrompts/PolyGuardMix`, `ToxicityPrompts/PolyGuardPrompts` | HF `cc-by-4.0` | 17 | 1,910,372 / test 29,325 | prompt/response harm, refusal, categories | — | **EXCLUDE.** `metadata.source` = wildguardmix_original/translated + `itw` (in-the-wild LMSYS/WildChat), so it falls under the audit's WildGuardMix exclusion. Translations partly by TowerInstruct (NC). |
| `allenai/wildjailbreak` | HF `odc-by`, gated; gate checkbox: "I agree to use this dataset for research purposes in accordance with the AI2 Responsible Use Guidelines" | en | 262k | vanilla/adversarial × harmful/benign | — | **EXCLUDE.** Research-purpose click-through. |
| `Edu-p/harmful-prompts-pt` | HF `mit`; README: "translated from a stratified 10% subset of the original WildJailbreak training split" | pt | 29,432 | harmful/benign | — | **EXCLUDE.** Derivative of WildJailbreak. |
| `ToxicityPrompts/XSafety` (GitHub `Jarviswang94/Multilingual_safety_benchmark`) | Mirror `odc-by`; GitHub LICENSE Apache-2.0 but README header "RESEARCH USE ONLY" | 10 | test 27,999 | 14 categories | — | **EXCLUDE.** Research only; commonsense part looks like SafeText (Reddit). |
| `LibrAI/do-not-answer` | HF `apache-2.0`, but README and GitHub `Libr-AI/do-not-answer`: "All datasets … released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0" | en | 939 | risk area / harm type | — | **EXCLUDE.** Card contradicts source; data is NC-SA. |
| `allenai/coconot` | Card has no licence tag; README says both "License: https://allenai.org/licenses/impact-lr" and "made available under the ODC-BY" | en | 11,477 | noncompliance category | — | **EXCLUDE.** Conflicting and custom (ImpACT) terms. |
| `mmathys/openai-moderation-api-evaluation`, `HRB25/pt-br-moderation-eval` (GitHub `openai/moderation-api-release` MIT) | MIT; paper (arXiv 2208.03274) footnote: evaluation set "sourced from CommonCrawl and model-generated data" | en / pt | 1,680 | S, H, V, HR, SH, S3, H2, V2 | — | **EXCLUDE (strict).** CommonCrawl web text; MIT comes from OpenAI, not the authors. The pt copy is a translation of it. |
| `sorry-bench/*`, `llm-jp/AnswerCarefully`, `Babelscape/ALERT`, `giskardai/do-not-answer-scenarios`, `kunishou/do-not-answer-ja` | `other` (gated) / custom ToU (gated manual) / `cc-by-nc-sa-4.0` | — | — | — | — | **EXCLUDE.** Custom, NC or SA terms. |
| `ai4privacy/pii-masking-400k`, `ai4privacy/open-pii-masking-500k-ai4privacy` (PII) | `other`; README: subject to the Llama Community License ("must include 'Llama' at the beginning of the model name") and "Commercial use … may require explicit written permission" | 6–8 | 325k / 464k | PII spans | — | **EXCLUDE.** Custom terms bind downstream models. |
| BeaverTails, PKU-SafeRLHF, toxic-chat, wildguardmix, PromptShield | NC / strict audit | — | — | — | — | **already excluded** |

### 5c. Spam / phishing / scam

| HF id (original source) | Licence + where verified | Languages | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| UCI SMS Spam Collection (`ucirvine/sms_spam`, `codesignal/sms-spam-collection`) | CC-BY-4.0 (audit) | en | 5,572 | ham/spam | — | **already used** (`sms_spam`); the codesignal copy is a duplicate |
| **`Dizzzy0x00/LLMGen-Phishing-Email-Dataset`** | HF `apache-2.0`; README: "generated using … DeepSeek for Chinese emails and OpenAI models for English emails" | en zh | 6,776 | `label` phishing/legit | Not a suite | **KEEP (caution).** Fully synthetic. |
| **`BothBosu/scam-dialogue`** | HF `apache-2.0` ("released under the Apache license 2.0"); generated with meta-llama-3-70b-instruct | en | 1,280 / test 320 | scam/non-scam + `type` | Not a suite | **KEEP (caution).** Llama 3 output terms bind the generator (audit rule). |
| **`BothBosu/multi-agent-scam-conversation`** | HF `apache-2.0`; two-agent synthetic dialogues | en | 1,280 / test 320 | scam/non-scam + type + personality | Not a suite | **KEEP (caution).** |
| **`shakeleoatmeal/phone-scam-detection-synthetic`** | HF `mit`; "Generated using Llama 8B model" | en | 1,259 / 361 / 180 | scam label + type | Not a suite | **KEEP (caution).** |
| **`alusci/sms-otp-spam-dataset`** | HF `mit`; "A synthetic dataset of 10,000 OTP-style SMS messages" | en | 10,000 | valid / spam-like | Not a suite | **KEEP (low value).** Templated and trivial. |
| `Virtue-AI-HUB/PolyGuard` config `cyber`, split `phishing` | see 5b | en | 1,462 | safe/unsafe | — | KEEP (covered by the 5b row) |
| `SetFit/enron_spam` (Enron-Spam, AUEB) | HF card: no licence; AUEB page lists files only, no terms | en | 31,716 / test 2,000 | ham/spam | — | **EXCLUDE.** No licence. Spam part comes from SpamAssassin/honeypots. |
| SpamAssassin public corpus | Project readme: "Copyright for the text in the messages remains with the original senders" | en | — | — | — | **EXCLUDE.** No licence grant. |
| `it4lia/PhishingEmailCuratedDatasets_Cleaned`, `kudzaiprichard/aura-phishing-email-corpus`, `nosadaniel/phishing-email-training-dataset` | `cc-by-4.0` / `mit` from the collectors | en | 181,907 / … | phishing/legit | — | **EXCLUDE.** Real e-mails from CEAS-08, TREC-05/06/07 (licence agreement), Nazario, Enron, Ling-Spam and SpamAssassin; the collector's licence cannot cover them. |
| `ealvaradob/phishing-dataset`, `FredZhang7/all-scam-spam`, `mshenoda/spam-messages`, `intelli-zen/spam_detect` | apache/mit tags | — | — | — | — | **EXCLUDE.** Aggregates of Kaggle, Enron, SpamAssassin, Telegram and YouTube spam sources. |
| `darkknight25/phishing_benign_email_dataset` | `mit`, but README: "for research and educational purposes only" | en | 200 | — | — | **EXCLUDE.** Research only. |
| `Deysi/spam-detection-dataset`, `CloveAI/india-spam-sms` | apache / mit, no provenance | en | 8,175 / 20,010 | — | — | **EXCLUDE (strict).** No provenance, as with S-Labs in the audit. Probably LLM or template text; could move to KEEP if the authors confirm. |
| `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`, `zefang-liu/phishing-email-dataset`, `UniqueData/spam-text-messages-dataset`, `puyang2025/seven-phishing-email-datasets` | GPL / LGPL / NC-ND / other | — | — | — | — | **EXCLUDE** |

### 5d. Prompt injection / jailbreak detection

| HF id (original source) | Licence + where verified | Languages | Train rows | Labels | Eval overlap | Verdict |
|---|---|---|---|---|---|---|
| `deepset/prompt-injections` | Apache-2.0 | en de | — | — | — | **already used** |
| **`reshabhs/SPML_Chatbot_Prompt_Injection`** | HF `mit`. GPT-4-generated system and user prompts; attack techniques from Gandalf (4,168 rows, Lakera MIT). README: "TensorTrust … not licensed for distribution, which precludes us from releasing" (so excluded by the authors). | en | 16,012 | `Prompt injection` 0/1 (12,542 / 3,470), `Degree` 0–10 | Not a suite | **KEEP.** Pair decision: does the user prompt attack this system prompt? |
| **`3nesdeniz/agentic-prompt-injection-5k`** | HF `cc-by-4.0`; "Human-designed attack templates … expanded deterministically"; benign hard negatives | en | 3,923 / 467 / 545 | benign/attack, attack_family, technique, severity | Not a suite | **KEEP.** |
| **`3nesdeniz/turkish-conversation-prompt-injection`** | HF `cc-by-4.0`; "All examples are synthetic … `synthetic_curated`" | tr | 530 / 100 / 120 | label 0/1, category, attack_family, source_context | Not a suite | **KEEP (small).** The only priority-language prompt-injection set found besides deepset. |
| **`nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1`** | HF `cc-by-4.0`; "This dataset is ready for commercial or non-commercial uses"; "fully synthetic with no seed data" | en | 1,272 | attack_category, injection_vector (all contain injections) | Not a suite | **KEEP.** Indirect injection in tool outputs; negatives can be built by removing the injected field. |
| **`Lakera/gandalf_ignore_instructions`** | HF `mit` ("distributed under the MIT License"); written by players of Lakera's own game | en | 777 / 111 / 112 | positives only (+ similarity) | Not a suite | **KEEP (caution).** Released by the platform owner, as with Civil Comments. Positives only. |
| **`Lakera/mosscap_prompt_injection`** | HF `mit`; player prompts to Lakera's Mosscap game, ChatGPT answers | en (mostly) | 223,533 / 27,683 / 27,729 | `level` 1–8; all rows are attack attempts | Not a suite | **KEEP (caution).** No benign/attack label, so use prompts as positives or level as difficulty. Drop `answer`/`raw_answer` (ChatGPT output). |
| `thu-coai/Safety-Prompts` instruction-attack part | see 5b | zh | 30k | goal hijacking, prompt leaking, role-play … | — | KEEP (covered by the 5b row) |
| `xTRam1/safe-guard-prompt-injection` | no licence | en | 8,236 | 0/1 | — | **EXCLUDE.** No licence; seeds include jackhhao and awesome-chatgpt-prompts. |
| `jackhhao/jailbreak-classification`, `TrustAIRLab/in-the-wild-jailbreak-prompts`, `walledai/JailbreakHub`, `rubend18/ChatGPT-Jailbreak-Prompts` | apache / mit / none | en | — | jailbreak/benign | — | **EXCLUDE.** Scraped from jailbreakchat.com, Reddit and Discord (platform text). |
| `Necent/llm-jailbreak-prompt-injection-dataset` | HF `mit`; README: "Each underlying source retains its original license" (30+ sources, including NC ones) | 27 | 1M+ | — | — | **EXCLUDE.** Aggregate with inherited licences. |
| `yanismiraoui/prompt_injections` | `apache-2.0`, no provenance | en fr de es pt it ro | 1,034 | positives only | — | **EXCLUDE (strict).** No provenance, as with S-Labs. |
| `hirundo-io/prompt-injection-purple-llama` (Meta PurpleLlama CyberSecEval) | no HF licence; GitHub `meta-llama/PurpleLlama` spdx NOASSERTION | en | 251 | — | — | **EXCLUDE.** |
| `rogue-security/prompt-injections-benchmark`, `OnerAYTAS/Turkish_prompt_injection_jailbreak_dataset`, `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT` | cc-by-nc-4.0 / cc-by-nc-sa-4.0 / cc-by-sa-4.0 | — | — | — | — | **EXCLUDE** |
| `neuralchemy/Prompt-injection-dataset`, `S-Labs/prompt-injection-dataset`, PromptShield | — | — | — | — | — | **already excluded** (audit) |

### Counts

- **KEEP (new):** 35 in total.
  - Toxicity/hate: 5 (DynaHate, GAHD-filtered, oasst2, ToxicCommons, toxic_conversations_50k as a duplicate).
  - Safety/moderation: 19.
  - Spam/phishing: 5.
  - Prompt injection: 6.
- **Already used:** civil_comments, HatemojiBuild, hh-rlhf, Nemotron Safety Guard v3, IndicGuard, sms_spam, deepset, agent_action_safety, shell-safety-v2.
- **EXCLUDE:** about 50 repos, listed in the tables above.

### Gaps in category 5 (synthetic data needed)

1. **Human-labelled toxicity and hate in comment style, outside en and de.** fr es it pt nl pl tr ru ar hi ja zh have no clean corpus: every one found is tweets, Reddit, YouTube or Wikipedia-talk text, or research-only. Clean multilingual coverage is limited to:
   - Aya red-teaming (harmful prompts: fr es ru ar hi);
   - oasst2 moderation votes (es ru zh de fr pt, sparse positives);
   - ToxicCommons (LLM labels on historical text: fr es de pl nl pt it);
   - Nemotron Safety Guard v3 (already used).

   Needed: synthetic or translated hateful and non-hateful comment pairs with targets, including contrast pairs in the DynaHate/GAHD style, for all 14 languages.
2. **Spam, phishing and scam in any language other than en and zh.** Every real e-mail or SMS corpus fails the licence rule. The zh coverage is synthetic only.
3. **Prompt injection outside en and tr.** No clean de fr es it pt nl pl ru ar hi ja sets; zh only via thu-coai instruction attacks.
4. **Harassment and cyberbullying** as a distinct label, and a **benign-comment** negative pool, in the 14 languages.
5. **Response-harm labels (assistant output) outside English.** Only Nemotron v3, which is already used and prompt-heavy.


## Part E: categories 6 (reading comprehension), 8 (topic) and 9 (intent / dialogue act)


Checked 2026-09-27 on pop-os. Method: HF metadata via `HfApi().dataset_info` + README + datasets-server
`/size` and `/first-rows` (scratchpad/hfcheck.py); upstream via GitHub API (`license.spdx_id`, README/LICENSE
files), project pages (WebFetch) and, where a page was unreachable, the search-engine snippet of that page
(marked "snippet"). No data downloaded except QuALITY train (11 MB, to read the per-article `license` field) and
NLU++ fold files (a few KB). Policy as in DATA_LICENSES.md: text provenance counts (Wikipedia = CC-BY-SA,
scraped news/exams/platform text out, collector "CC0/MIT" over third-party text out).

Eval suites touched by these categories: Belebele (FLORES passages; **351 of the 900 eng_Latn test passages
link to en.wikinews.org**, 276 wikibooks, 273 wikivoyage — counted from `link` field), SIB-200 (FLORES
sentences, same sources), AG News, HWU64 (NLU-Evaluation-Data; MASSIVE descends from SLURP/HWU), MASSIVE,
Banking77.

### 6. Reading comprehension (MC / boolean over a passage)

| HF id (original source) | Licence + where verified | Lang | Train | Labels | Eval overlap | Verdict |
|---|---|---|---:|---|---|---|
| theatticusproject/cuad-qa (Atticus Project CUAD v1; contracts from SEC EDGAR) | HF card `cc-by-4.0`; atticusprojectai.org/cuad: "41 clause types CC BY 4.0" | en | 22,450 (test 4,182) | per contract×41 clause categories: answer span or empty → yes/no "does the contract contain X" | none | **KEEP** — expert labels, public EDGAR filings (same basis as lex_glue/ledgar, already kept). HF repo is a loading script; data.zip from GitHub. Contracts are long: chunk to clause windows. |
| theatticusproject/maud (Atticus MAUD; merger agreements from EDGAR) | HF card `cc-by-4.0`; atticusprojectai.org/maud: "92 questions CC BY 4.0" | en | 25,827 (val 6,753, test 6,651) | MC answer per (question, subquestion) over agreement excerpt (`label`/`answer`) | none | **KEEP** — expert-annotated MC over passages. |
| emozilla/quality mirror → nyu-mll/quality (QuALITY v1.0.1) | project page nyu-mll.github.io/quality: "QuALITY is distributed under a CC BY 4.0 License"; per-article `license` field in train file: Gutenberg 236 sets ("for the use of anyone anywhere in the United States"), Slate/OANC 44 (OANC: free incl. commercial, snippet of anc.org), CC BY 4.0 20; mirror card has no licence → use GitHub files | en | 2,523 questions / 300 article-sets (dev 2,086) | 4-way MC `gold_label` | none | **KEEP** — long passages (~5k tokens): chunk or use `hard`/speed-validation subsets. Gutenberg items are US public domain (same basis as kept poem_sentiment). |
| WorkInTheDark/FairytaleQA (uci-soe/FairytaleQAData; 278 Project Gutenberg stories) | HF card `apache-2.0`; GitHub uci-soe/FairytaleQAData spdx `Apache-2.0`; README: "278 children's stories from Project Gutenberg" | en | 8,548 (val 1,025, test 1,007) | `ex-or-im` (explicit / implicit = is the answer stated in the section), `attribute` (character, setting, action, feeling, causal relationship, outcome resolution, prediction); free-text answers | none | **KEEP** — expert-written; decision tasks are explicit-vs-implicit and question-type; answer verification needs constructed negatives. |
| allenai/openbookqa | HF card `unknown` ("More Information Needed"); GitHub allenai/OpenBookQA `Apache-2.0` covers the experiment code repo; allenai.org/data page redirects to the HF card | en | 4,957 | 4-way MC (+`fact1`) | none | EXCLUDE (uncertain) — no licence stated for the data itself; owner could accept on the repo licence. Also not passage RC. |
| community-datasets/quarel, allenai/wiqa | cards: licence "[More Information Needed]"; no upstream GitHub repo; allenai.org pages redirect to Semantic Scholar / HF card | en | 1,941 / ~30k | 2-way MC / more-less-no effect | none | EXCLUDE — missing licence. |
| tasksource/proofwriter (AI2 ProofWriter V2020.12.3) | mirror card no licence; zip README (read via HTTP range) has no licence; allenai/ruletaker GitHub `Apache-2.0` | en | 585,552 | True/False/Unknown over rule text | none | EXCLUDE — missing licence for ProofWriter itself; the same task is already covered by `ruletaker` (Apache-2.0, in mixture). |
| allenai/ropes | card `cc-by-4.0`, but README: background passages "scraped ... from science textbooks and Wikipedia" (sample text is CK-12 style "Figure below") | en | 10,924 | extractive | none | EXCLUDE — passage provenance (Wikipedia CC-BY-SA, CK-12 CC-BY-NC). |
| qiaojin/PubMedQA (pubmedqa/pubmedqa) | card `mit`, GitHub `MIT`; contexts are PubMed abstracts (publisher copyright) | en | 1,000 labelled / 211k artificial | yes/no/maybe | none | EXCLUDE — MIT covers annotations, not the abstracts. |
| google/boolq; aps/super_glue boolq/multirc | card `cc-by-sa-3.0` (BoolQ); super_glue card `other` ("research context") | en | 9,427 / 27,243 | yes/no | none | EXCLUDE — SA / research-only; Wikipedia passages. |
| deepset/germanquad; google-research-datasets/tydiqa; coastalcph/tydi_xor_rc; rajpurkar/squad_v2; google/xquad | cards cc-by-4.0 / apache-2.0 / mit / cc-by-sa-4.0 / cc-by-sa-4.0; all passages are Wikipedia paragraphs (README of each) | de; 11 langs; en; many | 11.5k / 167k / … | extractive / answerability | none | EXCLUDE — Wikipedia CC-BY-SA text (same rule as CONDAQA). |
| sagnikrayc/mctest (MCTest) | card `other`, README: "Microsoft Research License Agreement" | en | 1,480 | 4-way MC | none | EXCLUDE — MSR licence (non-commercial). |
| nlpdata/c3 (C³) | GitHub license.txt: "intended for non-commercial research purpose only" | zh | — | MC | none | EXCLUDE — NC. |
| tasksource/reclor; LogiQA (datatune/LogiQA2.0, baber/logiqa2) | reclor card `other` (research); logiqa2 cards `mit` / `cc-by-sa-4.0`, text from Chinese civil-service exams | en/zh | — | MC | none | EXCLUDE — research-only / SA / exam provenance. |
| allenai/qasper, allenai/qasper-yesno | card `cc-by-4.0`; passages are full arXiv/S2ORC NLP papers (per-paper licences, arXiv default = non-exclusive distribution only) | en | 319 yes/no | yes/no | none | EXCLUDE — full-text licence not permissive. |
| deepmind/narrativeqa; Maluuba/newsqa; ibm-research/duorc; allenai/quac | apache-2.0 / mit / mit / mit cards; texts = movie scripts, CNN articles, Wikipedia/IMDb plots, Wikipedia | en | — | free-form | none | EXCLUDE — text provenance (and not decision-type). |
| nguha/legalbench (+ mteb/*LegalBench*) | card `cc-by-4.0` but README: "LegalBench tasks are subject to different licenses" | en | tiny train (few-shot) | many | none | EXCLUDE — mixed per-task licences, benchmark; CUAD/MAUD used directly instead. |
| CohereLabs/Global-MMLU, openai/MMMLU, CohereLabs/include-base-44 | cards apache-2.0 / mit / apache-2.0 | 42 / 14 / 44 langs | test (+dev) only | 4-way MC | none (benchmarks) | EXCLUDE — not passage RC; MMLU/INCLUDE questions are collected exam material (same rule as medmcqa/cloth); test-only benchmarks. |
| facebook/belebele | card `cc-by-sa-4.0` | 122 | test only | MC | **is an eval suite** | EXCLUDE — eval + SA. |

**Category 6 result: 4 KEEP, all English.** Multilingual passage RC with clean text: **none found** (every
multilingual set uses Wikipedia, FLORES or exam text). → synthetic gap. Already in mixture and relevant:
ruletaker, babi_nli, spartqa, social_i_qa, quartz, qasc.

### 8. Topic / news / document classification

| HF id (original source) | Licence + where verified | Lang | Train | Labels | Eval overlap | Verdict |
|---|---|---|---:|---|---|---|
| Fumika/Wikinews-multilingual (PrimerAI/primer-research/wikinews; Wikinews) | HF card `cc-by-2.5`; Wikinews:Copyright: "after September 25, 2005 ... Creative Commons Attribution 2.5", earlier public domain, after 2024-12-16 CC BY 4.0 | 33: en es fr de pt pl it zh ru ja nl tr ar fa … | 15,200 (5,240 en + 9,960 linked) | English Wikinews categories per `pageid` (Politics and conflicts, Crime and law, Economy and business, Disasters and accidents, Science and technology, Sports, Health, Culture and entertainment, Environment, …) + place/date categories to filter | **Belebele/SIB-200: FLORES was drawn from en.wikinews.org (351/900 Belebele passages link there)** | **KEEP (train with dedup)** — drop any article whose URL is a Belebele `link` or whose text contains a FLORES dev/devtest sentence. Only multilingual clean topic set found. |
| gfissore/arxiv-abstracts-2021 (arXiv metadata) | HF card `cc0-1.0`; arXiv API ToU: "descriptive metadata ... CC0 1.0 ... includes ... title, abstract, authors, identifiers, and classification terms" | en | 1,999,486 | arXiv `categories` (primary = first) | none | **KEEP** — title+abstract only (never full text). |
| NortheasternUniversity/big_patent (US patents) | HF card `cc-by-4.0`; USPTO patents | en | 1,207,222 | CPC section = config a,b,c,d,e,f,g,h,y (9 classes) | none | **KEEP** — use abstract (+ start of description). |
| ddrg/super_eurlex (EUR-Lex) | HF card `mit` (body: "License: [More Information Needed]"); EUR-Lex legal notice (snippet): "Reuse of the EUR-Lex data for commercial or non-commercial purposes is authorised provided the source is acknowledged" (Decision 2011/833/EU; editorial content CC BY 4.0) | 24 EU langs incl. de fr es it pt nl pl en | ~4.6M docs (meta_data/3.parquet alone 220,148) | `directory_code`, `subject_matter`, `eurovoc` (multi-label) | none | **KEEP** — attribution "© European Union, https://eur-lex.europa.eu"; text files are GB-sized, stream per sector/language. Replaces MultiEURLEX (SA). |
| joelniklaus/covid19_emergency_event (legislation of 8 countries) | HF card `cc0-1.0`; texts are official legal acts | en fr hu it nb nl pl | 3,312 | 8 boolean measure types (multi-label) | none | **KEEP** |
| lyon-nlp/clustering-hal-s2s (HAL open archive titles) | HF card `apache-2.0`; HAL doc (doc.hal.science, conditions de réutilisation): "la licence CC-0 s'appliquera ... aux métadonnées de l'ensemble des dépôts" | fr (mostly) | 85,375 (single `test` split; not our eval) | HAL `domain` (shs, math, info, sdv, spi, …) | none (is an MTEB-fr eval set) | **KEEP** — titles only. |
| itacasehold/itacasehold (giustizia-amministrativa.it massime) | HF card `apache-2.0` ("The data sets are distributed under the Apache 2.0 License"); official court documents | it | 792 | `materia` (subject area, e.g. Processo amministrativo, Edilizia, Ambiente) | none | **KEEP** (small). |
| dvgodoy/CUAD_v1_Contract_Understanding_clause_classification (CUAD) | HF card `cc-by-4.0`; CUAD CC BY 4.0 (Atticus site) | en | 13,155 | 41 clause types | none | **KEEP** — clause-type classification (document-section topic); same source as CUAD-QA, dedup across the two. |
| coastalcph/multi_eurlex, nlpaueb/multi_eurlex, mteb/eurlex-multilingual | cards `cc-by-sa-4.0` | 23 | 55k | EuroVoc | none | EXCLUDE — SA (use super_eurlex instead). |
| coastalcph/lex_glue eurlex / scotus / ecthr | card `cc-by-4.0`, licensing section "More Information Needed"; eurlex = MultiEURLEX (Chalkidis 2021b, SA); scotus labels from Supreme Court Database — CC BY-NC 3.0 (snippet of scdb.wustl.edu; site 403); ECtHR HUDOC terms not verified | en | 55k / 5k / 9k | EuroVoc / issue area / articles | none | EXCLUDE — SA / NC / unverified. (ledgar, case_hold already kept.) |
| rcds/swiss_judgment_prediction, rcds/swiss_law_area_prediction, mteb/SwissJudgementClassification | rcds cards `cc-by-sa-4.0`; mteb mirror says cc-by-4.0 (conflicts) | de fr it | 60k+ | approval/dismissal, law area | none | EXCLUDE — SA. |
| Filippo/osdg_cd, albertmartinez/OSDG (OSDG Community Dataset) | cards `cc-by-4.0` / `mit`; texts are excerpts of third-party reports and journal abstracts | en | 34,420 | 16 SDGs | none | EXCLUDE (borderline) — collector licence over third-party text (headline_cause precedent). |
| rafalposwiata/plsc, mteb/Plsc* (Polish Library of Science) | cards `cc0-1.0`; abstracts from Biblioteka Nauki journals; no rights-holder CC0 statement found | pl | 159,767 | scientific fields / disciplines | none | EXCLUDE (borderline) — collector CC0 over third-party abstracts. Owner could re-check Biblioteka Nauki metadata terms. |
| antoinelb7/alloprof, lyon-nlp/alloprof | card `mit` / `apache-2.0` but citation block: "copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}" | fr | — | school subject | none | EXCLUDE — conflicting NC-SA. |
| ccdv/arxiv-classification, ccdv/patent-classification | no licence on cards; arxiv set = full paper text | en | 28k / 25k | 11 arXiv / 9 CPC | none | EXCLUDE — missing licence (use arXiv abstracts / BIGPATENT directly). |
| fancyzhx/ag_news (+ mteb/NewsClassification) | card `unknown` | en | 120k | 4 | **AG News is an eval suite** | EXCLUDE — eval + unknown. |
| Davlan/sib200 | card `cc-by-sa-4.0` | 205 | 701 | 7 topics | **is an eval suite** | EXCLUDE. |
| fancyzhx/dbpedia_14 (+ DeveloperOats/DBPedia_Classes "cc0") | `cc-by-sa-3.0`; CC0 repack conflicts | en | 560k | 14 | none | EXCLUDE — SA. |
| community-datasets/gnad10 (10kGNAD) | card `cc-by-nc-sa-4.0`; GitHub tblock/10kGNAD README: dataset CC BY-NC-SA 4.0 | de | 9,245 | 9 | none | EXCLUDE — NC-SA. |
| facebookresearch/MLDoc | README: subset of Reuters RCV1/RCV2 (Reuters licence required) | 8 | — | 4 | none | EXCLUDE — restricted. |
| SetFit/20_newsgroups, SetFit/bbc-news, community-datasets/yahoo_answers_topics, clue/clue (tnews, iflytek) | cards: none / none / `unknown` / `unknown` | en/zh | — | — | none | EXCLUDE — missing licence, scraped text. |
| mteb/HeadlineClassification (ru), mteb/SpanishNewsClassification, thegauravgiri/nepali-news-dataset, aisbergpublicorganization/telegram-news-ua-dataset, zeroshot/twitter-financial-news-topic | mit / mit / mit / cc-by-4.0 / mit cards over scraped news / Telegram / tweets | ru es ne uk en | — | — | none | EXCLUDE — scraped news / platform text. |
| CLS / CSL (mteb/CLSClustering*) | `apache-2.0` card; Chinese core-journal titles/abstracts | zh | — | discipline | none | EXCLUDE (borderline) — journal abstract copyright not addressed. |

**Category 8 result: 8 KEEP** (1 multilingual incl. ja/zh/ru/ar/tr but small; EU languages via EUR-Lex/COVID;
fr via HAL; it via ITA-CaseHold; rest en). **Gap:** topic data for ja, zh, ar, hi, tr, ru beyond ~a few hundred
Wikinews articles each. Buildable clean sources: full Wikinews dumps per language edition (CC BY 2.5, own
categories), Global Voices (CC BY 3.0, multilingual, topic tags), VOA (US federal work, public domain; exclude
AP/Reuters wire items).

### 9. Intent / dialogue act (beyond MASSIVE, Banking77)

| HF id (original source) | Licence + where verified | Lang | Train | Labels | Eval overlap | Verdict |
|---|---|---|---:|---|---|---|
| NLU++ (github PolyAI-LDN/task-specific-datasets/nlupp; no HF mirror) | GitHub spdx `CC-BY-4.0` | en | 3,080 (banking 2,071, hotels 1,009; 20 folds) | multi-label intents (62, e.g. repeat, request_info, how_long, pin, arrival, new) + slots | Banking77 is a different PolyAI set; dedup vs Banking77 test | **KEEP** |
| uoe-nlp/multi3-nlu (Multi3NLU++) | HF card `cc-by-4.0`; professional translation of NLU++ | am, mr, tr, es (+ en source) | ~3,080 per language (fold files) | same multi-label intents | dedup vs Banking77 test | **KEEP** — tr and es are priority languages. |
| google-research-datasets/taskmaster1/2/3 (github …/Taskmaster TM-1…TM-4) | HF card `cc-by-4.0`; each TM-x README: "made available under the Creative Commons Attribution 4.0 License" | en | >55,000 dialogs (repo README; TM-2 17,289) | domain / `instruction_id` per dialog, slot segments | none | **KEEP** — HF repos are loading scripts; take JSON from GitHub. Decision tasks: domain / service of a user turn, dialogue-state checks. |
| pfb30/multi_woz_v22 (budzianowski/multiwoz) | HF card `apache-2.0`; GitHub spdx `MIT` | en | 8,437 dialogs (≈10.4k total) | per-turn dialogue acts (Inform, Request, Book, …), active intent (find_hotel, book_train, …), domain | none | **KEEP** — loading script; use GitHub MultiWOZ_2.2 JSON. |
| kchawla123/casino (kushalchawla/CaSiNo) | HF card `cc-by-4.0`; GitHub spdx `CC-BY-4.0` | en | 1,030 dialogs (396 strategy-annotated, ~4.6k utterances) | negotiation strategies: small-talk, self-need, other-need, elicit-pref, vouch-fair, promote-coordination, no-need, uv-part, showing-empathy | none | **KEEP** — dialogue-act/strategy labels by experts. |
| ConvLab/crosswoz (thu-coai/CrossWOZ) | HF card `apache-2.0`; GitHub spdx `Apache-2.0` | zh | 5,012 dialogs (6k total, 102k utterances) | dialogue acts, domains (hotel, restaurant, attraction, metro, taxi) | none | **KEEP** — data.zip (16 MB) in the HF repo. |
| DeepPavlov/BiToD (HLTCHKUST/BiToD) | mirror card has no licence; upstream GitHub spdx `Apache-2.0` | en, zh | 28,946 en / 28,873 zh turn rows | user/system acts, intents, dialogue state | none | **KEEP (use upstream files)**. |
| OpenVoiceOS/hass-intent-templates (home-assistant/intents grammars) | HF card `apache-2.0`; upstream GitHub spdx `CC-BY-4.0` | ~60 incl. de 11,728, es 9,084, en 3,804, fr 3,548, ar 2,038, nl, pl, pt, ru, tr, it … (ja/zh/hi small or absent) | templates (expand) | Home Assistant intent ids (HassTurnOn, HassLightSet, timers, broadcast …) + slots | smart-home commands can coincide with MASSIVE iot/HWU64 → exact-dedup vs both tests | **KEEP** — human-written community grammars; expand templates, cap per intent. |
| TigreGotico/yes-no-multilingual; OpenVoiceOS/yes_no_answers | HF cards `apache-2.0`; README: "All utterances were generated directly by a large language model (Claude)" | 43 incl. all 14 priority langs | 8,600 / yesno_multilingual.csv 352 KB | yes / no / unclear (28 subtypes Y1–Y10, N1–N10, C1–C8) | none | **KEEP (LLM-generated, creator licence)** — answer-polarity dialogue act. |
| clips/VaccinChatNL | HF card `cc-by-4.0` | nl | 10,542 (val 1,171, test 1,170) | ~180 intents (chitchat + vaccine FAQ) from real chatbot users | none | **KEEP** |
| NABA-AI/LUB-Saudi-Arabic-Intent | HF card `cc-by-4.0`; README: "All examples in this dataset are synthetically generated" (generator not named) | ar (Saudi) | 1,500 | Intent_Type, emotion, sarcasm, polarity, complaint, escalation risk | none | **KEEP (synthetic, small)** — also useful for cat. 3/10. |
| Process-Venue/IntentClassification_Dataset_for_AI_Assistant_Prompt_Routing_Hindi | HF card `apache-2.0`; two human annotators; sentence origin undocumented | hi | 5,000 | 6 routing intents (rewrite, creative, code, information, content, translation) | none | **KEEP (uncertain provenance, like prompt-injections in the audit)** |
| masakhane/InjongoIntent | HF card `apache-2.0` (mteb mirror cc-by-4.0); `source_datasets: clinc/clinc_oos` | en + 16 African | 2,240/lang, en 1,779 | 40 CLINC-style intents | dedup vs clinc_oos (already trained) | KEEP (low priority, non-priority languages) |
| GoktugD/turkish-intent-classification-1m | HF card `cc0-1.0`; deterministic template generator | tr | 980,000 | e-commerce intents (siparis_durumu, iptal_talebi, …) | none | KEEP (low quality: strip the generated context prefix; sample a few k) |
| jpcorb20/multidogo (awslabs MultiDoGo) | HF card: "Community Data License Agreement – Permissive, Version 1.0"; GitHub README: "licensed under the CDLA Permissive License" | en | 170,592 utterances | intents + dialogue acts (openinggreeting, bookflight, contentonly …) | none | **HOLD** — CDLA-Permissive places no restriction on trained models, but it is not on the allowlist. Owner decision. |
| amazon-research/dstc11-track2-intent-induction | repo `Apache-2.0`; data dir `dstc11/LICENSE.md`: "Community Data License Agreement - Permissive - Version 2.0" (§3 "No Restrictions on Results") | en | dev insurance ≈11 MB dialogues; test banking/finance | customer intents on customer turns | Banking domain ≠ Banking77 texts; dedup anyway | **HOLD** — as above (CDLA-Permissive-2.0). |
| cartesinus/leyzer-fedcsis (Leyzer) | HF card `cc-by-4.0`; upstream GitHub LICENSE: "Attribution-NonCommercial-NoDerivatives 4.0 International" | en pl es | ~20k | 186 intents | none | EXCLUDE — card conflicts with upstream NC-ND. |
| GEM/RiSAWOZ (terryqj0107/RiSAWOZ) | GEM card `cc-by-4.0`; upstream README: "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)" | zh | 10,000 | acts/states | none | EXCLUDE — NC (also X-RiSAWOZ derivatives). |
| SEACrowd/xsid (mainlp/xsid) | upstream GitHub spdx `CC-BY-SA-4.0`; SNIPS+mTOP derivative | 13 | — | intents | treated as derivative in eval_zeroshot | EXCLUDE — SA. |
| google-research-datasets/schema_guided_dstc8; nu-dialogue/jmultiwoz; mtop | cards `cc-by-sa-4.0` | en / ja / 6 | — | acts / intents | none | EXCLUDE — SA. |
| SLURP (pswietojanski/slurp) | README: "Textual data ... released on CC BY 4.0" | en | 11.5k | scenario/intent | **MASSIVE is the localisation of SLURP → en texts = MASSIVE en (eval)** | EXCLUDE — duplicates MASSIVE, no gain. |
| deutsche-telekom/NLU-Evaluation-Data-en-de, deutsche-telekom/NLU-few-shot-benchmark-en-de, mteb/RuNLUIntentClassification, DeepPavlov/hwu64(-translated) | cards `cc-by-4.0` / none; all are NLU-Evaluation-Data (HWU64) or translations | en de ru | 1,280–25k | HWU64 intents | **HWU64 is an eval suite** | EXCLUDE — eval family. |
| salioneme/persian-banking77 and other banking77 translations | `cc-by-4.0` | fa | — | 77 | **Banking77 eval** | EXCLUDE — eval family. |
| bitext/Bitext-customer-support-llm-chatbot-training-dataset (+ retail, events) | card `cdla-sharing-1.0` | en | ~27k | 27 intents | none | EXCLUDE — Sharing variant, not allowlisted; template/LLM text. |
| NathanDuran/MRDA-Corpus, NathanDuran/Switchboard-Corpus | GitHub spdx `GPL-3.0`; Switchboard is LDC | en | — | dialogue acts | none | EXCLUDE — GPL / LDC. |
| Maluuba/frames, hellohaptik/HINT3, AmFamMLTeam/ACID, ibm-research/vira-dialog-acts-live, stanfordnlp/craigslist_bargains, microsoft/meta_woz, SEACrowd/globalwoz, spawn99/PersuasionForGood | GitHub NOASSERTION / NOASSERTION / none / card none / `unknown` / `other` / `unknown` / card `mit` with unverified upstream and no act labels in the mirror | en (+multi) | — | acts/intents | none | EXCLUDE — missing or unverified licence. |
| tuetschek/atis, MultiATIS++ | no card licence; ATIS is LDC | en (+8) | — | intents | none | EXCLUDE — LDC. |

**Category 9 result: 14 KEEP rows (15 JSON items; the two yes/no sets are listed separately) + 2 HOLD.** Priority-language coverage: en (many), zh (CrossWOZ, BiToD), es/tr
(Multi3NLU++), de/fr/es/it/nl/pl/pt/ru/tr/ar (HA templates, smart-home only), nl (VaccinChat), ar (LUB, small),
hi (routing, small), all 14 (yes/no, LLM). **Gap:** general-domain human intent data in ja, hi, ar, ru, de, fr,
it, pt, pl outside smart-home; dialogue-act data outside en/zh.

### Gaps for synthetic data (categories 6, 8, 9)

1. **Reading comprehension, all non-English languages** — no clean passage-MC/boolean set exists; all use Wikipedia/FLORES/exams. Generate passages + MC/yes-no questions per language (or translate the clean English sets QuALITY/FairytaleQA/CUAD/MAUD with a licence-clean MT model).
2. **English RC outside legal/fiction** — clean English RC is legal (CUAD, MAUD) or fiction (QuALITY, FairytaleQA); science/news/everyday passages missing.
3. **Topic classification in ja, zh, ar, hi, tr, ru** — only a few hundred Wikinews articles each; build from Wikinews dumps / Global Voices / VOA or synthesise.
4. **News-style topic taxonomy (AG-News-like) with train data** — none clean; Wikinews categories are the closest; must dedup against Belebele/SIB (FLORES from Wikinews).
5. **Intent in ja/hi/ar/ru/pt/it/pl/fr/de outside smart-home**, and **dialogue acts outside en/zh** — synthesise, or accept the two CDLA-Permissive sets if the owner extends the allowlist.


# Machine-readable KEEP list

Fields: id, config, split, licence, languages, category (numbered; two categories separated by `; `), label_field, text_fields, test_overlap_note, source_part (which research part verified it; `parent` = verified here). Items marked conditional/uncertain/caution in the tables carry that word in `test_overlap_note` or `licence`.

```json
[
 {
  "id": "google-research-datasets/poem_sentiment",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "1-sentiment",
  "label_field": "label",
  "text_fields": [
   "verse_text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts); already in mixture v5",
  "source_part": "A"
 },
 {
  "id": "ai4bharat/IndicSentiment",
  "config": "<lang json files>",
  "split": "validation+test",
  "licence": "CC0-1.0 (AI4Bharat/IndicBERT README; HF card has none)",
  "languages": [
   "en",
   "hi",
   "bn",
   "mr",
   "ta",
   "te",
   "ur",
   "gu",
   "kn",
   "ml",
   "or",
   "pa",
   "as",
   "bd"
  ],
  "category": "1-sentiment",
  "label_field": "LABEL",
  "text_fields": [
   "ENGLISH REVIEW",
   "INDIC REVIEW",
   "ASPECT COMBO"
  ],
  "test_overlap_note": "not an eval suite: both splits usable for training; no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "community-datasets/re_dial",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "1-sentiment",
  "label_field": "respondentQuestions/initiatorQuestions.liked (0 no,1 yes,2 did not say)",
  "text_fields": [
   "messages",
   "movieMentions"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "tanaos/synthetic-sentiment-analysis-dataset-v1",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "1-sentiment",
  "label_field": "labels",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "Novora/Tri-Class-Sentiment-Synthetic",
  "config": "default",
  "split": "train",
  "licence": "CC0-1.0",
  "languages": [
   "en"
  ],
  "category": "1-sentiment",
  "label_field": "sentiment",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "sutro/synthetic-product-reviews-20k",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "1-sentiment",
  "label_field": "rating_out_of_5",
  "text_fields": [
   "review_title",
   "review_text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "YiMeng-SYSU/chinese-logic-sentiment-dataset",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "zh"
  ],
  "category": "1-sentiment",
  "label_field": "label",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "dhruv0808/indic_sentiment_analyzer",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en",
   "hi",
   "te",
   "ta",
   "kn",
   "or",
   "bn",
   "gu",
   "pa",
   "ml",
   "mr",
   "as"
  ],
  "category": "1-sentiment",
  "label_field": "Label",
  "text_fields": [
   "Sentence"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts); contains ~1k/lang IndicSentiment rows -> dedup if both used",
  "source_part": "A"
 },
 {
  "id": "Kenshiii/synthetic-product-reviews",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "1-sentiment",
  "label_field": "sentiment; attributes_normalized",
  "text_fields": [
   "review_text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "KhiredNetworks/synthetic-product-reviews",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "1-sentiment",
  "label_field": "sentiment; rating",
  "text_fields": [
   "review_title",
   "review_text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "RichardSakaguchiMS/brazilian-customer-service-conversations",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "pt"
  ],
  "category": "3-complaint; 1-sentiment",
  "label_field": "metadata.sentiment; metadata.intent",
  "text_fields": [
   "messages"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "Adilbai/kz-gov-complaints-data-kz-ru",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "ru",
   "kk"
  ],
  "category": "3-complaint; 1-sentiment",
  "label_field": "category; urgency_level; sentiment",
  "text_fields": [
   "text_ru",
   "text_kz"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "leonvanbokhorst/synthetic-complaints-v2",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "3-complaint; 1-sentiment",
  "label_field": "topic; style; sentiment",
  "text_fields": [
   "output"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "github:asappresearch/abcd",
  "config": "data/abcd_v1.1.json.gz",
  "split": "train",
  "licence": "MIT (GitHub LICENSE)",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "scenario.flow (10); scenario.subflow (55)",
  "text_fields": [
   "original (customer turns)"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "tasksource/it-support-tickets",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0 (Zenodo 7648117)",
  "languages": [
   "en",
   "de",
   "pt",
   "es"
  ],
  "category": "3-complaint",
  "label_field": "label",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts); already in mixture v5",
  "source_part": "A"
 },
 {
  "id": "tasksource/help-desk-tickets",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0 (Mendeley btm76zndnt v3)",
  "languages": [
   "en",
   "mixed"
  ],
  "category": "3-complaint; 10-urgency",
  "label_field": "issue_priority; issue_type; issue_resolution",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts) | real tickets (Mendeley CC BY 4.0), 357 rows; no eval overlap",
  "source_part": "AB"
 },
 {
  "id": "PolyAI/minds14",
  "config": "all",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "cs",
   "de",
   "en",
   "es",
   "fr",
   "it",
   "ko",
   "nl",
   "pl",
   "pt",
   "ru",
   "zh"
  ],
  "category": "3-complaint",
  "label_field": "intent_class",
  "text_fields": [
   "transcription",
   "english_transcription"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts); already in extra-v1",
  "source_part": "A"
 },
 {
  "id": "BEE-spoke-data/consumer-finance-complaints",
  "config": "has-text",
  "split": "train",
  "licence": "CC0-1.0 card; CFPB US federal data, narratives published with consumer opt-in consent",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "Product; Issue; Sub-issue; Company response to consumer",
  "text_fields": [
   "Consumer complaint narrative"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "liri-uzh/cfpb-complaints-mini",
  "config": "default",
  "split": "train",
  "licence": "CC0-1.0 (CFPB public domain)",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "label",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "vic35get/nhtsa_complaints_dataset",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0 card; NHTSA US federal data",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "components",
  "text_fields": [
   "summary"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "alaminxpro/university-students-complaints",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "Category; severity; Aspects",
  "text_fields": [
   "Complaint_Description"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "hblim/customer-complaints",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "label",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "cngchis/Support-Ticket-Router-12K-Cleaned",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "label",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "OrSabbach/food-delivery-support-tickets",
  "config": "data/dataset.parquet",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "category",
  "text_fields": [
   "customer_message"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "Console-AI/IT-helpdesk-synthetic-tickets",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "3-complaint; 10-urgency",
  "label_field": "priority; category",
  "text_fields": [
   "subject",
   "description"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts) | uncertain: empty README, generator undocumented; no eval overlap",
  "source_part": "AB"
 },
 {
  "id": "s2pidape/support-ticket-dataset",
  "config": "humans_only",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "category; issue_type_id",
  "text_fields": [
   "issue_description"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "adiprog14/lingrow-support-tickets",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "3-complaint",
  "label_field": "error_category; error_code",
  "text_fields": [
   "customer_message"
  ],
  "test_overlap_note": "no eval-suite source; still exact-dedup vs all eval test splits (build_mixture test_texts)",
  "source_part": "A"
 },
 {
  "id": "brighter-dataset/BRIGHTER-emotion-categories",
  "config": "hin",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "hi"
  ],
  "category": "2-emotion",
  "label_field": "emotions (multi-label; also anger/disgust/fear/joy/sadness/surprise 0/1)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; only hin/mar configs are clean (others Reddit/Twitter/YouTube/Weibo/news); emotion training ends zero-shot status of DAIR Emotion/GoEmotions",
  "source_part": "B"
 },
 {
  "id": "brighter-dataset/BRIGHTER-emotion-categories",
  "config": "mar",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "mr"
  ],
  "category": "2-emotion",
  "label_field": "emotions",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "as hin",
  "source_part": "B"
 },
 {
  "id": "tanaos/synthetic-emotion-detection-dataset-v1",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en"
  ],
  "category": "2-emotion",
  "label_field": "labels (0-7: joy, anger, fear, sadness, surprise, disgust, excitement, neutral)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "synthetic; no eval overlap; dedup vs DAIR/GoEmotions test anyway; emotion caveat",
  "source_part": "B"
 },
 {
  "id": "JusteLeo/French-emotion",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "fr"
  ],
  "category": "2-emotion",
  "label_field": "label",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "Gemini-generated; no eval overlap; emotion caveat",
  "source_part": "B"
 },
 {
  "id": "shreyaspullehf/emotion-dataset-20-emotions",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en"
  ],
  "category": "2-emotion",
  "label_field": "emotion",
  "text_fields": [
   "sentence"
  ],
  "test_overlap_note": "LLM-generated; no eval overlap; cap (repetitive); emotion caveat",
  "source_part": "B"
 },
 {
  "id": "Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "zh"
  ],
  "category": "2-emotion",
  "label_field": "emotion",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "AI-generated, human-labelled; no eval overlap; emotion caveat",
  "source_part": "B"
 },
 {
  "id": "jmccardle/pulse-sofroniew-emotion-concept-texts",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "en"
  ],
  "category": "2-emotion",
  "label_field": "emotion (171 concepts)",
  "text_fields": [
   "story"
  ],
  "test_overlap_note": "Qwen3-32B stories; no eval overlap; emotion caveat",
  "source_part": "B"
 },
 {
  "id": "llm-for-emotion/Cultural-Emo",
  "config": "ara|deu|eng|hin|spn",
  "split": "test",
  "licence": "mit",
  "languages": [
   "ar",
   "de",
   "en",
   "hi",
   "es"
  ],
  "category": "2-emotion",
  "label_field": "emotion_eng (also sentiment_eng)",
  "text_fields": [
   "text_<lang>"
  ],
  "test_overlap_note": "test-only benchmark (400/lang); prefer as new eval suite; if trained on, cannot be used for eval",
  "source_part": "B"
 },
 {
  "id": "github:wwbp/empathic_reactions",
  "config": "data/responses/data/messages.csv",
  "split": "all",
  "licence": "cc-by-4.0",
  "languages": [
   "en"
  ],
  "category": "2-emotion",
  "label_field": "empathy, distress (scores)",
  "text_fields": [
   "essay"
  ],
  "test_overlap_note": "not on HF; crowd-written; no eval overlap; emotion caveat",
  "source_part": "B"
 },
 {
  "id": "SINAI/ALIA-es-discriminative-stance-detection",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "es"
  ],
  "category": "10-stance",
  "label_field": "majority_label (favor/against/neutral; drop empty)",
  "text_fields": [
   "target",
   "description",
   "comment"
  ],
  "test_overlap_note": "no eval overlap; source Decide Madrid open data CC BY 4.0 (per card)",
  "source_part": "B"
 },
 {
  "id": "pacoreyes/StanceSentences",
  "config": "default",
  "split": "train",
  "licence": "apache-2.0",
  "languages": [
   "en"
  ],
  "category": "10-stance",
  "label_field": "label (support/oppose)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "uncertain: labelling method undocumented; text = US presidential documents (public domain); no eval overlap",
  "source_part": "B"
 },
 {
  "id": "sweatSmile/sarcastic-dataset",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en"
  ],
  "category": "10-sarcasm",
  "label_field": "derived: sentence=non-sarcastic, translation=sarcastic",
  "text_fields": [
   "sentence",
   "translation"
  ],
  "test_overlap_note": "GPT-generated rewrites of Harvard sentences; tiny; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "elvanalabs/sarcasm-statements-90",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en"
  ],
  "category": "10-sarcasm",
  "label_field": "sarcastic",
  "text_fields": [
   "statement"
  ],
  "test_overlap_note": "90 rows; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "GoktugD/turkish-formality-rewrite-500k",
  "config": "default",
  "split": "train",
  "licence": "cc0-1.0",
  "languages": [
   "tr"
  ],
  "category": "10-formality",
  "label_field": "derived: informal vs formal column",
  "text_fields": [
   "informal",
   "formal"
  ],
  "test_overlap_note": "procedurally generated; cap; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "NagaYu/deference-keigo-corpus",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "ja"
  ],
  "category": "10-formality",
  "label_field": "n_errors>0 (keigo error yes/no), error_types",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "rule-generated; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "ronantakizawa/japanese-honorifics",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "ja"
  ],
  "category": "10-formality",
  "label_field": "derived: register column (plain/teineigo/sonkeigo/kenjogo)",
  "text_fields": [
   "base_sentence",
   "teineigo",
   "sonkeigo",
   "kenjogo"
  ],
  "test_overlap_note": "GPT-4o-generated; tiny; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "weijianzhg/email-triage-action-seed",
  "config": "default",
  "split": "train",
  "licence": "apache-2.0",
  "languages": [
   "en"
  ],
  "category": "10-urgency",
  "label_field": "priority (also category, action)",
  "text_fields": [
   "subject",
   "body"
  ],
  "test_overlap_note": "Claude-generated; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "AshenFdo/synthetic_blood_request_urgency_dataset",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en"
  ],
  "category": "10-urgency",
  "label_field": "label (emergency/not_emergency)",
  "text_fields": [
   "description"
  ],
  "test_overlap_note": "AI-generated; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "IDinsight/urgency_detection_maternal_health_synthetic",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en"
  ],
  "category": "10-urgency",
  "label_field": "matching_rule (35 urgent warning signs)",
  "text_fields": [
   "generated_user_message"
  ],
  "test_overlap_note": "Gemini-generated; needs non-urgent negatives; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "truthfulqa/truthful_qa",
  "config": "generation",
  "split": "validation",
  "licence": "apache-2.0",
  "languages": [
   "en"
  ],
  "category": "10-fact-check",
  "label_field": "derived: correct_answers=true, incorrect_answers=false",
  "text_fields": [
   "question",
   "correct_answers",
   "incorrect_answers"
  ],
  "test_overlap_note": "no eval-suite overlap; popular LLM benchmark, consider holding out",
  "source_part": "B"
 },
 {
  "id": "Eurolingua/truthfulqax",
  "config": "truthfulqa_gen_{DE,FR,ES,IT,PT,NL,PL}_validation.jsonl",
  "split": "validation",
  "licence": "apache-2.0",
  "languages": [
   "de",
   "fr",
   "es",
   "it",
   "pt",
   "nl",
   "pl"
  ],
  "category": "10-fact-check",
  "label_field": "derived: correct vs incorrect answers",
  "text_fields": [
   "question",
   "correct_answers",
   "incorrect_answers"
  ],
  "test_overlap_note": "translation of TruthfulQA (same questions); load JSONL directly (loading script)",
  "source_part": "B"
 },
 {
  "id": "seyled/Phantom_Hallucination_Detection",
  "config": "PhantomDataset/*_seed.csv",
  "split": "train",
  "licence": "apache-2.0",
  "languages": [
   "en"
  ],
  "category": "10-fact-check",
  "label_field": "ground_truth_label",
  "text_fields": [
   "query",
   "context",
   "answer"
  ],
  "test_overlap_note": "SEC filings context; many context-length variants of same seeds, dedup; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "zenodo:3609356 (ClaimBuster)",
  "config": "crowdsourced/groundtruth",
  "split": "all",
  "licence": "cc-by-4.0",
  "languages": [
   "en"
  ],
  "category": "10-claim-detection",
  "label_field": "Verdict (-1 NFS / 0 UFS / 1 CFS)",
  "text_fields": [
   "Text"
  ],
  "test_overlap_note": "uncertain: US presidential debate transcripts; load from Zenodo not the cc-by-sa HF mirror; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "nvidia/Nemotron-PII",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "en"
  ],
  "category": "10-pii",
  "label_field": "spans (PII types)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "synthetic; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "gretelai/synthetic_pii_finance_multilingual",
  "config": "default",
  "split": "train",
  "licence": "apache-2.0",
  "languages": [
   "en",
   "fr",
   "de",
   "nl",
   "es",
   "it",
   "sv"
  ],
  "category": "10-pii",
  "label_field": "pii_spans (29 types); document_type",
  "text_fields": [
   "generated_text"
  ],
  "test_overlap_note": "synthetic; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "gretelai/gretel-pii-masking-en-v1",
  "config": "default",
  "split": "train",
  "licence": "apache-2.0",
  "languages": [
   "en"
  ],
  "category": "10-pii",
  "label_field": "entities",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "synthetic; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "Wismut/nym-pii-multilingual-data",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en",
   "de",
   "fr",
   "es",
   "it",
   "pt",
   "nl",
   "pl",
   "sv",
   "cs",
   "ro",
   "tr",
   "fi",
   "da",
   "el",
   "ru",
   "uk",
   "ja",
   "zh",
   "ko",
   "ar",
   "hi"
  ],
  "category": "10-pii",
  "label_field": "entities",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "template+Faker; cap; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "E3-JSI/synthetic-multi-pii-ner-v1",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en",
   "fr",
   "de",
   "el",
   "nl",
   "it",
   "sl"
  ],
  "category": "10-pii",
  "label_field": "entities",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "LLM-generated; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "urchade/synthetic-pii-ner-mistral-v1",
  "config": "data.json",
  "split": "all",
  "licence": "apache-2.0",
  "languages": [
   "en",
   "fr",
   "it",
   "de",
   "es"
  ],
  "category": "10-pii",
  "label_field": "ner ([start,end,type] token spans)",
  "text_fields": [
   "tokenized_text"
  ],
  "test_overlap_note": "Mistral-generated; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "Helsinki-NLP/tatoeba",
  "config": "per language pair",
  "split": "train",
  "licence": "cc-by-2.0",
  "languages": [
   "300+ incl. all priority"
  ],
  "category": "10-language-id",
  "label_field": "derived: language code",
  "text_fields": [
   "translation.<lang>"
  ],
  "test_overlap_note": "not FLORES; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "CohereLabs/aya_dataset",
  "config": "default",
  "split": "train",
  "licence": "apache-2.0",
  "languages": [
   "65 incl. ar, de, en, fr, hi, it, ja, nl, pl, pt, ru, es, tr, zh"
  ],
  "category": "10-language-id",
  "label_field": "language",
  "text_fields": [
   "inputs",
   "targets"
  ],
  "test_overlap_note": "use annotation_type=original-annotations only; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "joelniklaus/german_argument_mining",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "de"
  ],
  "category": "10-argument-mining",
  "label_field": "label (conclusion/definition/subsumption/other)",
  "text_fields": [
   "input_sentence"
  ],
  "test_overlap_note": "German court judgments (official works); no eval overlap",
  "source_part": "B"
 },
 {
  "id": "nyu-mll/blimp",
  "config": "all 67",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "en"
  ],
  "category": "10-acceptability",
  "label_field": "derived: sentence_good=acceptable, sentence_bad=unacceptable",
  "text_fields": [
   "sentence_good",
   "sentence_bad"
  ],
  "test_overlap_note": "template-generated; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "pinialt/cefr-texts-10languages",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "en",
   "fr",
   "es",
   "de",
   "it",
   "pt",
   "nl",
   "ru",
   "zh",
   "ar"
  ],
  "category": "10-readability",
  "label_field": "label (A1-C2)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "GPT-4o-mini-generated; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "yoonholee/humor-greats-public-domain",
  "config": "default",
  "split": "train",
  "licence": "cc0-1.0",
  "languages": [
   "en"
  ],
  "category": "10-humor",
  "label_field": "none (all humorous; positives only)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "public-domain Gutenberg; needs negatives; no eval overlap",
  "source_part": "B"
 },
 {
  "id": "nyu-mll/multi_nli",
  "config": "default",
  "split": "train",
  "licence": "OANC licence (permissive, commercial OK) per MNLI paper/card; fiction genre mixed incl. CC-BY-SA-3.0",
  "languages": [
   "en"
  ],
  "category": "4-nli",
  "label_field": "label",
  "text_fields": [
   "premise",
   "hypothesis"
  ],
  "test_overlap_note": "no eval-suite overlap; MNLI is not an eval suite",
  "source_part": "C"
 },
 {
  "id": "ankitkupadhyay/XNLI",
  "config": "default (files <lang>_train.csv)",
  "split": "train",
  "licence": "apache-2.0 (card); content inherits MNLI/OANC terms",
  "languages": [
   "ar",
   "bg",
   "de",
   "el",
   "en",
   "es",
   "fr",
   "hi",
   "ru",
   "sw",
   "th",
   "tr",
   "ur",
   "vi",
   "zh"
  ],
  "category": "4-nli",
  "label_field": "label",
  "text_fields": [
   "premise",
   "hypothesis"
  ],
  "test_overlap_note": "no eval-suite overlap (no id/fa; no XNLI dev/test rows)",
  "source_part": "C"
 },
 {
  "id": "boun-tabi/nli_tr",
  "config": "multinli_tr",
  "split": "train",
  "licence": "same terms as MultiNLI (GitHub boun-tabi/NLI-TR README)",
  "languages": [
   "tr"
  ],
  "category": "4-nli",
  "label_field": "label",
  "text_fields": [
   "premise",
   "hypothesis"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "takehika/wanli-ja-nli",
  "config": "ja_only",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "ja"
  ],
  "category": "4-nli",
  "label_field": "gold",
  "text_fields": [
   "premise_ja",
   "hypothesis_ja"
  ],
  "test_overlap_note": "none (translation of WANLI, already in mixture in English)",
  "source_part": "C"
 },
 {
  "id": "maximoss/mnli-nineeleven-fr",
  "config": "default",
  "split": "train",
  "licence": "bsd-2-clause",
  "languages": [
   "fr"
  ],
  "category": "4-nli",
  "label_field": "label",
  "text_fields": [
   "premise",
   "hypothesis"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "hkust-nlp/SynCSE-scratch-NLI",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "en"
  ],
  "category": "4-nli; 7-similarity",
  "label_field": "derived: (sent0,sent1)=entailment/paraphrase, (sent0,nli_hard)=contradiction/non-paraphrase",
  "text_fields": [
   "sent0",
   "sent1",
   "nli_hard"
  ],
  "test_overlap_note": "none; GPT-3.5 generated; overlaps redis/llm-paraphrases (excluded)",
  "source_part": "C"
 },
 {
  "id": "jhu-cogsci/hans",
  "config": "plain_text",
  "split": "train",
  "licence": "MIT (GitHub tommccoy1/hans; HF card says unknown)",
  "languages": [
   "en"
  ],
  "category": "4-nli",
  "label_field": "label",
  "text_fields": [
   "premise",
   "hypothesis"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "sileod/attempto-nli",
  "config": "default",
  "split": "train",
  "licence": "apache-2.0",
  "languages": [
   "en"
  ],
  "category": "4-nli",
  "label_field": "race_label",
  "text_fields": [
   "premise",
   "hypothesis"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "MoritzLaurer/synthetic_zeroshot_mixtral_v0.1",
  "config": "mixtral_written_texts_for_tasks, mixtral_written_texts_for_tasks_v2, mixtral_written_texts_for_tasks_v3, mixtral_written_texts_for_tasks_v4",
  "split": "train",
  "licence": "apache-2.0 (Mixtral-8x7B outputs)",
  "languages": [
   "en"
  ],
  "category": "4-nli",
  "label_field": "labels",
  "text_fields": [
   "text",
   "hypothesis"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "GoktugD/turkish-nli-constructed-1.5m",
  "config": "default",
  "split": "train",
  "licence": "cc0-1.0",
  "languages": [
   "tr"
  ],
  "category": "4-nli",
  "label_field": "label",
  "text_fields": [
   "premise",
   "hypothesis"
  ],
  "test_overlap_note": "none; template-generated, low priority, cap hard",
  "source_part": "C"
 },
 {
  "id": "community-datasets/tapaco",
  "config": "all_languages (or per-language configs)",
  "split": "train",
  "licence": "cc-by-2.0 (Tatoeba CC-BY 2.0 FR)",
  "languages": [
   "en",
   "de",
   "fr",
   "es",
   "it",
   "pt",
   "nl",
   "pl",
   "tr",
   "ru",
   "ar",
   "hi",
   "ja",
   "zh"
  ],
  "category": "7-similarity",
  "label_field": "derived: same paraphrase_set_id = paraphrase; sampled other set = not",
  "text_fields": [
   "paraphrase"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "Helsinki-NLP/tatoeba",
  "config": "language pairs (load raw OPUS/Tatoeba export; HF repo is a script)",
  "split": "train",
  "licence": "cc-by-2.0 (Tatoeba CC-BY 2.0 FR)",
  "languages": [
   "en",
   "de",
   "fr",
   "es",
   "it",
   "pt",
   "nl",
   "pl",
   "tr",
   "ru",
   "ar",
   "hi",
   "ja",
   "zh"
  ],
  "category": "7-similarity",
  "label_field": "derived: aligned pair = translation; sampled pair = not",
  "text_fields": [
   "translation.<src>",
   "translation.<tgt>"
  ],
  "test_overlap_note": "none (not FLORES)",
  "source_part": "C"
 },
 {
  "id": "dell-research-harvard/headlines-semantic-similarity",
  "config": "default",
  "split": "train",
  "licence": "cc-by-2.0 (off-copyright US newspapers)",
  "languages": [
   "en"
  ],
  "category": "7-similarity",
  "label_field": "derived: same group_id = same story",
  "text_fields": [
   "headline"
  ],
  "test_overlap_note": "none; sample, OCR noise",
  "source_part": "C"
 },
 {
  "id": "tasksource/esci",
  "config": "default",
  "split": "train",
  "licence": "apache-2.0 (amazon-science/esci-data)",
  "languages": [
   "en",
   "es",
   "ja"
  ],
  "category": "7-similarity",
  "label_field": "esci_label",
  "text_fields": [
   "query",
   "product_title",
   "product_description"
  ],
  "test_overlap_note": "none; already in mixture as 'esci' (cap 2000), raise es/ja share and dedupe",
  "source_part": "C"
 },
 {
  "id": "napsternxg/wands",
  "config": "default",
  "split": "train",
  "licence": "MIT (wayfair/WANDS)",
  "languages": [
   "en"
  ],
  "category": "7-similarity",
  "label_field": "label",
  "text_fields": [
   "query",
   "product_name",
   "product_description"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "Avature/Job-Title-Similarity",
  "config": "de,en,es,fr,it,ja,ko,nl,pl,pt,zh",
  "split": "queries+corpus",
  "licence": "apache-2.0",
  "languages": [
   "de",
   "en",
   "es",
   "fr",
   "it",
   "ja",
   "nl",
   "pl",
   "pt",
   "zh",
   "ko"
  ],
  "category": "7-similarity",
  "label_field": "labels (relevant corpus indices)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "none; benchmark without train split, small",
  "source_part": "C"
 },
 {
  "id": "federetyk/MELO-Benchmark",
  "config": "48 configs (<country>_q_<lang>_c_<lang>)",
  "split": "queries+corpus",
  "licence": "mit",
  "languages": [
   "de",
   "fr",
   "es",
   "it",
   "nl",
   "pl",
   "pt",
   "en"
  ],
  "category": "7-similarity",
  "label_field": "labels (relevant ESCO title indices)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "none; ESCO text reuse terms not separately verified",
  "source_part": "C"
 },
 {
  "id": "matsuxr/JaGovFaqs-22k",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0 (Japanese government copyright policy)",
  "languages": [
   "ja"
  ],
  "category": "7-similarity",
  "label_field": "derived: question-answer pair = relevant; other answers = negatives",
  "text_fields": [
   "Question",
   "Answer"
  ],
  "test_overlap_note": "none",
  "source_part": "C"
 },
 {
  "id": "stjiris/IRIS_sts",
  "config": "default",
  "split": "train",
  "licence": "mit",
  "languages": [
   "pt"
  ],
  "category": "7-similarity",
  "label_field": "relatedness_score",
  "text_fields": [
   "sentence1",
   "sentence2"
  ],
  "test_overlap_note": "none; small, partly generated",
  "source_part": "C"
 },
 {
  "id": "washenkov/synthetic-chat-duplicates",
  "config": "default",
  "split": "train",
  "licence": "cc-by-4.0",
  "languages": [
   "en"
  ],
  "category": "7-similarity",
  "label_field": "dialog_type",
  "text_fields": [
   "conversation"
  ],
  "test_overlap_note": "none; synthetic",
  "source_part": "C"
 },
 {
  "id": "github:bvidgen/Dynamically-Generated-Hate-Speech-Dataset (v0.2.3.csv; NOT tasksource/dynahate mirror tagged gpl)",
  "config": "default",
  "split": "train (split column; dev/test also usable)",
  "licence": "CC-BY-4.0 (upstream README)",
  "languages": [
   "en"
  ],
  "category": "5-toxicity-hate",
  "label_field": "label (hate/nothate); type; target",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup + add mteb/multi-hatecheck test texts (11 langs) to banned set; not HateCheck-derived",
  "source_part": "D"
 },
 {
  "id": "jagoldz/gahd (filter via GitHub jagol/gahd gahd_disaggregated.csv)",
  "config": "default",
  "split": "train (split column), source in {dynabench, contrastive, translation}",
  "licence": "CC-BY-4.0",
  "languages": [
   "de"
  ],
  "category": "5-toxicity-hate",
  "label_field": "label",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup + add mteb/multi-hatecheck test texts (11 langs) to banned set; drop source=news",
  "source_part": "D"
 },
 {
  "id": "mteb/toxic_conversations_50k",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0 (Civil Comments text CC0)",
  "languages": [
   "en"
  ],
  "category": "5-toxicity-hate",
  "label_field": "label_text",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; duplicate of civil_comments already in mixture -> dedup",
  "source_part": "D"
 },
 {
  "id": "OpenAssistant/oasst2",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "en",
   "es",
   "ru",
   "zh",
   "de",
   "fr",
   "pt",
   "it",
   "ja",
   "pl",
   "ar",
   "nl"
  ],
  "category": "5-toxicity-moderation",
  "label_field": "labels.{spam,pii,not_appropriate,hate_speech,sexual_content,toxicity,violence} (mean crowd votes)",
  "text_fields": [
   "text",
   "lang",
   "role"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; drop synthetic=true; ignore detoxify scores",
  "source_part": "D"
 },
 {
  "id": "PleIAs/ToxicCommons",
  "config": "default",
  "split": "train",
  "licence": "MIT (LLM labels by Llama 3.1 8B; caution)",
  "languages": [
   "en",
   "fr",
   "es",
   "de",
   "pl",
   "nl",
   "pt",
   "it"
  ],
  "category": "5-toxicity-hate",
  "label_field": "scores (5 axes 0-3)",
  "text_fields": [
   "original_text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; caution: machine labels",
  "source_part": "D"
 },
 {
  "id": "nvidia/Aegis-AI-Content-Safety-Dataset-1.0",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "labels_0..labels_4 (Safe/Needs Caution/13 categories)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; dedup against hh-rlhf/Aegis/Salad/AttaQ (shared Anthropic red-team prompts)",
  "source_part": "D"
 },
 {
  "id": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
  "config": "default",
  "split": "train (only HH-traceable prompts; drop REDACTED/DAN/AART)",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "prompt_label; response_label (prefer response_label_source=human); violated_categories",
  "text_fields": [
   "prompt",
   "response"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; dedup against hh-rlhf/Aegis/Salad/AttaQ (shared Anthropic red-team prompts); apply extra-v1 lineage join",
  "source_part": "D"
 },
 {
  "id": "CohereForAI/aya_redteaming",
  "config": "aya_{eng,fra,spa,rus,arb,hin,srp,tgl}.jsonl",
  "split": "all (no split)",
  "licence": "Apache-2.0",
  "languages": [
   "en",
   "fr",
   "es",
   "ru",
   "ar",
   "hi",
   "sr",
   "tl"
  ],
  "category": "5-safety-moderation",
  "label_field": "harm_category (multi-label JSON list); global_or_local",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; all rows harmful",
  "source_part": "D"
 },
 {
  "id": "Anthropic/hh-rlhf",
  "config": "data_dir=red-team-attempts",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "rating (0-4); min_harmlessness_score_transcript; tags",
  "text_fields": [
   "transcript",
   "task_description"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; dedup against hh-rlhf/Aegis/Salad/AttaQ (shared Anthropic red-team prompts)",
  "source_part": "D"
 },
 {
  "id": "thu-coai/Safety-Prompts",
  "config": "typical_safety_scenarios.json + instruction_attack_scenarios.json (field per scenario)",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "zh"
  ],
  "category": "5-safety-moderation (+prompt-injection subset)",
  "label_field": "type (13 scenarios)",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; responses are ChatGPT outputs, use prompt only",
  "source_part": "D"
 },
 {
  "id": "bench-llms/or-bench",
  "config": "or-bench-80k, or-bench-hard-1k, or-bench-toxic",
  "split": "train",
  "licence": "CC-BY-4.0 (GitHub Apache-2.0)",
  "languages": [
   "en"
  ],
  "category": "5-safety-over-refusal",
  "label_field": "config (benign vs toxic); category",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "natolambert/xstest-v2-copy (paul-rottger/exaggerated-safety)",
  "config": "prompts",
  "split": "prompts",
  "licence": "CC-BY-4.0 (prompts only)",
  "languages": [
   "en"
  ],
  "category": "5-safety-over-refusal",
  "label_field": "type (safe vs contrast_ unsafe)",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; drop model completions",
  "source_part": "D"
 },
 {
  "id": "Bertievidgen/SimpleSafetyTests",
  "config": "default",
  "split": "test",
  "licence": "CC-BY-2.0 (GitHub CC-BY-4.0)",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "harm_area; category",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "JailbreakBench/JBB-Behaviors",
  "config": "behaviors",
  "split": "harmful + benign",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "split name (harmful/benign); Category",
  "text_fields": [
   "Goal"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "walledai/HarmBench (centerforaisafety/HarmBench)",
  "config": "standard",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "category",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; drop copyright/contextual",
  "source_part": "D"
 },
 {
  "id": "walledai/AdvBench (llm-attacks/llm-attacks)",
  "config": "default",
  "split": "train",
  "licence": "MIT (caution: generated by Wizard-Vicuna-30B-Uncensored)",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "all harmful",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "declare-lab/HarmfulQA",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0 (ChatGPT-generated; caution)",
  "languages": [
   "en"
  ],
  "category": "5-safety-response-harm",
  "label_field": "blue_conversations (harmless) vs red_conversations (harmful); topic",
  "text_fields": [
   "question",
   "blue_conversations",
   "red_conversations"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "declare-lab/CategoricalHarmfulQA",
  "config": "default",
  "split": "en, zh, vi",
  "licence": "Apache-2.0",
  "languages": [
   "en",
   "zh",
   "vi"
  ],
  "category": "5-safety-moderation",
  "label_field": "Category; Subcategory",
  "text_fields": [
   "Question"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "OpenSafetyLab/Salad-Data",
  "config": "base_set",
  "split": "train (source in {GPT-Gen, HH-harmless, HH-red-team, Advbench})",
  "licence": "Apache-2.0",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "1-category / 2-category / 3-category (auto-labelled)",
  "text_fields": [
   "question"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; dedup against hh-rlhf/Aegis/Salad/AttaQ (shared Anthropic red-team prompts); drop Do-Not-Answer/ToxicChat/DAN/GPTFuzzer/Multilingual rows",
  "source_part": "D"
 },
 {
  "id": "ibm-research/AttaQ",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "label (7 harm types)",
  "text_fields": [
   "input"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; dedup against hh-rlhf/Aegis/Salad/AttaQ (shared Anthropic red-team prompts)",
  "source_part": "D"
 },
 {
  "id": "DAMO-NLP-SG/MultiJail",
  "config": "default",
  "split": "train",
  "licence": "MIT (caution: 'solely for academic purposes' sentence)",
  "languages": [
   "en",
   "zh",
   "it",
   "vi",
   "ar",
   "ko",
   "th",
   "bn",
   "sw",
   "jv"
  ],
  "category": "5-safety-moderation",
  "label_field": "tags",
  "text_fields": [
   "en",
   "zh",
   "it",
   "vi",
   "ar",
   "ko",
   "th",
   "bn",
   "sw",
   "jv"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; dedup against hh-rlhf/Aegis/Salad/AttaQ (shared Anthropic red-team prompts)",
  "source_part": "D"
 },
 {
  "id": "Virtue-AI-HUB/PolyGuard",
  "config": "social_media, finance_input, finance_output, hr, law_input, law_output, education, regulation_input, regulation_output, code, cyber (not cve/mitre)",
  "split": "all *_safe / *_unsafe splits",
  "licence": "CC-BY-4.0 (LLM-generated; caution)",
  "languages": [
   "en"
  ],
  "category": "5-safety-policy-moderation",
  "label_field": "label (safe/unsafe); category",
  "text_fields": [
   "prompt",
   "response"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "allenai/prosocial-dialog",
  "config": "default",
  "split": "train (source == ethics_amt only)",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "5-safety-moderation",
  "label_field": "safety_label (5 levels)",
  "text_fields": [
   "context"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; drop socialchemistry/sbic/ethics_reddit",
  "source_part": "D"
 },
 {
  "id": "nvidia/CantTalkAboutThis-Topic-Control-Dataset",
  "config": "mixtral",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "5-safety-topic-control",
  "label_field": "on-topic vs distractor turn",
  "text_fields": [
   "system_instruction",
   "conversation"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "Dizzzy0x00/LLMGen-Phishing-Email-Dataset",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0 (synthetic; caution)",
  "languages": [
   "en",
   "zh"
  ],
  "category": "5-spam-phishing",
  "label_field": "label",
  "text_fields": [
   "content"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "BothBosu/scam-dialogue",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0 (Llama-3-70B generated; caution)",
  "languages": [
   "en"
  ],
  "category": "5-spam-scam",
  "label_field": "label; type",
  "text_fields": [
   "dialogue"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "BothBosu/multi-agent-scam-conversation",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0 (synthetic; caution)",
  "languages": [
   "en"
  ],
  "category": "5-spam-scam",
  "label_field": "labels; type",
  "text_fields": [
   "dialogue"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "shakeleoatmeal/phone-scam-detection-synthetic",
  "config": "default",
  "split": "train",
  "licence": "MIT (Llama 8B generated; caution)",
  "languages": [
   "en"
  ],
  "category": "5-spam-scam",
  "label_field": "label; type",
  "text_fields": [
   "dialogue"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "alusci/sms-otp-spam-dataset",
  "config": "default",
  "split": "train",
  "licence": "MIT (templated; low value)",
  "languages": [
   "en"
  ],
  "category": "5-spam-sms",
  "label_field": "label (valid/spam)",
  "text_fields": [
   "sms_text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "reshabhs/SPML_Chatbot_Prompt_Injection",
  "config": "default",
  "split": "train",
  "licence": "MIT",
  "languages": [
   "en"
  ],
  "category": "5-prompt-injection",
  "label_field": "Prompt injection (0/1); Degree",
  "text_fields": [
   "System Prompt",
   "User Prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "3nesdeniz/agentic-prompt-injection-5k",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "5-prompt-injection",
  "label_field": "label/class; attack_family; technique",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "3nesdeniz/turkish-conversation-prompt-injection",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "tr"
  ],
  "category": "5-prompt-injection",
  "label_field": "label; attack_family",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "5-prompt-injection (indirect)",
  "label_field": "attack_category (all positive; build negatives by removing injection)",
  "text_fields": [
   "environment",
   "injection",
   "responses_create_params"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "Lakera/gandalf_ignore_instructions",
  "config": "default",
  "split": "train",
  "licence": "MIT (platform-released player text; caution)",
  "languages": [
   "en"
  ],
  "category": "5-prompt-injection",
  "label_field": "positives only",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup",
  "source_part": "D"
 },
 {
  "id": "Lakera/mosscap_prompt_injection",
  "config": "default",
  "split": "train",
  "licence": "MIT (platform-released player text; caution)",
  "languages": [
   "en"
  ],
  "category": "5-prompt-injection",
  "label_field": "level (1-8); attempts only",
  "text_fields": [
   "prompt"
  ],
  "test_overlap_note": "not an eval suite; run build_mixture test_texts() exact dedup; drop ChatGPT answer fields",
  "source_part": "D"
 },
 {
  "id": "theatticusproject/cuad-qa",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "6-reading-comprehension",
  "label_field": "answers (empty = clause absent)",
  "text_fields": [
   "context",
   "question"
  ],
  "test_overlap_note": "no eval-suite overlap; loading script, fetch data.zip from github.com/TheAtticusProject/cuad; dedup against dvgodoy CUAD clause set",
  "source_part": "E"
 },
 {
  "id": "theatticusproject/maud",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "6-reading-comprehension",
  "label_field": "label",
  "text_fields": [
   "text",
   "question",
   "subquestion"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "nyu-mll/quality (github; mirror emozilla/quality)",
  "config": "v1.0.1 htmlstripped",
  "split": "train",
  "licence": "CC-BY-4.0 (articles: Gutenberg US-PD, OANC Slate, CC-BY-4.0)",
  "languages": [
   "en"
  ],
  "category": "6-reading-comprehension",
  "label_field": "questions[].gold_label",
  "text_fields": [
   "article",
   "questions[].question",
   "questions[].options"
  ],
  "test_overlap_note": "no eval-suite overlap; long passages need chunking",
  "source_part": "E"
 },
 {
  "id": "WorkInTheDark/FairytaleQA",
  "config": "plain_text",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "en"
  ],
  "category": "6-reading-comprehension",
  "label_field": "ex-or-im; attribute",
  "text_fields": [
   "story_section",
   "question",
   "answer1"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "Fumika/Wikinews-multilingual",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-2.5 (Wikinews)",
  "languages": [
   "en",
   "es",
   "fr",
   "de",
   "pt",
   "pl",
   "it",
   "zh",
   "ru",
   "ja",
   "nl",
   "tr",
   "ar",
   "fa",
   "cs",
   "sv",
   "ta",
   "sr",
   "ca",
   "he",
   "fi",
   "eo",
   "el",
   "hu",
   "uk",
   "no",
   "ko",
   "ro",
   "bg",
   "bs",
   "li",
   "sq",
   "th"
  ],
  "category": "8-topic",
  "label_field": "categories (keep Wikinews top-level topic categories)",
  "text_fields": [
   "title",
   "text"
  ],
  "test_overlap_note": "OVERLAP RISK: FLORES (SIB-200, Belebele) was drawn from en.wikinews.org; 351/900 Belebele eng passages link there. Drop articles whose url is a Belebele link or whose text contains any FLORES dev/devtest sentence",
  "source_part": "E"
 },
 {
  "id": "gfissore/arxiv-abstracts-2021",
  "config": "default",
  "split": "train",
  "licence": "CC0-1.0 (arXiv metadata)",
  "languages": [
   "en"
  ],
  "category": "8-topic",
  "label_field": "categories (first = primary)",
  "text_fields": [
   "title",
   "abstract"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "NortheasternUniversity/big_patent",
  "config": "a..y (one per CPC section)",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "8-topic",
  "label_field": "config name = CPC section",
  "text_fields": [
   "abstract",
   "description"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "ddrg/super_eurlex",
  "config": "per sector/language (meta_data + text_data/<LANG>/*_clean.parquet)",
  "split": "train",
  "licence": "MIT card; EUR-Lex reuse authorised incl. commercial with attribution (Decision 2011/833/EU)",
  "languages": [
   "bg",
   "cs",
   "da",
   "de",
   "el",
   "en",
   "es",
   "et",
   "fi",
   "fr",
   "ga",
   "hr",
   "hu",
   "it",
   "lt",
   "lv",
   "mt",
   "nl",
   "pl",
   "pt",
   "ro",
   "sk",
   "sl",
   "sv"
  ],
  "category": "8-topic",
  "label_field": "directory_code / subject_matter / eurovoc",
  "text_fields": [
   "text (clean)"
  ],
  "test_overlap_note": "no eval-suite overlap; files are GB-sized, stream selectively; attribute '© European Union, https://eur-lex.europa.eu'",
  "source_part": "E"
 },
 {
  "id": "joelniklaus/covid19_emergency_event",
  "config": "default",
  "split": "train",
  "licence": "CC0-1.0",
  "languages": [
   "en",
   "fr",
   "hu",
   "it",
   "nb",
   "nl",
   "pl"
  ],
  "category": "8-topic",
  "label_field": "event1..event8 (multi-label)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "lyon-nlp/clustering-hal-s2s",
  "config": "default",
  "split": "test (only split; not a Statim eval)",
  "licence": "Apache-2.0 card; HAL metadata CC0",
  "languages": [
   "fr"
  ],
  "category": "8-topic",
  "label_field": "domain",
  "text_fields": [
   "title"
  ],
  "test_overlap_note": "no Statim eval overlap (it is an MTEB-French eval set)",
  "source_part": "E"
 },
 {
  "id": "itacasehold/itacasehold",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "it"
  ],
  "category": "8-topic",
  "label_field": "materia",
  "text_fields": [
   "title",
   "summary",
   "doc"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "8-topic",
  "label_field": "label (41 clause types)",
  "text_fields": [
   "clause"
  ],
  "test_overlap_note": "no eval-suite overlap; same contracts as CUAD-QA",
  "source_part": "E"
 },
 {
  "id": "github:PolyAI-LDN/task-specific-datasets/nlupp",
  "config": "banking, hotels",
  "split": "all folds",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "intents (multi-label)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "dedup against Banking77 test (same company, different collection)",
  "source_part": "E"
 },
 {
  "id": "uoe-nlp/multi3-nlu",
  "config": "<language>/<domain>/fold*.json",
  "split": "all folds",
  "licence": "CC-BY-4.0",
  "languages": [
   "am",
   "mr",
   "tr",
   "es"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "intents (multi-label)",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "dedup against Banking77 test",
  "source_part": "E"
 },
 {
  "id": "google-research-datasets/taskmaster1|2|3 (github Taskmaster TM-1..TM-4)",
  "config": "per TM release",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "instruction_id / domain; segments",
  "text_fields": [
   "utterances[].text"
  ],
  "test_overlap_note": "no eval-suite overlap; HF repos are loading scripts",
  "source_part": "E"
 },
 {
  "id": "pfb30/multi_woz_v22 (github budzianowski/multiwoz)",
  "config": "v2.2",
  "split": "train",
  "licence": "MIT (upstream); Apache-2.0 (card)",
  "languages": [
   "en"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "turns.dialogue_acts / frames.state.active_intent",
  "text_fields": [
   "turns.utterance"
  ],
  "test_overlap_note": "no eval-suite overlap; loading script",
  "source_part": "E"
 },
 {
  "id": "kchawla123/casino",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "en"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "annotations (strategy labels)",
  "text_fields": [
   "chat_logs[].text"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "ConvLab/crosswoz (github thu-coai/CrossWOZ)",
  "config": "data.zip",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "zh"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "dialog_act / domain",
  "text_fields": [
   "content"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "github:HLTCHKUST/BiToD (mirror DeepPavlov/BiToD)",
  "config": "en, zh",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "en",
   "zh"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "user/system actions, intent",
  "text_fields": [
   "utterance"
  ],
  "test_overlap_note": "no eval-suite overlap; mirror card has no licence, use upstream",
  "source_part": "E"
 },
 {
  "id": "OpenVoiceOS/hass-intent-templates",
  "config": "<lang>",
  "split": "train",
  "licence": "Apache-2.0 card; upstream home-assistant/intents CC-BY-4.0",
  "languages": [
   "de",
   "es",
   "en",
   "fr",
   "ar",
   "nl",
   "pl",
   "pt",
   "ru",
   "tr",
   "it",
   "..."
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "intent_id",
  "text_fields": [
   "template (expand with expansions)"
  ],
  "test_overlap_note": "exact-dedup expanded utterances against MASSIVE test (all langs) and HWU64 test",
  "source_part": "E"
 },
 {
  "id": "TigreGotico/yes-no-multilingual",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0 (LLM-generated by Claude)",
  "languages": [
   "en",
   "de",
   "fr",
   "es",
   "it",
   "pt",
   "ru",
   "pl",
   "nl",
   "ja",
   "zh",
   "ar",
   "tr",
   "..."
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "agreement (yes/no/unclear); subtype",
  "text_fields": [
   "utterance"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "OpenVoiceOS/yes_no_answers",
  "config": "yesno_multilingual.csv",
  "split": "train",
  "licence": "Apache-2.0 (LLM-generated by Claude)",
  "languages": [
   "en",
   "de",
   "fr",
   "es",
   "it",
   "pt",
   "ru",
   "pl",
   "nl",
   "ja",
   "zh",
   "ar",
   "tr",
   "..."
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "agreement",
  "text_fields": [
   "utterance"
  ],
  "test_overlap_note": "no eval-suite overlap; dedup against TigreGotico/yes-no-multilingual",
  "source_part": "E"
 },
 {
  "id": "clips/VaccinChatNL",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0",
  "languages": [
   "nl"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "label",
  "text_fields": [
   "sentence1"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "NABA-AI/LUB-Saudi-Arabic-Intent",
  "config": "default",
  "split": "train",
  "licence": "CC-BY-4.0 (synthetic)",
  "languages": [
   "ar"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "Intent_Type (also Emotion, Sarcasm, complaint/escalation fields)",
  "text_fields": [
   "Text",
   "Context"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "Process-Venue/IntentClassification_Dataset_for_AI_Assistant_Prompt_Routing_Hindi",
  "config": "default",
  "split": "train",
  "licence": "Apache-2.0 (text provenance undocumented)",
  "languages": [
   "hi"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "Answer",
  "text_fields": [
   "Sentence"
  ],
  "test_overlap_note": "no eval-suite overlap",
  "source_part": "E"
 },
 {
  "id": "masakhane/InjongoIntent",
  "config": "eng + 16 African configs",
  "split": "train",
  "licence": "Apache-2.0",
  "languages": [
   "en",
   "am",
   "ee",
   "ha",
   "ig",
   "rw",
   "ln",
   "lg",
   "om",
   "sn",
   "st",
   "sw",
   "tw",
   "wo",
   "xh",
   "yo",
   "zu"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "intent",
  "text_fields": [
   "text"
  ],
  "test_overlap_note": "no eval-suite overlap; CLINC-derived intents, dedup vs clinc_oos already in mixture",
  "source_part": "E"
 },
 {
  "id": "GoktugD/turkish-intent-classification-1m",
  "config": "default",
  "split": "train",
  "licence": "CC0-1.0 (template-generated)",
  "languages": [
   "tr"
  ],
  "category": "9-intent-dialogue-act",
  "label_field": "label",
  "text_fields": [
   "text (strip generated context prefix before 'Kullanıcı iletisi:')"
  ],
  "test_overlap_note": "no eval-suite overlap; low quality, sample small",
  "source_part": "E"
 },
 {
  "id": "amyrmahdy/decima-synthetic-decisions",
  "config": "short|long|relabel",
  "split": "train",
  "licence": "cc-by-4.0 (card; fully synthetic, teacher Gemma-4-26B-A4B-it, Apache-2.0 model card)",
  "languages": [
   "en",
   "fa",
   "ar",
   "ru"
  ],
  "category": "10-typed-decisions",
  "label_field": "soft label per choice (kind: choose/verify/score/rank)",
  "text_fields": [
   "state",
   "question",
   "choices"
  ],
  "test_overlap_note": "no eval-suite overlap (synthetic); fa/ar text: exact-dedup against FarsTail and SemRel arb tests anyway; one-teacher distribution, cap",
  "source_part": "parent"
 }
]
```
