# Data licences

Released Statim weights are trained only on data whose licence permits commercial use **and** does
not impose ShareAlike or copyleft terms on the model: Apache-2.0, MIT, BSD, CC0, CC-BY, ODC-By,
AFL-3.0. Excluded: non-commercial or research-only terms, ShareAlike (CC-BY-SA), copyleft (GPL,
AGPL, MPL, ODbL), custom or unknown terms. Datasets used only to *measure* the model are listed
separately; the model never trains on them.

## Base models

| Model | Licence |
|---|---|
| [Laya English checkpoint](https://huggingface.co/convaiinnovations/laya) | Apache-2.0 |
| [Laya multilingual checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual) | Apache-2.0 |
| [ModernBERT-large (encoder of the English checkpoint)](https://huggingface.co/answerdotai/ModernBERT-large) | Apache-2.0 |
| [mmBERT-base (encoder of the multilingual checkpoint)](https://huggingface.co/jhu-clsp/mmBERT-base) | MIT |

## Training data of released weights

| Dataset | Licence | Notes |
|---|---|---|
| [Banking77, train split](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-4.0 | 77 banking intents. Attribution: Casanueva et al., 2020, PolyAI. |
| [MASSIVE, train split, 51 languages](https://huggingface.co/datasets/AmazonScience/massive) | CC-BY-4.0 | Loaded from the mteb/amazon_massive_intent mirror. Attribution: FitzGerald et al., 2022, Amazon. |
| [typed-decisions, train split](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | Replay data. |
| [nvidia/Nemotron-Safety-Guard-Dataset-v3](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3) | CC-BY-4.0 | 3000 items; ar, de, en, es, fr, hi, it, ja, ko, nl, th, zh; evidence: [card revision](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3/blob/a3f7ecb3433d1933701a83f18de16c36934a7f51/README.md) |
| [l3cube-pune/IndicGuard](https://huggingface.co/datasets/l3cube-pune/IndicGuard) | CC-BY-4.0 | 3000 items; bn, en, gu, hi, kn, ml, mr, pa, ta, te, ur; evidence: [card revision](https://huggingface.co/datasets/l3cube-pune/IndicGuard/blob/f60250767f0ab6ae8f1d99831516fe8a6b453f8d/README.md) |
| [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) | CC-BY-4.0 | 3000 items; cs, de, en, es, fr, it, ko, nl, pl, pt, ru, zh; evidence: [card revision](https://huggingface.co/datasets/PolyAI/minds14/blob/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md) |
| [benayas/snips](https://huggingface.co/datasets/benayas/snips) | APACHE-2.0 | 3000 items; en; evidence: [card revision](https://huggingface.co/datasets/benayas/snips/blob/16915b895754dd4028068abe52b6851a904977d7/README.md) |
| [tasksource typed-decisions mixture](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) | per source, see below | Only rows marked commercial whose every listed licence is permissive. |

### Mixture sources (163 sources, 179,750 items, at most 2000 per source)

Licence as recorded per row by tasksource; source names follow its build manifest.

| Source | Licence | Items |
|---|---|---:|
| `agent_action_safety` | apache-2.0 | 2000 |
| `babi_nli/basic-coreference` | bsd | 252 |
| `babi_nli/basic-deduction` | bsd | 243 |
| `babi_nli/basic-induction` | bsd | 241 |
| `babi_nli/compound-coreference` | bsd | 240 |
| `babi_nli/conjunction` | bsd | 247 |
| `babi_nli/counting` | bsd | 217 |
| `babi_nli/indefinite-knowledge` | bsd | 247 |
| `babi_nli/lists-sets` | bsd | 242 |
| `babi_nli/path-finding` | bsd | 237 |
| `babi_nli/positional-reasoning` | bsd | 222 |
| `babi_nli/simple-negation` | bsd | 255 |
| `babi_nli/single-supporting-fact` | bsd | 222 |
| `babi_nli/size-reasoning` | bsd | 247 |
| `babi_nli/three-arg-relations` | bsd | 250 |
| `babi_nli/three-supporting-facts` | bsd | 253 |
| `babi_nli/time-reasoning` | bsd | 242 |
| `babi_nli/two-arg-relations` | bsd | 251 |
| `babi_nli/two-supporting-facts` | bsd | 230 |
| `babi_nli/yes-no-questions` | bsd | 253 |
| `balanced-copa` | cc-by-4.0, BSD 2-Clause License (DPI) | 1028 |
| `circa` | cc-by-4.0 | 2000 |
| `civil_comments/identity_attack_share` | cc0-1.0 | 2000 |
| `civil_comments/insult_share` | cc0-1.0 | 2000 |
| `civil_comments/obscene_share` | cc0-1.0 | 2000 |
| `civil_comments/severe_toxicity_share` | cc0-1.0 | 2000 |
| `civil_comments/sexual_explicit_share` | cc0-1.0 | 2000 |
| `civil_comments/threat_share` | cc0-1.0 | 2000 |
| `civil_comments/toxicity_share` | cc0-1.0 | 2000 |
| `cladder` | mit | 2000 |
| `clcd-english` | apache-2.0 | 2000 |
| `clinc_oos/plus` | cc-by-3.0 | 2000 |
| `codah/codah` | odc-by | 2000 |
| `commonsense_qa` | mit | 2000 |
| `commonsense_qa_2.0` | cc-by-4.0 | 2000 |
| `corr2cause` | mit | 2000 |
| `defeasible-nli/atomic` | apache-2.0, MIT License (DPI) | 2000 |
| `dnd_style_intents` | apache-2.0 | 2000 |
| `e-CARE` | BSD 2-Clause License (DPI) | 2000 |
| `esci` | apache-2.0 | 2000 |
| `ethics/deontology` | mit | 1452 |
| `ethics/justice` | mit | 1441 |
| `ethics/virtue` | mit | 1473 |
| `feasibilityQA` | mit | 2000 |
| `fig-qa` | mit | 2000 |
| `FOL-nli` | apache-2.0 | 2000 |
| `HatemojiBuild` | cc-by-4.0 | 2000 |
| `HelpSteer/coherence` | cc-by-4.0, CC BY 4.0 (DPI) | 1268 |
| `HelpSteer/complexity` | cc-by-4.0, CC BY 4.0 (DPI) | 1280 |
| `HelpSteer/correctness` | cc-by-4.0, CC BY 4.0 (DPI) | 1273 |
| `HelpSteer/helpfulness` | cc-by-4.0, CC BY 4.0 (DPI) | 1274 |
| `HelpSteer/verbosity` | cc-by-4.0, CC BY 4.0 (DPI) | 1290 |
| `HelpSteer2/coherence` | cc-by-4.0 | 1150 |
| `HelpSteer2/complexity` | cc-by-4.0 | 1186 |
| `HelpSteer2/correctness` | cc-by-4.0 | 1153 |
| `HelpSteer2/helpfulness` | cc-by-4.0 | 1149 |
| `HelpSteer2/verbosity` | cc-by-4.0 | 1165 |
| `HelpSteer3/edit_quality` | cc-by-4.0 | 2000 |
| `HelpSteer3/feedback` | cc-by-4.0 | 1362 |
| `HelpSteer3/preference` | cc-by-4.0 | 2000 |
| `HelpSteer3/preference_strength` | cc-by-4.0 | 1142 |
| `HelpSteer3/principle` | cc-by-4.0 | 1691 |
| `hh-rlhf/harmless-base` | MIT License (DPI) | 2000 |
| `hh-rlhf/helpful-base` | MIT License (DPI) | 2000 |
| `hh-rlhf/helpful-online` | MIT License (DPI) | 2000 |
| `hh-rlhf/helpful-rejection-sampled` | MIT License (DPI) | 2000 |
| `I2D2` | apache-2.0 | 2000 |
| `it-support-tickets` | cc-by-4.0 | 1826 |
| `lex_glue/case_hold` | cc-by-4.0 | 2000 |
| `lex_glue/ledgar` | cc-by-4.0 | 1606 |
| `logical-entailment` | apache-2.0 | 2000 |
| `lonli` | mit | 2000 |
| `math_qa` | apache-2.0 | 2000 |
| `mindgames` | apache-2.0, Apache License 2.0 (DPI) | 2000 |
| `multilingual/xcopa/et` | cc-by-4.0 | 55 |
| `multilingual/xcopa/ht` | cc-by-4.0 | 57 |
| `multilingual/xcopa/id` | cc-by-4.0 | 64 |
| `multilingual/xcopa/it` | cc-by-4.0 | 54 |
| `multilingual/xcopa/qu` | cc-by-4.0 | 58 |
| `multilingual/xcopa/sw` | cc-by-4.0 | 58 |
| `multilingual/xcopa/ta` | cc-by-4.0 | 55 |
| `multilingual/xcopa/th` | cc-by-4.0 | 56 |
| `multilingual/xcopa/tr` | cc-by-4.0 | 59 |
| `multilingual/xcopa/translation-et` | cc-by-4.0 | 55 |
| `multilingual/xcopa/translation-ht` | cc-by-4.0 | 57 |
| `multilingual/xcopa/translation-id` | cc-by-4.0 | 58 |
| `multilingual/xcopa/translation-it` | cc-by-4.0 | 55 |
| `multilingual/xcopa/translation-sw` | cc-by-4.0 | 56 |
| `multilingual/xcopa/translation-ta` | cc-by-4.0 | 61 |
| `multilingual/xcopa/translation-th` | cc-by-4.0 | 56 |
| `multilingual/xcopa/translation-tr` | cc-by-4.0 | 57 |
| `multilingual/xcopa/translation-vi` | cc-by-4.0 | 56 |
| `multilingual/xcopa/translation-zh` | cc-by-4.0 | 58 |
| `multilingual/xcopa/vi` | cc-by-4.0 | 58 |
| `multilingual/xcopa/zh` | cc-by-4.0 | 58 |
| `multilingual/xcsr/X-CODAH-ar` | mit | 164 |
| `multilingual/xcsr/X-CODAH-de` | mit | 167 |
| `multilingual/xcsr/X-CODAH-en` | mit | 167 |
| `multilingual/xcsr/X-CODAH-es` | mit | 162 |
| `multilingual/xcsr/X-CODAH-fr` | mit | 169 |
| `multilingual/xcsr/X-CODAH-hi` | mit | 165 |
| `multilingual/xcsr/X-CODAH-it` | mit | 167 |
| `multilingual/xcsr/X-CODAH-jap` | mit | 168 |
| `multilingual/xcsr/X-CODAH-nl` | mit | 163 |
| `multilingual/xcsr/X-CODAH-pl` | mit | 163 |
| `multilingual/xcsr/X-CODAH-pt` | mit | 166 |
| `multilingual/xcsr/X-CODAH-ru` | mit | 168 |
| `multilingual/xcsr/X-CODAH-sw` | mit | 170 |
| `multilingual/xcsr/X-CODAH-ur` | mit | 170 |
| `multilingual/xcsr/X-CODAH-vi` | mit | 167 |
| `multilingual/xcsr/X-CODAH-zh` | mit | 165 |
| `multilingual/xcsr/X-CSQA-ar` | mit | 403 |
| `multilingual/xcsr/X-CSQA-de` | mit | 396 |
| `multilingual/xcsr/X-CSQA-en` | mit | 396 |
| `multilingual/xcsr/X-CSQA-es` | mit | 405 |
| `multilingual/xcsr/X-CSQA-fr` | mit | 411 |
| `multilingual/xcsr/X-CSQA-hi` | mit | 401 |
| `multilingual/xcsr/X-CSQA-it` | mit | 404 |
| `multilingual/xcsr/X-CSQA-jap` | mit | 399 |
| `multilingual/xcsr/X-CSQA-nl` | mit | 390 |
| `multilingual/xcsr/X-CSQA-pl` | mit | 397 |
| `multilingual/xcsr/X-CSQA-pt` | mit | 396 |
| `multilingual/xcsr/X-CSQA-ru` | mit | 402 |
| `multilingual/xcsr/X-CSQA-sw` | mit | 401 |
| `multilingual/xcsr/X-CSQA-ur` | mit | 404 |
| `multilingual/xcsr/X-CSQA-vi` | mit | 387 |
| `multilingual/xcsr/X-CSQA-zh` | mit | 399 |
| `PARARULE-Plus` | mit | 2000 |
| `patent-phrase-similarity` | cc-by-4.0 | 2000 |
| `poem_sentiment` | cc-by-4.0 | 1154 |
| `prm800k_dpo/solution` | mit | 2000 |
| `prm800k_dpo/step` | mit | 2000 |
| `procedural-typed-decisions/arithmetic` | apache-2.0 | 2000 |
| `procedural-typed-decisions/entity_belief_tracking` | apache-2.0 | 2000 |
| `procedural-typed-decisions/event_state_reconstruction` | apache-2.0 | 2000 |
| `procedural-typed-decisions/evidence_sufficiency` | apache-2.0 | 2000 |
| `procedural-typed-decisions/multi_view_adjudication` | apache-2.0 | 2000 |
| `procedural-typed-decisions/needle_retrieval` | apache-2.0 | 2000 |
| `procedural-typed-decisions/partial_observation_calibration` | apache-2.0 | 2000 |
| `procedural-typed-decisions/policy_applicability` | apache-2.0 | 2000 |
| `procedural-typed-decisions/policy_under_uncertainty` | apache-2.0 | 2000 |
| `procedural-typed-decisions/record_aggregation` | apache-2.0 | 2000 |
| `procedural-typed-decisions/state_perturbation` | apache-2.0 | 2000 |
| `procedural-typed-decisions/table_lookup` | apache-2.0 | 2000 |
| `prompt-injections` | apache-2.0 | 606 |
| `prost` | apache-2.0, Apache License 2.0 (DPI) | 2000 |
| `qasc` | cc-by-4.0 | 2000 |
| `quartz` | cc-by-4.0 | 2000 |
| `ruletaker` | apache-2.0, Apache License 2.0 (DPI) | 2000 |
| `SDOH-NLI` | cc-by-4.0 | 2000 |
| `shell-safety-v2` | mit | 2000 |
| `sms_spam` | CC BY 4.0 (DPI) | 2000 |
| `snips_built_in_intents` | cc0-1.0, CC0 1.0 (DPI) | 396 |
| `social_i_qa` | CC BY 4.0 (DPI) | 2000 |
| `SpaceNLI` | mit | 2000 |
| `spartqa-mchoice` | mit | 2000 |
| `spartqa-yn` | apache-2.0 | 2000 |
| `stepgame` | mit | 2000 |
| `temporal-nli` | apache-2.0 | 2000 |
| `utilitarianism` | mit | 2000 |
| `WANLI` | cc-by-4.0 | 2000 |
| `winodict` | cc-by-4.0 | 1541 |
| `winowhy` | mit | 2000 |

Filter statistics of this build: rows 2,500,000, not commercial 1,124,896, audit excluded 489,942, not permissive 219,847, too long 6,489, test duplicate 7.

## Used only for evaluation (never trained on)

| Dataset | Licence |
|---|---|
| [Banking77 test](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-4.0 |
| [MASSIVE test / validation](https://huggingface.co/datasets/AmazonScience/massive) | CC-BY-4.0 |
| [typed-decisions test](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 |
| [AG News test](https://huggingface.co/datasets/fancyzhx/ag_news) | unknown |
| [DAIR Emotion test / validation](https://huggingface.co/datasets/dair-ai/emotion) | unknown on HF |
| [tyqiangz multilingual sentiment test / valid](https://github.com/tyqiangz/multilingual-sentiment-datasets) | aggregate; includes research-only corpora |

## Not used for released weights

| Data | Reason |
|---|---|
| tyqiangz multilingual sentiment (train) | Aggregates Amazon Reviews Multi (academic research only), Yelp, IMDB, SemEval and DAIR Emotion; the repository's Apache-2.0 tag does not cover the underlying data. |
| AG News and tweet_eval texts for distillation | Licence unknown; replaced by texts from the licence-filtered mixture. |

### Mixture sources excluded by the licence audit (116)

Their recorded tags looked permissive, but the origin of the text or labels is not: origin of text and labels checked against upstream cards, papers and repositories; tasksource jev_sources.yaml used to map sources to upstreams.

| Source | Reason |
|---|---|
| `acceptability-prediction/binary_votes` | Lau et al. 2015: BNC sentences round-trip machine-translated; BNC licence restricts commercial use |
| `acceptability-prediction/rating_votes` | Lau et al. 2015: BNC sentences round-trip machine-translated; BNC licence restricts commercial use |
| `amazon_counterfactual/en` | Amazon review sentences; upstream amazon-research repo licensed CC BY-SA 4.0 |
| `amazon_polarity/amazon_polarity` | Amazon reviews (SNAP/McAuley scrape); apache tag wrong |
| `arct` | arguments = NYT 'Room for Debate' reader comments (platform/news terms) |
| `args_me` | args.me crawled debate portals (debate.org, idebate, debatewise, debatepedia CC BY-SA); platform/SA text |
| `autotnli` | premise tables from InfoTabS = Wikipedia infoboxes (CC BY-SA) |
| `blog_authorship_corpus/age` | Blogger.com posts; corpus licensed for non-commercial research only |
| `cicero` | dialogues from DailyDialog (CC BY-NC-SA 4.0), MuTual, DREAM (exam material) |
| `cloth` | CLOTH passages from Chinese school English exams scraped from websites; no rights-holder grant (cf. RACE research-only) |
| `CONDAQA` | passages from English Wikipedia (CC BY-SA); tag apache-2.0 covers annotations only |
| `cosmos_qa` | contexts are Spinn3r ICWSM blog posts (research-only corpus); CC BY 4.0 covers annotations |
| `cycic_classification` | CycIC (Cycorp/AI2) release ships no licence; Cyc-KB derived, terms unknown |
| `cycic_multiplechoice` | CycIC (Cycorp/AI2) release ships no licence; Cyc-KB derived, terms unknown |
| `defeasible-nli/snli` | SNLI premises/hypotheses (CC BY-SA 4.0) |
| `defeasible-nli/social` | Social-Chem-101 situations/RoTs (CC BY-SA 4.0; Reddit-sourced situations) |
| `dgen` | compiled from SciQ (CC BY-NC 3.0), MCQL, AI2 science Qs, vocabulary.com; mixed/NC |
| `discovery/discovery` | sentence pairs mined from Aranea/Depcc web crawl; no licence for underlying web text |
| `disrpt/eng.dep.scidtb.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `doc-nli` | DocNLI built from CNN/DailyMail, DUC news + SQuAD (Wikipedia CC BY-SA) |
| `docred` | documents from Wikipedia (CC BY-SA) |
| `dynasent/r1_votes` | Round 1 = Yelp Academic Dataset review sentences (Yelp terms) |
| `dynasent/r2_votes` | Round 2 partly crowd edits of Yelp prompt sentences; Yelp-derived |
| `equate` | subsets from CNN news (NewsNLI), Reddit (RedditNLI), RTE |
| `ethics/commonsense` | commonsense subset includes long Reddit r/AITA posts |
| `FLUTE` | premises from EmpatheticDialogues (CC BY-NC 4.0) + simile/metaphor sentences from mixed prior corpora; card afl-3.0 undocumented |
| `fool-me-twice` | evidence sentences from Wikipedia (CC BY-SA) |
| `github-issue-similarity` | GitHub issue text (user content under GitHub ToS); MIT tag by repackager |
| `goal-step-wikihow/goal` | wikiHow text (CC BY-NC-SA); mit tag wrong |
| `goal-step-wikihow/order` | wikiHow text (CC BY-NC-SA); mit tag wrong |
| `goal-step-wikihow/step` | wikiHow text (CC BY-NC-SA); mit tag wrong |
| `headline_cause/en_simple` | scraped news headlines; CC0 claim by collector not rights holder |
| `health_fact` | PUBHEALTH claims/explanations from fact-check & news sites (Snopes, Politifact, Reuters) |
| `hlgd` | news headlines from 34 outlets; card says licensing relies on fair use |
| `hope_edi/english` | YouTube comments (platform terms) |
| `hover` | HoVer: Wikipedia-derived claims/evidence, dataset CC BY-SA 4.0 |
| `hover-3way/nli` | HoVer: Wikipedia-derived claims/evidence, dataset CC BY-SA 4.0 |
| `hyperpartisan_news` | full news articles (publisher copyright) |
| `jigsaw_toxicity` | Wikipedia talk-page comments (CC BY-SA 3.0) |
| `lex_glue/unfair_tos` | verbatim Terms of Service of YouTube/Facebook/eBay etc.; no rights-holder licence |
| `medmcqa` | exam MCQs + explanations 'collected from open websites and books' (textbook text); apache tag by collector |
| `monotonicity-entailment` | MED (Yanaka 2019) licensed CC BY-SA 4.0 |
| `moral_stories/full` | norms taken verbatim from Social-Chem-101 (CC BY-SA 4.0) |
| `multilingual/AfriSenti-twitter-sentiment/amh` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/arq` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/ary` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/hau` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/ibo` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/kin` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/pcm` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/por` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/swa` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/tso` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/twi` | tweets (X/Twitter terms) |
| `multilingual/AfriSenti-twitter-sentiment/yor` | tweets (X/Twitter terms) |
| `multilingual/disrpt/deu.rst.pcc.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/eus.rst.ert.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/fas.rst.prstc.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/fra.sdrt.annodis.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/nld.rst.nldt.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/por.rst.cstn.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/rus.rst.rrt.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/spa.rst.rststb.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/tha.pdtb.tdtb.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/disrpt/zho.rst.gcdt.rels` | DISRPT corpora: SciDTB (no licence), PCC/ANNODIS/RRT/RST-STB (CC BY-NC-SA), PRSTC (CC BY-NC), CSTNews (research only), LST20/NLDT/GCDT/ERT news/mixed; not permissive |
| `multilingual/masakhanews/amh` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/eng` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/fra` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/hau` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/ibo` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/lin` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/lug` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/orm` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/pcm` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/run` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/sna` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/som` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/swa` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/tir` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/xho` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/masakhanews/yor` | news articles from BBC/VOA/other outlets (publisher copyright) |
| `multilingual/MLMA_hate_speech` | tweets (X/Twitter terms) |
| `multilingual/offenseval_dravidian/kannada` | YouTube comments (platform terms) |
| `multilingual/offenseval_dravidian/malayalam` | YouTube comments (platform terms) |
| `multilingual/offenseval_dravidian/tamil` | YouTube comments (platform terms) |
| `multilingual/x-stance` | upstream ZurichNLP README: dataset CC BY-NC 4.0 (c) smartvote.ch; MIT covers code only |
| `naturallogic` | built from MED/FraCaS sentences (MED CC BY-SA 4.0) |
| `numer_sense` | strict policy (unverified or mixed provenance): OMCS sentence licence not verified |
| `PARADISE` | built from wikiHow warnings/tips (wikiHow CC BY-NC-SA) |
| `prompt-injection-dataset` | strict policy (unverified or mixed provenance): S-Labs card has no provenance; content looks author-generated |
| `Prompt-injection-dataset/full` | neuralchemy card lists WildGuard/JudgeComparison component under 'Research' licence; mixed upstream |
| `PromptShield` | benign/attack data from Alpaca (CC BY-NC 4.0), databricks-dolly (CC BY-SA 3.0), LMSYS-Chat-1M (custom licence) per PromptShield paper |
| `propsegment/nli` | WikiText-103 (Wikipedia CC BY-SA) + NewSHead news (URL-only, re-crawled) |
| `proto_qa/proto_qa` | training questions scraped from Family Feud fan transcription sites |
| `puzzte` | puzzles copied from edugoog.com/testbook.com etc. (source URLs in data) |
| `resnli` | strict policy (unverified or mixed provenance): upstream repo licence not verified; tasksource tag CC-BY-4.0 |
| `scone` | built on MoNLI -> SNLI sentences (CC BY-SA 4.0) |
| `sem_eval_2010_task_8` | strict policy (unverified or mixed provenance): HF card licence blank; CC0 only via DPI; web-sourced sentences |
| `sherliic` | strict policy (unverified or mixed provenance): CC-BY-4.0 via DPI only; patterns mined from ClueWeb |
| `stackoverflow-questions` | Stack Overflow (CC BY-SA); apache tag wrong |
| `starcon` | StArCon: debate-portal arguments; licence unknown on card |
| `sts-companion` | STS 2012-17 companion: forums, news headlines, MT data with mixed/unknown licences |
| `subjectivity` | sentences from copyrighted news articles (Antici et al. 2023) |
| `syntactic-augmentation-nli` | strict policy (unverified or mixed provenance): MNLI premises: small fiction portion (Seven Swords) is CC BY-SA 3.0 |
| `Touche23-ValueEval` | arguments pooled from IBM-ArgQ-30k (CC BY-SA 3.0), NYT, Zhihu, Group Discussion Ideas; mixed/SA/news origin |
| `tracie` | premises are ROCStories stories (no open licence) |
| `TuringBench` | human-written side = scraped news articles (CNN etc.); repack jana4/turingbench-humanized unknown |
| `twitter-financial-news-sentiment` | tweets (X/Twitter terms) |
| `UltraFeedback-paired` | 3rd-party repack of UltraFeedback: prompts from FLAN/ShareGPT/Evol-Instruct (mixed/NC), completions from Llama-2/Bard/GPT etc. under custom model terms |
| `UNLI` | premise/hypothesis pairs are SNLI (CC BY-SA 4.0) |
| `wikimedqa/medwiki` | Wikipedia medical articles (CC BY-SA) |
| `wildguardmix-cleaned/prompt_harm` | strict policy (unverified or mixed provenance): AI2 ODC-By (gated, Responsible Use); ~11% in-the-wild prompts from LMSYS-Chat-1M/WildChat and GPT-4-generated items; 3rd-party cleaned repack |
| `wildguardmix-cleaned/response_harm` | strict policy (unverified or mixed provenance): AI2 ODC-By (gated, Responsible Use); ~11% in-the-wild prompts from LMSYS-Chat-1M/WildChat and GPT-4-generated items; 3rd-party cleaned repack |
| `wildguardmix-cleaned/response_refusal` | strict policy (unverified or mixed provenance): AI2 ODC-By (gated, Responsible Use); ~11% in-the-wild prompts from LMSYS-Chat-1M/WildChat and GPT-4-generated items; 3rd-party cleaned repack |
| `wnut_17/wnut_17` | Twitter/Reddit/YouTube/StackExchange text |
| `wouldyourather` | scraped 'would you rather' site questions + vote counts; CC0 tag unverified |

Research checkpoints trained before this policy (the Banking77 v1/v3 and multi-task runs documented in the README) used some of the data above and are not released.

