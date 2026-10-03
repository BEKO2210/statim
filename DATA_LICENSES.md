# Data licences

Released Statim weights are trained only on data whose licence permits commercial use **and** does
not impose ShareAlike or copyleft terms on the model: Apache-2.0, MIT, BSD, CC0, CC-BY, ODC-By,
AFL-3.0. Excluded: non-commercial or research-only terms, ShareAlike (CC-BY-SA), copyleft (GPL,
AGPL, MPL, ODbL), custom or unknown terms. Datasets used only to *measure* the model are listed
separately; the model never trains on them. The weights listed under [Licence findings](#licence-findings-2026-10-03) do not meet
this rule.

## Licence findings (2026-10-03)

A licence re-audit traced every source of mixtures v5 (163 sources) and v6 (111 sources) back to its
upstream original: the original repository's LICENSE or README, the paper, the project website, or
the terms of the model that generated the text. Where the dataset card and the upstream disagree, the
stricter licence applies.

### Sources that break the data policy

| Source | Mixture | Rows | Recorded as | Found upstream | Problem |
|---|---|---:|---|---|---|
| [ankitkupadhyay/XNLI](https://huggingface.co/datasets/ankitkupadhyay/XNLI) | v6 | 6,200 | apache-2.0 (dataset card) | [facebookresearch/XNLI LICENSE](https://github.com/facebookresearch/XNLI/blob/main/LICENSE): "Attribution-NonCommercial 4.0 International" | non-commercial |
| [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | v6 | 6,200 | OANC licence | the corpus card: most of it under the OANC licence, but the FICTION section includes *Seven Swords* under "Creative Commons Share-Alike 3.0 Unported" | ShareAlike in part of one genre; the mixture did not keep the genre, so fiction rows cannot be ruled out |
| [boun-tabi/nli_tr](https://huggingface.co/datasets/boun-tabi/nli_tr) (`multinli_tr`) | v6 | 6,200 | same terms as MultiNLI | [boun-tabi/NLI-TR](https://github.com/boun-tabi/NLI-TR): "licensed under the same terms as MultiNLI"; the rows keep MultiNLI's order, and about a fifth of the sampled rows are fiction | as above; the loader had no genre filter |
| [dhruv0808/indic_sentiment_analyzer](https://huggingface.co/datasets/dhruv0808/indic_sentiment_analyzer) | v6 | 6,200 | CC-BY-4.0 (card metadata) | the card's own License section: "Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)" | non-commercial |
| [OrSabbach/food-delivery-support-tickets](https://huggingface.co/datasets/OrSabbach/food-delivery-support-tickets) | v6 | 6,200 | MIT | the card: every free-text field was generated with Qwen2.5-3B-Instruct, whose [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) allows non-commercial use only, including for outputs used to train a model | generator licence is non-commercial |
| [Fumika/Wikinews-multilingual](https://huggingface.co/datasets/Fumika/Wikinews-multilingual), Arabic and Persian rows | v6 | 52 | CC-BY-2.5 | the ar.wikinews.org and fa.wikinews.org site licence: "Creative Commons Attribution-Share Alike 4.0" (the other editions are CC BY) | ShareAlike |
| [SDOH-NLI](https://github.com/google-research-datasets/SDOH-NLI) | v5 | 2,000 | cc-by-4.0 | the repository is CC BY 4.0, but the paper ([arXiv 2310.18431](https://arxiv.org/abs/2310.18431)) built it from medical reports scraped from mtsamples.com, which grants no licence | unknown licence of the texts |
| [lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue) `ledgar` | v5 | 1,606 | cc-by-4.0 | neither the LEDGAR paper ([LREC 2020](https://aclanthology.org/2020.lrec-1.155/)) nor the LexGLUE repository states a data licence; the card's Licensing Information says "More Information Needed" | unknown licence |

### Sources that could not be verified

The creator's licence is permissive in each case, but the origin of the text or the terms of the model
that wrote it could not be confirmed:

- v5: circa (the README says CC BY 4.0 but links CC BY-SA 4.0), temporal-nli (no upstream found),
  deepset/prompt-injections (no provenance), sms_spam (its component corpora carry no licence),
  dnd_style_intents and shell-safety-v2 (generator not named);
- v6: Console-AI/IT-helpdesk-synthetic-tickets, Process-Venue Prompt_Routing_Hindi,
  lyon-nlp/clustering-hal-s2s, SINAI/ALIA-es-discriminative-stance-detection, and 15 synthetic sets
  whose generator is not named (sutro, leonvanbokhorst, tanaos ×2, RichardSakaguchiMS, shreyaspullehf,
  declare-lab/CategoricalHarmfulQA, E3-JSI, AshenFdo, Johnson8187, takehika/wanli-ja-nli, Wismut,
  NABA-AI, 3nesdeniz/turkish-conversation-prompt-injection, alusci).

Several sources that pass the licence check contain outputs of models whose terms restrict using them
to train other models: OpenAI (WANLI, prosocial-dialog, Salad-Data GPT-Gen, SPML, hblim, cngchis,
sweatSmile, parts of IRIS_sts and BRIGHTER hin/mar), Gemini (agent_action_safety, Adilbai, IDinsight,
JusteLeo) and Doubao (YiMeng).

### Affected weights

| Weights | Trained on | Affected by |
|---|---|---|
| statim-decide-en-large 0.5.0 | mixture v5 | SDOH-NLI, ledgar |
| statim-decide-multilingual-base 0.4.0 | mixture v5 | SDOH-NLI, ledgar |
| statim-decide-multilingual-base 0.7.0 and the pii, emotion and safety adapters built on it | mixture v8 (v6 and v5) | every row of the first table |

From 2026-10-03 these weights are offered only for noncommercial use under PolyForm Noncommercial
1.0.0. The Small Business and Free Trial licences and the commercial licence do not apply to them,
because rights in the training data that Statim does not hold cannot be passed on. They will be
replaced by models trained only on data that passes the rule below.

### Rule for every future mixture

A source enters a mixture only if all three hold:

1. its upstream licence is confirmed at the source and is one of the licences named at the top of
   this page;
2. the origin of its text is documented, and that text carries no stricter licence;
3. if a model generated or labelled it, the model is named and its terms allow training other
   models on its outputs.

Sources that cannot be verified are left out, and so are outputs of OpenAI and Gemini models, of
Llama, Gemma (up to version 3) and other models under restrictive terms.

### Grounded synthetic pilot inputs

These sources are inputs to the urgency/NLI pilot, not to released weights. The loader reads only
the named training split at the immutable revision and stores downloads outside `data/`.

| Source | Revision and licence evidence | Text used |
|---|---|---|
| FiscalNote/billsum | [`3d8510441c06a3d9dfb32eb0d7f80151730bcc4f`, CC0-1.0](https://huggingface.co/datasets/FiscalNote/billsum/blob/3d8510441c06a3d9dfb32eb0d7f80151730bcc4f/README.md) | US bills, `train` only; `ca_test` is never read |
| launch/gov_report | [`32feeaede49fed993aef070bc4da09263fd0429a`, CC-BY-4.0](https://huggingface.co/datasets/launch/gov_report/blob/32feeaede49fed993aef070bc4da09263fd0429a/README.md) | GAO and CRS reports, `gao_train` and `crs_train` only |
| dennlinger/eur-lex-sum | [`33ecb2d630298e3f912d067aaaa71aaf4ee92404`, CC-BY-4.0](https://huggingface.co/datasets/dennlinger/eur-lex-sum/blob/33ecb2d630298e3f912d067aaaa71aaf4ee92404/README.md) | EU legal acts, language-specific `train` files only |

Qwen/Qwen3-8B (Apache-2.0) writes each item and microsoft/Phi-4-mini-instruct (MIT)
independently labels it. Only matching labels survive. The item-keyed provenance sidecars retain
both model identities and Ollama digests, the source revision and passage hash, and prompt version.

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

### Mixture v6 sources (111 sources, 534,231 items, at most 6200 per source)

Licence checked at the source for every entry (registry `tools/finetune/sources/v6-keep.json`, review notes and attribution in `tools/finetune/sources/v6-research.md`). Every text that occurs in an evaluation suite was removed first (862,438 banned texts).

| Source | Category | Licence | Languages | Items |
|---|---|---|---|---:|
| [3nesdeniz/agentic-prompt-injection-5k](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k) | prompt-injection | CC-BY-4.0 | en | 6,200 |
| [3nesdeniz/turkish-conversation-prompt-injection](https://huggingface.co/datasets/3nesdeniz/turkish-conversation-prompt-injection) | prompt-injection | CC-BY-4.0 | tr | 530 |
| [Adilbai/kz-gov-complaints-data-kz-ru](https://huggingface.co/datasets/Adilbai/kz-gov-complaints-data-kz-ru) | complaint, sentiment | Apache-2.0 | ru, kk | 1,200 |
| [adiprog14/lingrow-support-tickets](https://huggingface.co/datasets/adiprog14/lingrow-support-tickets) | complaint | MIT | en | 6,200 |
| [ai4bharat/IndicSentiment](https://huggingface.co/datasets/ai4bharat/IndicSentiment) | sentiment | CC0-1.0 (AI4Bharat/IndicBERT README; HF card has none) | en, hi, bn, mr, ta, te, ur, gu +6 | 6,200 |
| [alaminxpro/university-students-complaints](https://huggingface.co/datasets/alaminxpro/university-students-complaints) | complaint | CC-BY-4.0 | en | 546 |
| [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) | safety-moderation | CC-BY-4.0 | en | 6,200 |
| [alusci/sms-otp-spam-dataset](https://huggingface.co/datasets/alusci/sms-otp-spam-dataset) | spam-sms | MIT (templated; low value) | en | 6,200 |
| [amyrmahdy/decima-synthetic-decisions](https://huggingface.co/datasets/amyrmahdy/decima-synthetic-decisions) | typed-decisions | cc-by-4.0 (card; fully synthetic, teacher Gemma-4-26B-A4B-it, Apache-2.0 model card) | en, fa, ar, ru | 6,200 |
| [ankitkupadhyay/XNLI](https://huggingface.co/datasets/ankitkupadhyay/XNLI) | nli | apache-2.0 (card); content inherits MNLI/OANC terms | ar, bg, de, el, en, es, fr, hi +7 | 6,200 |
| [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | safety-moderation | MIT | en | 6,200 |
| [AshenFdo/synthetic_blood_request_urgency_dataset](https://huggingface.co/datasets/AshenFdo/synthetic_blood_request_urgency_dataset) | urgency | mit | en | 2,500 |
| [Avature/Job-Title-Similarity](https://huggingface.co/datasets/Avature/Job-Title-Similarity) | similarity | apache-2.0 | de, en, es, fr, it, ja, nl, pl +3 | 4,563 |
| [BEE-spoke-data/consumer-finance-complaints](https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints) | complaint | CC0-1.0 card; CFPB US federal data, narratives published with consumer opt-in consent | en | 6,200 |
| [boun-tabi/nli_tr](https://huggingface.co/datasets/boun-tabi/nli_tr) | nli | same terms as MultiNLI (GitHub boun-tabi/NLI-TR README) | tr | 6,200 |
| [brighter-dataset/BRIGHTER-emotion-categories](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories) | emotion | cc-by-4.0 | hi | 3,629 |
| [brighter-dataset/BRIGHTER-emotion-categories](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories) | emotion | cc-by-4.0 | mr | 3,768 |
| [clips/VaccinChatNL](https://huggingface.co/datasets/clips/VaccinChatNL) | intent-dialogue-act | CC-BY-4.0 | nl | 6,200 |
| [cngchis/Support-Ticket-Router-12K-Cleaned](https://huggingface.co/datasets/cngchis/Support-Ticket-Router-12K-Cleaned) | complaint | Apache-2.0 | en | 6,200 |
| [CohereForAI/aya_redteaming](https://huggingface.co/datasets/CohereForAI/aya_redteaming) | safety-moderation | Apache-2.0 | en, fr, es, ru, ar, hi, sr, tl | 494 |
| [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | language-id | apache-2.0 | 65 incl. ar, de, en, fr, hi, it, ja, nl, pl, pt, ru, es, tr, zh | 6,200 |
| [community-datasets/re_dial](https://huggingface.co/datasets/community-datasets/re_dial) | sentiment | CC-BY-4.0 | en | 6,200 |
| [community-datasets/tapaco](https://huggingface.co/datasets/community-datasets/tapaco) | similarity | cc-by-2.0 (Tatoeba CC-BY 2.0 FR) | en, de, fr, es, it, pt, nl, pl +6 | 6,200 |
| [Console-AI/IT-helpdesk-synthetic-tickets](https://huggingface.co/datasets/Console-AI/IT-helpdesk-synthetic-tickets) | complaint, urgency | MIT | en | 1,000 |
| ConvLab/crosswoz (github thu-coai/CrossWOZ) | intent-dialogue-act | Apache-2.0 | zh | 6,200 |
| [ddrg/super_eurlex](https://huggingface.co/datasets/ddrg/super_eurlex) | topic | MIT card; EUR-Lex reuse authorised incl. commercial with attribution (Decision 2011/833/EU) | bg, cs, da, de, el, en, es, et +16 | 6,200 |
| [declare-lab/CategoricalHarmfulQA](https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA) | safety-moderation | Apache-2.0 | en, zh, vi | 550 |
| [dell-research-harvard/headlines-semantic-similarity](https://huggingface.co/datasets/dell-research-harvard/headlines-semantic-similarity) | similarity | cc-by-2.0 (off-copyright US newspapers) | en | 6,200 |
| [dhruv0808/indic_sentiment_analyzer](https://huggingface.co/datasets/dhruv0808/indic_sentiment_analyzer) | sentiment | CC-BY-4.0 | en, hi, te, ta, kn, or, bn, gu +4 | 6,200 |
| [dvgodoy/CUAD_v1_Contract_Understanding_clause_classification](https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_clause_classification) | topic | CC-BY-4.0 | en | 6,200 |
| [E3-JSI/synthetic-multi-pii-ner-v1](https://huggingface.co/datasets/E3-JSI/synthetic-multi-pii-ner-v1) | pii | mit | en, fr, de, el, nl, it, sl | 2,971 |
| [elvanalabs/sarcasm-statements-90](https://huggingface.co/datasets/elvanalabs/sarcasm-statements-90) | sarcasm | mit | en | 90 |
| [Fumika/Wikinews-multilingual](https://huggingface.co/datasets/Fumika/Wikinews-multilingual) | topic | CC-BY-2.5 (Wikinews) | en, es, fr, de, pt, pl, it, zh +25 | 6,200 |
| [gfissore/arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) | topic | CC0-1.0 (arXiv metadata) | en | 6,200 |
| [asappresearch/abcd](https://github.com/asappresearch/abcd) | complaint | MIT (GitHub LICENSE) | en | 6,200 |
| [bvidgen/Dynamically-Generated-Hate-Speech-Dataset (v0.2.3.csv; NOT tasksource/dynahate mirror tagged gpl)](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) | toxicity-hate | CC-BY-4.0 (upstream README) | en | 6,200 |
| [HLTCHKUST/BiToD (mirror DeepPavlov/BiToD)](https://github.com/HLTCHKUST/BiToD) | intent-dialogue-act | Apache-2.0 | en, zh | 6,200 |
| [PolyAI-LDN/task-specific-datasets/nlupp](https://github.com/PolyAI-LDN/task-specific-datasets) | intent-dialogue-act | CC-BY-4.0 | en | 705 |
| [wwbp/empathic_reactions](https://github.com/wwbp/empathic_reactions) | emotion | cc-by-4.0 | en | 3,719 |
| [GoktugD/turkish-formality-rewrite-500k](https://huggingface.co/datasets/GoktugD/turkish-formality-rewrite-500k) | formality | cc0-1.0 | tr | 6,200 |
| [GoktugD/turkish-intent-classification-1m](https://huggingface.co/datasets/GoktugD/turkish-intent-classification-1m) | intent-dialogue-act | CC0-1.0 (template-generated) | tr | 6,200 |
| [GoktugD/turkish-nli-constructed-1.5m](https://huggingface.co/datasets/GoktugD/turkish-nli-constructed-1.5m) | nli | cc0-1.0 | tr | 6,200 |
| [google-research-datasets/poem_sentiment](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) | sentiment | CC-BY-4.0 | en | 892 |
| google-research-datasets/taskmaster1/2/3 (github Taskmaster TM-1..TM-4) | intent-dialogue-act | CC-BY-4.0 | en | 6,200 |
| [gretelai/gretel-pii-masking-en-v1](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | pii | apache-2.0 | en | 6,200 |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | pii | apache-2.0 | en, fr, de, nl, es, it, sv | 6,200 |
| [hblim/customer-complaints](https://huggingface.co/datasets/hblim/customer-complaints) | complaint | MIT | en | 1,260 |
| [Helsinki-NLP/tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba) | language-id | cc-by-2.0 | 300+ incl. all priority | 6,200 |
| [Helsinki-NLP/tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba) | similarity | cc-by-2.0 (Tatoeba CC-BY 2.0 FR) | en, de, fr, es, it, pt, nl, pl +6 | 6,200 |
| [ibm-research/AttaQ](https://huggingface.co/datasets/ibm-research/AttaQ) | safety-moderation | MIT | en | 1,402 |
| [IDinsight/urgency_detection_maternal_health_synthetic](https://huggingface.co/datasets/IDinsight/urgency_detection_maternal_health_synthetic) | urgency | mit | en | 6,200 |
| jagoldz/gahd (filter via GitHub jagol/gahd gahd_disaggregated.csv) | toxicity-hate | CC-BY-4.0 | de | 5,441 |
| [jmccardle/pulse-sofroniew-emotion-concept-texts](https://huggingface.co/datasets/jmccardle/pulse-sofroniew-emotion-concept-texts) | emotion | cc-by-4.0 | en | 6,200 |
| [joelniklaus/covid19_emergency_event](https://huggingface.co/datasets/joelniklaus/covid19_emergency_event) | topic | CC0-1.0 | en, fr, hu, it, nb, nl, pl | 1,202 |
| [joelniklaus/german_argument_mining](https://huggingface.co/datasets/joelniklaus/german_argument_mining) | argument-mining | cc-by-4.0 | de | 6,200 |
| [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset) | emotion | mit | zh | 6,200 |
| [JusteLeo/French-emotion](https://huggingface.co/datasets/JusteLeo/French-emotion) | emotion | mit | fr | 6,200 |
| [kchawla123/casino](https://huggingface.co/datasets/kchawla123/casino) | intent-dialogue-act | CC-BY-4.0 | en | 3,643 |
| [Kenshiii/synthetic-product-reviews](https://huggingface.co/datasets/Kenshiii/synthetic-product-reviews) | sentiment | CC-BY-4.0 | en | 687 |
| [KhiredNetworks/synthetic-product-reviews](https://huggingface.co/datasets/KhiredNetworks/synthetic-product-reviews) | sentiment | MIT | en | 6,200 |
| [leonvanbokhorst/synthetic-complaints-v2](https://huggingface.co/datasets/leonvanbokhorst/synthetic-complaints-v2) | complaint, sentiment | MIT | en | 6,200 |
| [liri-uzh/cfpb-complaints-mini](https://huggingface.co/datasets/liri-uzh/cfpb-complaints-mini) | complaint | CC0-1.0 (CFPB public domain) | en | 6,200 |
| [llm-for-emotion/Cultural-Emo](https://huggingface.co/datasets/llm-for-emotion/Cultural-Emo) | emotion | mit | ar, de, en, hi, es | 3,999 |
| [lyon-nlp/clustering-hal-s2s](https://huggingface.co/datasets/lyon-nlp/clustering-hal-s2s) | topic | Apache-2.0 card; HAL metadata CC0 | fr | 6,200 |
| [masakhane/InjongoIntent](https://huggingface.co/datasets/masakhane/InjongoIntent) | intent-dialogue-act | Apache-2.0 | en, am, ee, ha, ig, rw, ln, lg +9 | 6,200 |
| [matsuxr/JaGovFaqs-22k](https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k) | similarity | cc-by-4.0 (Japanese government copyright policy) | ja | 6,200 |
| [maximoss/mnli-nineeleven-fr](https://huggingface.co/datasets/maximoss/mnli-nineeleven-fr) | nli | bsd-2-clause | fr | 3,988 |
| [MoritzLaurer/synthetic_zeroshot_mixtral_v0.1](https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1) | nli | apache-2.0 (Mixtral-8x7B outputs) | en | 6,200 |
| [mteb/toxic_conversations_50k](https://huggingface.co/datasets/mteb/toxic_conversations_50k) | toxicity-hate | CC-BY-4.0 (Civil Comments text CC0) | en | 6,200 |
| [NABA-AI/LUB-Saudi-Arabic-Intent](https://huggingface.co/datasets/NABA-AI/LUB-Saudi-Arabic-Intent) | intent-dialogue-act | CC-BY-4.0 (synthetic) | ar | 1,500 |
| [NagaYu/deference-keigo-corpus](https://huggingface.co/datasets/NagaYu/deference-keigo-corpus) | formality | cc-by-4.0 | ja | 3,186 |
| [napsternxg/wands](https://huggingface.co/datasets/napsternxg/wands) | similarity | MIT (wayfair/WANDS) | en | 6,200 |
| [NortheasternUniversity/big_patent](https://huggingface.co/datasets/NortheasternUniversity/big_patent) | topic | CC-BY-4.0 | en | 6,200 |
| [Novora/Tri-Class-Sentiment-Synthetic](https://huggingface.co/datasets/Novora/Tri-Class-Sentiment-Synthetic) | sentiment | CC0-1.0 | en | 6,200 |
| [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | safety-moderation | CC-BY-4.0 | en | 6,200 |
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | safety-moderation | CC-BY-4.0 | en | 6,200 |
| [nvidia/CantTalkAboutThis-Topic-Control-Dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset) | safety-topic-control | CC-BY-4.0 | en | 1,073 |
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | pii | cc-by-4.0 | en | 6,200 |
| [nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) | prompt-injection (indirect) | CC-BY-4.0 | en | 1,220 |
| [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | nli | OANC licence (permissive, commercial OK) per MNLI paper/card; fiction genre mixed incl. CC-BY-SA-3.0 | en | 6,200 |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | toxicity-moderation | Apache-2.0 | en, es, ru, zh, de, fr, pt, it +4 | 6,200 |
| [OpenSafetyLab/Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) | safety-moderation | Apache-2.0 | en | 6,200 |
| [OrSabbach/food-delivery-support-tickets](https://huggingface.co/datasets/OrSabbach/food-delivery-support-tickets) | complaint | MIT | en | 6,200 |
| [pacoreyes/StanceSentences](https://huggingface.co/datasets/pacoreyes/StanceSentences) | stance | apache-2.0 | en | 972 |
| pfb30/multi_woz_v22 (github budzianowski/multiwoz) | intent-dialogue-act | MIT (upstream); Apache-2.0 (card) | en | 6,200 |
| [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) | complaint | CC-BY-4.0 | cs, de, en, es, fr, it, ko, nl +4 | 6,200 |
| [Process-Venue/IntentClassification_Dataset_for_AI_Assistant_Prompt_Routing_Hindi](https://huggingface.co/datasets/Process-Venue/IntentClassification_Dataset_for_AI_Assistant_Prompt_Routing_Hindi) | intent-dialogue-act | Apache-2.0 (text provenance undocumented) | hi | 4,998 |
| [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | prompt-injection | MIT | en | 6,200 |
| [RichardSakaguchiMS/brazilian-customer-service-conversations](https://huggingface.co/datasets/RichardSakaguchiMS/brazilian-customer-service-conversations) | complaint, sentiment | Apache-2.0 | pt | 1,510 |
| [s2pidape/support-ticket-dataset](https://huggingface.co/datasets/s2pidape/support-ticket-dataset) | complaint | CC-BY-4.0 | en | 6,200 |
| [shreyaspullehf/emotion-dataset-20-emotions](https://huggingface.co/datasets/shreyaspullehf/emotion-dataset-20-emotions) | emotion | mit | en | 6,200 |
| [sileod/attempto-nli](https://huggingface.co/datasets/sileod/attempto-nli) | nli | apache-2.0 | en | 6,200 |
| [SINAI/ALIA-es-discriminative-stance-detection](https://huggingface.co/datasets/SINAI/ALIA-es-discriminative-stance-detection) | stance | cc-by-4.0 | es | 2,850 |
| [stjiris/IRIS_sts](https://huggingface.co/datasets/stjiris/IRIS_sts) | similarity | mit | pt | 3,334 |
| [sutro/synthetic-product-reviews-20k](https://huggingface.co/datasets/sutro/synthetic-product-reviews-20k) | sentiment | MIT | en | 6,200 |
| [sweatSmile/sarcastic-dataset](https://huggingface.co/datasets/sweatSmile/sarcastic-dataset) | sarcasm | mit | en | 1,440 |
| [takehika/wanli-ja-nli](https://huggingface.co/datasets/takehika/wanli-ja-nli) | nli | cc-by-4.0 | ja | 6,200 |
| [tanaos/synthetic-emotion-detection-dataset-v1](https://huggingface.co/datasets/tanaos/synthetic-emotion-detection-dataset-v1) | emotion | mit | en | 6,200 |
| [tanaos/synthetic-sentiment-analysis-dataset-v1](https://huggingface.co/datasets/tanaos/synthetic-sentiment-analysis-dataset-v1) | sentiment | MIT | en | 6,200 |
| [tasksource/esci](https://huggingface.co/datasets/tasksource/esci) | similarity | apache-2.0 (amazon-science/esci-data) | en, es, ja | 6,200 |
| [tasksource/help-desk-tickets](https://huggingface.co/datasets/tasksource/help-desk-tickets) | complaint, urgency | CC-BY-4.0 (Mendeley btm76zndnt v3) | en, mixed | 357 |
| [tasksource/it-support-tickets](https://huggingface.co/datasets/tasksource/it-support-tickets) | complaint | CC-BY-4.0 (Zenodo 7648117) | en, de, pt, es | 1,568 |
| [theatticusproject/cuad-qa](https://huggingface.co/datasets/theatticusproject/cuad-qa) | reading-comprehension | CC-BY-4.0 | en | 6,200 |
| [theatticusproject/maud](https://huggingface.co/datasets/theatticusproject/maud) | reading-comprehension | CC-BY-4.0 | en | 6,200 |
| [uoe-nlp/multi3-nlu](https://huggingface.co/datasets/uoe-nlp/multi3-nlu) | intent-dialogue-act | CC-BY-4.0 | am, mr, tr, es | 5,636 |
| [urchade/synthetic-pii-ner-mistral-v1](https://huggingface.co/datasets/urchade/synthetic-pii-ner-mistral-v1) | pii | apache-2.0 | en, fr, it, de, es | 6,200 |
| [vic35get/nhtsa_complaints_dataset](https://huggingface.co/datasets/vic35get/nhtsa_complaints_dataset) | complaint | Apache-2.0 card; NHTSA US federal data | en | 6,200 |
| [Wismut/nym-pii-multilingual-data](https://huggingface.co/datasets/Wismut/nym-pii-multilingual-data) | pii | mit | en, de, fr, es, it, pt, nl, pl +14 | 6,200 |
| [WorkInTheDark/FairytaleQA](https://huggingface.co/datasets/WorkInTheDark/FairytaleQA) | reading-comprehension | Apache-2.0 | en | 6,200 |
| [YiMeng-SYSU/chinese-logic-sentiment-dataset](https://huggingface.co/datasets/YiMeng-SYSU/chinese-logic-sentiment-dataset) | sentiment | Apache-2.0 | zh | 2,176 |
| [3609356 (ClaimBuster)](https://zenodo.org/records/3609356) | claim-detection | cc-by-4.0 | en | 1,032 |

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
