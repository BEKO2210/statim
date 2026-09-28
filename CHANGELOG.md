# Changelog

All notable changes to Statim are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0 the HTTP API may still change
between minor versions; every change is listed here.

## [Unreleased]

### Added
- Mixture v6: five licence-checked sources for the three weakest held-out categories (fact-check
  0.313, emotion 0.586, topic 0.607 in 0.7.0). Licence and provenance evidence for each is in
  `tools/finetune/sources/v6-research.md` (Part F), and each is registered in `v6-keep.json`
  (`source_part` F):
  - Emotion: Horizon-Labs multilingual-zeroshot-synthetic, the Qwen-generated emotion subset, in
    all 14 languages, which adds the first emotion data for pt, ru, it, nl, ar and tr; and NAIST
    LIFE STORY (ja), human-written and CC BY 4.0, from four pinned quarterly files.
  - Fact-check: agentlans/fact-or-opinion, DeepSeek-written rows only, in 11 languages. It is asked
    as the new task `claim_detection` (fact / opinion / both / neither, plus "does it state a
    verifiable fact?"), with templates and glosses in all 14 languages.
  - Topic: US bills 2017-2023 with their CRS policy area (en), and Japanese statutes with their
    e-Gov law field (ja). Both have single-label gold.
- `tools/finetune/mixture_v6/emotion_taxonomy.py`: one emotion taxonomy (Ekman's six emotions plus
  love and neutral). Emotion sources opt in with `"emotion_taxonomy": "basic8"`. A row whose label
  has no class is dropped rather than forced into a class. The existing emotion sources are
  unchanged.
- Offline tests for the new loaders, adapters and taxonomy (`test_label_fixes.py`). A new test
  also checks that no new source is a `bench/eval_categories.py` held-out source.

## [0.7.0] - 2026-09-28

### Added
- `statim-decide-multilingual-base` 0.7.0, fine-tuned from 0.4.0 on the fixed mixture v6 (111 of 112
  licence-checked sources, 534,231 items) plus the licence-filtered v5 mixture, with stronger
  distillation from 0.4.0 so it keeps what 0.4.0 knew. Held-out results against 0.4.0:
  - decision categories (14 categories, 35 language cells, texts that occur in training removed):
    0.748 macro accuracy, up from 0.559 (+18.5 points pooled); reading 0.927, stance and urgency
    0.893, PII 0.856 over 11 languages, similarity 0.833, sentiment 0.800, formality 0.773,
    complaint 0.767, intent 0.753, NLI 0.747, safety 0.727, topic 0.607, emotion 0.586, fact-check
    0.313;
  - typed-decisions 0.763 (0.7585), Banking77 0.914 (0.9035), MASSIVE 0.800 (0.772); AG News 0.9295
    and DAIR Emotion 0.504 within noise of 0.9315 and 0.5265;
  - ticket triage (500 Banking77 tickets, q8_0 on CPU): 0.908 (0.896); at `min_confidence` 0.9,
    2 % escalated and 0.920 on the rest (4 % and 0.915 before);
  - gate: 23 significant gains, 66 within noise, no regression after Holm-Bonferroni (one nominal
    drop, Belebele German, near chance for both models); validation mean 0.7340 (0.7404).
- Model cards list the decision categories per language cell; the site's gate grid shows them as
  their own group. Both use the gate's corrected regression rule.
- `tools/finetune/mixture_v6/audit.py`, a content audit that loads a sample of every enabled
  source. It flags serialized options, numeric options, constant labels and question/option
  mismatches. The `mixture-audit` workflow runs it together with the offline adapter tests.
- `bench/eval_categories.py`: one held-out suite per decision category (sentiment, emotion,
  complaint, NLI, safety, reading comprehension, similarity, topic, intent, stance, formality,
  urgency, fact-check, PII), built from the test or otherwise unused splits of the mixture v6
  sources through the v6 adapters (first instruction paraphrase, fixed option order), 150 seeded
  stratified items per suite and language, evaluated over HTTP. Every pooled suite text is in the
  mixture's banned set (`eval_texts.py`), and `gate.py` checks the suites as a `categories` family.
  `--exclude-mixture` (`gate.py eval --mixture`) drops items that share a text with the training
  mixture; gate.py compares a category cell only on the same pool fingerprint and keeps zero-shot
  or biased cells (reported with a reason) out of the family.

### Changed
- Promotion gate (`tools/finetune/gate.py`): per-suite regressions are one-sided z-tests with
  Holm-Bonferroni over all compared suites (family-wise error 5 %); the validation mean may fall by at
  most one point; at least one family must improve significantly. Family-level pooled regressions
  still reject. With 88 suites the old rules (any suite beyond 2 SE, validation mean strictly higher)
  rejected an equally good challenger about 94 % of the time; a null simulation in
  `tools/finetune/test_gate.py` shows 89 % false rejections for the old per-suite rule and 5 % with
  Holm.

### Fixed
- Mixture v6 labels. Several v6 adapters wrote a constant or wrong gold label, and training
  would have learned it:
  - toxicity/moderation: "not toxic", prosocial-dialog `__casual__`, Aegis 1.0 votes and the
    oasst2 crowd votes all became "violates a policy";
  - similarity: every pair of tapaco, headlines, JaGovFaqs, ESCI and WANDS was "unrelated";
  - PII: span lists were used as class labels;
  - ClassLabel ids and card-documented ints were offered as options "0", "1", "2";
  - star ratings were offered as sentiment options;
  - multi-label rows (BRIGHTER, aya_redteaming, Wikinews, arXiv, NHTSA) got an arbitrary first
    label as gold;
  - the complaint sources always answered "is this a complaint?" with yes;
  - MAUD and FairytaleQA gave constant answers;
  - reading contexts revealed the answer by their length.

  The fixes:
  - Hub sources are sampled over the whole split (seeded strata over the parquet revision)
    instead of the head of the stream, so sorted and label-cycling splits yield every label.
  - A yes/no question that one source answers the same way in >= 97 % of its items is dropped.
  - Disabled with a reason: SimpleSafetyTests, MultiJail and OR-Bench (benchmarks), QuALITY
    (articles exceed the window), humor-greats and Lakera gandalf (positives only), MELO and
    washenkov (no pairs can be built), hass-intent-templates (template syntax).
- Category topic suite: after the v6 adapter fixes, the registry maps big_patent's CPC letters itself;
  the suite no longer renames them first, so the topic pool is no longer empty (#21).

## [0.6.2] - 2026-09-28

### Added
- Films on the site. A new "Watch it decide" section plays the 60-second film (16:9 on landscape
  screens, 9:16 on portrait screens) next to the 45-second story, and "One pass. Every option scored."
  plays the 30-second showreel in the same two formats. Nothing loads before the play button: the
  posters are 12 to 28 kB and the home page stays at 593 kB. Open Graph video tags let chat apps
  preview the film.
- README: a ten-second animation (animated WebP, portrait on narrow screens) that links to the film.
  GitHub removes `<video>` from READMEs, so the film itself plays on the site.
- Model cards (`tools/release/hf_publish.py`) embed the film as a `<video>` element.

## [0.6.1] - 2026-09-28

### Fixed
- The `security` and `security_model` tests no longer fail when run as root (Docker and many cloud
  sandboxes): root can read a mode-000 file, so the unreadable-key-file check is skipped for root
  and still runs for every other user.
- `REPRODUCE.md`, corrected by an independent clean-room run on a fresh CPU-only machine (#16):
  section 1 clones the repository, uses v0.5.2 and checks the model against its `SHA256SUMS`;
  section 2 initialises the ggml submodule, installs the converter packages and expects 10 CPU tests
  (4 more per GPU backend); `huggingface-cli`, which exits 1 in `huggingface_hub` 1.x, is replaced by
  `hf download`; notes for CPU-only machines. The run is in `docs/reproductions/clean-room.md`: ticket
  triage 0.896 and 0.915 exactly, typed-decisions and Banking77 exactly, AG News and Emotion within
  1 and 4 rows of 2,000 (PyTorch reference on CPU instead of CUDA).

### Added
- `tools/finetune/data_licenses.py --v6 / --synth`: `DATA_LICENSES.md` can list the mixture v6
  sources (source, category, licence, languages, items) and the synthetic gap data (generator, its
  licence, verification rate).

## [0.6.0] - 2026-09-28

### Added
- Opt-in server-side micro-batching for concurrent `POST /v1/systemone` calls via
  `--batch-window-ms` (default 0/off) and `--max-batch` (default 16). Compatible requests share the
  existing packed batch execution path while retaining independent responses, deadlines, admission
  accounting and request IDs. Prometheus exposes `statim_batch_size` and `statim_batch_wait_ms`
  summaries, and a CPU live-server test gates packed/unpacked answers at 1e-4.
- Mixture v6 builder (`tools/finetune/mixture_v6/`): 121 training sources, each with a
  commercial-use, non-ShareAlike licence checked at the source (registry
  `tools/finetune/sources/v6-keep.json`, review notes in `v6-research.md`), task-specific adapters and
  question templates in 14 languages, removal of every text that occurs in an evaluation suite
  (862k texts from 20 suites), one dev/train side per text across all sources, and one worker
  process per source so memory stays flat over the whole build.
- Synthetic gap data v2 (`tools/synth/`), generated only by a local model (qwen3:8b through Ollama)
  for categories without enough licence-clean human data: target labels weighted by acceptance
  rate, a blind second answer that must match the gold label, fixed question paraphrases, varied
  openings and details, near-duplicate removal with multilingual-e5-small (cosine >= 0.92), and a
  quality report that ends in SCALE UP, REVISE or INSUFFICIENT DATA.

### Changed
- `train_multitask.py`: with `--budget mixture=N`, a v6 mixture is shared over its categories by
  temperature mixing (`--mixture-temperature`, default 2) and dev accuracy is logged per category.
  The mixture is read as a stream (about 2 GB less RAM for a mixture of a million rows), and
  distillation texts no longer come from the held-out dev slice.

## [0.5.2] - 2026-09-27

### Changed
- Playground: the review threshold defaults to 0.6, so answers the model is unsure about show
  "Needs review" out of the box (the demo's German ticket splits urgency between "soon" and
  "critical"); it can still be switched off.
- `docs/API.md` states that `action.act_probability` saturates at 1.0 on the published
  checkpoints and points to `min_confidence` / `escalate` as the abstain mechanism.

## [0.5.1] - 2026-09-27

### Added
- `REPRODUCE.md`: how to check every published number, from the ten-minute ticket-triage run to the
  full gate evaluation of a published model and its comparison with the base checkpoint, with the
  values to expect.

### Changed
- `gate.py` runs on machines without a Vulkan build: `STATIM_BIN`, `STATIM_GATE_DEVICE`,
  `STATIM_GATE_PORT` and `STATIM_CONVERT_PY` override the binary, device, port and converter.
- Website hero shows the 0.5.0 English model's real answers; the demo Space Dockerfile defaults to
  the current release.

## [0.5.0] - 2026-09-27

The first English Statim Decide model, published weights on Hugging Face, a public demo, a CUDA
backend, and a playground anyone can use without writing JSON.

### Results
- `statim-decide-en-large` (ModernBERT-large encoder, fine-tuned from the English Laya checkpoint
  with `train_multitask.py --clean` and 8-bit AdamW; best epoch 11 of 12, chosen on validation data
  only). The gate reports PROMOTE against the English base checkpoint on 54 held-out suites: 11
  significant gains, 0 regressions. typed-decisions 0.361 → 0.768, on par with the best published
  result (meraGPT 0.768; laya-typed-decisions 0.766, Jev 0.727); Banking77 0.550 → 0.928
  (supervised MPNet 0.941); MASSIVE English 0.533 → 0.867; HWU64 0.607 → 0.833. Suites never trained
  on stay within noise: pooled zero-shot +1.25 points (rows) / +0.76 (suites), AG News
  0.9425 → 0.939, DAIR Emotion 0.5945 → 0.588.
- Correction: the 0.4.0 model scores 0.9315 on AG News zero-shot, not 0.9385 as the 0.4.0 notes,
  roadmap and site said; 0.9385 belonged to the earlier Banking77 fine-tune.
- Known weakness, measured: sentiment is deliberately never trained (it serves as a zero-shot
  suite), and the models rarely choose a "mixed" sentiment even for explicitly mixed reviews.

### Added
- CUDA backend (`-DSTATIM_CUDA=ON`) with an exact f32 mode by default (TF32 disabled, non-flash
  attention, because CUDA flash attention converts K/V to f16); the four `*_cuda` parity tests pass
  like the CPU and Vulkan gates. `--gpu-fast` enables the fast f16 paths.
- Selective prediction via the `min_confidence` decision option and `--min-confidence` server default;
  responses annotate answers below an active threshold with `escalate: true`. Python and TypeScript
  clients send it and parse `escalate`.
- Published weights on Hugging Face under professional names: `Beko2210/statim-decide-multilingual-base`
  (the 0.4.0 weights) and `Beko2210/statim-decide-en-large` (this release), each as f32 and q8_0
  GGUF plus the checkpoint, with model cards generated from the gate evaluation
  (`tools/release/hf_publish.py`); the API reports these names.
- Public demo: a Hugging Face Docker Space (`deploy/hf-space`) that runs the release binary and the
  published model with public-demo limits.
- Playground rebuilt for people who do not write JSON: text on one side, questions as cards with
  a type switch and option inputs, each answer shown inside its question with bars that resolve
  together, a review threshold, a developer view (request, response, cURL, Python, history), four
  worked examples including German, and the brand design with embedded font subsets.
- `examples/ticket-triage`: real Banking77 tickets routed with three questions in one request; the
  published q8_0 model measures 0.896 on a seeded 500-ticket sample and 0.915 on the 96 % answered
  at a 0.9 review threshold.
- Synthetic training data pipeline (`tools/synth`) that uses only a local Apache-2.0 generator
  (qwen3:8b via Ollama), with attribute-driven seeds, balanced answer targets, an independent
  verify pass, near-duplicate and test-overlap filters, and a quality report with a scale-up verdict
  (`tools/synth/report.py`).
- `train_multitask.py --optim adamw8bit` (ModernBERT-large now trains on an 8 GB GPU) and int32
  token id storage.

### Changed
- Model weights can now be used under PolyForm Small Business 1.0.0 (free commercial use for
  companies below 100 people and 1 M USD revenue) and PolyForm Free Trial 1.0.0 (any company may
  evaluate them for fewer than 32 days), in addition to PolyForm Noncommercial 1.0.0 and the
  commercial licence. Licence texts are embedded verbatim in `LICENSE-MODEL.md`.
- The website moved into the repository (`site/`) with a QA suite and deployment from `main`.

### Fixed
- The release workflow's upload job sets `GH_REPO`, so release binaries attach without a checkout.
- The SDK integration tests no longer pin the server version.

## [0.4.0] - 2026-09-27

A stronger licence-clean model trained on five times more audited data, official client SDKs,
release packaging with binaries on every release, and a stricter no-harm gate.

### Results
- 0.4.0 model (multilingual checkpoint, `train_multitask.py --clean` on mixture v5: 163 audited
  tasksource sources plus Nemotron-Safety, IndicGuard, MINDS-14 and SNIPS, MASSIVE 2,000 rows per
  language, 20 epochs with early stopping; best epoch 19). Against 0.3.0 the gate reports
  PROMOTE: typed-decisions 0.6905 → 0.7585 (above Jev's 0.727 and the dataset's teacher agreement
  0.735), MASSIVE over 12 languages 0.733 → 0.772, Banking77 0.891 → 0.903, HWU64 0.760 → 0.820,
  Belebele 0.273 → 0.310. Zero-shot suites pooled: −0.3 points (rows) / −1.6 (suites), both within
  two standard errors; the next run strengthens distillation to reverse that trend. Against the
  base checkpoint: trained tasks +41 points, sentiment +3.4 (significant), zero-shot within noise.
  Chart: `assets/diagrams/results-0.4.0-*.svg`. Weights are not yet published.
- Context length was measured, not assumed: 13.4 % of training items exceed 512 tokens, 2.4 % 1,024
  and 0.2 % 2,048, and no evaluation item exceeds 1,024. typed-decisions: 0.756 at max_len 512,
  0.7585 at 1,024 and at 2,048 (identical).

### Added
- `gate.py` also pools each suite family (trained, zero-shot, sentiment), row- and suite-weighted,
  so a drift spread over many small suites counts as a regression even when no single suite is
  significant.
- `train_multitask.py --massive-langs` and `--max-len` (per-language MASSIVE selection; context
  length override saved with the model).
- Official client SDKs for the HTTP API: Python package `statim` in `clients/python`
  (standard library only) and TypeScript package `@statim/client` in `clients/js`
  (`fetch`, no runtime dependencies). Both expose `decide`, `decide_batch`, `models`,
  `health`, and `ready`, typed choice, score, and yes/no answers, request IDs, and
  retries with backoff for HTTP 503.
- GitHub release packaging (runs when a release is published) for portable Linux x86-64 CPU and Vulkan binaries, including
  licence and deployment documents plus published SHA-256 checksums; model weights remain separate.
- A non-root Vulkan container image with Mesa and NVIDIA Container Toolkit deployment options.
- A production deployment guide covering hardened systemd and Docker operation, TLS reverse proxying,
  authenticated Prometheus scraping, health/readiness probes, and resource ceilings.

## [0.3.0] - 2026-09-27

A licence-clean multi-task model that passes a no-harm gate, a licensing model for commercial use,
enterprise security hardening, a zero-shot benchmark and full API documentation.

### Results
- The 0.3.0 model (multilingual checkpoint fine-tuned with `train_multitask.py --clean` on
  licence-audited data only) against the base checkpoint, evaluated by `tools/finetune/gate.py` on
  54 held-out suites: MASSIVE intents over 12 languages 0.340 → 0.733 (Arabic 0.200 → 0.613,
  Hindi 0.267 → 0.673), Banking77 0.5175 → 0.891, typed-decisions 0.351 → 0.6905, HWU64 (sibling
  of MASSIVE, overlapping rows removed) 0.500 → 0.760; zero-shot suites never trained on stay
  within noise or improve (GoEmotions +6.0, SIB-200 +2.5, SemRel +2.9, Belebele +2.7, FarsTail
  −2.0, DAIR Emotion −1.5, AG News +0.1 points). 16 significant gains, 0 significant
  regressions; calibration error roughly halves. Chart: `assets/diagrams/results-0.3.0-*.svg`.
  Weights are reproducible with the scripts and not yet published.

### Added
- Release results chart generated from gate evaluations (`tools/diagrams/gate_chart.py`).
- Updated `docs/API.md` and `docs/openapi.yaml` for the hardened server (Codex, checked against a
  running server).
- Licensing model: source code stays Apache-2.0; model weights published by Statim are licensed
  under PolyForm Noncommercial 1.0.0 (`LICENSE-MODEL.md`, verbatim official text), commercial use
  needs a paid licence (`COMMERCIAL.md`).
- `DATA_LICENSES.md`, generated by `tools/finetune/data_licenses.py`: base models, training data of
  released weights with licences, evaluation-only data, and data excluded from releases.
- `train_multitask.py --clean`: commercial-clean training (no tyqiangz sentiment, distillation texts
  from the licence-filtered mixture).

### Changed
- `build_mixture.py` keeps only rows whose every listed licence is permissive (Apache-2.0, MIT, BSD,
  CC0, CC-BY, ODC-By, AFL-3.0); ShareAlike, copyleft, custom and unknown terms are excluded.

### Security
- Fail closed when any explicitly configured key file/environment value cannot supply valid
  keys. Log unauthenticated local mode explicitly; require the production environment file/key.
- Preflight JSON with depth, node, object-width, key-length, and duplicate-key checks before
  constructing the ordered DOM, preventing deeply nested crashes and quadratic wide-object parsing.
- Bound HTTP workers and pending sockets; authenticate and admit inference requests before
  body reception. Reject oversized declared bodies without draining, and enforce absolute
  header/body deadlines even when a peer keeps sending bytes.
- Bound field sizes and aggregate batch/ensemble/calibration/consensus work, token budgets,
  attention-memory estimates, and response bytes. Add cooperative inference deadlines and
  systemd memory/CPU/task/file-descriptor ceilings without changing graph packing or math.
- Replace the entry-count-only calibration cache with a byte-bounded LRU keyed on the validated
  question only, so unknown fields (still ignored, as by laya.serve) are never retained; a test
  sends 64 MiB of ignored metadata and checks that memory stays flat.
- Allowlist request IDs and serialize log records as JSON to prevent log injection.
- Validate signed/unsigned token and ensemble budgets before integer narrowing, including
  CLI defaults; reject effective sequence lengths beyond model capacity with 422.
- Upgrade vendored cpp-httplib from 0.26.0 to 0.58.0. Explicitly reject simultaneous
  Content-Length/Transfer-Encoding (including zero lengths) and duplicate Content-Length headers.
- Require bearer auth for `/metrics` and `/v1/models` when configured. Keep `/health` and
  `/ready` open; reduce `/health` to status and version. `/v1/models` now also reports each
  model's compute device, and the playground reads models and device from there.
- Use local timestamp storage with `gmtime_r` (`gmtime_s` on Windows) for concurrent logs.
- Install a global HTTP exception handler with fixed client errors and server-only details.
- Add C++, CPU model-backed, and live HTTP security regressions alongside existing parity gates.

## [0.2.1] - 2026-09-27

Diagrams in the brand style for the README.

### Added
- README diagrams in the brand style, each in a light and a dark variant (switched by the reader's
  colour scheme): request-to-decision architecture (replaces the ASCII sketch), GPU vs. CPU
  throughput and latency, and the fine-tuning results. Sources in `assets/diagrams/`.

## [0.2.0] - 2026-09-27

GPU inference, a production-grade playground, and a reproducible fine-tuning and evaluation toolkit.

### Added
- **Vulkan GPU backend.** `--device cpu|gpu|vulkan|<name>` (or `STATIM_DEVICE`) selects the ggml
  backend; weights are copied to VRAM once. Exact f32 by default: ggml-vulkan's f16 matmul paths
  moved logits by up to 0.12, so they are disabled unless `--gpu-fast` / `STATIM_GPU_FAST=1`.
  Build with `-DSTATIM_VULKAN=ON`, which also registers the parity gates on the GPU. RTX 3070 vs.
  Ryzen 7 5800X: 7.7–8.9× HTTP throughput, parity 240/240 at max |Δlogit| 8.8e-5 / 1.6e-4.
- **Option budget per request.** `max_len` / `head_max_len` request fields (as in Laya's
  `predict_batch`) and server defaults `--max-len` / `--head-max-len`. Many-option questions
  (e.g. 77 intents) otherwise see one subword per option.
- **Playground redesign.** Two-pane layout, live server status, model details from `/v1/models`,
  API key dialog, light/dark theme, examples (including a German support ticket with a separate
  revenue-impact question), question templates, JSON linting, per-type result views with
  confidence meters, JSON and cURL tabs, session history, keyboard shortcut.
- **Fine-tuning toolkit** (`tools/finetune/`):
  - `train_banking77.py`: Laya's RLCD recipe on one task, frozen token embeddings, typed-decisions
    replay and `--distill` learning-without-forgetting replay.
  - `train_multitask.py`: Banking77 + MASSIVE (51 languages) + multilingual sentiment (12
    languages); train rows that also occur in a test split are dropped (14,121 MASSIVE and 452
    sentiment rows, per language in `bench/results/multitask_train_test_overlap.json`); per-epoch
    task budgets, LR warmup and EMA weights.
  - `merge.py`: model soup, task arithmetic and TIES merging of checkpoints from one base.
  - `build_mixture.py`: licence-clean training mixture from `tasksource/tasksource-jev-typed-decisions`
    (commercial rows only, per-source cap, evaluation and emotion sources excluded, exact-match
    dedup against every reported test split).
  - `eval_laya.py` (test suites on GPU) and `eval_dev.py` (model selection on validation data only).
- **Multilingual benchmark** `bench/eval_multilingual.py`: MASSIVE intents (59 labels) and
  multilingual sentiment, 12 languages each, seeded stratified samples with identical MASSIVE rows
  across languages; results for the base checkpoints in `bench/results/`.
- `CHANGELOG.md` and `docs/ROADMAP.md`.
- **Brand identity** (`assets/brand/`): logo mark (one pass meeting a column of options, one of
  them chosen), custom monoline wordmark whose i-dot repeats the decision point, light and dark
  lockups, favicon, PNG icons (16–512 px) and a 1280×640 social preview. Used in the README header
  and the playground.

### Changed
- `/health` and the startup log report the actual compute device instead of a fixed `"cpu"`.
- The engine raises `max_len` with a raised `head_max_len` so the state keeps at least 128 tokens.
  This never binds at the checkpoint defaults; CPU and GPU parity are unchanged.
- The model parity test prints per-state deviations (`STATIM_VERBOSE`) and can run one graph per
  item (`STATIM_BATCH1`).

### Fixed
- Mixture training: items sharing a document with the held-out mixture slice are dropped, so the
  slice no longer inflates checkpoint selection (found by a pre-training audit).
- `eval_dev.py` refuses models trained with fewer than 500 held-out Banking77 rows.
- `--gpu-fast` help text understated the logit drift (up to ~0.12, not ~1e-2).
- Playground: the mode selector silently overwrote the chosen model. `consensus` is now a model
  option, offered only when both checkpoints are loaded.

### Results (weights are reproducible with the scripts; not shipped)
- Banking77 fine-tune v3 (`--distill 6000 --epochs 5`), first 2,000 test rows: 0.4885 → 0.8655,
  ECE 0.37 → 0.04; held-out Emotion 0.532 → 0.528 and AG News 0.938 → 0.9385 (within noise).
- Multi-task checkpoint (experimental): MASSIVE macro over 12 languages 0.340 → 0.689, multilingual
  sentiment 0.559 → 0.654, but Banking77 0.8275 (−3.8 vs. v3). Not yet recommended.

## [0.1.0] - 2026-09-26

First release: C++20 runtime for Laya decision checkpoints on ggml, native tokenizer (identical to
Hugging Face on 3,906 cases), parity gates against the official package, HTTP server with the
Jev/Laya `POST /v1/systemone` protocol, batching, consensus mode, contextual calibration, worker
pool, auth, Prometheus metrics, playground, Docker and systemd packaging.

[Unreleased]: https://github.com/BEKO2210/statim/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/BEKO2210/statim/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/BEKO2210/statim/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/BEKO2210/statim/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/BEKO2210/statim/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/BEKO2210/statim/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/BEKO2210/statim/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/BEKO2210/statim/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/BEKO2210/statim/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/BEKO2210/statim/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/BEKO2210/statim/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/BEKO2210/statim/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/BEKO2210/statim/releases/tag/v0.1.0
