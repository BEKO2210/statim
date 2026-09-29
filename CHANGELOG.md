# Changelog

All notable changes to Statim are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0 the HTTP API may still change
between minor versions; every change is listed here.

## [Unreleased]

## [0.8.7] - 2026-09-29

### Changed
- The README reads well on a phone. On narrow screens GitHub squeezes tables, so cells of prose
  wrapped word by word; "At a glance" took 181 px per row at 390 px width.
  - Every table now holds short cells. Protocols, conditions and ratios moved to short lists
    directly below. Rendered with GitHub's own Markdown API at 390 px, no table exceeds 69 px per
    row or scrolls sideways.
  - The link line became a list: live demo, models and adapters, the example, and reproduction.
  - "At a glance" compares Statim with Qwen3-8B.
  - Models, API, performance, baselines and the model gates use narrow, natural labels.
  - The licence is a list.
  - Every number of the previous README is still there and keeps its condition next to it, for
    example 2.1× with 4 clients. A script and a read-only review checked this.
- `check_docs.py` FACTS can be scoped to a section heading, so the same label in two README tables
  is read in its own table. A test covers it.

## [0.8.6] - 2026-09-29

### Changed
- A first contact in three steps. The README quick start is download (with checksum checks), start
  and ask. It no longer clones the repository, and it no longer starts a second server with another
  model file.
  - The quick start shows the real answer of the published q8_0 model, with the `legend` that gives
    `score` its scale, and one sentence on what `choice`, `score` and `noul` return.
  - The block was run verbatim in an empty folder with the v0.8.5 release: both checksums OK, the
    answer identical to the excerpt.
- The README opens with what Statim does in plain words, and states the release binary's platform:
  x86-64 with AVX2.
- Building from source, GPU backends and quantization moved to `docs/BUILD.md`. It says that release
  binaries exist only for Linux x86-64 and everything else builds from source.
- The research detail of "Results" (gate method and charts, consensus, calibration and quantization,
  many-option tasks) moved verbatim to `docs/RESULTS.md`. The README keeps the evidence tables.

### Fixed
- The engine binary is 5.5 MB (5,460,960 bytes in the v0.8.5 CPU release), not the 3.3 MB measured
  on 2026-09-26, in the README and on the site. A `check_docs` fact keeps the copies equal.

## [0.8.5] - 2026-09-29

### Added
- The safety adapter for statim-decide-multilingual-base 0.7.0 is published
  ([Beko2210/statim-decide-multilingual-base-safety](https://huggingface.co/Beko2210/statim-decide-multilingual-base-safety)).
  Its first run gained +8.0 points on 150 items, inside the noise band. A replication, registered
  in `docs/ADAPTERS.md` before it ran, tested the same file on 1,350 fresh items: 0.708 to 0.806,
  +9.78 points, 2 SE 3.28, promote. On the published f32 base file it reproduces the replication
  cell for cell. On q8_0, as runtime LoRA, it gains +10.30 points.
- `bench/eval_categories.py --skip N` and `tools/finetune/lora_experiment.py --eval-skip N` score
  only items an earlier run never saw: the stratified draw is prefix-stable, so a draw of 1,500 starts
  with the 150 items of the first run. Adapter cards state the fresh sample.
- Adapter cards name sources and licences without the registry's working notes, and show Qwen3-8B
  only when the items are the ones BASELINES measured.

## [0.8.4] - 2026-09-29

### Added
- The first published category adapters for statim-decide-multilingual-base 0.7.0, on Hugging Face:
  [PII](https://huggingface.co/Beko2210/statim-decide-multilingual-base-pii) (+5.46 points over
  11 languages, 0.856 to 0.910) and
  [emotion](https://huggingface.co/Beko2210/statim-decide-multilingual-base-emotion) (+4.75 over
  8). On the published f32 base file they reproduce the experiment cell for cell. On the q8_0 file,
  as runtime LoRA, they gain +5.52 and +5.33 points. Each repository carries the GGUF, the PEFT
  source, the training record, the evaluation files, a NOTICE with the CC-BY attributions, and
  checksums.
- `tools/release/hf_publish_adapter.py` builds such a package from a LoRA experiment's outputs:
  - It refuses an adapter the gate did not promote.
  - It checks that every published base file loads the adapter with the fingerprint the adapter
    records, and it copies the evaluations of the published files.
  - It writes the model card and NOTICE from the experiment's files, and it removes local paths.
  - With `--upload`, it compares the remote SHA-256 with the local checksums.
  - `tools/release/test_hf_publish_adapter.py` covers it, and CI runs it.
- `check_docs.py` also guards the Statim and Qwen3-8B numbers that ROADMAP repeats for the five
  weak categories, and the published adapters' numbers.

## [0.8.3] - 2026-09-29

### Added
- LoRA adapters in the official Python and TypeScript SDKs. `decide` and `decide_batch` take
  `adapter` (a loaded adapter's name, `"auto"` or `"none"`). Results carry `routing.adapter` and
  `routing.adapter_reason`, and `models()` lists each model's adapters (`Adapter`: id, source, mode,
  rank, alpha, pairs, pairs_applied, categories, bytes). Responses from servers without adapters
  parse as before. Both suites have a live test against a server with an adapter
  (`STATIM_ADAPTER_URL`, `STATIM_ADAPTER_NAME`).

## [0.8.2] - 2026-09-29

### Added
- `tools/release/check_versions.py`: the engine version in `CMakeLists.txt` must match every
  place that repeats it (server, SDKs, Space Dockerfile, site, release downloads in README,
  REPRODUCE.md and the ticket-triage example, client READMEs, API reference and OpenAPI spec);
  CI runs it first, and `--set x.y.z` bumps all of them for a release.
- `tools/docs/check_docs.py`, run by CI before the build: every tracked Markdown file and the site
  are checked for links and anchors (also github.com links into the repository), repository paths,
  `statim` and script flags in documented commands, and release downloads of the current version.
  `docs/API.md` and `docs/openapi.yaml` are checked against the C++ source: routes, request and
  question fields, every HTTP error message, the metric families with the `/metrics` example's
  TYPE and HELP lines, and every serve flag. Published numbers repeated across README, ROADMAP,
  BASELINES and the site must equal their source (17 facts). `tools/docs/test_check_docs.py`
  plants each kind of drift and expects the check to report it.
- LoRA training for per-category specialists (`tools/finetune/train_lora.py`): PEFT LoRA on the
  encoder's `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, `mlp.Wo` (88 modules; `bias="none"`, no
  `modules_to_save`, no DoRA, `init_lora_weights` true or gaussian), decision head and token
  embeddings frozen, trained on one category's rows of a built mixture with the RLCD + CE step of
  the full fine-tunes, best adapter by dev accuracy, saved with `save_pretrained` (safetensors).
  The output converts with `tools/convert_lora.py`. Defaults are sized for an 8 GB RTX 3070 (bf16
  autocast, gradient checkpointing, `--max-tokens 8192 --max-rows 64 --accum 2`, r 16); see the
  module docstring. Needs `pip install peft`.
  - `tools/finetune/lora_experiment.py` runs train, convert, serve with the adapter and
    `bench/eval_categories.py` for base and adapter per category, and decides with the gate's Holm
    logic (`gate.adapter_decision`, factored out of `gate.compare`).
  - `bench/eval_categories.py --adapter NAME` sends `"adapter"` with every request, checks
    `routing.adapter` and records the adapter, so base and adapter runs stay apart.
  - `tools/finetune/test_train_lora.py`: category selection, dev split, decision; an opt-in test
    trains and converts a real adapter (`STATIM_LORA_BASE_DIR`, `STATIM_LORA_BASE_GGUF`).
- Mixture v6, Part G: training data for the categories where 0.7.0 trails Qwen3-8B zero-shot.
  Evidence and every examined candidate are in `tools/finetune/sources/v6-research.md` (Part G),
  and the sources are registered in `v6-keep.json` (`source_part` G):
  - PII: `naeyn/nobody-pii-synth-de` (Apache-2.0, generated from templates and Faker, no model
    output), train split at a pinned commit. 9,133 items: per-type probes de 4,710, en 2,151,
    nl 1,746, plus 526 choice items. The loader re-derives the language of PII-free rows, which are
    partly English under a `de` tag.
  - PII: `Powpowpow23/ru-pii-ner-data` (Apache-2.0; DeepSeek-written templates filled with
    fictitious Faker/custom-generator values), 104,111 Russian train rows at a pinned commit.
    Its 25 nested-span types reuse the existing PII vocabulary; probes respect each row's
    `supervised_types`, and only the 2,400 explicit negative examples supply document-level “no”.
    The normal per-source cap prevents the large source from dominating the mixture.
  - Fact-check: no source passed the licence and label checks (38 candidates examined). Emotion,
    sentiment and safety: nothing kept; 41 checked candidates are recorded as rejected.
- Offline tests for the Part G loaders and adapters and for the evidence and held-out rules
  (`test_label_fixes.py`).

- `CLAUDE.md`: how to build, test, document and release Statim, and the rules for data, models and
  the API, for coding agents working in the repository.
- `docs/ADAPTERS.md`: the first category-adapter experiment on 0.7.0. The PII adapter (+5.46
  points over 11 languages) and the emotion adapter (+4.75 over 8) pass the gate; safety (+8.0 on
  one cell of 150 items), sentiment and fact-check stay within noise. The adapters are not
  published yet.

### Removed
- `docs/WEITERMACHEN.md`, internal handoff notes from 2026-09-26. README, ROADMAP and CHANGELOG
  carry the current state.

### Fixed
- Fact-check options `check worthy` and `non factual` had no description, in the training items and
  in the held-out suite: their gloss keys kept the hyphen (`check-worthy`, `non-factual`) that
  `registry.canon()` turns into a space. `templates.describe()` now matches labels regardless of
  separators, and a test requires every gloss to be reachable in every language. This changes the
  fact-check suite's items and the category pool fingerprint. The baselines answered the 150
  fact-check items again: Qwen3-8B 0.493 → 0.513 (its 14-category mean 0.704 → 0.706); Statim
  (0.313) and mDeBERTa-XNLI (0.347) are unchanged.
- `docs/API.md` still gave 0.2.1 and 0.3.0 as the compiled-in version, and its `/metrics` example and
  check snippet predated the TYPE lines of `statim_workers_busy` and `statim_model_info` (0.8.0), so
  the snippet failed against a real server.
- Documentation audit, every file against the code, the CLI and the releases:
  - `REPRODUCE.md` and the ticket-triage example downloaded v0.7.0. The example's "What you should
    see" named the 0.4.0 model file, and its demo transcript came from 0.4.0; it is re-recorded
    with the published 0.7.0 model.
  - The site showed MASSIVE 0.772 (the 0.4.0 model) for the 0.7.0 model, now 0.800.
  - `docs/openapi.yaml` gave 0.4.0 and 0.2.1 as versions and lacked the 422
    `inference cancelled` response. Both API documents lacked the micro-batch summaries in the
    `/metrics` example and several serve flags.
  - ROADMAP listed shipped 0.9 work as open. SECURITY now lists fuzz finding F8 (0.8.1) and dates
    its validation counts. The client READMEs name the server version and say that adapters are
    not wrapped yet.

## [0.8.1] - 2026-09-29

LoRA adapters are bound to the exact checkpoint they were trained on, and the follow-ups from the
fuzzing review (#27).

### Fixed
- An adapter could load onto a checkpoint that differs from its base only in the matrices, e.g. one
  with another adapter merged in: the 0.8.0 fingerprint covers the vectors (norms and biases) only.
  `tools/convert_laya.py` now records `statim.checkpoint_sha256`, a SHA-256 over every source
  tensor before type conversion, so all weight types of one checkpoint share it; `statim-quantize`
  keeps it, `tools/convert_lora.py` copies it into the adapter, and the engine refuses a mismatch
  when both files carry it. Files converted before keep the fingerprint check alone.
  `statim info` prints it.
- `statim-quantize` accepts only the matrix types the loader accepts (one list,
  `src/weight_types.h`); types such as `iq4_nl` used to produce files the engine refuses.
- A model or adapter path that is not valid UTF-8 no longer stops `statim serve` at startup: the
  `model_loaded` and `adapter_loaded` log lines replace invalid bytes instead of throwing.
- `docs/API.md` lists the 422 `inference cancelled` response (client gone while waiting in a
  micro-batch).
- A model file with an empty tensor (e.g. a zero-length `act_head.2.bias`) no longer reaches undefined
  behaviour (`memcpy` with a null pointer) while `Model::load` converts it, before validation
  rejects the file; empty inputs to the SHA-256 are skipped for the same reason. Found by the
  fuzzer with the new q4_0/q8_0 seeds; the input is a regression test.

### Changed
- Fuzzing: q4_0 and q8_0 variants of the tiny model are gguf seeds, so the CPU repack path in
  `Model::load` is fuzzed as well. Tests: `quantize_types`, `lora_checkpoint_binding` (tiny models
  with equal vectors and different matrices), and a non-UTF-8 model path in `security_http`.
- CI converts the Laya checkpoints again (model cache key `laya-models-v2`) so they carry the
  checkpoint SHA-256.

## [0.8.0] - 2026-09-29

Per-category LoRA adapters, a measured comparison with a general LLM, fuzzing and hardened model
loading, and licence-checked training data for the weakest categories. The published models are
unchanged.

### Added
- `bench/baselines.py` and `docs/BASELINES.md`: Statim against a local LLM (Qwen3-8B, zero-shot)
  and a zero-shot NLI classifier (mDeBERTa-v3 XNLI) on the gate's 11,550 held-out items, with the
  same questions and options for every system. Over 14 decision categories: 0.748, 0.704 and 0.488;
  Banking77: 0.913, 0.650 and 0.224. On the same GPU, Statim answers about 68 decisions per second
  and the LLM about 6. README section "Against general models".
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
- LoRA adapters: one base model plus small per-category adapters, chosen per request.
  - `tools/convert_lora.py` converts a PEFT adapter (safetensors; LoRA on the encoder's
    `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, `mlp.Wo`) into a `statim-lora-v1` GGUF. The PEFT scale
    (`lora_alpha / r`, rsLoRA, `rank_pattern` / `alpha_pattern`) is folded into `lora_b`; f16 and
    bf16 factors are widened exactly. Rejected: trained biases, `modules_to_save`, LoRA on other
    modules, `fan_in_fan_out`, the LoRA variants whose inference is not `W + B·A` (DoRA, aLoRA,
    QALoRA, BD-LoRA, KaSA, Arrow) and initialisations that change the base weights (PiSSA, OLoRA,
    CorDA, LoRA-GA, LoftQ) unless PEFT converted the adapter into a plain LoRA; VeLoRA, MonteCLoRA
    and MiCA adapters convert as the plain LoRA they are at inference. `--base` is required: it
    checks the shapes and records the checkpoint's fingerprint, a SHA-256 over its vectors (norms
    and biases) that is the same for its f32 and quantized files and differs between fully
    fine-tuned checkpoints (not for a LoRA merged into the matrices, which keeps the vectors);
    the engine refuses an adapter whose fingerprint does not match (`statim info` prints a
    model's). `--category` sets the question families for auto routing.
  - `statim serve --adapter [model:]name=file.gguf` (repeatable) and `--adapter-mode merge|runtime`;
    `decide`, `bench` and `info` take one `--adapter`. Merge computes `W + B·A` with ggml at load
    and runs the unchanged base graph: base latency, one copy of the adapted weights per adapter
    (438 MB f32 / 116 MB q8_0 for the multilingual model). Runtime keeps only the factors (3.3 MB
    at rank 4) and adds `B·(A·x)` in the graph, 17-28 % slower per request on CPU. The default is
    merge on f32/f16/bf16 weights and runtime on quantized weights, where merging rounds the delta
    to the quantization grid: with the test adapter it loses 14 % of the adapter's effect on q8_0
    and 91 % on q4_0, while runtime stays as close to the f32 reference as the quantized base is.
    Zero-delta pairs are skipped, so an untrained adapter is bit-identical to the base.
  - Request field `adapter`: a loaded name, `"auto"` or `null`/`"none"`. `"auto"` routes by
    question family: keywords of the 14 decision categories in the question ID, then in the
    instructions; a request whose questions do not all share one family uses the base weights.
    A named adapter that only one model carries selects that model (`routing.reason` `adapter`).
  - `routing.adapter` and `routing.adapter_reason` in responses, `adapters` per model in
    `GET /v1/models`, and `statim_adapter_info`, `statim_adapter_bytes` and `statim_engines` in
    `/metrics`, all only when the server has adapters loaded (or the request sets `adapter`);
    responses of servers without adapters are unchanged. Adapters share the base model's
    `--workers` slots and engines: the base and all its adapters hold at most `--workers` compute
    buffers, and an idle engine of another adapter is dropped when one is needed. Micro-batches
    never mix adapters. The `--consensus` default does not apply to a request that names an
    adapter.
  - Tests (CPU): `lora_convert` (converter scaling, dtypes, fingerprint and rejections),
    `lora_parity` (a zero adapter is bit-identical to the base; a random rank-4 adapter matches a
    PyTorch merge of the same adapter into the official Laya model within 1e-4 in weights and
    logits, in both modes; adapters for another checkpoint or with wrong shapes are refused),
    `lora_quantized` (q8_0 and q4_0 bases: runtime accuracy, bit-exact merged weights, repacked
    merge) and `server_lora` (selection, auto routing, errors, batch endpoint, micro-batching
    across adapters, authentication, consensus, one engine per worker). CI runs them verbose.
- libFuzzer harnesses under ASan + UBSan for request bodies (`POST /v1/systemone` and `/batch`,
  through the handler's own parsing, validation, tokenization, packing, inference and response
  serialization), the tokenizer (arbitrary bytes, both BPE pipelines, real Laya vocabularies) and
  GGUF model loading (`fuzz/`, `-DSTATIM_FUZZ=ON`, `fuzz/run.sh`). The hand-written seeds and every
  crash input are committed and replayed by ctest (`fuzz_regressions_*`) in ordinary builds.
- CI job `fuzz`: every harness for 60 s per push, continuing from the corpus grown in earlier runs
  (kept with `actions/cache`); crash inputs are uploaded as an artifact.
- `tests/test_model_validation.cpp`: 20 malformed-model cases that must be rejected at load.
- Startup warning `auth_off_on_network` when the server listens beyond loopback without API keys
  (the Docker images do this by default).

### Fixed
- A malformed or hostile model file can no longer abort the process. Every GGUF metadata read
  checks the stored type first (`gguf_get_*` abort on a mismatch), and `Model::load` now rejects,
  with an error, files whose tensors do not match the hyperparameters (shapes, types), whose
  special token ids or vocabulary exceed the embedding table, whose hyperparameters are out of
  range, or whose calibration tables are malformed. Before, such files either aborted at load or
  loaded and then aborted (`GGML_ASSERT`) or read out of bounds on the first request.
- `laya.temperature_by_options` and `laya.lang_temperatures` are validated at load. A malformed
  table used to throw from the engine constructor, and a short per-language `temperature` array was
  indexed out of bounds for requests that set `lang`.
- A model whose `general.name` is not valid UTF-8 is rejected at load. It is echoed as `"model"` in
  every response, whose serialization then threw, so every request answered 500.
- Bearer-key comparison runs over a fixed length, so its timing no longer depends on the configured
  keys' lengths.
- Error logging cannot throw on invalid UTF-8 in an exception message.

### Changed
- README rewritten: results at a glance with their protocol, one results section, CPU and GPU
  performance together, LoRA adapters, security and robustness. Every number comes from the document
  it links to.
- Request parsing and validation moved from the HTTP handler into `parse_decide_request()`
  (`statim/security.h`) so the fuzzer runs exactly the server's code. Behaviour is unchanged.

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

[Unreleased]: https://github.com/BEKO2210/statim/compare/v0.8.7...HEAD
[0.8.7]: https://github.com/BEKO2210/statim/compare/v0.8.6...v0.8.7
[0.8.6]: https://github.com/BEKO2210/statim/compare/v0.8.5...v0.8.6
[0.8.5]: https://github.com/BEKO2210/statim/compare/v0.8.4...v0.8.5
[0.8.4]: https://github.com/BEKO2210/statim/compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com/BEKO2210/statim/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/BEKO2210/statim/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/BEKO2210/statim/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/BEKO2210/statim/compare/v0.7.0...v0.8.0
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
