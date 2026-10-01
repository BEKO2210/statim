# Readiness for 1.0

Statim 1.0 ships only when every item marked P0 below is closed and each closure is backed by a
reproducible proof: a CI job, a test, a measurement with its protocol, or an API query. A new
feature cannot substitute for a proof. The external review of 0.9.0 summed up the current state
as "production-shaped, not production-proven". This list is how that changes.

The items come from a read-only audit of 0.9.2 in eight areas:

- security;
- supply chain;
- reliability;
- operability;
- correctness;
- evaluation;
- documentation;
- platform coverage.

**Priorities.** P0 blocks 1.0. P1 should be in 1.0. Items are closed only with the proof in the
last column.

**Status (2026-10-01).** All nine P0 items are closed. The server-reliability items P1 #23 to #27
(soak, overload, leaks at exit, SIGTERM, dropped clients) are now also release criteria: 1.0 waits
for them like a P0.

**Effort.** S is up to a day, M a few days, L a week or more.

## P0 — blocks 1.0

| # | Item | Area | Effort | Status | Proof |
|---|---|---|---|---|---|
| 1 | A server bound to a non-loopback address refuses to start without an API key, including the container's default `--host 0.0.0.0`. Only `--allow-unauthenticated` keeps today's warning. | Security | M | **closed** | `test_non_loopback_without_key_fails_closed`, `test_non_loopback_allow_unauthenticated`, `test_non_loopback_env_key`, `test_loopback_ipv4_without_key`, `test_loopback_ipv6_without_key` in `tests/security/test_http.py`; loopback predicate cases in `tests/test_security.cpp` |
| 2 | On a CPU without the build's instruction set (AVX2, FMA, F16C, BMI2 for the x86-64 release), Statim exits with a message naming the missing features before any inference, instead of `Illegal instruction`. | Reliability, platform | M | **closed** | see [Proofs](#proofs); `tests/test_cpu_check.cpp` |
| 3 | `main` accepts changes only through pull requests with green CI (`build-test`, `fuzz`, and since #62 and #66 `sanitize` and `vendored-cves`), and nobody can bypass that. Release tags cannot be deleted or moved. | Supply chain | S | **closed** (2026-09-30) | see [Proofs](#proofs) |
| 4 | Every GitHub Action is pinned to a full commit SHA. | Supply chain | S | **closed** (#51) | `.github/workflows/*.yml`; Dependabot keeps the pins current |
| 5 | A root `SECURITY.md` gives the contact, an acknowledgement window and the supported versions; private vulnerability reporting is on. | Security | S | **closed** | `SECURITY.md` in the root; `gh api repos/BEKO2210/statim/private-vulnerability-reporting` returns `{"enabled":true}` |
| 6 | CI fails on a known high or critical CVE in the vendored ggml, cpp-httplib and nlohmann/json. | Security | M | **closed** (#66) | the `vendored-cves` job; planted advisories in `tools/security/test_vendored_cves.py` |
| 7 | The security, validation, GGUF-preflight and fuzz-regression tests run under ASan and UBSan in CI, not only the 60-second fuzz job. | Security | M | **closed** (#62) | the `sanitize` job in `.github/workflows/ci.yml` |
| 8 | The support matrix in the README equals what CI builds and tests and what the release ships. ARM NEON and CUDA are claimed but not tested today. | Documentation, platform | S | **closed** | README [Platform support](../README.md#platform-support): ARM NEON is no longer claimed; Vulkan and CUDA list the parity tests run on an RTX 3070 on 2026-10-01 |
| 9 | The model promotion gate uses paired statistics (McNemar or a paired bootstrap) on stored per-item outcomes, and a test fails if the unpaired formula returns. The published "0 regressions" is recomputed or withdrawn. | Evaluation | L | **closed** | `tools/finetune/test_gate.py` (guard test `test_no_unpaired_two_proportion_standard_error_outside_allowlist`); [paired recomputation](reproductions/paired-gate-2026-10-01.md): both models and all three adapters keep their decision, 0 regressions |

## P1 — should be in 1.0

### Security

| # | Item | Effort | Status |
|---|---|---|---|
| 10 | Threat model (assets, actors, trust boundaries, residual risks), linked from `SECURITY.md` | S | open |
| 11 | CI fails when `security_http` skips | S | open |
| 12 | Key scopes (a metrics key cannot call inference), and a key id in the logs | M | open |
| 13 | API keys shorter than 32 characters are rejected | S | open |
| 14 | The playground no longer keeps the bearer in `localStorage`; the server sends CSP, `nosniff` and `frame-ancestors 'none'` | S | open |
| 15 | A tested nginx configuration in deploy/ (`nginx -t`); TLS stays at the proxy | S | open |
| 16 | A weekly fuzz campaign longer than 60 s, with a coverage note per harness | M | open |
| 17 | Release builds with PIE, a stack protector, `_FORTIFY_SOURCE=2` and full RELRO, checked | S | **closed** (#71): `STATIM_HARDEN`; `tools/release/check_hardening.py` in CI and release |
| 18 | Secret-scanning push protection, plus a CI scan of the diff | S | partly: push protection is on |

### Supply chain

| # | Item | Effort | Status |
|---|---|---|---|
| 19 | A release archive built twice has an identical SHA-256; the compiler package is pinned | M | open |
| 20 | An SPDX SBOM next to `SHA256SUMS` | S | done in #51; proven by the next release |
| 21 | Build-provenance attestations on the release archives, with the verify command documented | M | done in #51; proven by the next release |
| 22 | Hash-locked tools and pinned inputs: Python CI tools, `httplib.h` against `httplib.version`, a hash for `json.hpp`, checksums in `fetch_models.sh`, a digest-pinned Docker `FROM` | M | partly: the Python CI tools are hash-locked (#51) |

### Reliability

| # | Item | Effort | Status |
|---|---|---|---|
| 23 | A 24 h soak with an RSS ceiling and a `/ready` poll. **Required for 1.0** | M | running: `bench/soak.py`, 24 h on belkis-home since 2026-10-01 12:28 |
| 24 | A load test beyond `--max-concurrent` that expects 503 with `Retry-After`, then a clean 200. **Required for 1.0** | M | **closed** (#69): `server_lifecycle`, overload case |
| 25 | LeakSanitizer on the HTTP suite and a few hundred inferences. **Required for 1.0** | M | **closed** (#69): `server_lifecycle`, every exit checked for sanitizer reports, under ASan in the `sanitize` job |
| 26 | SIGTERM drains or cancels within a bound; `TimeoutStopSec` matches; a test sends the signal. **Required for 1.0** | M | **closed** (#69): `server_lifecycle`, 5x SIGTERM and SIGINT with requests in flight; `TimeoutStopSec=150` |
| 27 | A client dropped mid-request, then the same request again, gives an identical 200. **Required for 1.0** | M | **closed** (#69): `server_lifecycle`, dropped-client case |
| 28 | A Docker `HEALTHCHECK` on `/health`, and a compose file with the systemd unit's limits | S | **closed** (#74): static `/health` probe in both images; `deploy/docker-compose.yml` with the unit's limits |

### Operability

| # | Item | Effort | Status |
|---|---|---|---|
| 29 | an alerts file in deploy/, checked with `promtool`: ready, 503s, latency, busy workers | S | open |
| 30 | A runbook for SIGILL, OOM, 503, a bad GGUF and key rotation, using the server's real event names | M | open |
| 31 | Upgrade and rollback steps; the 1.x promise for the GGUF formats (`statim-decision-v1`, `statim-lora-v1`) | S | open |
| 32 | The fingerprint (and the checkpoint SHA-256) on `/v1/models` and in `statim_model_info` | S | open |
| 33 | A request counter labelled by adapter | S | open |

### Correctness

| # | Item | Effort | Status |
|---|---|---|---|
| 34 | A line-coverage artifact for `src/security.cpp`, `http_security.h` and `src/server.cpp`, with a baseline | M | open |
| 35 | A performance and memory regression gate against the latest release | M | **closed** (#60): `bench/perf_gate.py`, required for every PR that can affect speed or memory (CLAUDE.md) |
| 36 | The README's parity tolerance equals the CI tolerance; green runs archive the worst \|Δlogit\| | S | open |
| 37 | The Python and TypeScript clients' tests run in CI, both the hermetic cases and live cases against a started server | M | open |

### Evaluation

| # | Item | Effort | Status |
|---|---|---|---|
| 38 | A contamination report for the 1.0 weights: exact matches and near-duplicates against every evaluation split, as a release gate | M | open |
| 39 | A specialist baseline on the same items, or no "same protocol" wording for external supervised numbers | L | open |
| 40 | The glance table names model 0.7.0 and trained against zero-shot | S | open |

### Documentation

| # | Item | Effort | Status |
|---|---|---|---|
| 41 | A deprecation policy: one minor version of warning, removal only in a major | S | open |
| 42 | A commercial term sheet, or no wording that implies the paid terms are in the repository | M | open |

### Platform coverage

| # | Item | Effort | Status |
|---|---|---|---|
| 43 | Vulkan `ctest` on real hardware before the Vulkan asset is uploaded; CUDA the same, or out of the status line | M | partly: Vulkan and CUDA parity pass on an RTX 3070 (2026-10-01, manual); not yet a release step |
| 44 | An ARM test run (planned on a Galaxy A15 and a Galaxy Tab S9 Ultra), or no ARM NEON claim | S | partly: Galaxy A15 passes the native suite and both parity tests (2026-10-01), CI cross-builds Android arm64; the Tab S9 Ultra (i8mm) is next |

### From the external reviews (2026-10-01)

Findings of the Qwen and Gemini reviews and of the ORT work that the audit did not list. Claims in
those reviews that the code refuted are not listed (constant-time key comparison exists, there are
no aligned vector loads, the scratch buffers are per thread).

| # | Item | Area | Effort | Status |
|---|---|---|---|---|
| 45 | GGUF preflight, ggml and the mmap open the model path separately; a file swapped between them skips the preflight. Open once and check the same file (descriptor, or device, inode and size) | Security | M | **closed** (#70): one `MappedFile` per load; `test_gguf_preflight` covers a rename over the mapped path, directories, FIFOs and empty files |
| 46 | ThreadSanitizer on the server with 32 concurrent clients, micro-batching and adapter switches | Reliability | M | open |
| 47 | The inference deadline is checked inside the custom SGEMM op too, so a long batch cannot overrun it by a whole matrix product | Reliability | S | open |
| 48 | The start-up log names the active matrix-product path (custom SGEMM or ggml) and the CPU features in use | Operability | S | open |
| 49 | A native C++ ONNX Runtime benchmark next to the Python one, so the binding overhead is excluded by construction | Correctness | M | open |
| 50 | Claim hygiene: the Hugging Face cards' q8_0 lines ("faster on CPU", "4x smaller") match the measurements (q8_0 is slower than f32 on AVX2, about 2.6x smaller); every speed claim names its hardware and protocol | Documentation | S | **closed** (#72): the cards compute the q8_0 size ratio from the files and state where q8_0 is slower (AVX2) and faster (ARM dotprod, CUDA); README corrected |
| 51 | The paired evaluation files (`eval.json`, `eval-items.jsonl.gz`) and regenerated cards on Hugging Face, so third parties can run the paired comparison | Evaluation | S | open; at the 1.0 release |
| 52 | GPU: parity on a self-hosted runner (the RTX 3070 on pop-os) for `main` and release tags only, never for fork pull requests; Vulkan on the belkis-home Intel iGPU; GPU cells in the perf gate; ONNX Runtime CUDA and TensorRT in the comparison | Platform | L | open |
| 53 | A 72 h soak with cancellations and adapter switches before the 1.0 tag, after the 24 h run of #23 passes | Reliability | M | open |
| 54 | A request whose client disconnects is still computed to the end and holds its worker (found by `server_lifecycle`). Cancel the inference when the connection closes, as the deadline already does | Reliability | M | open |

## Proofs

### P0 #5–#9 (2026-10-01)

- **#5, security policy (#61).** `SECURITY.md` names GitHub private vulnerability reporting
  (enabled), response targets and supported versions.
- **#6, known CVEs (#66).** The `vendored-cves` job runs on every push and daily, and is a required
  check. Its tests plant advisories that must fail the job. A live control query (cpp-httplib
  v0.43.0 must have CVEs) proves that the database still answers for these packages. The 13
  llama.cpp advisories are triaged with evidence in `tools/security/cve-triage.json`.
- **#7, sanitizers (#62).** The `sanitize` job runs 15 suites under ASan and UBSan and is a
  required check. Its local run found nothing.
- **#8, support matrix (#64, #65).** The README lists only what CI builds and tests or what ran on
  real hardware: x86-64 CPU, Vulkan and CUDA on an RTX 3070, Android arm64 on a Galaxy A15. The
  `android-arm64` job cross-compiles every target.
- **#9, paired gate (#67).** [docs/reproductions/paired-gate-2026-10-01.md](reproductions/paired-gate-2026-10-01.md)
  recomputes every published decision. Both models and all three adapters keep their decisions,
  with 0 regressions.

### P0 #3: branch and tag protection (2026-09-30)

The repository rulesets:

- **24274330, "main: PR and green CI".** Pull requests only, squash merges only, required checks
  `build-test`, `fuzz`, `sanitize` and `vendored-cves` (the last two added on 2026-10-01 after their
  first green runs), no force pushes, no deletion. There are no bypass actors:
  `gh api repos/BEKO2210/statim/rulesets/24274330` reports `"current_user_can_bypass": "never"`
  for the owner's admin account. #51 was the first merge under this rule.
- **24274331, "release tags are immutable".** Tags `v*` cannot be deleted, updated or
  force-pushed, and there are no bypass actors. Tested with the admin account on a throwaway tag
  `v0.0.0-ruleset-test`:
  - the delete was rejected with `422 Cannot delete this tag`;
  - the move to another commit was rejected with `422 Cannot force-push to this tag`.

  Creating new release tags stays allowed.
- **Break glass.** A tag or `main` can only be changed against these rules by setting the ruleset's
  enforcement to disabled, acting, and setting it back to active. That is recorded in the
  repository's audit log. It was used once, to delete the test tag.

### P0 #2: unsupported CPUs (2026-10-01)

A portable build (`-DSTATIM_NATIVE=OFF`, as released) was run on two CPUs without AVX2:

- **i5-2520M (Sandy Bridge, lenovo).** `statim decide`, `statim serve` and `statim-quantize` exit 1
  with: "this build of Statim needs an x86-64 CPU with AVX2, FMA, F16C and BMI2 (x86-64-v3 …). This
  CPU lacks: avx2, fma, f16c, bmi2. Build from source on this machine instead: …".
- **i3-3227U (Ivy Bridge, homemini2).** The same, naming `avx2, fma, bmi2` (Ivy Bridge has F16C).
- **0.9.2, for comparison.** It died on the same lenovo with `Illegal instruction (core dumped)`,
  exit 132.

`statim version` still answers on both. `tests/test_cpu_check.cpp` covers the feature logic and the
message.

### P0 #4: pinned actions (#51)

Every `uses:` line is a full commit SHA, with its version as a comment. Each SHA was resolved from
the action's own repository (`gh api`, `git ls-remote`) and equals the commit that its floating
major tag pointed to on 2026-09-30. The checkout action was corrected from an older v4.2.2 to
v4.4.0 in review. Dependabot (`.github/dependabot.yml`) proposes updates weekly.

## Test machines

| Machine | CPU | Role |
|---|---|---|
| pop-os | AMD Ryzen 7 5800X, Zen 3, AVX2 | development and reference measurements |
| belkis-home | Intel Xeon E3-1505M v5, Skylake, AVX2 | second x86 platform, soak and load tests |
| lenovo | Intel Core i5-2520M, Sandy Bridge, no AVX2 | unsupported-CPU behaviour (P0 #2) |
| homemini2 | Intel Core i3-3227U, Ivy Bridge, no AVX2 | light functional and container tests |
| Galaxy A15, Galaxy Tab S9 Ultra | ARMv8.2 and ARMv9 (dotprod, i8mm) | ARM proof (P1 #44) |
