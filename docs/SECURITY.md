# Security hardening

To report a vulnerability, see the [security policy](../SECURITY.md).

The HTTP API retains successful Jev/Laya response shapes; `/health` is intentionally
reduced to `status` and `version`. Limits and flags are documented in README → API.
The playground denies framing unless the operator explicitly lists allowed origins with
`--frame-ancestors`; other responses keep their existing security headers.

| Finding | Fix | Regression coverage |
|---|---|---|
| 1 — authentication fails open or uses weak keys | Validate each configured key source before model loading; require every key to be 32–4096 printable ASCII characters without whitespace; abort on unreadable/missing/empty/comment-only/weak/invalid sources. Refuse non-loopback listeners without keys before model loading unless `--allow-unauthenticated` explicitly opts in and logs `auth_off_on_network`. Mandatory systemd env file and valid key. | C++ 31/32-character boundary, multi-key file, secret-free error, file/env, and loopback-predicate tests; HTTP startup subprocesses cover weak and invalid sources before model loading, fail-closed `0.0.0.0`, explicit opt-in, authenticated non-loopback, IPv4 loopback, and IPv6 loopback when supported. |
| 2 — deep JSON | SAX preflight caps depth and nodes before constructing/copying/serializing the ordered DOM. | 30,000 nested arrays; exact depth boundary; node-count boundary; live HTTP 413 and server survival. |
| 3 — connection exhaustion | Fixed HTTP worker count, bounded pending socket queue, early auth/admission, absolute header/body deadlines, immediate close on rejected bodies. | Socket-free real HTTP processing verifies 401/413 without reads or draining; deterministic full-queue rejection; live admission saturation and drip-fed headers/bodies with health checks. |
| 4 — batch amplification | Bound fields, aggregate evaluations/tokens, conservative attention and response estimates; final response cap; cooperative inference/queue deadlines; deployment ceilings. Existing graph packing is preserved. | C++ field/work/token/attention/response limits, ensemble/calibration/consensus multipliers; live oversized batch rejection; CPU model deadline, bit-identical finite-deadline output, cancellation and executor recovery. |
| 5 — calibration growth | Exact semantic cache keys built from the validated question (unknown fields stay ignored, as in laya.serve, and are never retained); LRU bounded by bytes and entries. Return cached values safely by value. | C++ eviction/byte/entry ceilings, oversized entries, 5,000 insertions; ignored 1 MiB metadata leaves decisions unchanged; live test sends 64 MiB of ignored metadata and checks flat memory; identical cache-hit/miss decisions. |
| 6 — wide ordered JSON | SAX member/node limits before ordered-map insertion; duplicate key rejection. | 30,000-member object rejected in C++ and HTTP; duplicate-key and node boundary tests. |
| 7 — request-ID log injection | Short ASCII allowlist and generated replacements; complete records serialized as JSON; upgraded header parser. | C++ CRLF, percent encoding, quotes, size tests; HTTP malicious IDs replaced and logs parsed/checked. |
| 8 — integer narrowing / capacity | Check signed/unsigned values before narrowing; strict CLI integer parsing; validate effective sequence capacity including 128 state tokens. | Positive/negative wraparound, UINT64_MAX, wrong types, CLI overflow, incompatible head/sequence budgets, preserved automatic state room. |
| 9 — HTTP framing / dependency | Unmodified cpp-httplib 0.58.0 plus explicit rejection of all CL+TE combinations, duplicate lengths and malformed lengths. Source URL/checksum in `third_party/httplib.version`. | Real socket-free parser and HTTP replay of nonzero/zero CL+TE, equal/unequal duplicate lengths, comma lists and negative lengths. Chunked oversized body stops without draining. |
| 10 — operational endpoints | Auth on metrics/models; public minimal health and readiness. | C++ middleware and HTTP anonymous/authenticated endpoint checks; exact health field set. |
| 11 — timestamp race | Local `tm` with `gmtime_r`/`gmtime_s`. | 16 concurrent threads generating 16,000 timestamps. |
| 12 — exception disclosure | Global HTTP handler returns a fixed detail and removes EXCEPTION_WHAT; details logged server-side as JSON. | Real socket-free routing exception containing a private sentinel; assert fixed JSON and no sentinel/header in response. |

## Fuzzing (pre-1.0)

`fuzz/` holds libFuzzer harnesses for request bodies, the tokenizer and GGUF loading, built with
ASan + UBSan (`fuzz/README.md`). CI runs each for 60 s per push from the corpus grown in earlier
runs (`actions/cache`); the committed seeds and every crash input are replayed by ctest in ordinary
builds. Initial campaign (clang 18, 3 harnesses in
parallel, 35 min each): `fuzz_request` 562,522 executions, `fuzz_tokenizer`
622,148, `fuzz_gguf` 502,099 after the F5 fix (plus 246,999 in the first run, which stopped on
F5). A follow-up run (0.8.1) added q4_0/q8_0 repack seeds to `fuzz_gguf` and found one more (F8).
Findings:

| Finding | Fix | Regression coverage |
|---|---|---|
| F1 — model metadata of the wrong GGUF type aborts the process (`gguf_get_*` `GGML_ASSERT`) | Every read checks the stored type (`src/model.cpp` `key()`/`arr_key()`, `src/tokenizer_gguf.cpp`). | `test_model_validation` (5 type cases) |
| F2 — tensors are not checked against the hyperparameters; a mismatched file loads, then aborts in `ggml_mul_mat`/`ggml_get_rows` on the first request | `validate()` in `Model::load`: exact shapes and supported types for every tensor, head counts, ranges. | `test_model_validation` (shape/range cases) |
| F3 — special token ids and the tokenizer vocabulary are not checked against the embedding rows (`GGML_ASSERT` in `get_rows`, or out-of-bounds reads without asserts) | Load rejects ids outside `[0, rows)` and a vocabulary larger than the table. | `test_model_validation` |
| F4 — calibration tables parsed lazily: malformed `laya.temperature_by_options` throws from `Engine()`; a short per-language `temperature` array is indexed out of bounds on requests that set `lang` | Both tables validated at load. | `fuzz/regressions/gguf/engine-ctor-bad-temperature-json.gguf`, `test_model_validation` |
| F5 — `general.name` with invalid UTF-8 is echoed as `"model"`; `dump()` throws, so every request and `/v1/models` answered 500 | Load requires valid UTF-8. | `fuzz/regressions/gguf/general-name-invalid-utf8.gguf`, `test_model_validation` |
| F6 — empty mask token silently disables scrubbing the mask piece from caller text | Rejected at load. | `test_model_validation` |
| F7 — tensor-bounds check `off + nbytes > size` can wrap; `fstat` result unchecked | Overflow-free comparison; `fstat` checked. | `fuzz_gguf` (no committed input) |
| U1, U2 — upstream ggml `gguf_init_from_reader`: the element-count overflow guard itself overflows (`ggml_nelements`), and the tensor type is loaded into `enum ggml_type` before its range check | Rejected by Statim's structural GGUF preflight before ggml parses the file. Reported upstream in [llama.cpp #29383](https://github.com/ggml-org/llama.cpp/issues/29383#issuecomment-5909941199), which already covered U1. | `test_gguf_preflight` (explicit U1/U2 files), `fuzz/regressions/gguf/ggml-nelements-overflow-in-guard.gguf` |
| F8 — a tensor with zero elements makes `Model::load`'s conversion, and the checkpoint SHA-256, `memcpy` through a null `data()` pointer (undefined behaviour) | Both skip zero-length copies (`src/model.cpp`, `src/sha256.cpp`). | `fuzz/regressions/gguf/empty-tensor-to-f32-memcpy-null.gguf` |
| F9 — a checkpoint whose encoder layers are all global (or all local) leaves the other attention mask out of the allocated graph; uploading it aborted in `ggml_backend_tensor_set` on the first request | Upload a mask only when the graph allocated it (`src/model.cpp`); results for real checkpoints, which use both masks, are unchanged. Found by `fuzz_gguf` in 0.9.1 work. | `test_model_validation` (all-global model answers) |

The request and tokenizer harnesses found no crash, hang, leak, UB or contract violation (every
rejection was `HttpError` 400/413/422 or `QuestionError`). The tokenizer is linear on adversarial
50,000-character states (≤ 16 ms for runs of one letter, punctuation, continuation bytes, CJK).

## Design decisions and limits

- Aggregate token limits bound the existing materialized rows before encoding. The engine's
  length sorting and graph packing stay unchanged to avoid changing inference arithmetic.
- Calibration uses exact, byte-bounded semantic strings rather than fixed-size lossy hashes.
  This removes memory amplification without introducing hash-collision changes to decisions.
- CPU cancellation occurs between ggml operations; GPU checks occur between graphs. A running
  operation is not forcibly preempted. Systemd ceilings bound the process independently.
- Network-level flood protection and frontend/backend proxy integration remain deployment
  concerns. The framing regressions do not claim a deployed proxy exploit or any new CVE.
- GPU parity is not run, as requested. GPU math and tensor kernels are unchanged.
- The live HTTP suite exits 77 if localhost binding is forbidden; CTest reports a skip and the
  README provides the manual command. Localhost binding succeeded in this run.

## Validation

The run that validated the 0.3.0 fixes (the suites have grown since; `ctest` prints today's
counts). Release configure/build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` and
`cmake --build build -j16` passed. Final CTest parity results are recorded after the run.

- Socket-free security: **83 checks passed**.
- HTTP/startup security: **60 checks passed**, including **12 startup failure checks**;
  localhost binding succeeded (no skip).
- CPU model-backed security: **89 checks passed** (includes the socket-free checks).
  Finite-deadline inference and calibration cache hits matched bit-for-bit; the CPU
  executor recovered after cancellation with identical logits and action probabilities.
- `git diff --check` and Python syntax compilation passed.

Findings come from a review of 0.2.1 by a separate AI coding agent (Codex), not a third-party audit (12 issues: 5 high, 4 medium, 3 low); every fix has a regression test in `tests/test_security.cpp` or `tests/security/test_http.py`, both part of `ctest`.

## Supply chain

Statim hardens its continuous integration and release pipeline against supply-chain tampering:

- **Pinned actions.** Every workflow action (`uses:`) across `.github/workflows/` is pinned to an immutable full commit SHA, resolved directly from the action's official repository, with the semantic release tag recorded in a trailing comment. Action pins are monitored and kept current weekly via Dependabot (`.github/dependabot.yml`).
- **Least privilege.** Workflows declare `contents: read` by default. Elevated permissions are scoped strictly to the specific jobs that require them: `contents: write` for uploading release assets, and `id-token: write` / `attestations: write` for generating artifact attestations.
- **Pinned CI dependencies.** Python tooling dependencies in CI are pinned to exact versions with SHA-256 integrity hashes (`requirements-ci.txt`) and installed with `--require-hashes`.
- **Software Bill of Materials (SBOM).** Each release archive has a companion SPDX 2.3 JSON SBOM generated from the finished archive by the standard-library-only `tools/release/sbom.py` and published alongside the release assets. It inventories every shipped file with SHA-1 and SHA-256 checksums; the statically linked ggml, cpp-httplib, nlohmann/json and libstdc++ components with versions and licences; and the operating-system libraries reported by the archived binaries' ELF dynamic sections. It also records the compiler and minimum required glibc symbol version reported by `readelf`.
- **Build provenance.** Release archives are signed with build provenance attestations via `actions/attest-build-provenance`. Provenance can be verified using the GitHub CLI:
  ```bash
  gh attestation verify statim-0.9.3-linux-x86_64-cpu.tar.gz --owner BEKO2210
  ```
- **Secret scanning.** GitHub push protection rejects pushes that contain known credential formats. CI job `secrets` also runs gitleaks 8.30.1 (pinned, SHA-256-checked) over the commits each push or pull request adds, and over the whole history daily. `.gitleaks.toml` adds only a narrow allowlist: test keys of the form `<name>-test-key-<hex>` and the description hashes in `docs/api-v1.contract.json`.
- **Checksums.** Each release attaches `SHA256SUMS` covering all released archives:
  ```bash
  sha256sum --check --ignore-missing SHA256SUMS
  ```
- **Vendored CVE scanning and triage.** CI runs `tools/security/vendored_cves.py` daily and on every change to detect known high or critical vulnerabilities across vendored dependencies (`cpp-httplib`, `nlohmann/json`, and `ggml`) using OSV.dev and GitHub Security Advisories. For upstream advisories that cannot be matched automatically to a pinned commit (such as ggml and GGUF advisories filed against `ggml-org/llama.cpp`), each advisory concerning ggml or GGUF must have an explicit entry in `tools/security/cve-triage.json` recording a decision (`fixed-in-pinned`, `not-affected`, or `accepted-risk`), commit and file/line evidence, and a `review_by` expiry date. To triage a new advisory, locate the upstream fixing commit or PR, inspect whether that change is present in `third_party/ggml` or whether the affected subsystem (such as RPC or llama-server) is unbuilt, add the entry to `tools/security/cve-triage.json` with the evidence, and set a future `review_by` date.
- **Binary hardening.** Release binaries (`statim` and `statim-quantize` on Linux x86-64 and in the Android cross-build) are built with defense-in-depth compile and link flags enabled by default (`STATIM_HARDEN`): Position Independent Executables (`-fPIE` / `-pie` via CMake `check_pie_supported()`), stack protector (`-fstack-protector-strong`), compile-time and runtime buffer fortification (`-D_FORTIFY_SOURCE=2`, or `=3` when supported by the compiler; skipped under ASan to prevent interceptor conflicts), full RELRO (`-Wl,-z,relro,-z,now`), and a non-executable stack (`-Wl,-z,noexecstack`). The standard-library checker `tools/release/check_hardening.py` inspects every release binary and archive using `readelf` or `llvm-readelf` in CI and release workflows, verifying `Type: DYN`, `GNU_RELRO`, `BIND_NOW` or `FLAGS_1 NOW`, `GNU_STACK` (without executable `E` flag), `__stack_chk_fail`, and fortified imported symbols.
