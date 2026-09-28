# Security hardening

The HTTP API retains successful Jev/Laya response shapes; `/health` is intentionally
reduced to `status` and `version`. Limits and flags are documented in README → API.

| Finding | Fix | Regression coverage |
|---|---|---|
| 1 — key sources fail open | Validate each configured source before model loading; abort on unreadable/missing/empty/comment-only/invalid sources. Explicit auth-off log only with no source. Mandatory systemd env file and nonempty key. | C++ file/env tests (including unreadable file); HTTP startup subprocesses with invalid sources, multiple sources, and auth-off startup/log check. |
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
F5). Findings:

| Finding | Fix | Regression coverage |
|---|---|---|
| F1 — model metadata of the wrong GGUF type aborts the process (`gguf_get_*` `GGML_ASSERT`) | Every read checks the stored type (`src/model.cpp` `key()`/`arr_key()`, `src/tokenizer_gguf.cpp`). | `test_model_validation` (5 type cases) |
| F2 — tensors are not checked against the hyperparameters; a mismatched file loads, then aborts in `ggml_mul_mat`/`ggml_get_rows` on the first request | `validate()` in `Model::load`: exact shapes and supported types for every tensor, head counts, ranges. | `test_model_validation` (shape/range cases) |
| F3 — special token ids and the tokenizer vocabulary are not checked against the embedding rows (`GGML_ASSERT` in `get_rows`, or out-of-bounds reads without asserts) | Load rejects ids outside `[0, rows)` and a vocabulary larger than the table. | `test_model_validation` |
| F4 — calibration tables parsed lazily: malformed `laya.temperature_by_options` throws from `Engine()`; a short per-language `temperature` array is indexed out of bounds on requests that set `lang` | Both tables validated at load. | `fuzz/regressions/gguf/engine-ctor-bad-temperature-json.gguf`, `test_model_validation` |
| F5 — `general.name` with invalid UTF-8 is echoed as `"model"`; `dump()` throws, so every request and `/v1/models` answered 500 | Load requires valid UTF-8. | `fuzz/regressions/gguf/general-name-invalid-utf8.gguf`, `test_model_validation` |
| F6 — empty mask token silently disables scrubbing the mask piece from caller text | Rejected at load. | `test_model_validation` |
| F7 — tensor-bounds check `off + nbytes > size` can wrap; `fstat` result unchecked | Overflow-free comparison; `fstat` checked. | `fuzz_gguf` (no committed input) |
| U1, U2 — upstream ggml `gguf_init_from_reader`: the element-count overflow guard itself overflows (`ggml_nelements`), and the tensor type is loaded into `enum ggml_type` before its range check | Not ours to patch in the vendored tree; both wrap/are rejected in practice. Suppressed by exact function in `fuzz/ubsan.supp`; to be reported upstream. | `fuzz/regressions/gguf/ggml-nelements-overflow-in-guard.gguf` |

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

Release configure/build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` and
`cmake --build build -j16` passed. Final CTest parity results are recorded after the run.

- Socket-free security: **83 checks passed**.
- HTTP/startup security: **60 checks passed**, including **12 startup failure checks**;
  localhost binding succeeded (no skip).
- CPU model-backed security: **89 checks passed** (includes the socket-free checks).
  Finite-deadline inference and calibration cache hits matched bit-for-bit; the CPU
  executor recovered after cancellation with identical logits and action probabilities.
- `git diff --check` and Python syntax compilation passed.

Findings come from an independent review of 0.2.1 (12 issues: 5 high, 4 medium, 3 low); every fix has a regression test in `tests/test_security.cpp` or `tests/security/test_http.py`, both part of `ctest`.
