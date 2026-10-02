# Threat model

Statim is a single process that loads model files and answers HTTP requests. This page lists:

- what it protects;
- whom it protects against;
- where the trust boundaries are;
- which control covers each threat, and the test that proves it;
- what remains the operator's job.

Report a vulnerability as described in [SECURITY.md](../SECURITY.md). The fix history is in
[docs/SECURITY.md](SECURITY.md).

## Assets

| Asset | Why it matters |
|---|---|
| Availability of the decision service | Applications block on its answers |
| Integrity of answers | A wrong decision is a business error; answers must equal the reviewed model's |
| API keys | Whoever holds one can run inference and read metrics |
| Request contents (texts, JSON states) | Can contain personal or confidential data |
| Model and adapter files | Their integrity decides every answer; their weights may be licensed |
| The host | A memory-safety bug in parsing could become code execution |

## Actors

1. **Anonymous network client.** Reaches the port, holds no key.
2. **Authenticated client.** Holds a valid key. Trusted to call the API, not trusted with the host.
3. **Malicious or corrupted model/adapter file.** A file from an untrusted source, or damaged in
   transit.
4. **Local user on the host** without root, or another tenant.
5. **Supply-chain attacker.** Targets the vendored libraries, the build or the release artifacts.

Out of scope: an attacker with root or with write access to the binary, the unit file or the key
file. They control the process anyway.

## Trust boundaries

```
client ──TLS──▶ reverse proxy ──HTTP (loopback/private)──▶ statim serve ──mmap──▶ model files
                (operator's)                                │
                                                            └──▶ logs, /metrics (operator's)
```

- **Network → Statim.** All request bytes are untrusted until they pass authentication and the
  limits.
- **Files → Statim.** Model and adapter files are untrusted input to the parser. They are not
  trusted until the structural preflight and the load-time validation have accepted them.
- **Statim → operator.** Logs record per request the path, status, timing, token count, request id
  and remote address, plus error details. They never record request bodies or keys. Metrics hold
  counters only.

## Threats and controls

| Threat | Actor | Control | Proof |
|---|---|---|---|
| Unauthenticated use of a network-reachable server | 1 | The server refuses to start on a non-loopback address without a key, unless `--allow-unauthenticated` is given | `test_non_loopback_without_key_fails_closed` and four more cases in `tests/security/test_http.py` |
| Guessing a key through response timing | 1 | Fixed-length constant-time comparison (`bearer_authorized`, `include/statim/http_security.h`) | security unit tests |
| Resource exhaustion: huge, deep or wide JSON, many fields, long texts, big batches | 1, 2 | Every request is bounded before inference: SAX preflight limits, token and work budgets, a response cap, `--max-concurrent` with 503 and `Retry-After`, absolute header and body deadlines, an inference deadline | `security`, `security_http`, `server_lifecycle` (overload); fuzzing of request bodies |
| Slowloris, dropped clients, connection floods | 1 | Fixed worker pool, a bounded socket queue, header and body deadlines | `security_http` (drip-fed headers and bodies), `server_lifecycle` (dropped clients) |
| HTTP request smuggling | 1 | Every CL+TE combination, duplicate and malformed lengths rejected | `security_http` framing cases |
| Log injection, leaked exception text | 1, 2 | Request ids restricted to an ASCII allowlist; logs written as JSON; a fixed error body for internal errors | `security` and `security_http` |
| Memory corruption from a crafted model or adapter file | 3 | Structural preflight before ggml parses anything; the file is opened once and mapped, so the checked bytes are the parsed bytes; shape, type, range and UTF-8 validation at load | `gguf_preflight`, `model_validation`, the `fuzz_gguf` corpus, all under ASan and UBSan in CI |
| An adapter applied to the wrong base model | 3 | Adapters record the base fingerprint and checkpoint SHA-256; a mismatch is refused | `lora_checkpoint_binding` |
| Swapping a model file during loading | 4 | Open once, map once, never reopen by name | the rename case in `test_gguf_preflight` |
| Reading the API key from the process or files | 4 | Key from a root-owned env file or key file; the systemd unit runs `DynamicUser` with `ProtectSystem=strict`, `NoNewPrivileges` and no capabilities; the container is non-root, read-only, with all capabilities dropped | `deploy/statim.service`, `deploy/docker-compose.yml` |
| Tampered release artifacts | 5 | `SHA256SUMS`, build-provenance attestations, an SPDX SBOM per archive, hardened binaries checked before upload | `release.yml`; `check_hardening.py` |
| Tampered CI, actions or dependencies | 5 | Actions pinned to commit SHAs, hash-locked Python tools, least-privilege workflow permissions, protected `main` and immutable tags | READINESS P0 #3 and #4 |
| A known vulnerability in vendored code | 5 | The `vendored-cves` job on every push and daily, with a triage file | READINESS P0 #6 |

## Residual risks (the operator's part, or open)

- **TLS.** Statim speaks plain HTTP. TLS, client certificates and network-level rate limiting
  belong to the reverse proxy ([DEPLOY.md](DEPLOY.md#tls-reverse-proxy)).
- **One key, full access.** A key grants inference and metrics alike. There are no scopes yet
  (READINESS #12). Rotate keys by restarting with the new key source.
- **Minimum key length.** Short keys are not rejected yet (READINESS #13). Use at least 32 random
  characters, for example `openssl rand -hex 32`.
- **The playground.** It is public by design and sends a strict Content-Security-Policy (only its own
  hash-pinned script, no framing); it keeps an API key only in the tab's `sessionStorage`. A key typed
  into it is still readable by anything that runs in that tab. Run with `--no-playground` where
  untrusted users reach the server.
- **Request data in memory.** Request texts stay in process memory and in the bounded calibration
  cache. A core dump would contain them. Disable core dumps on hosts that handle personal data.
- **Untrusted model files.** The parser is hardened, fuzzed and sanitized, but a model file is still
  code-adjacent input. Load models only from sources you verify by checksum, such as the published
  `SHA256SUMS`.
- **Denial of service by an authenticated client.** Limits bound each request, but a key holder can
  keep the server busy up to `--max-concurrent`. A dropped request is still computed to the end
  (READINESS #54). Give each application its own key and server, or rate-limit at the proxy.
- **Side channels between tenants on one host.** These are out of scope. Run sensitive tenants on
  separate hosts.
