# Runbook

What to do when Statim misbehaves in production. Each entry gives:

- the **signal**: the log event, metric or exit status you see;
- the **check**: a command that confirms the cause;
- the **fix**.

Event names and metrics are the real ones from the server (`docs/API.md` lists the metrics). Logs
are one JSON object per line on stderr; with systemd, `journalctl -u statim -o cat | jq` reads them.

## Quick health check

```sh
curl -fsS http://127.0.0.1:8080/health          # {"status":"ok","version":"..."}: the process is up
curl -fsS http://127.0.0.1:8080/ready           # 200 {"ready":true}: it accepts work; 503 while saturated
curl -fsS -H "Authorization: Bearer $KEY" http://127.0.0.1:8080/metrics | grep -E 'statim_(in_flight|workers_busy|rejected_busy_total|model_info|adapter_requests_total)'
journalctl -u statim -o cat --since -10min | jq -c 'select(.level != "info")'
```

## The process does not start

### `Illegal instruction`, or "this build of Statim needs an x86-64 CPU with AVX2, FMA, F16C and BMI2"

- **Signal.**
  - Releases after 0.9.2 exit 1 with that message, naming the missing CPU features.
  - 0.9.2 and older crash with SIGILL (exit 132).
- **Check.** `grep -o -w 'avx2\|fma\|f16c\|bmi2' /proc/cpuinfo | sort -u` lists what the CPU has.
- **Fix.** The release binaries need x86-64-v3 (Haswell or newer). On older CPUs, build from source
  with `-DSTATIM_NATIVE=ON` ([BUILD.md](BUILD.md#older-x86-cpus)).

### Exit status 2: a configuration error

Statim checks its configuration before it loads a model. The message names the problem.

| Message contains | Cause | Fix |
|---|---|---|
| `is not loopback and no API key is configured` | `--host 0.0.0.0` (or a LAN address) without a key | Set `STATIM_API_KEY` or `--api-key-file`. `--allow-unauthenticated` only on a trusted network ([SECURITY.md](../SECURITY.md)). |
| `unknown argument` | A flag that this version does not know, often after a rollback | Remove it from the unit or compose file ([COMPATIBILITY.md](COMPATIBILITY.md#rolling-back)). |
| key file errors (unreadable, empty, comment-only) | The key source is broken | Fix the file; the unit expects `/etc/statim/env` with `STATIM_API_KEY=` set. |

### `cannot read GGUF model '<path>': GGUF preflight: ...`

- **Signal.** Exit before `listening`. The preflight rejected the file's structure.
- **Check.** `sha256sum <model.gguf>` against the published `SHA256SUMS`, and `ls -l`: a truncated
  or partially copied file is the usual cause.
- **Fix.** Download it again and verify the checksum. The file is opened once and checked as a
  whole, so a file changed during the start cannot slip through. Restart after replacing it.

### `'<path>' is not a Statim decision model` / `LoRA adapter ... was converted for another checkpoint`

- **Cause.** The file is a GGUF but not one Statim converted, or an adapter belongs to a different
  base model. The fingerprint and checkpoint SHA-256 must match.
- **Fix.** Convert with `tools/convert_laya.py`. Use the adapter that was published for this exact
  base, or convert it again with `tools/convert_lora.py --base <model.gguf>`.

## It runs, but requests fail

### 503 `{"detail":"server busy, try again later"}` with `Retry-After: 1`

- **Signal.** `statim_rejected_busy_total` rising, `statim_in_flight` at `--max-concurrent`, and
  `/ready` answering 503.
- **Meaning.** Admission control works as designed: more concurrent requests than
  `--max-concurrent` (default 16). This is not a failure of Statim.
- **Fix.**
  - Clients should honour `Retry-After`; the official SDKs retry.
  - If it persists, check `statim_workers_busy` against `--workers`. If every worker is always
    busy, the host lacks compute: add workers (each needs its own compute buffers, see the
    `model_loaded` event's `threads_per_worker`) or another instance behind the proxy.
  - Raising `--max-concurrent` alone only makes requests wait longer.

### 422 `{"detail":"inference deadline exceeded"}`

- **Meaning.** A request did not finish within `--inference-timeout` (default 120 s), including the
  time it queued for a worker.
- **Check.**
  - `statim_request_duration_ms` buckets;
  - request sizes: states per batch and text lengths (`usage.input_tokens` on successful answers).
- **Fix.** Smaller batches, more workers or a faster device. Under sustained overload, lower the
  load before raising the timeout.

### 422 with `inference_cancelled` in the log

- **Meaning.** The client disconnected before its answer, and Statim stopped computing it.
- **Check.** The client's or the proxy's timeout: NGINX `proxy_read_timeout` must exceed the
  inference timeout ([DEPLOY.md](DEPLOY.md#tls-reverse-proxy)).

### 401 `invalid or missing bearer token`

- **Check.** The client sends `Authorization: Bearer <key>`, and the proxy passes the header on.
- **Fix.** Set the key in the client. See [key rotation](#rotating-api-keys) if a key changed.

### 500 `{"detail":"inference failed"}` with an `inference_failed` event

- **Meaning.** An unexpected error inside inference. The event carries the request id and the
  internal error text; the client gets only the fixed detail.
- **Fix.**
  - Collect the event and the request id, and the request if it is safe to share.
  - Report it ([SECURITY.md](../SECURITY.md) if it looks exploitable, an issue otherwise).
  - The server keeps serving other requests.

## Memory and restarts

### Killed by the OOM killer, or the container restarts

- **Signal.** `journalctl -k | grep -i oom`; systemd reports `oom-kill`; the container's
  `State.OOMKilled` is true.
- **Check.** RSS against the model:
  - an f32 multilingual model uses about 0.9 GB of weights;
  - each adapter merged on f32 weights adds a full copy per engine;
  - each worker adds compute buffers.

  Batch size drives the activation memory.
- **Fix.**
  - Lower `--workers` or the batch limits.
  - Use q8_0 weights, which also run adapters at runtime instead of merging them.
  - Or raise `MemoryMax` (unit) / `memory` (compose).

  A 24 h soak with two adapters held 2.9–3.0 GB after warm-up on the multilingual f32 model.

### Unhealthy container

- **Signal.** Docker reports `unhealthy`. The image's probe calls `/health` every 30 s.
- **Check.** `docker inspect --format '{{json .State.Health}}' <container>` shows the probe output.
  A model still loading counts as starting for 60 s.
- **Fix.** If `/health` itself fails, the process is wedged: collect the logs, then restart. A busy
  server stays healthy: the probe does not use `/ready`.

## Rotating API keys

Keys are read at start-up. Several keys can be valid at once, one per line in `--api-key-file`, so
a rotation needs no downtime for clients.

1. Generate the new key: `openssl rand -hex 32`. Use at least 32 random characters.
2. Add it as a second line to the key file (or switch from `STATIM_API_KEY` to a key file), then
   restart. In-flight requests drain before the old process exits.
3. Move every client to the new key.
4. Remove the old key from the file and restart again.

If a key leaked, skip step 3's grace period: replace the key and restart at once, then update the
clients.

## Upgrades and rollbacks

See [COMPATIBILITY.md](COMPATIBILITY.md#upgrading).

## What to collect before you report

- `statim version`;
- the unit or compose file;
- the start-up lines of the log: `listening`, `model_loaded`, `adapter_loaded`;
- the failing request's id and its `request` event;
- `/metrics` at the time;
- the model's SHA-256.
