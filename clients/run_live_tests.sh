#!/usr/bin/env bash
# Run both SDK test suites, hermetic and live, against three started servers (READINESS P1 #37):
#   8190  no key (most live cases)
#   8191  with a bearer key (authentication cases)
#   8192  with a test LoRA adapter (adapter routing cases)
#
#   clients/run_live_tests.sh BINARY MODEL ADAPTER_GGUF
#   e.g. clients/run_live_tests.sh build/statim models/laya-multilingual-f32.gguf build/lora/random.gguf
#
# Needs the Python SDK's test requirements (clients/python/requirements-test.txt) and Node 18+ with
# clients/js installed (`npm ci`). Exits non-zero if a suite fails or skips a live case.
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 BINARY MODEL ADAPTER_GGUF" >&2; exit 2; }
BIN=$(realpath "$1"); MODEL=$(realpath "$2"); ADAPTER=$(realpath "$3")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOGS=${STATIM_SDK_LOGS:-$(mktemp -d)}
KEY=sdk-test-key-0123456789abcdef0123
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

"$BIN" serve -m multilingual="$MODEL" --port 8190 >"$LOGS/s8190.log" 2>&1 & PIDS+=($!)
STATIM_API_KEY=$KEY "$BIN" serve -m multilingual="$MODEL" --port 8191 >"$LOGS/s8191.log" 2>&1 & PIDS+=($!)
"$BIN" serve -m multilingual="$MODEL" --adapter multilingual:random="$ADAPTER" --port 8192 \
  >"$LOGS/s8192.log" 2>&1 & PIDS+=($!)

for port in 8190 8191 8192; do
  for _ in $(seq 1 120); do
    curl -fsS "http://127.0.0.1:$port/ready" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS "http://127.0.0.1:$port/ready" >/dev/null || { echo "server on $port not ready" >&2; cat "$LOGS/s$port.log" >&2; exit 1; }
done

export STATIM_URL=http://127.0.0.1:8190 STATIM_AUTH_URL=http://127.0.0.1:8191 STATIM_API_KEY_TEST=$KEY
export STATIM_ADAPTER_URL=http://127.0.0.1:8192 STATIM_ADAPTER_NAME=random

echo "== Python SDK"
(cd "$ROOT/clients/python" && PYTHONPATH=src python3 -m pytest -q -rs tests | tee "$LOGS/python.log")
# A skipped live case means the server setup above is incomplete, not that the SDK passed.
if grep -E '^SKIPPED' "$LOGS/python.log" | grep -v 'is not one of the 14 auto-rule families'; then
  echo "Python SDK: unexpected skip" >&2; exit 1
fi

echo "== TypeScript SDK"
(cd "$ROOT/clients/js" && npm run -s build && npm test 2>&1 | tee "$LOGS/js.log")
skipped=$(grep -E '^ℹ skipped [0-9]+' "$LOGS/js.log" | grep -oE '[0-9]+$' || echo 0)
auto=$(grep -c 'is not one of the 14 auto-rule families' "$LOGS/js.log" || true)
if [ "$skipped" -gt "$auto" ]; then echo "TypeScript SDK: $skipped skipped, only $auto expected" >&2; exit 1; fi
echo "SDK suites passed (logs in $LOGS)"
