#!/usr/bin/env bash
# nginx -t, then a live Statim behind deploy/nginx/statim.conf.
# Needs: docker, curl, openssl, python3, ss, and a statim binary
# (build/statim, or $STATIM_BIN) plus a GGUF ($STATIM_MODEL, otherwise the
# first of the q8 English file and the f32 checkpoints fetch_models.sh writes).
#
# Image: nginx:1.28, binary nginx/1.28.3, inspected 2026-10-02 on linux/amd64.
#   docker pull nginx:1.28
#   docker image inspect nginx:1.28 --format '{{index .RepoDigests 0}}'
# Pinned by digest so a moved tag cannot change what CI tests.
NGINX_IMAGE=nginx@sha256:146adea4768b83c607d0bdfa4188464e3da6e0a3ad4475db1d1d8f64f27c29cc

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STATIM_BIN=${STATIM_BIN:-$ROOT/build/statim}
STATIM_THREADS=${STATIM_THREADS:-4}
NGINX_NAME="statim-nginx-test-$$"
TMP=$(mktemp -d)
STATIM_PID=""

cleanup() {
    local ec=$?
    if [[ -n "$NGINX_NAME" ]]; then
        docker rm -f "$NGINX_NAME" >/dev/null 2>&1 || true
    fi
    if [[ -n "$STATIM_PID" ]]; then
        kill "$STATIM_PID" >/dev/null 2>&1 || true
        wait "$STATIM_PID" >/dev/null 2>&1 || true
    fi
    rm -rf "$TMP"
    exit "$ec"
}
trap cleanup EXIT

fail() {
    echo "FAIL $*" >&2
    if docker inspect "$NGINX_NAME" >/dev/null 2>&1; then
        echo "---- nginx logs ----" >&2
        docker logs "$NGINX_NAME" 2>&1 | tail -100 >&2 || true
    fi
    if [[ -f "$TMP/statim.log" ]]; then
        echo "---- statim log (tail) ----" >&2
        tail -50 "$TMP/statim.log" >&2 || true
    fi
    exit 1
}

pass() { echo "PASS $*"; }

port_busy() {
    local want=$1
    ss -ltn | awk -v want="$want" 'NR > 1 {
        n = split($4, a, ":")
        if (a[n] == want) { found = 1 }
    } END { exit !found }'
}

pick_model() {
    if [[ -n "${STATIM_MODEL:-}" ]]; then
        [[ -s "$STATIM_MODEL" ]] || fail "STATIM_MODEL is missing: $STATIM_MODEL"
        printf '%s\n' "$STATIM_MODEL"
        return
    fi
    local rel
    for rel in \
        models/laya-english-q8_0.gguf \
        models/laya-multilingual-f32.gguf \
        models/laya-english-f32.gguf
    do
        if [[ -s "$ROOT/$rel" ]]; then
            printf '%s\n' "$ROOT/$rel"
            return
        fi
    done
    fail "no GGUF found; set STATIM_MODEL or run tools/fetch_models.sh"
}

echo "nginx image: $NGINX_IMAGE"
docker image inspect "$NGINX_IMAGE" >/dev/null 2>&1 || docker pull "$NGINX_IMAGE"

for p in 80 443 8080; do
    if port_busy "$p"; then
        fail "port $p is already in use; not touching it"
    fi
done

[[ -x "$STATIM_BIN" ]] || fail "statim binary is not executable: $STATIM_BIN"
MODEL=$(pick_model)
echo "statim: $STATIM_BIN"
echo "model: $MODEL"

CERT=$TMP/certs
CFG=$TMP/conf.d
mkdir -p "$CERT" "$CFG"
# Throwaway certificate. Not a committed key.
openssl req -x509 -newkey rsa:2048 -sha256 -days 1 -nodes \
    -keyout "$CERT/privkey.pem" -out "$CERT/fullchain.pem" \
    -subj "/CN=statim.example.com" >/dev/null 2>&1
umask 077
openssl rand -hex 32 > "$TMP/api_key"
umask 022

cp "$ROOT/deploy/nginx/statim-zones.conf" "$CFG/statim-zones.conf"
sed \
    -e 's#/etc/letsencrypt/live/statim.example.com/fullchain.pem#/certs/fullchain.pem#g' \
    -e 's#/etc/letsencrypt/live/statim.example.com/privkey.pem#/certs/privkey.pem#g' \
    "$ROOT/deploy/nginx/statim.conf" > "$CFG/statim.conf"
if grep -q '/etc/letsencrypt/' "$CFG/statim.conf"; then
    fail "certificate placeholder was not substituted"
fi

echo "nginx -t"
docker run --rm \
    -v "$CFG:/etc/nginx/conf.d:ro" \
    -v "$CERT:/certs:ro" \
    "$NGINX_IMAGE" nginx -t
pass "nginx -t"

echo "starting statim on 127.0.0.1:8080"
"$STATIM_BIN" serve \
    -m "$MODEL" \
    --host 127.0.0.1 --port 8080 \
    --threads "$STATIM_THREADS" \
    --api-key-file "$TMP/api_key" \
    >"$TMP/statim.log" 2>&1 &
STATIM_PID=$!

healthy=0
for _ in $(seq 1 180); do
    if curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
        healthy=1
        break
    fi
    if ! kill -0 "$STATIM_PID" 2>/dev/null; then
        fail "statim exited before /health"
    fi
    sleep 1
done
[[ "$healthy" == 1 ]] || fail "statim /health did not return 200 within 180s"
pass "statim /health on loopback"

docker run -d --name "$NGINX_NAME" --network host \
    -e NGINX_ENTRYPOINT_QUIET_LOGS=1 \
    -v "$CFG:/etc/nginx/conf.d:ro" \
    -v "$CERT:/certs:ro" \
    "$NGINX_IMAGE" >/dev/null

up=0
for _ in $(seq 1 40); do
    code=$(curl -4 -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1/health || true)
    if [[ "$code" == "301" ]]; then
        up=1
        break
    fi
    sleep 0.5
done
[[ "$up" == 1 ]] || fail "nginx did not answer on port 80 (last code ${code:-none})"
pass "nginx is listening"

# --- checks through the proxy ---

code=$(curl -4 -sk -o "$TMP/health.json" -w '%{http_code}' --max-time 10 https://127.0.0.1/health || true)
[[ "$code" == "200" ]] || fail "/health through nginx returned $code"
pass "/health 200"

code=$(curl -4 -sk -o "$TMP/ready.json" -w '%{http_code}' --max-time 10 https://127.0.0.1/ready || true)
[[ "$code" == "200" ]] || fail "/ready through nginx returned $code"
pass "/ready 200"

cat > "$TMP/decide.json" <<'EOF'
{"state":"The refund was posted twice.","questions":{"refund":{"type":"noul","instructions":"Does the text ask for a refund?"}}}
EOF

code=$(curl -4 -sk -o "$TMP/decision.json" -w '%{http_code}' --max-time 180 \
    -H "Authorization: Bearer $(cat "$TMP/api_key")" \
    -H 'Content-Type: application/json' \
    --data-binary @"$TMP/decide.json" \
    https://127.0.0.1/v1/systemone || true)
[[ "$code" == "200" ]] || fail "POST /v1/systemone with a key returned $code ($(head -c 200 "$TMP/decision.json" 2>/dev/null || true))"
python3 -c 'import json,sys; body=json.load(open(sys.argv[1])); assert "answers" in body, body' "$TMP/decision.json" \
    || fail "200 body has no decision"
pass "POST /v1/systemone 200 with a decision"

code=$(curl -4 -sk -o "$TMP/unauth.json" -w '%{http_code}' --max-time 30 \
    -H 'Content-Type: application/json' \
    --data-binary @"$TMP/decide.json" \
    https://127.0.0.1/v1/systemone || true)
[[ "$code" == "401" ]] || fail "POST /v1/systemone without a key returned $code"
pass "POST /v1/systemone without a key 401"

# A loopback client matches `allow 127.0.0.1`, so the deny has to be probed from
# another address. Prefer a global address on this host. If that connect fails
# (some CI runners do not hairpin), ask from a container on the docker bridge:
# nginx is on the host network and sees that container's address, not 127.0.0.1.
metrics_from_outside() {
    local ip code gw
    if [[ "${STATIM_NGINX_OUTSIDE:-}" != "bridge" ]]; then
        ip=$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1 || true)
        if [[ -n "$ip" ]]; then
            code=$(curl -4 --interface "$ip" -sk -o "$TMP/metrics.txt" -w '%{http_code}' --max-time 10 \
                "https://$ip/metrics" || true)
            if [[ "$code" =~ ^[0-9]{3}$ && "$code" != "000" ]]; then
                echo "metrics client: $ip" >&2
                printf '%s\n' "$code"
                return
            fi
            echo "metrics probe via $ip failed (${code:-none}); trying the docker bridge" >&2
        fi
    fi
    gw=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
    [[ -n "$gw" ]] || fail "docker bridge has no gateway"
    code=$(docker run --rm --network bridge "$NGINX_IMAGE" \
        curl -4 -sk -o /dev/null -w '%{http_code}' --max-time 10 "https://$gw/metrics" || true)
    echo "metrics client: bridge container via $gw" >&2
    printf '%s\n' "$code"
}
code=$(metrics_from_outside)
[[ "$code" == "403" ]] || fail "/metrics from outside 127.0.0.1 returned $code, want 403 ($(head -c 160 "$TMP/metrics.txt" 2>/dev/null || true))"
pass "/metrics from outside 127.0.0.1 is 403"

redir=$(curl -4 -s -D - -o /dev/null --max-time 10 http://127.0.0.1/health || true)
printf '%s\n' "$redir" | head -n 1 | grep -q ' 301 ' || fail "http:// did not return 301 ($redir)"
printf '%s\n' "$redir" | grep -qi '^location: https://' || fail "301 Location is not https:// ($redir)"
pass "http:// redirects to https:// with 301"

headers=$(curl -4 -sk -D - -o /dev/null --max-time 10 https://127.0.0.1/health || true)
printf '%s\n' "$headers" | grep -qi '^strict-transport-security:' || fail "missing Strict-Transport-Security"
printf '%s\n' "$headers" | grep -qi '^x-content-type-options:[[:space:]]*nosniff' || fail "missing nosniff"
pass "response has HSTS and nosniff"

python3 -c 'import sys; open(sys.argv[1], "wb").write(b"x" * (2 * 1024 * 1024 + 1))' "$TMP/big" 
code=$(curl -4 -sk -o "$TMP/big.body" -w '%{http_code}' --max-time 60 \
    -H 'Content-Type: application/octet-stream' \
    --data-binary @"$TMP/big" \
    https://127.0.0.1/v1/systemone || true)
[[ "$code" == "413" ]] || fail "oversized body returned $code, want 413"
grep -q 'nginx' "$TMP/big.body" || fail "413 body was not generated by nginx"
pass "body over 2 MiB is 413 from nginx"

logs=$(docker logs "$NGINX_NAME" 2>&1 || true)
printf '%s\n' "$logs" | grep -F 'POST /v1/systemone' | grep -q ' 200 ' \
    || fail "access log has no successful POST /v1/systemone"
if printf '%s\n' "$logs" | grep -q 'Bearer'; then
    fail "access log contains Bearer"
fi
pass "access log has the decision and no Bearer"

echo "all checks passed"
