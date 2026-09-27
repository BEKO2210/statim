#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
binary="$repo_root/build/statim"
model="$repo_root/models/laya-multilingual-big1-f32.gguf"

for command in docker curl python3; do
  command -v "$command" >/dev/null || { echo "error: $command is required" >&2; exit 1; }
done
[[ -x "$binary" ]] || { echo "error: missing executable $binary" >&2; exit 1; }
[[ -f "$model" ]] || { echo "error: missing model $model" >&2; exit 1; }

context=$(mktemp -d)
export BUILDX_CONFIG="$context/buildx"
suffix="$$-$(date +%s)"
image_name="statim-hf-space-test:$suffix"
container_name="statim-hf-space-test-$suffix"
response_file="$context/response.json"

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
  docker image rm -f "$image_name" >/dev/null 2>&1 || true
  rm -rf -- "$context"
}
trap cleanup EXIT INT TERM

cp "$script_dir/Dockerfile" "$context/Dockerfile"
cp --reflink=auto "$binary" "$context/statim"
cp --reflink=auto "$model" "$context/model.gguf"

echo "Building local-source Space image..."
docker build \
  --progress=plain \
  --build-arg STATIM_SOURCE=local \
  --build-arg MODEL_SOURCE=local \
  --tag "$image_name" \
  "$context"

port=$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)

docker run --detach --name "$container_name" \
  --publish "127.0.0.1:${port}:7860" "$image_name" >/dev/null

if ! docker inspect --format '{{.State.Running}}' "$container_name" | grep -qx true; then
  docker logs "$container_name" >&2
  echo "error: container exited during startup" >&2
  exit 1
fi

configured_user=$(docker inspect --format '{{.Config.User}}' "$container_name")
[[ "$configured_user" == "1000:1000" ]] || {
  echo "error: image user is $configured_user, expected 1000:1000" >&2
  exit 1
}
runtime_uid=$(docker exec "$container_name" busybox id -u)
[[ "$runtime_uid" == "1000" ]] || {
  echo "error: runtime UID is $runtime_uid, expected 1000" >&2
  exit 1
}

echo "Waiting for readiness on http://127.0.0.1:${port}/ready ..."
ready=false
for _ in $(seq 1 120); do
  if curl --fail --silent --show-error --max-time 2 \
      "http://127.0.0.1:${port}/ready" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null | grep -qx true; then
    docker logs "$container_name" >&2
    echo "error: container exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done
[[ "$ready" == true ]] || {
  docker logs "$container_name" >&2
  echo "error: readiness timeout" >&2
  exit 1
}

# /proc/net/tcp{,6}: state 0A is LISTEN and the local-address suffix is the port.
docker exec "$container_name" busybox awk '
  FNR > 1 && $4 == "0A" {
    split($2, address, ":")
    if (toupper(address[length(address)]) != "1EB4") {
      print "unexpected listening socket: " $2 > "/dev/stderr"
      bad = 1
    }
    found = 1
  }
  END {
    if (!found) { print "no listening socket found" > "/dev/stderr"; exit 1 }
    exit bad
  }
' /proc/net/tcp /proc/net/tcp6

response_time=$(curl --fail --silent --show-error --max-time 120 \
  --output "$response_file" --write-out '%{time_total}' \
  --header 'Content-Type: application/json' \
  "http://127.0.0.1:${port}/v1/systemone" \
  --data-binary @- <<'JSON'
{
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    }
  }
}
JSON
)

choice=$(python3 - "$response_file" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    response = json.load(handle)
answers = response.get("answers")
if not isinstance(answers, dict):
    raise SystemExit("response has no answers object")
choice = answers.get("department", {}).get("choice")
if not isinstance(choice, str) or not choice:
    raise SystemExit("response has no non-empty answers.department.choice")
print(choice)
PY
)

playground=$(curl --fail --silent --show-error --max-time 10 \
  "http://127.0.0.1:${port}/")
grep -Fq '<title>Statim Playground</title>' <<<"$playground" || {
  echo "error: GET / did not return the playground HTML" >&2
  exit 1
}

image_bytes=$(docker image inspect --format '{{.Size}}' "$image_name")
image_size=$(python3 - "$image_bytes" <<'PY'
import sys
size = int(sys.argv[1])
print(f"{size / 1024 / 1024:.1f} MiB ({size} bytes)")
PY
)

echo "PASS"
echo "  UID: $runtime_uid (configured as $configured_user)"
echo "  listeners: 7860 only"
echo "  ready: OK"
echo "  decision choice: $choice"
echo "  decision response time: ${response_time}s"
echo "  playground: OK"
echo "  image size: $image_size"
