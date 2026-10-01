# Production deployment

Statim should listen on a private interface, require bearer authentication, and sit behind a
TLS-terminating reverse proxy. Model weights are separate artifacts: obtain or convert them as
described in the README, verify their provenance, and mount them read-only. Release archives and
container images do not contain weights.

## systemd

Install the CPU or Vulkan release binary and a model, then install the supplied unit:

```sh
sudo install -Dm0755 statim /usr/local/bin/statim
sudo install -Dm0644 model.gguf /var/lib/statim/multilingual.gguf
sudo install -Dm0644 deploy/statim.service /etc/systemd/system/statim.service
sudo install -d -m0750 /etc/statim
printf 'STATIM_API_KEY=%s\n' 'replace-with-a-long-random-key' | sudo tee /etc/statim/env >/dev/null
sudo chmod 0600 /etc/statim/env
sudo systemctl daemon-reload
sudo systemctl enable --now statim
```

The unit binds to `127.0.0.1:8080`, runs as a dynamic unprivileged user, and requires a nonempty
`STATIM_API_KEY`. A comma-separated value supports key rotation. An explicitly configured empty or
invalid environment value makes startup fail closed.

The shipped unit starts two workers and enforces `MemoryHigh=6G`, `MemoryMax=8G`, `CPUQuota=400%`,
`TasksMax=256`, and `LimitNOFILE=4096`. Tune the workers and ceilings together after measuring the
chosen model. Keep application limits explicit when increasing them, for example:

```ini
[Service]
ExecStart=
ExecStart=/usr/local/bin/statim serve -m multilingual=/var/lib/statim/multilingual.gguf --host 127.0.0.1 --port 8080 --workers 2 --max-concurrent 16 --http-queue 32 --max-request-work 4096 --max-request-tokens 1048576 --max-attention-mib 1024 --max-response-bytes 16777216 --request-timeout 30 --inference-timeout 120
```

Place overrides in `/etc/systemd/system/statim.service.d/limits.conf`, then run `systemctl daemon-reload`
and restart the service. The request, JSON, state, question, and option limits in the README remain in
force; do not treat the process memory ceiling as a replacement for them.

On `SIGTERM` or `SIGINT`, Statim immediately stops accepting new connections (subsequent attempts are
refused) and drains admitted in-flight requests to completion, returning full HTTP 200 responses before
exiting with status 0 and logging `{"event":"shutdown"}`. In-flight requests are bounded by
`--inference-timeout` (default 120 s); the systemd unit sets `TimeoutStopSec=150` (the 120 s inference
deadline plus 30 s headroom for network flush and process exit) so systemd does not prematurely issue a
`SIGKILL` while in-flight inferences drain.

## Docker CPU

Build the existing distroless CPU image and run it with a read-only model mount and host ceilings:

```sh
docker build -t statim:1.0 .
docker run -d --name statim-cpu --restart unless-stopped \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --memory 8g --memory-reservation 6g --cpus 4 --pids-limit 256 \
  --ulimit nofile=4096:4096 --security-opt no-new-privileges \
  --env-file /etc/statim/container.env \
  -v /srv/statim/model.gguf:/models/model.gguf:ro \
  -p 127.0.0.1:8080:8080 statim:1.0
```

`/etc/statim/container.env` must contain a valid `STATIM_API_KEY` and should be readable only by the
administrator. Alternatively, mount a key file and override the image command so the server reads
it directly:

```sh
docker run -d --name statim-cpu --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v /srv/statim/model.gguf:/models/model.gguf:ro \
  -v /etc/statim/api-key:/run/secrets/statim-api-key:ro \
  -p 127.0.0.1:8080:8080 statim:1.0 \
  serve -m /models/model.gguf --host 0.0.0.0 --port 8080 \
  --api-key-file /run/secrets/statim-api-key
```

The CPU and Vulkan images deliberately keep `--host 0.0.0.0`. With neither a valid
`STATIM_API_KEY` nor a valid mounted `--api-key-file`, their default server command prints a clear
configuration error and exits with status 2 before loading the model. Both images run as an
unprivileged user (distroless `nonroot` in the CPU image and UID/GID 65532 in the Vulkan image), so
the mounted key must be readable by that user.

## Docker GPU (Vulkan)

Build `Dockerfile.vulkan`, then expose only the GPU devices needed by the container:

```sh
docker build -f Dockerfile.vulkan -t statim-vulkan:1.0 .
docker run -d --name statim-gpu --restart unless-stopped \
  --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --memory 8g --memory-reservation 6g --cpus 4 --pids-limit 256 \
  --ulimit nofile=4096:4096 --security-opt no-new-privileges \
  --env-file /etc/statim/container.env \
  -v /srv/statim/model.gguf:/models/model.gguf:ro \
  -p 127.0.0.1:8080:8080 statim-vulkan:1.0
```

For NVIDIA, install and configure NVIDIA Container Toolkit on the host and replace the `--device`
and `--group-add` arguments with:

```sh
--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

The host driver must provide a Vulkan-capable ICD. The image also includes Mesa Vulkan drivers for
`/dev/dri` devices. It runs as UID/GID 65532 and defaults to `--device vulkan`; no GPU is required to
build the image. The same `STATIM_API_KEY` environment-file or mounted `--api-key-file` patterns
shown for the CPU image apply. Budget VRAM for all loaded models (the two f32 checkpoints need
about 3.3 GB).

On a trusted network only, an operator can make the old unauthenticated behavior explicit by
overriding the image command with `serve ... --host 0.0.0.0 --allow-unauthenticated` (the image
entrypoint supplies `statim`). Statim then starts and logs `auth_off_on_network`.

## Docker Compose

`deploy/docker-compose.yml` provides a production-ready single-service definition mirroring the
resource limits and security profile of `deploy/statim.service`:

- Memory ceiling of 8 GB (`deploy.resources.limits.memory: 8G`, matching `MemoryMax=8G`). systemd's
  `MemoryHigh=6G` has no Compose equivalent: a memory reservation is a guarantee, not a throttle;
- 4 CPU quota (`cpus: "4"`, matching `CPUQuota=400%`);
- 256 PID limit (`pids: 256`, matching `TasksMax=256`);
- File descriptor limit of 4096 (`nofile: 4096:4096`, matching `LimitNOFILE=4096`);
- 150-second graceful stop timeout (`stop_grace_period: 150s`, matching `TimeoutStopSec=150`);
- Hardened sandbox: `read_only: true`, `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`,
  non-root user `65532:65532` (`nonroot`), and `tmpfs` `/tmp:rw,noexec,nosuid,size=64m`.

The port binds to `127.0.0.1:8080` by default. To expose the server behind a TLS reverse proxy (such
as NGINX, Traefik, or Caddy), attach the service to a shared internal Docker bridge network or bind
the host port to a private VPC/LAN interface IP (e.g. `10.0.0.1:8080:8080`).

Model weights are mounted as a read-only volume (`${STATIM_MODEL:-./models/model.gguf}`). The API
key is supplied via a Docker secret (`${STATIM_API_KEY_FILE:-./statim_api_key.txt}`, mounted at
`/run/secrets/statim_api_key`) or via an environment file (`env_file:`), never inline:

```sh
STATIM_MODEL=/srv/statim/model.gguf STATIM_API_KEY_FILE=/etc/statim/api-key \
  docker compose -f deploy/docker-compose.yml up -d
```

### Container health check

Both `Dockerfile` and `Dockerfile.vulkan` define a container `HEALTHCHECK` with `--interval=30s`,
`--timeout=5s`, `--start-period=60s`, and `--retries=3`.

The runtime images are minimal (distroless Debian 12 and unprivileged Ubuntu 24.04 without curl,
wget, or shell in the CPU image). The health check runs `/usr/local/bin/statim-healthcheck`, a tiny
static C++ binary built during the builder stage (`tools/docker/healthcheck.cpp`). It connects to
`127.0.0.1:8080`, issues `GET /health HTTP/1.1`, enforces a 3-second socket timeout, and exits 0 on
HTTP 200 (or 1 on connection refusal, timeout, or non-200 responses).

**Why liveness uses `/health` and not `/ready`:**
- Container liveness monitors whether the process is alive and responsive. `GET /health` is public
  and returns HTTP 200 as long as the server event loop is servicing requests.
- `GET /ready` reports whether the admission controller currently has capacity. Under peak load,
  when `--max-concurrent` requests are being scored, `/ready` returns HTTP 503 (`{"ready":false}`).
  Probing `/ready` for container liveness would trigger container restarts precisely when the server
  is saturated with valid inference work, terminating in-flight requests and causing cascading failures.
  `/ready` is intended for external load balancers routing new ingress connections.
- The 60-second `--start-period` ensures model weight preflight and cold mmap loading on slower storage
  do not prematurely count as failed retries.

## TLS reverse proxy

Keep Statim bound to loopback or a private container network. In the NGINX `http` context, define
connection and request-rate zones:

```nginx
limit_req_zone $binary_remote_addr zone=statim_rate:10m rate=10r/s;
limit_conn_zone $binary_remote_addr zone=statim_conn:10m;
```

Then use a TLS server block such as:

```nginx
server {
    listen 443 ssl http2;
    server_name statim.example.com;
    ssl_certificate /etc/letsencrypt/live/statim.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/statim.example.com/privkey.pem;

    client_max_body_size 2m;
    limit_req zone=statim_rate burst=20 nodelay;
    limit_conn statim_conn 20;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Request-ID $request_id;
        proxy_connect_timeout 5s;
        proxy_read_timeout 125s;
        proxy_send_timeout 35s;
    }
}
```

Add a separate port-80 redirect and manage certificates with the mechanism appropriate to the
site. The proxy passes `Authorization` by default; never log bearer values. Align proxy timeouts
with Statim's 30-second request and 120-second inference deadlines, and apply network-level rate
limits because application admission control is not a complete denial-of-service boundary.

## Probes and metrics

Liveness and readiness are deliberately unauthenticated so an orchestrator can probe them without
holding an API key:

```sh
curl --fail --silent http://127.0.0.1:8080/health
curl --fail --silent http://127.0.0.1:8080/ready
```

`/health` reports only status and version. `/ready` becomes successful only when the service can
accept traffic. Inference, `/v1/models`, and `/metrics` require the bearer key whenever auth is
configured. Configure Prometheus with a dedicated rotatable key:

```yaml
scrape_configs:
  - job_name: statim
    static_configs:
      - targets: ["127.0.0.1:8080"]
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/statim.token
```

Protect the token file with restrictive permissions and do not expose `/metrics` through the public
proxy. Alert on readiness failures, HTTP 503s, latency, in-flight requests, and busy workers. See
`docs/SECURITY.md` for the threat boundaries and regression coverage behind these recommendations.
