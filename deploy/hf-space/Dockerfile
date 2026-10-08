# syntax=docker/dockerfile:1

ARG STATIM_SOURCE=release
ARG STATIM_VERSION=0.11.0
ARG MODEL_SOURCE=hub
ARG MODEL_REPO=Beko2210/statim-decide-multilingual-base
ARG MODEL_FILE=statim-decide-multilingual-base-q8_0.gguf
# Pinned to the Hub commit of the released model and checked by SHA-256. A /resolve/main/ URL let
# the Space keep a cached older model after the repository moved on (0.9.3 served a pre-0.7.0 file).
ARG MODEL_REVISION=bfb190cda3ed71e278beeb295d279e7fd6900138
ARG MODEL_SHA256=97a275d7033382cb0dc443598b1507f1d9ecdaf75dcfa1237b489d489bf70dd6
# The actions LoRA adapter (games on the site's live mode), pinned and checked the same way.
ARG ADAPTER_SOURCE=hub
ARG ADAPTER_REPO=Beko2210/statim-decide-multilingual-base-actions
ARG ADAPTER_FILE=statim-decide-multilingual-base-actions.lora.gguf
ARG ADAPTER_REVISION=91eb69665d725fa356df81e0efb8b8ef8fca3497
ARG ADAPTER_SHA256=ed7b2a639fbf4876a171bfed62c8d5e8184d92024acda34b1ca7f8d8dab2a3b3

FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251 AS statim-release
ARG STATIM_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /download
RUN set -eu; \
    case "$STATIM_VERSION" in (*[!0-9A-Za-z._+-]*|'') echo "invalid STATIM_VERSION" >&2; exit 2;; esac; \
    archive="statim-${STATIM_VERSION}-linux-x86_64-cpu.tar.gz"; \
    release_url="https://github.com/BEKO2210/statim/releases/download/v${STATIM_VERSION}"; \
    curl -fL --retry 3 -o "$archive" "$release_url/$archive"; \
    curl -fL --retry 3 -o SHA256SUMS "$release_url/SHA256SUMS"; \
    checksum="$(awk -v file="$archive" '$2 == file || $2 == "*" file { print $1 }' SHA256SUMS)"; \
    test -n "$checksum"; \
    test "$(printf '%s\n' "$checksum" | wc -l)" -eq 1; \
    printf '%s  %s\n' "$checksum" "$archive" | sha256sum -c -; \
    tar -xzf "$archive" --strip-components=1 "statim-${STATIM_VERSION}-linux-x86_64-cpu/statim"; \
    install -Dm0755 statim /out/statim

FROM scratch AS statim-local
COPY statim /out/statim

FROM statim-${STATIM_SOURCE} AS statim

FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251 AS model-hub
ARG MODEL_REPO
ARG MODEL_FILE
ARG MODEL_REVISION
ARG MODEL_SHA256
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "$MODEL_REPO" in (*[!0-9A-Za-z._/-]*|'') echo "invalid MODEL_REPO" >&2; exit 2;; esac; \
    case "$MODEL_FILE" in (*[!0-9A-Za-z._-]*|'') echo "invalid MODEL_FILE" >&2; exit 2;; esac; \
    case "$MODEL_REVISION" in (*[!0-9a-f]*|'') echo "invalid MODEL_REVISION" >&2; exit 2;; esac; \
    case "$MODEL_SHA256" in (*[!0-9a-f]*|'') echo "invalid MODEL_SHA256" >&2; exit 2;; esac; \
    mkdir -p /out; \
    curl -fL --retry 3 -o /out/model.gguf \
      "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}"; \
    echo "${MODEL_SHA256}  /out/model.gguf" | sha256sum -c -

FROM scratch AS model-local
COPY model.gguf /out/model.gguf

FROM model-${MODEL_SOURCE} AS model

FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251 AS adapter-hub
ARG ADAPTER_REPO
ARG ADAPTER_FILE
ARG ADAPTER_REVISION
ARG ADAPTER_SHA256
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "$ADAPTER_REPO" in (*[!0-9A-Za-z._/-]*|'') echo "invalid ADAPTER_REPO" >&2; exit 2;; esac; \
    case "$ADAPTER_FILE" in (*[!0-9A-Za-z._-]*|'') echo "invalid ADAPTER_FILE" >&2; exit 2;; esac; \
    case "$ADAPTER_REVISION" in (*[!0-9a-f]*|'') echo "invalid ADAPTER_REVISION" >&2; exit 2;; esac; \
    case "$ADAPTER_SHA256" in (*[!0-9a-f]*|'') echo "invalid ADAPTER_SHA256" >&2; exit 2;; esac; \
    mkdir -p /out; \
    curl -fL --retry 3 -o /out/actions.lora.gguf \
      "https://huggingface.co/${ADAPTER_REPO}/resolve/${ADAPTER_REVISION}/${ADAPTER_FILE}"; \
    echo "${ADAPTER_SHA256}  /out/actions.lora.gguf" | sha256sum -c -

FROM scratch AS adapter-local
COPY actions.lora.gguf /out/actions.lora.gguf

FROM adapter-${ADAPTER_SOURCE} AS adapter

FROM debian:trixie-slim@sha256:a99cfc517144bc59b1978475ec53b46ecabec7e43635402ee5b77cc54cd1b20a AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends busybox libgomp1 libstdc++6 \
    && groupadd --gid 1000 statim \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin statim \
    && rm -rf /var/lib/apt/lists/*
COPY --from=statim --chmod=0755 /out/statim /usr/local/bin/statim
COPY --from=model --chown=1000:1000 /out/model.gguf /models/model.gguf
COPY --from=adapter --chown=1000:1000 /out/actions.lora.gguf /models/actions.lora.gguf

EXPOSE 7860
USER 1000:1000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["busybox", "wget", "-q", "--spider", "http://127.0.0.1:7860/ready"]

ENTRYPOINT ["/usr/local/bin/statim"]
# The public demo is intentionally unauthenticated; production images fail closed instead.
# --cors-origin lets the site's live game mode (beko2210.github.io) call this server from the browser.
CMD ["serve", "-m", "multilingual=/models/model.gguf", "--adapter", "multilingual:actions=/models/actions.lora.gguf", "--host", "0.0.0.0", "--port", "7860", "--allow-unauthenticated", "--frame-ancestors", "https://huggingface.co", "--cors-origin", "https://beko2210.github.io", "--threads", "2", "--workers", "1", "--max-concurrent", "2", "--http-queue", "8", "--max-request-work", "128", "--max-request-tokens", "131072", "--max-attention-mib", "1024", "--max-response-bytes", "1048576", "--request-timeout", "15", "--inference-timeout", "120"]
