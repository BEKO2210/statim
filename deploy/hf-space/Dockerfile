# syntax=docker/dockerfile:1

ARG STATIM_SOURCE=release
ARG STATIM_VERSION=0.8.3
ARG MODEL_SOURCE=hub
ARG MODEL_REPO=Beko2210/statim-decide-multilingual-base
ARG MODEL_FILE=statim-decide-multilingual-base-q8_0.gguf

FROM debian:bookworm-slim AS statim-release
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

FROM debian:bookworm-slim AS model-hub
ARG MODEL_REPO
ARG MODEL_FILE
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "$MODEL_REPO" in (*[!0-9A-Za-z._/-]*|'') echo "invalid MODEL_REPO" >&2; exit 2;; esac; \
    case "$MODEL_FILE" in (*[!0-9A-Za-z._-]*|'') echo "invalid MODEL_FILE" >&2; exit 2;; esac; \
    mkdir -p /out; \
    curl -fL --retry 3 -o /out/model.gguf \
      "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"

FROM scratch AS model-local
COPY model.gguf /out/model.gguf

FROM model-${MODEL_SOURCE} AS model

FROM debian:trixie-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends busybox libgomp1 libstdc++6 \
    && groupadd --gid 1000 statim \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin statim \
    && rm -rf /var/lib/apt/lists/*
COPY --from=statim --chmod=0755 /out/statim /usr/local/bin/statim
COPY --from=model --chown=1000:1000 /out/model.gguf /models/model.gguf

EXPOSE 7860
USER 1000:1000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["busybox", "wget", "-q", "--spider", "http://127.0.0.1:7860/ready"]

ENTRYPOINT ["/usr/local/bin/statim"]
CMD ["serve", "-m", "multilingual=/models/model.gguf", "--host", "0.0.0.0", "--port", "7860", "--threads", "2", "--workers", "1", "--max-concurrent", "2", "--http-queue", "8", "--max-request-work", "128", "--max-request-tokens", "131072", "--max-attention-mib", "1024", "--max-response-bytes", "1048576", "--request-timeout", "15", "--inference-timeout", "120"]
