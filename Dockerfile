# syntax=docker/dockerfile:1
# Multi-stage build: portable x86-64-v3 binary (AVX2/FMA), distroless runtime, non-root.
FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTATIM_NATIVE=OFF -DSTATIM_BUILD_TESTS=OFF \
      -DCMAKE_CXX_FLAGS="-march=x86-64-v3" -DCMAKE_C_FLAGS="-march=x86-64-v3" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    && cmake --build build --target statim statim-quantize \
    && g++ -O2 -s -static tools/docker/healthcheck.cpp -o build/statim-healthcheck \
    && strip build/statim build/statim-quantize

FROM gcr.io/distroless/cc-debian12:nonroot@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f
COPY --from=build /src/build/statim /src/build/statim-quantize /src/build/statim-healthcheck /usr/local/bin/
EXPOSE 8080
USER nonroot
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["/usr/local/bin/statim-healthcheck"]
# Mount a converted model; provide STATIM_API_KEY or override CMD with --api-key-file for this listener.
ENTRYPOINT ["/usr/local/bin/statim"]
CMD ["serve", "-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080"]
