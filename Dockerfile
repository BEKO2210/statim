# syntax=docker/dockerfile:1
# Multi-stage build: portable x86-64-v3 binary (AVX2/FMA), distroless runtime, non-root.
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTATIM_NATIVE=OFF -DSTATIM_BUILD_TESTS=OFF \
      -DCMAKE_CXX_FLAGS="-march=x86-64-v3" -DCMAKE_C_FLAGS="-march=x86-64-v3" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    && cmake --build build --target statim statim-quantize \
    && strip build/statim build/statim-quantize

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/build/statim /src/build/statim-quantize /usr/local/bin/
EXPOSE 8080
USER nonroot
# Mount a converted model at /models/model.gguf; set STATIM_API_KEY to require bearer auth.
ENTRYPOINT ["/usr/local/bin/statim"]
CMD ["serve", "-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080"]
