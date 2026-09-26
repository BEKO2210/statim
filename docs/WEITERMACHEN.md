# Statim – hier weitermachen

Stand: 26.09.2026, nachts gebaut auf belkis-home. Repo: https://github.com/BEKO2210/statim (privat).

## Was steht

- C++20-Engine auf ggml, lädt Laya-Checkpoints als eine GGUF-Datei (mmap, kein Python zur Laufzeit).
- Tokenizer in C++ – 100 % identisch mit HuggingFace (3.906 Fälle + 140.000 Fuzz-Strings), ~10× schneller.
- Paritäts-Tore gegen das offizielle Laya-Paket, beide Checkpoints (mehrsprachig + englisch):
  240/240 identische Token-Folgen, Antworten ≤ 1e-4 Abweichung. Läuft in CI (`ctest`).
- Server `statim serve`: `/v1/systemone` (Jev/Laya-kompatibel), `/v1/systemone/batch`, `/v1/models`,
  `/health`, `/ready`, `/metrics` (Prometheus), Bearer-Keys, 503+Retry-After bei Überlast, JSON-Logs,
  Auto-Routing Englisch → großes Modell, Web-Playground unter `/`.
- Paketierung: Dockerfile (distroless, non-root), gehärtete systemd-Unit, CI-Workflow, `statim-quantize`.

## Befehle

```bash
cd ~/statim
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build                       # Paritäts-Tore
./build/statim serve -m multilingual=models/laya-multilingual-f32.gguf -m english=models/laya-english-f32.gguf --port 8412
```

Referenz-Python (Laya original) liegt in `.venv-ref/` – nur für Tests/Benchmarks.

## Ehrliche Befunde

MESSWERTE_PLATZHALTER

## Offene Punkte

- GPU-Build (CUDA/Vulkan) – Graph ist backend-neutral, Toolchain fehlt auf belkis-home.
- 4-Bit-Quantisierung kostet Genauigkeit (Details in README) – f32/f16 bleiben Standard.
- Repo ist privat. Öffentlich schalten: `gh repo edit BEKO2210/statim --visibility public --accept-visibility-change-consequences`.
