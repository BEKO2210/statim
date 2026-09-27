# Statim – hier weitermachen

Stand: 26.09.2026, nachts gebaut auf belkis-home. Repo: https://github.com/BEKO2210/statim (öffentlich seit 27.09.2026).

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

Gemessen auf belkis-home (Xeon E3-1505M v5, Turbo aus), gleiches Modell, gleiche Eingaben:

| | Laya | Statim |
|---|---|---|
| Kaltstart bis erste Antwort | 11,35 s | 0,76 s |
| Server-RAM | 3.915 MB | 650 MB |
| HTTP-Durchsatz (1 Client) | 0,43 req/s | 1,07 req/s |
| HTTP p95 (1 Client) | 4.994 ms | 1.280 ms |
| Latenz im Prozess (Mittel) | 1.186 ms | 1.139 ms – praktisch gleich |

Genauigkeit (je 400 Fälle, Laya's eigene Konstruktion), Konsens-Modus vs. bestes Laya-Modell:
AG News 0,950 = 0,950 · Emotion 0,600 vs. 0,5925 · Banking77 0,4875 vs. 0,425.

Was nicht geholfen hat (gemessen, deshalb nicht Standard): Options-Rotation (Emotion 0,5375 → 0,525),
Kontext-Kalibrierung (gemischt), 4-Bit (verfälscht Antworten). Rohe Rechenleistung ist gleich –
PyTorch/MKL ist auf diesem CPU schon am Limit. Der Vorsprung kommt aus allem drumherum.

## Offene Punkte

- GPU: Vulkan läuft (26.09.2026, pop-os, RTX 3070): `-DSTATIM_VULKAN=ON`, `--device vulkan`, ~8× Durchsatz,
  Parität 240/240 (exakt-f32 als Standard, `--gpu-fast` = f16). CUDA-Backend noch offen (bräuchte nvcc + gcc ≤ 12).
- 4-Bit-Quantisierung kostet Genauigkeit (Details in README) – f32/f16 bleiben Standard.
- Laya's volle Sprach-Erkennung (Router) und das dritte Modell `typed-decisions` fehlen noch.
- Banking77 (26.09.2026): Ursache war v. a. `head_max_len` (77 Optionen → 1 Subword je Intent).
  v3 (`--distill 6000 --epochs 5`, 35 min auf 3070): Banking77 0,4885 → 0,8655 (n=2000), Emotion/AG News
  unverändert (Distillation verhindert Vergessen, v1 ohne verlor 3 Pkt. Emotion). Modelle lokal:
  `models/laya-multilingual-banking77*` (v3), `-v1` (alt). Offen: HF-Upload, mehr Domänen (MASSIVE, CLINC).
