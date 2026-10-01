# Build, GPU, and quantization

Release binaries exist only for Linux x86-64 (CPU and Vulkan); everything else builds from source.
This page covers source builds, optional GPU support, and model quantization. Return to the
[README](../README.md), or see [REPRODUCE.md](../REPRODUCE.md#2-the-engine-matches-the-python-reference)
for the parity tests.

### Build from source

```bash
git clone --recursive https://github.com/BEKO2210/statim && cd statim
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
pip install numpy safetensors gguf            # converter only; not needed at runtime
tools/fetch_models.sh multilingual english    # download from Hugging Face + convert to GGUF
ctest --test-dir build                        # parity gates against the official package (as a regular user, see REPRODUCE.md)
./build/statim serve -m english=models/laya-english-f32.gguf -m multilingual=models/laya-multilingual-f32.gguf --consensus
  # open http://127.0.0.1:8080/ for the playground
```

Quantized variants: `build/statim-quantize models/laya-multilingual-f32.gguf out.gguf q8_0`. See
[quantization results](RESULTS.md#consensus-calibration-and-quantization) before choosing 4-bit weights.

### Older x86 CPUs

The release binaries are built for x86-64-v3: AVX2, FMA, F16C and BMI2, which means Intel Haswell,
AMD Excavator or newer. On an older CPU, `statim` and `statim-quantize` stop before any inference,
name the missing features and exit with status 1. Build from source on that machine with
`-DSTATIM_NATIVE=ON` instead.

### Android arm64

Statim builds for Android with the NDK (r27 or newer). CI cross-compiles every target on each
push; there is no release binary.

```bash
cmake -S . -B build-android -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DSTATIM_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENMP=OFF
cmake --build build-android
adb push build-android/statim models/laya-multilingual-f32.gguf /data/local/tmp/   # no root needed
```

Tested on a Galaxy A15 5G (MediaTek Dimensity 6100+: 2 Cortex-A76 and 6 Cortex-A55, 4 GB, Android
16) on 2026-10-01:

- **Correctness.** The native test suite passes: security, model validation, GGUF preflight,
  tokenizer, and multilingual and English parity. Both parity tests agree on all 240 argmaxes, with
  a max |Δlogit| of 3.2e-4 (multilingual) and 2.0e-4 (English), within the 1e-3 tolerance.
  `statim-quantize` runs on the phone too.
- **Speed (indicative only; a phone's clock and scheduler move these by up to 2×).** The
  multilingual model takes about 1.0–1.2 s per golden item in q8_0 and 2.1–3.9 s in f32. On this
  CPU, q8_0 is the faster format: the dot-product instructions do the int8 work, and the custom f32
  kernel is x86-only.
- **Replays.** The fuzz replay binaries need `STATIM_FUZZ_DATA` set to the copied `fuzz/data`
  directory, because their build-time path does not exist on the phone.

### GPU backends

GPU support is optional. Vulkan requires Vulkan headers, `glslc`, SPIR-V headers, and a runtime
driver. CUDA requires the CUDA toolkit and `-DSTATIM_CUDA=ON`.

```bash
cmake -S . -B build-vk -DSTATIM_VULKAN=ON && cmake --build build-vk
ctest --test-dir build-vk                      # CPU gates + the same gates on the GPU (*_vulkan)
./build-vk/statim serve --device vulkan -m english=models/laya-english-f32.gguf -m multilingual=models/laya-multilingual-f32.gguf
```

### CPU performance switches

- **f32 projections.** On x86-64 CPUs with AVX2 and FMA, Statim computes the encoder's f32
  projections with its own packed GEMM (`src/kernels.cpp`). `STATIM_SGEMM=0` falls back to ggml's
  matrix multiply, for A/B measurements.
- **Profiling.** `STATIM_PROFILE=FILE` writes the time of every graph node, one JSON record per
  scoring call. `bench/profile_short.py` uses it to compare Statim with ONNX Runtime per operation
  ([docs/ORT.md](ORT.md)).
