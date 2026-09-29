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

### GPU backends

GPU support is optional. Vulkan requires Vulkan headers, `glslc`, SPIR-V headers, and a runtime
driver. CUDA requires the CUDA toolkit and `-DSTATIM_CUDA=ON`.

```bash
cmake -S . -B build-vk -DSTATIM_VULKAN=ON && cmake --build build-vk
ctest --test-dir build-vk                      # CPU gates + the same gates on the GPU (*_vulkan)
./build-vk/statim serve --device vulkan -m english=models/laya-english-f32.gguf -m multilingual=models/laya-multilingual-f32.gguf
```
