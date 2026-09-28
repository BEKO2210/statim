// Statim — ModernBERT encoder + Laya decision head, executed with ggml.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "statim/tokenizer.h"

namespace statim {

struct HParams {
    // encoder
    int n_embd = 0, n_layer = 0, n_head = 0, n_ff = 0, local_window = 128;
    std::vector<bool> layer_is_global;
    float rope_theta_global = 160000.f, rope_theta_local = 10000.f, norm_eps = 1e-5f;
    int max_position = 8192;
    // decision head
    int head_n_layer = 2, head_n_head = 12, head_n_ff = 3072, n_act = 2;
    // prompt layout
    int max_len = 512, head_max_len = 192;
    // calibration
    std::vector<float> temperature{1.f, 1.f, 1.f};
    std::string temperature_by_options_json = "{}";
    std::string lang_temperatures_json = "{}";
    // special ids
    int32_t cls_id = -1, sep_id = -1, mask_id = -1, pad_id = 0;
    std::string mask_token = "<mask>";
    std::string name, weight_type;
};

// One encoder row: [CLS] head [SEP] ([MASK] option)* [SEP] state [SEP]
struct Item {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;  // positions of the per-option [MASK] tokens
    int qtype = 0;                 // 0 choice, 1 score, 2 noul
};

struct ItemResult {
    std::vector<float> logits;     // raw scorer logits, one per marker
    std::vector<float> act_probs;  // softmax over act head
};

struct RunOptions {
    int n_threads = 0;          // 0 = hardware concurrency
    bool flash_attn = true;     // fused flash-attention kernel (false: explicit softmax path)
};

// How a LoRA adapter is applied to the base weights (docs/API.md, "LoRA adapters").
//   merge:   W' = W + B·A is computed once at load (with ggml) into a copy of each adapted
//            projection; requests run exactly the base graph, so latency is the base latency,
//            at the cost of one copy of the adapted weights per adapter.
//   runtime: the graph adds B·(A·x) after each adapted projection; memory is only the LoRA
//            factors, every request pays two extra small matmuls per adapted projection.
enum class AdapterMode { merge, runtime };

struct AdapterInfo {
    std::string name;                     // general.name of the adapter file
    std::string path;
    std::vector<std::string> categories;  // question families served in "adapter": "auto" routing
    AdapterMode mode = AdapterMode::merge;
    int rank = 0;
    float alpha = 0;
    int n_pairs = 0;     // LoRA pairs in the file
    int n_applied = 0;   // pairs with a non-zero delta (merged or evaluated at runtime)
    size_t bytes = 0;    // memory added on top of the shared base weights
    double load_ms = 0;
};

class Model {
public:
    // device: "cpu", "gpu" (first GPU), a backend name ("vulkan", "cuda") or a device name
    // ("Vulkan0"). Empty: $STATIM_DEVICE, else "cpu". GPU devices get a copy of the weights.
    static std::shared_ptr<Model> load(const std::string& gguf_path, const std::string& device = "");

    // The base model with a LoRA adapter (tools/convert_lora.py) applied. Shares the tokenizer and
    // every tensor the adapter does not touch with `base` and keeps `base` alive. Pairs whose
    // delta is exactly zero leave the base tensor in place, so a zero adapter is bit-identical to
    // the base. Throws if the adapter was converted for another model or its shapes do not match.
    static std::shared_ptr<Model> with_adapter(std::shared_ptr<Model> base, const std::string& adapter_gguf,
                                               AdapterMode mode = AdapterMode::merge, int n_threads = 0);
    ~Model();

    const HParams& hparams() const;
    const Tokenizer& tokenizer() const;
    size_t weight_bytes() const;
    const std::string& device() const;  // e.g. "cpu", "Vulkan0 (NVIDIA GeForce RTX 3070)"
    const AdapterInfo* adapter() const;  // nullptr for a base model

    // f32 copy of a weight as this model evaluates it (tests and tooling). For an adapter view the
    // adapted encoder projections include B·A in both modes. Throws for weights held only in the
    // CPU repack buffer after a merge (they cannot be read back).
    std::vector<float> weight_f32(const std::string& name) const;

    struct Impl;
    Impl* impl() const { return impl_.get(); }

private:
    Model();
    std::unique_ptr<Impl> impl_;
};

// Per-thread executor: owns the compute allocator. Not thread-safe; use one per worker.
class Runner {
public:
    explicit Runner(std::shared_ptr<Model> model, RunOptions opts = {});
    ~Runner();
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    // Items in one call are padded to the longest and evaluated in one graph.
    std::vector<ItemResult> run(const std::vector<Item>& items);

    void set_deadline(std::chrono::steady_clock::time_point deadline);
    const Model& model() const { return *model_; }

private:
    struct Impl;
    std::shared_ptr<Model> model_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace statim
