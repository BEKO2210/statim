// Statim — ModernBERT encoder + Laya decision head, executed with ggml.
#pragma once

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

class Model {
public:
    // device: "cpu", "gpu" (first GPU), a backend name ("vulkan", "cuda") or a device name
    // ("Vulkan0"). Empty: $STATIM_DEVICE, else "cpu". GPU devices get a copy of the weights.
    static std::shared_ptr<Model> load(const std::string& gguf_path, const std::string& device = "");
    ~Model();

    const HParams& hparams() const;
    const Tokenizer& tokenizer() const;
    size_t weight_bytes() const;
    const std::string& device() const;  // e.g. "cpu", "Vulkan0 (NVIDIA GeForce RTX 3070)"

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

    const Model& model() const { return *model_; }

private:
    struct Impl;
    std::shared_ptr<Model> model_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace statim
