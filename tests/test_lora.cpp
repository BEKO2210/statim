// LoRA adapter gates (CPU):
//   1. a zero adapter (lora_B = 0) is bit-identical to the base: raw logits and decide() JSON;
//   2. a random rank-4 adapter matches a PyTorch reference that merged the same adapter into the
//      official Laya model (tests/lora/gen_reference.py):
//      - merged weights at sampled positions and the per-tensor delta sum, within 1e-4,
//      - the logits within 1e-4 and the same argmax on every item,
//      in both merge and runtime mode, and merge vs. runtime within 1e-4;
//   3. binding to the base: SHA-256 test vectors, the checkpoint fingerprint and full checkpoint
//      SHA-256, and rejection of an adapter for another checkpoint, one without a fingerprint and
//      one whose shapes do not fit;
//   4. load errors: a model file as adapter, stacked adapters.
// With --quantized, adapters on quantized base weights instead (q8_0, q4_0; on an AVX2 CPU q4_0
// lives in the repack buffer): the files share the f32 checkpoint's fingerprint; runtime LoRA, the
// default there, keeps the logits as close to the f32 reference as the base's quantization noise;
// --adapter-mode merge writes exactly quantize(dequantize(W) + B·A), and the repacked copy computes
// the same logits as the plain one. How much of the adapter merging loses is printed.
//
// usage: test_lora base.gguf zero.gguf random.gguf golden_inputs.json golden_base.jsonl
//                  golden_lora_random.jsonl lora_random_weights.json
//        test_lora --quantized base.gguf random.gguf golden_lora_random.jsonl quantized.gguf...
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "json.hpp"
#include "sha256.h"
#include "statim/engine.h"
#include "statim/model.h"

using json = nlohmann::json;

namespace {

int failures = 0;

// quantized base + runtime adapter vs the f32 reference, relative to the quantized base's own noise
constexpr double kQuantizedTolerance = 1.5;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) ++failures;
}

std::string fmt(const char* f, double v) {
    char buf[128];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}

struct Rec {
    int state;
    std::string question;
    statim::Item item;
    std::vector<double> logits;
};

std::vector<Rec> read_jsonl(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<Rec> out;
    std::string line;
    while (std::getline(in, line)) {
        json r = json::parse(line);
        out.push_back({r["state_index"].get<int>(), r["question"].get<std::string>(),
                       {r["ids"].get<std::vector<int32_t>>(), r["markers"].get<std::vector<int32_t>>(), r["qtype"].get<int>()},
                       r["logits"].get<std::vector<double>>()});
    }
    return out;
}

// One batched graph per state, like test_model_parity.
std::vector<std::vector<float>> run(const std::shared_ptr<statim::Model>& m, const std::vector<Rec>& recs) {
    statim::Runner runner(m);
    std::vector<std::vector<float>> out(recs.size());
    for (size_t i = 0; i < recs.size();) {
        size_t j = i;
        std::vector<statim::Item> items;
        while (j < recs.size() && recs[j].state == recs[i].state) items.push_back(recs[j++].item);
        auto res = runner.run(items);
        for (size_t k = 0; k < res.size(); ++k) out[i + k] = res[k].logits;
        i = j;
    }
    return out;
}

bool bit_identical(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].size() != b[i].size() || std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) != 0) return false;
    return true;
}

double max_diff(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t k = 0; k < a[i].size(); ++k) d = std::max(d, std::fabs(double(a[i][k]) - b[i][k]));
    return d;
}

template <class F>
bool throws(F&& f, const std::string& needle) {
    try {
        f();
    } catch (const std::exception& e) {
        std::printf("        (%s)\n", e.what());
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// Writes the adapter at src to dst with its metadata changed by edit, the tensor of the same name
// swapped for `replace` (if given) and `extra` tensors added.
void rewrite_adapter(const std::string& src, const std::string& dst, const std::function<void(gguf_context*)>& edit,
                     const ggml_tensor* replace = nullptr, const std::vector<const ggml_tensor*>& extra = {}) {
    ggml_context* data = nullptr;
    gguf_context* in = gguf_init_from_file(src.c_str(), {/*no_alloc=*/false, &data});
    if (!in) throw std::runtime_error("cannot read " + src);
    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, in);
    edit(out);
    for (int64_t i = 0; i < gguf_get_n_tensors(in); ++i) {
        const char* name = gguf_get_tensor_name(in, i);
        gguf_add_tensor(out, replace && std::strcmp(name, ggml_get_name(replace)) == 0 ? replace : ggml_get_tensor(data, name));
    }
    for (const ggml_tensor* t : extra) gguf_add_tensor(out, t);
    const bool ok = gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
    gguf_free(in);
    ggml_free(data);
    if (!ok) throw std::runtime_error("cannot write " + dst);
}

double mean_diff(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    double d = 0;
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t k = 0; k < a[i].size(); ++k, ++n) d += std::fabs(double(a[i][k]) - b[i][k]);
    return n ? d / n : 0;
}

size_t same_argmax(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    size_t same = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        size_t x = 0, y = 0;
        for (size_t k = 0; k < a[i].size(); ++k) {
            if (a[i][k] > a[i][x]) x = k;
            if (b[i][k] > b[i][y]) y = k;
        }
        same += x == y;
    }
    return same;
}

// LoRA on quantized base weights (see the header).
int quantized(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s --quantized base.gguf random.gguf golden_lora_random.jsonl quantized.gguf...\n", argv[0]);
        return 2;
    }
    auto base = statim::Model::load(argv[2], "cpu");
    const std::string random_path = argv[3];
    const std::vector<Rec> ref = read_jsonl(argv[4]);
    std::vector<std::vector<float>> ref_logits;
    for (const Rec& r : ref) ref_logits.emplace_back(r.logits.begin(), r.logits.end());
    const auto lb_f32 = run(base, ref);
    const std::map<std::string, ggml_type> types = {{"q8_0", GGML_TYPE_Q8_0}, {"q4_0", GGML_TYPE_Q4_0}};
    for (int i = 5; i < argc; ++i) {
        auto qbase = statim::Model::load(argv[i], "cpu");
        const std::string wt = qbase->hparams().weight_type;
        std::printf("%s base (%s)\n", wt.c_str(), argv[i]);
        check(types.count(wt) == 1, wt + ": a quantized type this test knows");
        check(qbase->fingerprint() == base->fingerprint(), wt + ": same fingerprint as the f32 checkpoint");

        // default: runtime LoRA, the exact f32 delta on top of the quantized base
        auto runtime = statim::Model::with_adapter(qbase, random_path);
        check(runtime->adapter()->mode == statim::AdapterMode::runtime && runtime->adapter()->n_applied == 87,
              wt + ": runtime is the default mode on quantized weights");
        const auto lb = run(qbase, ref), lr = run(runtime, ref);
        const double noise = mean_diff(lb, lb_f32), effect = mean_diff(lr, lb), dev = mean_diff(lr, ref_logits);
        std::printf("  mean |dlogit|: %s base vs f32 base %.4f; adapter effect %.4f; %s + adapter vs the PyTorch f32 merge %.4f\n",
                    wt.c_str(), noise, effect, wt.c_str(), dev);
        check(effect > 1e-2, wt + ": the adapter changes the logits");
        check(dev <= kQuantizedTolerance * noise,
              wt + ": with the adapter, the logits stay as close to the f32 reference as the base's own quantization noise (" +
                  fmt("%.2f", dev / noise) + "x, limit " + fmt("%.2f", kQuantizedTolerance) + "x)");

        // --adapter-mode merge: quantize(dequantize(W) + B·A), checked bit for bit on weights outside the
        // repack buffer (STATIM_NO_REPACK), and the repacked copy must compute the same logits
        auto merged = statim::Model::with_adapter(qbase, random_path, statim::AdapterMode::merge);
        setenv("STATIM_NO_REPACK", "1", 1);
        auto qplain = statim::Model::load(argv[i], "cpu");
        unsetenv("STATIM_NO_REPACK");
        auto plain = statim::Model::with_adapter(qplain, random_path, statim::AdapterMode::merge);
        const std::string name = "encoder.layers.5.attn.Wqkv.weight";
        const int64_t in = qbase->hparams().n_embd, out = 3 * in;
        const std::vector<float> exact = runtime->weight_f32(name);
        std::vector<uint8_t> q(ggml_row_size(types.at(wt), in) * out);
        ggml_quantize_chunk(types.at(wt), exact.data(), q.data(), 0, out, in, nullptr);
        std::vector<float> want(exact.size());
        ggml_get_type_traits(types.at(wt))->to_float(q.data(), want.data(), static_cast<int64_t>(want.size()));
        check(plain->weight_f32(name) == want, wt + ": merge writes quantize(dequantize(W) + B·A), bit for bit (" + name + ")");
        // the CPU's repacked int8 GEMM and the generic kernel do not round identically, so compare the
        // merged pair against the same pair of base models
        const auto lm = run(merged, ref), lp = run(plain, ref), lbp = run(qplain, ref);
        const double kernels = mean_diff(lb, lbp), merged_kernels = mean_diff(lm, lp);
        check(merged_kernels <= 2 * kernels + 1e-5,
              wt + ": merged weights in the CPU's layout vs the file layout differ as much as the base's do (mean |dlogit| " +
                  fmt("%.2e", merged_kernels) + " vs " + fmt("%.2e", kernels) + ")");
        const double lost = mean_diff(lm, lr);
        std::printf("  merge vs runtime mean |dlogit| %.4f = %.0f %% of the adapter's effect; merged vs base %.4f; argmax "
                    "merge = runtime on %zu/%zu (merging rounds the delta to the %s grid: the reason runtime is the default)\n",
                    lost, 100 * lost / effect, mean_diff(lm, lb), same_argmax(lm, lr), ref.size(), wt.c_str());
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--quantized") == 0) return quantized(argc, argv);
    if (argc < 8) {
        std::fprintf(stderr, "usage: %s base.gguf zero.gguf random.gguf golden_inputs.json golden_base.jsonl "
                             "golden_lora_random.jsonl lora_random_weights.json\n", argv[0]);
        return 2;
    }
    const std::string base_path = argv[1], zero_path = argv[2], random_path = argv[3];
    auto base = statim::Model::load(base_path, "cpu");
    const std::vector<Rec> ref = read_jsonl(argv[6]);
    std::map<std::pair<int, std::string>, const Rec*> base_ref;
    const std::vector<Rec> base_all = read_jsonl(argv[5]);
    for (const Rec& r : base_all) base_ref[{r.state, r.question}] = &r;
    std::printf("reference: %zu items\n", ref.size());
    for (const Rec& r : ref) {
        const Rec* b = base_ref.at({r.state, r.question});
        if (b->item.ids != r.item.ids) throw std::runtime_error("golden_base and golden_lora tokenize differently");
    }

    const auto logits_base = run(base, ref);
    {
        double d = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const Rec* b = base_ref.at({ref[i].state, ref[i].question});
            for (size_t k = 0; k < b->logits.size(); ++k) d = std::max(d, std::fabs(logits_base[i][k] - b->logits[k]));
        }
        std::printf("base vs PyTorch on these items (context, gated by test_model_parity): max |diff| %.2e\n", d);
    }

    std::printf("zero adapter\n");
    for (auto mode : {statim::AdapterMode::merge, statim::AdapterMode::runtime}) {
        const char* mname = mode == statim::AdapterMode::merge ? "merge" : "runtime";
        auto zero = statim::Model::with_adapter(base, zero_path, mode);
        const auto* info = zero->adapter();
        check(info && info->n_pairs == 88 && info->n_applied == 0 && info->bytes == 0,
              std::string(mname) + ": 88 pairs, none applied, no extra memory");
        if (mode == statim::AdapterMode::merge) {
            check(bit_identical(run(zero, ref), logits_base), "merge: logits bit-identical to the base");
            std::ifstream in(argv[4]);
            json inputs = json::parse(in);
            statim::DecideOptions opts;
            opts.return_logits = true;
            statim::Engine eb(base), ez(zero);
            bool same = true;
            for (int si : {0, 2, 11, 21, 25})
                same &= eb.decide(inputs["states"][si], inputs["questions"], opts).dump() ==
                        ez.decide(inputs["states"][si], inputs["questions"], opts).dump();
            check(same, "merge: decide() answers byte-identical to the base");
        }
    }

    std::printf("random adapter vs PyTorch merge\n");
    std::ifstream wf(argv[7]);
    const json wref = json::parse(wf);
    std::vector<std::vector<float>> by_mode[2];
    for (auto mode : {statim::AdapterMode::merge, statim::AdapterMode::runtime}) {
        const int mi = mode == statim::AdapterMode::merge ? 0 : 1;
        const std::string mname = mi == 0 ? "merge" : "runtime";
        auto rnd = statim::Model::with_adapter(base, random_path, mode);
        const auto* info = rnd->adapter();
        std::printf("  %s: %d/%d pairs applied, %.1f MB extra, loaded in %.0f ms\n", mname.c_str(), info->n_applied,
                    info->n_pairs, info->bytes / 1e6, info->load_ms);
        check(info->n_pairs == 88 && info->n_applied == 87, mname + ": 87 of 88 pairs applied (layer 0 mlp.Wo is zero)");

        double wmax = 0, smax = 0;
        for (auto& [name, t] : wref["tensors"].items()) {
            const std::vector<float> w = rnd->weight_f32(name);
            const std::vector<float> w0 = base->weight_f32(name);
            const auto idx = t["index"].get<std::vector<size_t>>();
            const auto val = t["value"].get<std::vector<double>>();
            for (size_t k = 0; k < idx.size(); ++k) wmax = std::max(wmax, std::fabs(w[idx[k]] - val[k]));
            double sum = 0;
            for (size_t k = 0; k < w.size(); ++k) sum += double(w[k]) - double(w0[k]);
            smax = std::max(smax, std::fabs(sum - t["delta_sum"].get<double>()));
        }
        check(wmax <= 1e-4, mname + ": merged weights at sampled positions, max |diff| " + fmt("%.2e", wmax) +
                                " (if this fails by far, the fixture differs from the one gen_reference.py saw)");
        check(smax <= 1e-4, mname + ": per-tensor delta sums, max |diff| " + fmt("%.2e", smax));

        by_mode[mi] = run(rnd, ref);
        const auto& ours = by_mode[mi];
        double abs_max = 0, eff_mean = 0;
        size_t n = 0, argmax_ok = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const Rec* b = base_ref.at({ref[i].state, ref[i].question});
            size_t am = 0, am_ref = 0;
            for (size_t k = 0; k < ours[i].size(); ++k) {
                abs_max = std::max(abs_max, std::fabs(ours[i][k] - ref[i].logits[k]));
                eff_mean += std::fabs(ref[i].logits[k] - b->logits[k]);
                ++n;
                if (ours[i][k] > ours[i][am]) am = k;
                if (ref[i].logits[k] > ref[i].logits[am_ref]) am_ref = k;
            }
            argmax_ok += am == am_ref;
        }
        eff_mean /= n;
        check(eff_mean > 1e-2, mname + ": the adapter changes the logits (mean |effect| " + fmt("%.3f", eff_mean) + ")");
        check(abs_max <= 1e-4 && argmax_ok == ref.size(),
              mname + ": logits vs PyTorch, max |diff| " + fmt("%.2e", abs_max) + ", argmax " + std::to_string(argmax_ok) + "/" +
                  std::to_string(ref.size()));
    }
    check(max_diff(by_mode[0], by_mode[1]) <= 1e-4, "merge vs runtime logits, max |diff| " + fmt("%.2e", max_diff(by_mode[0], by_mode[1])));

    std::printf("base binding\n");
    auto sha = [](const std::string& s) {
        statim::Sha256 h;
        h.update(s.data(), s.size());
        return h.hex();
    };
    statim::Sha256 million;
    const std::string thousand(1000, 'a');
    for (int i = 0; i < 1000; ++i) million.update(thousand.data(), thousand.size());
    check(sha("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" &&
              sha("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" &&
              sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" &&
              million.hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "SHA-256 test vectors (FIPS 180-2: empty, abc, 448 bits, a million 'a' in 1000-byte pieces)");
    check(base->fingerprint().size() == 64 && statim::Model::with_adapter(base, zero_path)->fingerprint() == base->fingerprint(),
          "the base has a fingerprint (" + base->fingerprint().substr(0, 16) + "...), its adapter views report it");
    const std::string edited = random_path + ".edited.gguf";
    rewrite_adapter(random_path, edited, [](gguf_context* g) {
        gguf_set_val_str(g, "statim.lora.base_fingerprint", std::string(64, 'f').c_str());
    });
    check(throws([&] { statim::Model::with_adapter(base, edited); }, "was converted for another checkpoint"),
          "an adapter converted for another checkpoint is rejected");
    if (!base->checkpoint_sha256().empty()) {
        rewrite_adapter(random_path, edited, [](gguf_context* g) {
            gguf_set_val_str(g, "statim.lora.base_checkpoint_sha256", std::string(64, 'f').c_str());
        });
        check(throws([&] { statim::Model::with_adapter(base, edited); }, "the vectors match but the matrices do not"),
              "an adapter for different checkpoint matrices is rejected");
    } else {
        std::printf("  note base model has no statim.checkpoint_sha256; skipping matrix binding check\n");
    }
    rewrite_adapter(random_path, edited, [](gguf_context* g) { gguf_remove_key(g, "statim.lora.base_fingerprint"); });
    check(throws([&] { statim::Model::with_adapter(base, edited); }, "does not record its base model"),
          "an adapter without a base fingerprint is rejected");
    {
        ggml_context* c = ggml_init({ggml_tensor_overhead() + 100 * 4 * sizeof(float) + 64, nullptr, false});
        ggml_tensor* a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 100, 4);  // the base projection takes 768 inputs
        ggml_set_name(a, "encoder.layers.3.mlp.Wi.weight.lora_a");
        for (int64_t k = 0; k < ggml_nelements(a); ++k) static_cast<float*>(a->data)[k] = 0.01f;
        rewrite_adapter(random_path, edited, [](gguf_context*) {}, a);
        ggml_free(c);
    }
    check(throws([&] { statim::Model::with_adapter(base, edited); }, "encoder.layers.3.mlp.Wi.weight has shapes A 100x4"),
          "an adapter whose shapes do not fit the base is rejected");
    {
        ggml_context* c = ggml_init({2 * ggml_tensor_overhead() + 2 * 2304 * 4 * sizeof(float) + 64, nullptr, false});
        ggml_tensor* a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 768, 4);
        ggml_tensor* b = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 2304);
        ggml_set_name(a, "encoder.layers.03.mlp.Wi.weight.lora_a");  // a second name for layer 3
        ggml_set_name(b, "encoder.layers.03.mlp.Wi.weight.lora_b");
        std::memset(a->data, 0, ggml_nbytes(a));
        std::memset(b->data, 0, ggml_nbytes(b));
        rewrite_adapter(random_path, edited, [](gguf_context*) {}, nullptr, {a, b});
        ggml_free(c);
    }
    check(throws([&] { statim::Model::with_adapter(base, edited); }, "unsupported tensor 'encoder.layers.03.mlp.Wi.weight.lora_a'"),
          "a layer number with a leading zero is not a second name for the same projection");
    rewrite_adapter(random_path, edited, [](gguf_context* g) { gguf_set_val_str(g, "statim.lora.rank", "4"); });
    check(throws([&] { statim::Model::with_adapter(base, edited); }, "has the wrong type"),
          "adapter metadata of the wrong type is an error, not an abort");
    std::remove(edited.c_str());

    std::printf("errors\n");
    check(throws([&] { statim::Model::with_adapter(base, base_path); }, "not a Statim LoRA adapter"),
          "a model file is rejected as adapter");
    check(throws([&] { statim::Model::with_adapter(statim::Model::with_adapter(base, zero_path), zero_path); }, "cannot stack"),
          "adapters do not stack");
    check(throws([&] { statim::Model::with_adapter(base, "/nonexistent.gguf"); }, "cannot read"), "missing file");

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
