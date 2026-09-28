// LoRA adapter gates (CPU):
//   1. a zero adapter (lora_B = 0) is bit-identical to the base: raw logits and decide() JSON;
//   2. a random rank-4 adapter matches a PyTorch reference that merged the same adapter into the
//      official Laya model (tests/lora/gen_reference.py):
//      - merged weights at sampled positions and the per-tensor delta sum, within 1e-4,
//      - the logits within 1e-4 and the same argmax on every item,
//      in both merge and runtime mode, and merge vs. runtime within 1e-4;
//   3. load errors: a model file as adapter, stacked adapters.
//
// usage: test_lora base.gguf zero.gguf random.gguf golden_inputs.json golden_base.jsonl
//                  golden_lora_random.jsonl lora_random_weights.json
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
#include "statim/engine.h"
#include "statim/model.h"

using json = nlohmann::json;

namespace {

int failures = 0;

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

}  // namespace

int main(int argc, char** argv) {
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

    std::printf("errors\n");
    check(throws([&] { statim::Model::with_adapter(base, base_path); }, "not a Statim LoRA adapter"),
          "a model file is rejected as adapter");
    check(throws([&] { statim::Model::with_adapter(statim::Model::with_adapter(base, zero_path), zero_path); }, "cannot stack"),
          "adapters do not stack");
    check(throws([&] { statim::Model::with_adapter(base, "/nonexistent.gguf"); }, "cannot read"), "missing file");

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
