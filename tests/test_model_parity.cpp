// Parity gate: Statim's forward pass vs. the official Laya package (tests/data/golden_*.jsonl).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "json.hpp"
#include "statim/model.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf golden.jsonl [tol] [threads]\n", argv[0]);
        return 2;
    }
    const double tol = argc > 3 ? std::atof(argv[3]) : 1e-3;
    statim::RunOptions ro;
    ro.n_threads = argc > 4 ? std::atoi(argv[4]) : 0;
    if (const char* f = std::getenv("STATIM_FLASH")) ro.flash_attn = std::string(f) == "1";
    auto t0 = std::chrono::steady_clock::now();
    auto model = statim::Model::load(argv[1]);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("loaded %s (%s, %.1f MB weights) in %.0f ms\n", model->hparams().name.c_str(),
                model->hparams().weight_type.c_str(), model->weight_bytes() / 1e6,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    statim::Runner runner(model, ro);

    std::ifstream in(argv[2]);
    std::map<int, std::vector<json>> by_state;
    std::string line;
    while (std::getline(in, line)) {
        json r = json::parse(line);
        by_state[r["state_index"].get<int>()].push_back(std::move(r));
    }
    size_t n = 0, argmax_ok = 0;
    double max_diff = 0, max_act = 0, total_ms = 0;
    for (auto& [si, recs] : by_state) {
        std::vector<statim::Item> items;
        for (auto& r : recs) items.push_back({r["ids"].get<std::vector<int32_t>>(), r["markers"].get<std::vector<int32_t>>(), r["qtype"].get<int>()});
        auto a = std::chrono::steady_clock::now();
        auto res = runner.run(items);  // one state = one batched graph, like Laya's system_one
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        total_ms += ms;
        size_t maxlen = 0;
        for (auto& it : items) maxlen = std::max(maxlen, it.ids.size());
        if (std::getenv("STATIM_VERBOSE")) std::printf("  state %2d: %zu rows, L=%4zu, %8.1f ms\n", si, items.size(), maxlen, ms);
        for (size_t i = 0; i < recs.size(); ++i) {
            auto ref = recs[i]["logits"].get<std::vector<double>>();
            auto refa = recs[i]["act"].get<std::vector<double>>();
            size_t am_ref = 0, am = 0;
            for (size_t k = 0; k < ref.size(); ++k) {
                max_diff = std::max(max_diff, std::fabs(ref[k] - res[i].logits[k]));
                if (ref[k] > ref[am_ref]) am_ref = k;
                if (res[i].logits[k] > res[i].logits[am]) am = k;
            }
            for (size_t k = 0; k < refa.size(); ++k) max_act = std::max(max_act, std::fabs(refa[k] - res[i].act_probs[k]));
            argmax_ok += am == am_ref;
            ++n;
            if (si == 0 && i < 2) {
                std::printf("  state0 %-10s ref:", recs[i]["question"].get<std::string>().c_str());
                for (double v : ref) std::printf(" %8.4f", v);
                std::printf("\n  %*s ours:", 17, "");
                for (float v : res[i].logits) std::printf(" %8.4f", v);
                std::printf("\n");
            }
        }
    }
    std::printf("items %zu | argmax agree %zu/%zu | max |dlogit| %.2e | max |dact| %.2e | %.1f ms/state avg\n", n, argmax_ok, n,
                max_diff, max_act, total_ms / by_state.size());
    bool ok = argmax_ok == n && max_diff <= tol;
    std::printf("%s (tol %.1e)\n", ok ? "PASS" : "FAIL", tol);
    return ok ? 0 : 1;
}
