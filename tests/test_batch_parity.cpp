// Packing several states must not change any decoded answer beyond the public 1e-4 contract.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "statim/engine.h"

using statim::ojson;

static double worst = 0;

static bool compare(const ojson& alone, const ojson& packed, const std::string& path) {
    if (alone.is_number() && packed.is_number()) {
        const double diff = std::fabs(alone.get<double>() - packed.get<double>());
        worst = std::max(worst, diff);
        if (diff <= 1e-4) return true;
        std::fprintf(stderr, "%s differs by %.8g\n", path.c_str(), diff);
        return false;
    }
    if (alone.type() != packed.type() || alone.size() != packed.size()) {
        std::fprintf(stderr, "%s type/size differs\n", path.c_str());
        return false;
    }
    if (alone.is_object()) {
        bool ok = true;
        for (auto it = alone.begin(); it != alone.end(); ++it) {
            if (!packed.contains(it.key())) {
                std::fprintf(stderr, "%s.%s is missing\n", path.c_str(), it.key().c_str());
                ok = false;
            } else {
                ok &= compare(it.value(), packed[it.key()], path + "." + it.key());
            }
        }
        return ok;
    }
    if (alone.is_array()) {
        bool ok = true;
        for (size_t i = 0; i < alone.size(); ++i)
            ok &= compare(alone[i], packed[i], path + "[" + std::to_string(i) + "]");
        return ok;
    }
    if (alone == packed) return true;
    std::fprintf(stderr, "%s differs: %s vs %s\n", path.c_str(), alone.dump().c_str(), packed.dump().c_str());
    return false;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s model.gguf golden_inputs.json\n", argv[0]);
        return 2;
    }
    auto model = statim::Model::load(argv[1], "cpu");
    statim::Engine engine(model);
    std::ifstream in(argv[2]);
    ojson fixture = ojson::parse(in);
    std::vector<ojson> states;
    for (const auto& state : fixture["states"])
        if (!(state.is_string() && state.get<std::string>().empty()) && states.size() < 6) states.push_back(state);
    ojson questions = ojson::object();
    for (auto it = fixture["questions"].begin(); it != fixture["questions"].end() && questions.size() < 3; ++it)
        questions[it.key()] = it.value();

    std::vector<ojson> alone;
    for (const auto& state : states) alone.push_back(engine.decide(state, questions));
    const auto packed = engine.decide_batch(states, questions);
    bool ok = alone.size() == packed.size();
    for (size_t i = 0; i < alone.size() && i < packed.size(); ++i)
        ok &= compare(alone[i], packed[i], "state" + std::to_string(i));
    std::printf("batch parity: %zu states, worst |diff| %.2e: %s\n", states.size(), worst, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
