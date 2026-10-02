// End-to-end gate: state + questions -> token ids and answer JSON, vs. laya.Agent.system_one.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>

#include "statim/engine.h"

using statim::ojson;

static double worst = 0;
static int mismatches = 0;

static void cmp(const ojson& ref, const ojson& got, const std::string& path, double tol) {
    if (ref.is_number() && got.is_number()) {
        double d = std::fabs(ref.get<double>() - got.get<double>());
        worst = std::max(worst, d);
        if (d > tol) {
            ++mismatches;
            std::printf("  %s: ref %.4f got %.4f\n", path.c_str(), ref.get<double>(), got.get<double>());
        }
        return;
    }
    if (ref.type() != got.type()) {
        ++mismatches;
        std::printf("  %s: type differs (%s vs %s)\n", path.c_str(), ref.dump().c_str(), got.dump().c_str());
        return;
    }
    if (ref.is_object()) {
        auto rk = ref.begin(), gk = got.begin();
        for (; rk != ref.end() && gk != got.end(); ++rk, ++gk) {
            if (rk.key() != gk.key()) {
                ++mismatches;
                std::printf("  %s: key order/name differs (%s vs %s)\n", path.c_str(), rk.key().c_str(), gk.key().c_str());
                return;
            }
            cmp(rk.value(), gk.value(), path + "." + rk.key(), tol);
        }
        if (ref.size() != got.size()) {
            ++mismatches;
            std::printf("  %s: size differs\n", path.c_str());
        }
        return;
    }
    if (ref != got) {
        ++mismatches;
        std::printf("  %s: %s vs %s\n", path.c_str(), ref.dump().c_str(), got.dump().c_str());
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s model.gguf golden_inputs.json golden.jsonl [tol]\n", argv[0]);
        return 2;
    }
    const double tol = argc > 4 ? std::atof(argv[4]) : 2e-3;
    auto model = statim::Model::load(argv[1]);
    statim::Engine engine(model);
    std::ifstream fi(argv[2]);
    ojson inputs = ojson::parse(fi);
    std::map<std::pair<int, std::string>, ojson> golden;
    std::ifstream fg(argv[3]);
    std::string line;
    while (std::getline(fg, line)) {
        ojson r = ojson::parse(line);
        golden[{r["state_index"].get<int>(), r["question"].get<std::string>()}] = r;
    }
    int id_ok = 0, id_total = 0, choice_ok = 0, choice_total = 0;
    const ojson& qs = inputs["questions"];
    for (size_t si = 0; si < inputs["states"].size(); ++si) {
        const ojson& st = inputs["states"][si];
        auto items = engine.encode(st, qs);
        size_t qi = 0;
        for (auto it = qs.begin(); it != qs.end(); ++it, ++qi) {
            const ojson& g = golden.at({static_cast<int>(si), it.key()});
            ++id_total;
            bool same = g["ids"].get<std::vector<int32_t>>() == items[qi].ids && g["markers"].get<std::vector<int32_t>>() == items[qi].markers;
            id_ok += same;
            if (!same) std::printf("state %zu %s: token ids differ\n", si, it.key().c_str());
        }
        ojson res = engine.decide(st, qs);
        for (auto it = qs.begin(); it != qs.end(); ++it) {
            const ojson& ref = golden.at({static_cast<int>(si), it.key()})["answer"];
            const ojson& got = res["answers"][it.key()];
            cmp(ref, got, "state" + std::to_string(si) + "." + it.key(), tol);
            if (ref.contains("choice")) {
                ++choice_total;
                choice_ok += ref["choice"] == got["choice"];
            }
        }
    }
    std::printf("token ids identical: %d/%d | choice identical: %d/%d | answer fields over tol: %d | worst |diff| %.2e\n",
                id_ok, id_total, choice_ok, choice_total, mismatches, worst);
    bool ok = id_ok == id_total && choice_ok == choice_total && mismatches == 0;
    std::printf("%s (tol %.1e)\n", ok ? "PASS" : "FAIL", tol);
    return ok ? 0 : 1;
}
