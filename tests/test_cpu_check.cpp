// The CPU requirement check of portable x86-64 builds (statim/cpu_check.h).
#include "statim/cpu_check.h"

#include <cstdio>
#include <set>
#include <string>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        failures += !ok;
    };
    const std::set<std::string> sandy_bridge = {};             // AVX only: none of the four
    const std::set<std::string> haswell = {"avx2", "fma", "f16c", "bmi2"};
    const std::set<std::string> no_fma = {"avx2", "f16c", "bmi2"};
    auto with = [](const std::set<std::string>& s) { return [&s](const char* f) { return s.count(f) > 0; }; };
    check(statim::missing_cpu_features(with(haswell)).empty(), "an x86-64-v3 CPU passes");
    check(statim::missing_cpu_features(with(sandy_bridge)) == "avx2, fma, f16c, bmi2", "a Sandy Bridge CPU lacks all four");
    check(statim::missing_cpu_features(with(no_fma)) == "fma", "a single missing feature is named");
    const std::string message = statim::cpu_requirement_message("avx2, fma");
    check(message.find("This CPU lacks: avx2, fma.") != std::string::npos, "the message names what is missing");
    check(message.find("-DSTATIM_NATIVE=ON") != std::string::npos, "the message says how to build for this CPU");
    // the real CPU running this test is a supported one (CI and the release builders)
    check(statim::missing_cpu_features().empty(), "this machine passes");
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
