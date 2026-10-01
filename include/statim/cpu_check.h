#pragma once
// Release builds are compiled for x86-64-v3 (AVX2, FMA, F16C, BMI2). On an older CPU the first
// vector instruction would kill the process with SIGILL and no explanation, so the executables
// check the CPU first and say what is missing.
#include <string>

namespace statim {

// The x86-64-v3 features this build needs that `has` reports missing, comma-separated; empty when
// the CPU is sufficient or the build makes no such assumption.
template <class Has>
std::string missing_cpu_features(Has has) {
    std::string missing;
#if defined(STATIM_REQUIRE_X86_V3)
    for (const char* feature : {"avx2", "fma", "f16c", "bmi2"})
        if (!has(feature)) missing += (missing.empty() ? "" : ", ") + std::string(feature);
#else
    (void)has;
#endif
    return missing;
}

inline std::string missing_cpu_features() {
#if defined(STATIM_REQUIRE_X86_V3) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return missing_cpu_features([](const char* f) {
        const std::string name = f;
        if (name == "avx2") return __builtin_cpu_supports("avx2") != 0;
        if (name == "fma") return __builtin_cpu_supports("fma") != 0;
        if (name == "f16c") return __builtin_cpu_supports("f16c") != 0;
        return __builtin_cpu_supports("bmi2") != 0;
    });
#else
    return {};
#endif
}

inline std::string cpu_requirement_message(const std::string& missing) {
    return "this build of Statim needs an x86-64 CPU with AVX2, FMA, F16C and BMI2 (x86-64-v3: Intel "
           "Haswell, AMD Excavator or newer). This CPU lacks: " + missing + ". Build from source on this "
           "machine instead: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTATIM_NATIVE=ON (docs/BUILD.md).";
}

}  // namespace statim
