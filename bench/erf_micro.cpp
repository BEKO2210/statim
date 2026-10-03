#include "kernels.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <vector>

namespace {

constexpr long kRows = 8 * 144;
constexpr long kFf = 1152;
constexpr int kRuns = 31;

using Kernel = void (*)(float*, const float*, long, long);

__attribute__((noinline)) void geglu_libm(float* dst, const float* src, long rows, long ff) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    for (long r = 0; r < rows; ++r) {
        const float* in = src + r * 2 * ff;
        const float* gate = in + ff;
        float* out = dst + r * ff;
#pragma omp simd
        for (long i = 0; i < ff; ++i) {
            const float x = in[i];
            out[i] = 0.5f * x * (1.0f + std::erf(x * kInvSqrt2)) * gate[i];
        }
    }
}

__attribute__((noinline)) void geglu_approx(float* dst, const float* src, long rows, long ff) {
    statim::geglu_rows(dst, src, rows, ff, 0, rows);
}

void pin_to_current_cpu() {
    const int cpu = sched_getcpu();
    if (cpu < 0) {
        std::fprintf(stderr, "sched_getcpu: %s\n", std::strerror(errno));
        std::exit(2);
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::fprintf(stderr, "sched_setaffinity: %s\n", std::strerror(errno));
        std::exit(2);
    }
    std::printf("pinned_cpu: %d\n", cpu);
}

double median_ns_per_element(Kernel kernel, float* dst, const float* src) {
    for (int i = 0; i < 5; ++i) kernel(dst, src, kRows, kFf);
    std::array<double, kRuns> samples{};
    for (double& sample : samples) {
        const auto begin = std::chrono::steady_clock::now();
        kernel(dst, src, kRows, kFf);
        const auto end = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::nano>(end - begin).count() /
                 static_cast<double>(kRows * kFf);
    }
    std::sort(samples.begin(), samples.end());
    return samples[kRuns / 2];
}

}  // namespace

int main() {
    pin_to_current_cpu();
    std::vector<float> src(static_cast<size_t>(kRows) * 2 * kFf);
    std::vector<float> libm(static_cast<size_t>(kRows) * kFf);
    std::vector<float> approx(static_cast<size_t>(kRows) * kFf);
    std::mt19937 rng(0x5a17u);
    std::normal_distribution<float> activation(0.0f, 1.25f);
    std::normal_distribution<float> gate(0.0f, 1.0f);
    for (long r = 0; r < kRows; ++r) {
        for (long i = 0; i < kFf; ++i) {
            src[static_cast<size_t>(r) * 2 * kFf + i] = activation(rng);
            src[static_cast<size_t>(r) * 2 * kFf + kFf + i] = gate(rng);
        }
    }

    const double libm_ns = median_ns_per_element(geglu_libm, libm.data(), src.data());
    const double approx_ns = median_ns_per_element(geglu_approx, approx.data(), src.data());

    double checksum = 0.0;
    for (size_t i = 0; i < approx.size(); i += 257) checksum += approx[i] + libm[i];
    std::printf("shape: rows=%ld ff=%ld, median of %d runs, 1 thread\n", kRows, kFf, kRuns);
    std::printf("implementation       ns/element   relative\n");
    std::printf("libm erff             %9.4f      1.000x\n", libm_ns);
    std::printf("statim erf_approx     %9.4f      %.3fx\n", approx_ns, approx_ns / libm_ns);
    std::printf("checksum: %.9g\n", checksum);
    return approx_ns <= libm_ns ? 0 : 1;
}
