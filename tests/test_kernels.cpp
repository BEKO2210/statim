#include "kernels.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

uint32_t ulp_distance(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<uint32_t>::max();
    const uint32_t ua = std::bit_cast<uint32_t>(a);
    const uint32_t ub = std::bit_cast<uint32_t>(b);
    const uint32_t oa = (ua & 0x80000000u) ? ~ua : ua | 0x80000000u;
    const uint32_t ob = (ub & 0x80000000u) ? ~ub : ub | 0x80000000u;
    return oa > ob ? oa - ob : ob - oa;
}

}  // namespace

int main() {
#if defined(__x86_64__) || defined(_M_X64)
    // geglu_rows is built with -mavx2 -mfma in portable builds; this file is not, so the
    // check runs before any AVX2 instruction (the sgemm test does the same).
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        std::printf("SKIP: CPU lacks AVX2/FMA\n");
        return 77;
    }
#endif
    bool ok = true;
    uint32_t worst_ulp = 0;
    float worst_x = 0.0f, worst_got = 0.0f, worst_ref = 0.0f;
    auto check_erf = [&](float x) {
        const float got = statim::erf_approx(x);
        const float ref = static_cast<float>(std::erf(static_cast<double>(x)));
        const uint32_t ulp = ulp_distance(got, ref);
        if (ulp > worst_ulp) {
            worst_ulp = ulp;
            worst_x = x;
            worst_got = got;
            worst_ref = ref;
        }
        ok &= ulp <= 2;
    };

    // One value from every 64 consecutive float encodings covers all exponents,
    // signs, subnormals, and mantissa regions in a bounded test runtime.
    for (uint64_t bits = 0; bits <= 0xffffffffull; bits += 64) {
        const uint32_t word = static_cast<uint32_t>(bits);
        if ((word & 0x7fffffffu) < 0x7f800000u)
            check_erf(std::bit_cast<float>(word));
    }

    // Densely cover the approximation joins, where adjacent formulae can expose
    // otherwise hard-to-hit rounding maxima.
    for (uint32_t join : {0x31800000u, 0x3f580000u, 0x3f800000u, 0x3fa00000u,
                          0x4036db6eu, 0x407ae148u, 0x40800000u}) {
        for (int delta = -4096; delta <= 4096; ++delta) {
            const uint32_t mag = static_cast<uint32_t>(static_cast<int64_t>(join) + delta);
            check_erf(std::bit_cast<float>(mag));
            check_erf(std::bit_cast<float>(mag | 0x80000000u));
        }
    }

    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    bool special_ok = true;
    special_ok &= std::bit_cast<uint32_t>(statim::erf_approx(0.0f)) == 0x00000000u;
    special_ok &= std::bit_cast<uint32_t>(statim::erf_approx(-0.0f)) == 0x80000000u;
    special_ok &= statim::erf_approx(inf) == 1.0f;
    special_ok &= statim::erf_approx(-inf) == -1.0f;
    const uint32_t nan_result = std::bit_cast<uint32_t>(statim::erf_approx(nan));
    special_ok &= (nan_result & 0x7f800000u) == 0x7f800000u && (nan_result & 0x007fffffu) != 0;
    special_ok &= statim::erf_approx(4.0f) == 1.0f && statim::erf_approx(-4.0f) == -1.0f;
    ok &= special_ok;
    if (!special_ok) {
        std::printf("special bits: +0=%08x -0=%08x +inf=%08x -inf=%08x nan=%08x +4=%08x -4=%08x\n",
                    std::bit_cast<uint32_t>(statim::erf_approx(0.0f)),
                    std::bit_cast<uint32_t>(statim::erf_approx(-0.0f)),
                    std::bit_cast<uint32_t>(statim::erf_approx(inf)),
                    std::bit_cast<uint32_t>(statim::erf_approx(-inf)), nan_result,
                    std::bit_cast<uint32_t>(statim::erf_approx(4.0f)),
                    std::bit_cast<uint32_t>(statim::erf_approx(-4.0f)));
    }

    // The shipping GeGLU path: production widths are multiples of 8 (1152, 2624), so test the
    // vectorized loop itself, through the 1.0 and 3.92 polynomial joins and the saturation knee
    // where 1 + erf cancels, with gate = 1 and gate = x. The bound is on the absolute error of
    // 1 + erf (2^-22, two ulp of 1) scaled by |0.5 * x * gate|, plus 4 ulp of the result: a
    // few ulp of erf near -1 is the most any float erf (also glibc's erff) can promise there.
    double geglu_worst_excess = 0.0;
    float geglu_worst_x = 0.0f;
    for (long ff : {8L, 1152L}) {
        constexpr long rows = 2;
        std::vector<float> src(rows * 2 * ff), got(rows * ff);
        for (long start = 0; start < 2 * 7000000; start += rows * ff) {
            for (long r = 0; r < rows; ++r) {
                for (long i = 0; i < ff; ++i) {
                    // x/sqrt(2) spans [-5.6, 5.6] in steps of about 8e-7 (all knots and the knee)
                    const long k = start + r * ff + i;
                    const float x = -8.0f + 16.0f * static_cast<float>(k % 7000000) / 7000000.0f;
                    src[r * 2 * ff + i] = x;
                    src[r * 2 * ff + ff + i] = k >= 7000000 ? x : 1.0f;
                }
            }
            statim::geglu_rows(got.data(), src.data(), rows, ff, 0, rows);
            for (long r = 0; r < rows; ++r) {
                for (long i = 0; i < ff; ++i) {
                    const double x = src[r * 2 * ff + i];
                    const double gate = src[r * 2 * ff + ff + i];
                    const double ref = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))) * gate;
                    const float out = got[r * ff + i];
                    const double err = std::fabs(static_cast<double>(out) - ref);
                    const double bound = std::ldexp(1.0, -22) * std::fabs(0.5 * x * gate) +
                                         4.0 * std::fabs(ref) * std::ldexp(1.0, -23);
                    if (err > bound && err / bound > geglu_worst_excess) {
                        geglu_worst_excess = err / bound;
                        geglu_worst_x = static_cast<float>(x);
                    }
                    ok &= err <= bound;
                }
            }
        }
    }
    // The input that flipped the sign under the rational approximation (Grok review of #121).
    {
        const float x = -0x1.50205p+2f;
        float in[16], out[8];
        for (int i = 0; i < 8; ++i) { in[i] = x; in[8 + i] = x; }
        statim::geglu_rows(out, in, 1, 8, 0, 1);
        const double ref = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))) * x;
        ok &= (out[0] > 0) == (ref > 0);
    }
    if (geglu_worst_excess > 0)
        std::printf("GeGLU bound exceeded %.3gx at x=%a\n", geglu_worst_excess, geglu_worst_x);

    std::printf("erf sweep max %u ulp at x=%a (got=%a ref=%a); specials %s; GeGLU %s; overall %s\n",
                worst_ulp, worst_x, worst_got, worst_ref, special_ok ? "PASS" : "FAIL",
                geglu_worst_excess > 0 ? "FAIL" : "PASS", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
