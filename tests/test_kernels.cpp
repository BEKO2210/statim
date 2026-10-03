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
    for (uint32_t join : {0x31800000u, 0x3f580000u, 0x3fa00000u,
                          0x4036db6eu, 0x40800000u}) {
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

    constexpr long rows = 3, ff = 257;
    std::vector<float> src(rows * 2 * ff), got(rows * ff);
    for (long r = 0; r < rows; ++r) {
        for (long i = 0; i < ff; ++i) {
            src[r * 2 * ff + i] = -1.0f + 2.0f * static_cast<float>(r * ff + i) /
                                                    static_cast<float>(rows * ff - 1);
            src[r * 2 * ff + ff + i] = 0.25f + static_cast<float>((17 * i + r) % 31) / 16.0f;
        }
    }
    statim::geglu_rows(got.data(), src.data(), rows, ff, 0, rows);
    uint32_t geglu_worst = 0;
    for (long r = 0; r < rows; ++r) {
        for (long i = 0; i < ff; ++i) {
            const double x = src[r * 2 * ff + i];
            const double gate = src[r * 2 * ff + ff + i];
            const float ref = static_cast<float>(0.5 * x *
                              (1.0 + std::erf(x / std::sqrt(2.0))) * gate);
            const uint32_t ulp = ulp_distance(got[r * ff + i], ref);
            geglu_worst = std::max(geglu_worst, ulp);
            ok &= ulp <= 4;
        }
    }

    std::printf("erf sweep max %u ulp at x=%a (got=%a ref=%a); specials %s; GeGLU max %u ulp: %s\n",
                worst_ulp, worst_x, worst_got, worst_ref, special_ok ? "PASS" : "FAIL",
                geglu_worst, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
