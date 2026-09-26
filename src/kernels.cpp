// Fused elementwise kernels, compiled with -O3 -ffast-math so erff vectorises through glibc libmvec.
#include "kernels.h"

#include <cmath>

namespace statim {

// dst[r, i] = gelu_erf(src[r, i]) * src[r, ff + i]   (ModernBERT GeGLU: act(input) * gate)
void geglu_rows(float* dst, const float* src, long rows, long ff, long row_begin, long row_end) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    for (long r = row_begin; r < row_end && r < rows; ++r) {
        const float* in = src + r * 2 * ff;
        const float* gate = in + ff;
        float* out = dst + r * ff;
#pragma omp simd
        for (long i = 0; i < ff; ++i) {
            const float x = in[i];
            out[i] = 0.5f * x * (1.0f + erff(x * kInvSqrt2)) * gate[i];
        }
    }
}

}  // namespace statim
