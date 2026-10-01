// CPU kernels.  This file is compiled with -O3 -ffast-math; the SGEMM uses explicit
// AVX2/FMA intrinsics so its reduction order and exceptional-value behavior do not
// depend on the compiler's scalar reassociation choices.
#include "kernels.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

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

#if defined(__x86_64__) || defined(_M_X64)

namespace {

constexpr int kMR = 16;   // output columns held in two YMM registers
constexpr int kNR = 6;    // activation rows; 6x16 uses 12 accumulator registers
constexpr int kKC = 256;  // 16 KiB W micro-panel + 6 KiB X micro-panel in L1d
constexpr int kMC = 128;  // W block is 128 KiB, leaving room for X in the 512 KiB L2
constexpr int kNC = 120;

#if defined(__GNUC__) || defined(__clang__)
#define STATIM_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define STATIM_AVX2_FMA
#endif

STATIM_AVX2_FMA inline __m256i lane_mask(int n) {
    return _mm256_setr_epi32(n > 0 ? -1 : 0, n > 1 ? -1 : 0, n > 2 ? -1 : 0, n > 3 ? -1 : 0,
                             n > 4 ? -1 : 0, n > 5 ? -1 : 0, n > 6 ? -1 : 0, n > 7 ? -1 : 0);
}

STATIM_AVX2_FMA inline void transpose8x8(__m256 (&r)[8]) {
    const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]);
    const __m256 t1 = _mm256_unpackhi_ps(r[0], r[1]);
    const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]);
    const __m256 t3 = _mm256_unpackhi_ps(r[2], r[3]);
    const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]);
    const __m256 t5 = _mm256_unpackhi_ps(r[4], r[5]);
    const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]);
    const __m256 t7 = _mm256_unpackhi_ps(r[6], r[7]);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xee);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xee);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xee);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xee);
    r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// Packed A is [K,6], packed B is [K,16].  K is deliberately not padded: doing
// arithmetic on padding would turn an otherwise-unused 0*Inf into a NaN.
STATIM_AVX2_FMA inline void microkernel_6x16(float* c, int64_t ldc, const float* a,
                                             const float* b, int kb, bool first,
                                             int nr, int mr) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = c00, c11 = c00, c20 = c00, c21 = c00;
    __m256 c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    const bool full = nr == kNR && mr == kMR;
    const __m256i ml = lane_mask(std::min(mr, 8));
    const __m256i mh = lane_mask(std::max(0, mr - 8));
    if (!first) {
#define LOAD_ROW(R, LO, HI) do { \
        if ((R) < nr) { \
            (LO) = full ? _mm256_loadu_ps(c + (R) * ldc) : _mm256_maskload_ps(c + (R) * ldc, ml); \
            (HI) = full ? _mm256_loadu_ps(c + (R) * ldc + 8) : _mm256_maskload_ps(c + (R) * ldc + 8, mh); \
        } \
    } while (0)
        LOAD_ROW(0, c00, c01); LOAD_ROW(1, c10, c11); LOAD_ROW(2, c20, c21);
        LOAD_ROW(3, c30, c31); LOAD_ROW(4, c40, c41); LOAD_ROW(5, c50, c51);
#undef LOAD_ROW
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 4
#endif
    for (int k = 0; k < kb; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b + k * kMR);
        const __m256 b1 = _mm256_loadu_ps(b + k * kMR + 8);
#define FMA_ROW(R, LO, HI) do { \
        const __m256 av = _mm256_broadcast_ss(a + k * kNR + (R)); \
        (LO) = _mm256_fmadd_ps(av, b0, (LO)); \
        (HI) = _mm256_fmadd_ps(av, b1, (HI)); \
    } while (0)
        FMA_ROW(0, c00, c01); FMA_ROW(1, c10, c11); FMA_ROW(2, c20, c21);
        FMA_ROW(3, c30, c31); FMA_ROW(4, c40, c41); FMA_ROW(5, c50, c51);
#undef FMA_ROW
    }
#define STORE_ROW(R, LO, HI) do { \
        if ((R) < nr) { \
            if (full) { \
                _mm256_storeu_ps(c + (R) * ldc, (LO)); \
                _mm256_storeu_ps(c + (R) * ldc + 8, (HI)); \
            } else { \
                _mm256_maskstore_ps(c + (R) * ldc, ml, (LO)); \
                _mm256_maskstore_ps(c + (R) * ldc + 8, mh, (HI)); \
            } \
        } \
    } while (0)
    STORE_ROW(0, c00, c01); STORE_ROW(1, c10, c11); STORE_ROW(2, c20, c21);
    STORE_ROW(3, c30, c31); STORE_ROW(4, c40, c41); STORE_ROW(5, c50, c51);
#undef STORE_ROW
}

size_t apack_floats(int64_t N, int64_t K) {
    return static_cast<size_t>((K + kKC - 1) / kKC * ((N + kNR - 1) / kNR) * kKC * kNR);
}

// Packs the K block p of weight tile tj ([16 outputs, K]) into dst as [kb, 16].
STATIM_AVX2_FMA void pack_w_tile(float* dst, const float* w, int64_t M, int64_t K, int64_t p, int64_t tj) {
    const int kb = static_cast<int>(std::min<int64_t>(kKC, K - p * kKC));
    const int64_t m0 = tj * kMR;
    int k = 0;
    for (; k + 8 <= kb; k += 8) {
        __m256 lo[8], hi[8];
        for (int j = 0; j < 8; ++j) {
            lo[j] = m0 + j < M ? _mm256_loadu_ps(w + (m0 + j) * K + p * kKC + k) : _mm256_setzero_ps();
            hi[j] = m0 + 8 + j < M ? _mm256_loadu_ps(w + (m0 + 8 + j) * K + p * kKC + k) : _mm256_setzero_ps();
        }
        transpose8x8(lo);
        transpose8x8(hi);
        for (int q = 0; q < 8; ++q) {
            _mm256_storeu_ps(dst + (k + q) * kMR, lo[q]);
            _mm256_storeu_ps(dst + (k + q) * kMR + 8, hi[q]);
        }
    }
    for (; k < kb; ++k)
        for (int j = 0; j < kMR; ++j)
            dst[k * kMR + j] = m0 + j < M ? w[(m0 + j) * K + p * kKC + k] : 0.0f;
}

// Phase 1, all threads: pack the activations once, as [K block][6-row tile][kb, 6]. They are
// shared by every thread and fit in L3 for the sequence lengths Statim serves.
STATIM_AVX2_FMA void pack_avx2(float* workspace, const float* x, int64_t N, int64_t K, int ith, int nth,
                              const SgemmAbort* abort = nullptr) {
    const int64_t nt = (N + kNR - 1) / kNR;
    const int64_t kp = (K + kKC - 1) / kKC;
    for (int64_t index = ith; index < kp * nt; index += nth) {
        if (abort && abort->requested()) return;
        const int64_t p = index / nt, ti = index % nt;
        const int kb = static_cast<int>(std::min<int64_t>(kKC, K - p * kKC));
        float* dst = workspace + index * kKC * kNR;
        const int64_t n0 = ti * kNR;
        for (int k = 0; k < kb; ++k)
            for (int r = 0; r < kNR; ++r)
                dst[k * kNR + r] = n0 + r < N ? x[(n0 + r) * K + p * kKC + k] : 0.0f;
    }
}

// Phase 2: each thread owns a contiguous range of 16-output tiles. Per K block it packs one
// MC x KC weight block into a thread-local buffer that stays in L2 and streams every activation
// tile through it (GotoBLAS order), so the weights cross memory once and are never written back.
STATIM_AVX2_FMA void compute_avx2(float* y, const float* workspace, const float* w,
                                   int64_t M, int64_t N, int64_t K, int ith, int nth,
                                   const SgemmAbort* abort = nullptr) {
    const int64_t mt = (M + kMR - 1) / kMR;
    const int64_t nt = (N + kNR - 1) / kNR;
    const int64_t mbeg = mt * ith / nth;
    const int64_t mend = mt * (ith + 1) / nth;
    if (mbeg == mend) return;
    const int64_t kp = (K + kKC - 1) / kKC;
    const int64_t mc_tiles = kMC / kMR;
    thread_local std::vector<float> wbuf;
    wbuf.resize(static_cast<size_t>(mc_tiles * kKC * kMR));
    for (int64_t p = 0; p < kp; ++p) {
        const int kb = static_cast<int>(std::min<int64_t>(kKC, K - p * kKC));
        for (int64_t mt0 = mbeg; mt0 < mend; mt0 += mc_tiles) {
            if (abort && abort->requested()) return;
            const int64_t mt1 = std::min(mend, mt0 + mc_tiles);
            for (int64_t tj = mt0; tj < mt1; ++tj)
                pack_w_tile(wbuf.data() + (tj - mt0) * kKC * kMR, w, M, K, p, tj);
            for (int64_t ti = 0; ti < nt; ++ti) {
                const float* a = workspace + (p * nt + ti) * kKC * kNR;
                const int nr = static_cast<int>(std::min<int64_t>(kNR, N - ti * kNR));
                for (int64_t tj = mt0; tj < mt1; ++tj) {
                    const int mr = static_cast<int>(std::min<int64_t>(kMR, M - tj * kMR));
                    microkernel_6x16(y + ti * kNR * M + tj * kMR, M, a, wbuf.data() + (tj - mt0) * kKC * kMR,
                                     kb, p == 0, nr, mr);
                }
            }
        }
    }
}

}  // namespace

bool packed_sgemm_available() {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return true;  // x86-64-v3 release builds provide both features
#endif
}

size_t packed_sgemm_workspace_floats(int64_t, int64_t N, int64_t K) {
    return apack_floats(N, K);
}

void packed_sgemm_pack(float* workspace, const float* x, const float* w,
                       int64_t M, int64_t N, int64_t K, int ith, int nth) {
    (void)w; (void)M;
    pack_avx2(workspace, x, N, K, ith, nth);
}

void packed_sgemm_compute(float* y, const float* workspace, const float* w,
                          int64_t M, int64_t N, int64_t K, int ith, int nth) {
    compute_avx2(y, workspace, w, M, N, K, ith, nth);
}

#else

bool packed_sgemm_available() { return false; }
size_t packed_sgemm_workspace_floats(int64_t, int64_t, int64_t) { return 0; }
void packed_sgemm_pack(float*, const float*, const float*, int64_t, int64_t, int64_t, int, int) {}
void packed_sgemm_compute(float*, const float*, const float*, int64_t, int64_t, int64_t, int, int) {}

#endif

void sgemm_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const ggml_tensor* w = dst->src[0];
    const ggml_tensor* x = dst->src[1];
    auto* sync = static_cast<SgemmOpSync*>(userdata);
    const int64_t M = w->ne[1], N = ggml_nrows(x), K = w->ne[0];
    // Read the generation before arriving, so the last arriver's publish is always seen as a change.
    const uint32_t generation = sync->generation.load(std::memory_order_acquire);
    pack_avx2(sync->workspace, static_cast<const float*>(x->data), N, K, ith, nth, sync->abort);
    if (sync->arrived.fetch_add(1, std::memory_order_acq_rel) == nth - 1) {
        sync->arrived.store(0, std::memory_order_relaxed);  // ready for the next evaluation
        sync->generation.store(generation + 1, std::memory_order_release);
        sync->generation.notify_all();
    } else {
        sync->generation.wait(generation, std::memory_order_acquire);
    }
    if (sync->abort && sync->abort->requested()) return;
    compute_avx2(static_cast<float*>(dst->data), sync->workspace,
                 static_cast<const float*>(w->data), M, N, K, ith, nth, sync->abort);
}

}  // namespace statim
