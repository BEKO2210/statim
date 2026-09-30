#pragma once

#include <cstddef>
#include <cstdint>

namespace statim {
void geglu_rows(float* dst, const float* src, long rows, long ff, long row_begin, long row_end);

// Y[N, M] = X[N, K] * W[M, K]^T.  ggml stores the corresponding tensors as
// [M, N], [K, N], and [K, M], so dimension zero is the contiguous dimension.
// Each caller handles one of nth disjoint groups of 16-column output tiles.
bool packed_sgemm_available();
size_t packed_sgemm_workspace_floats(int64_t M, int64_t N, int64_t K);
void packed_sgemm_pack(float* workspace, const float* x, const float* w,
                       int64_t M, int64_t N, int64_t K, int ith, int nth);
void packed_sgemm_compute(float* y, const float* workspace,
                          int64_t M, int64_t N, int64_t K, int ith, int nth);
}
