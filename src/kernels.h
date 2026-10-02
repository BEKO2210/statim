#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>

struct ggml_tensor;

namespace statim {
struct SgemmAbort {
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    const std::atomic<bool>* cancelled = nullptr;
    bool requested() const {
        return (cancelled && cancelled->load(std::memory_order_relaxed)) ||
               (deadline != std::chrono::steady_clock::time_point::max() &&
                std::chrono::steady_clock::now() >= deadline);
    }
};
void geglu_rows(float* dst, const float* src, long rows, long ff, long row_begin, long row_end);

// Y[N, M] = X[N, K] * W[M, K]^T.  ggml stores the corresponding tensors as
// [M, N], [K, N], and [K, M], so dimension zero is the contiguous dimension.
// Each caller handles one of nth disjoint groups of 16-column output tiles.
bool packed_sgemm_available();
bool packed_sgemm_enabled();
std::vector<std::string> matrix_cpu_features();
size_t packed_sgemm_workspace_floats(int64_t M, int64_t N, int64_t K);
void packed_sgemm_pack(float* workspace, const float* x, const float* w,
                       int64_t M, int64_t N, int64_t K, int ith, int nth);
// ggml custom-op body for dst = W·x with src = {W [K, M], x [K, N...]}: every thread
// packs its share of x, waits at the barrier, then computes its output tiles. userdata is a
// SgemmOpSync owned by the graph; it may be reused for any number of evaluations.
struct SgemmOpSync {
    std::atomic<int> arrived{0};
    std::atomic<uint32_t> generation{0};
    const SgemmAbort* abort = nullptr;
    // Packed-activation workspace. The graph runs one node at a time, so every projection of a
    // graph shares one buffer; a graph tensor would stay allocated for the whole graph instead.
    float* workspace = nullptr;
    size_t workspace_floats = 0;  // needed by this node; the shared buffer is at least this large
};
void sgemm_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata);

void packed_sgemm_compute(float* y, const float* workspace, const float* w,
                          int64_t M, int64_t N, int64_t K, int ith, int nth);
}
