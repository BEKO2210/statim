#include "kernels.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using Shape = std::tuple<int, int, int>;

std::vector<double> reference(const std::vector<float>& x, const std::vector<float>& w,
                              int M, int N, int K) {
    std::vector<double> y(static_cast<size_t>(M) * N);
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) sum += static_cast<double>(x[n*K+k]) * w[m*K+k];
            y[n*M+m] = sum;
        }
    return y;
}

std::vector<float> run(const std::vector<float>& x, const std::vector<float>& w,
                       int M, int N, int K, int nth) {
    std::vector<float> y(static_cast<size_t>(M) * N, -123.0f);
    std::vector<float> workspace(statim::packed_sgemm_workspace_floats(M, N, K));
    std::vector<std::thread> workers;
    for (int ith = 0; ith < nth; ++ith)
        workers.emplace_back([&, ith] { statim::packed_sgemm_pack(workspace.data(), x.data(), w.data(), M, N, K, ith, nth); });
    for (auto& worker : workers) worker.join();
    workers.clear();
    for (int ith = 0; ith < nth; ++ith)
        workers.emplace_back([&, ith] { statim::packed_sgemm_compute(y.data(), workspace.data(), w.data(), M, N, K, ith, nth); });
    for (auto& worker : workers) worker.join();
    return y;
}

}  // namespace

int main() {
    if (!statim::packed_sgemm_available()) {
        std::puts("SKIP: AVX2/FMA unavailable");
        return 77;
    }
    const std::vector<Shape> shapes = {
        {1, 1, 1}, {7, 5, 17}, {17, 1, 7}, {64, 128, 64},
        {767, 5, 767}, {768, 1, 768}, {2304, 1, 768}, {17, 1024, 2304},
    };
    const int thread_counts[] = {1, 3, 4, 8};
    std::mt19937 rng(0x5a17u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    double worst_abs = 0.0, worst_rel = 0.0;
    bool ok = true;
    for (const auto& [M, N, K] : shapes) {
        std::vector<float> x(static_cast<size_t>(N) * K), w(static_cast<size_t>(M) * K);
        std::generate(x.begin(), x.end(), [&] { return dist(rng); });
        std::generate(w.begin(), w.end(), [&] { return dist(rng); });
        const auto ref = reference(x, w, M, N, K);
        for (int nth : thread_counts) {
            const auto got = run(x, w, M, N, K, nth);
            for (size_t i = 0; i < got.size(); ++i) {
                const double ae = std::fabs(static_cast<double>(got[i]) - ref[i]);
                const double re = ae / std::max(1.0, std::fabs(ref[i]));
                worst_abs = std::max(worst_abs, ae);
                worst_rel = std::max(worst_rel, re);
                ok &= ae <= 3e-4 * std::max(1.0, std::fabs(ref[i]));
            }
        }
    }

    // Exceptional values must reach every dot product that consumes them.  No
    // padded K arithmetic is allowed to manufacture exceptions in finite cells.
    constexpr int M = 4, N = 4, K = 4;
    std::vector<float> x(static_cast<size_t>(N) * K, 1.0f), w(static_cast<size_t>(M) * K, 1.0f);
    x[0] = std::numeric_limits<float>::quiet_NaN();
    x[K + 1] = std::numeric_limits<float>::infinity();
    w[2*K + 2] = std::numeric_limits<float>::quiet_NaN();
    w[3*K + 3] = -std::numeric_limits<float>::infinity();
    const auto got = run(x, w, M, N, K, 3);
    for (int m = 0; m < M; ++m) ok &= std::isnan(got[m]);
    for (int m = 0; m < M; ++m) ok &= std::isinf(got[M + m]) || std::isnan(got[M + m]);
    for (int n = 0; n < N; ++n) ok &= std::isnan(got[n*M + 2]);
    for (int n = 2; n < N; ++n) ok &= std::isinf(got[n*M + 3]) && std::signbit(got[n*M + 3]);
    ok &= std::isfinite(got[2*M]) && std::isfinite(got[3*M]);

    // The op as ggml runs it: one custom node, every pool thread inside the barrier, and the same
    // node and SgemmOpSync evaluated twice (the barrier must reset itself).
    for (const auto& [GM, GN, GK] : std::vector<Shape>{{17, 1, 7}, {45, 25, 300}, {767, 13, 769}}) {
        std::vector<float> gx(static_cast<size_t>(GN) * GK), gw(static_cast<size_t>(GM) * GK);
        std::generate(gx.begin(), gx.end(), [&] { return dist(rng); });
        std::generate(gw.begin(), gw.end(), [&] { return dist(rng); });
        const auto ref = reference(gx, gw, GM, GN, GK);
        for (int nth : {1, 3, 8}) {
            ggml_init_params params{size_t(64) << 20, nullptr, false};
            ggml_context* ctx = ggml_init(params);
            ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GK, GM);
            ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GK, GN);
            std::vector<float> workspace(statim::packed_sgemm_workspace_floats(GM, GN, GK));
            std::copy(gw.begin(), gw.end(), static_cast<float*>(w->data));
            std::copy(gx.begin(), gx.end(), static_cast<float*>(x->data));
            statim::SgemmOpSync sync;
            sync.workspace = workspace.data();
            sync.workspace_floats = workspace.size();
            ggml_tensor* args[] = {w, x};
            ggml_tensor* y = ggml_custom_4d(ctx, GGML_TYPE_F32, GM, GN, 1, 1, args, 2,
                                            statim::sgemm_custom_op, GGML_N_TASKS_MAX, &sync);
            ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            for (int pass = 0; pass < 2; ++pass) {
                std::fill_n(static_cast<float*>(y->data), GM * GN, -123.0f);
                ok &= ggml_graph_compute_with_ctx(ctx, gf, nth) == GGML_STATUS_SUCCESS;
                const float* got = static_cast<const float*>(y->data);
                for (size_t i = 0; i < ref.size(); ++i)
                    ok &= std::fabs(got[i] - ref[i]) <= 3e-4 * std::max(1.0, std::fabs(ref[i]));
            }
            ggml_free(ctx);
        }
    }

    // An already-expired large op must take only the packing barrier, not the matrix product.
    {
        constexpr int CM = 2048, CN = 256, CK = 2048, CTHREADS = 8;
        ggml_init_params params{size_t(96) << 20, nullptr, false};
        ggml_context* ctx = ggml_init(params);
        ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CK, CM);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CK, CN);
        std::vector<float> workspace(statim::packed_sgemm_workspace_floats(CM, CN, CK));
        statim::SgemmAbort abort{std::chrono::steady_clock::now() - std::chrono::seconds(1), nullptr};
        statim::SgemmOpSync sync;
        sync.workspace = workspace.data();
        sync.abort = &abort;
        ggml_tensor* args[] = {w, x};
        ggml_tensor* y = ggml_custom_4d(ctx, GGML_TYPE_F32, CM, CN, 1, 1, args, 2,
                                        statim::sgemm_custom_op, GGML_N_TASKS_MAX, &sync);
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for (int ith = 0; ith < CTHREADS; ++ith)
            threads.emplace_back([&, ith] { statim::sgemm_custom_op(y, ith, CTHREADS, &sync); });
        for (auto& thread : threads) thread.join();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        ok &= elapsed < std::chrono::seconds(1);
        ok &= sync.generation.load() == 1;
        ggml_free(ctx);
    }

    // Hold one participant back so the others are known to be at the generation barrier,
    // cancel, then let the final participant release them. No thread may remain waiting.
    {
        constexpr int CM = 128, CN = 32, CK = 256, CTHREADS = 8;
        ggml_init_params params{size_t(8) << 20, nullptr, false};
        ggml_context* ctx = ggml_init(params);
        ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CK, CM);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CK, CN);
        std::vector<float> workspace(statim::packed_sgemm_workspace_floats(CM, CN, CK));
        std::atomic<bool> cancelled{false};
        statim::SgemmAbort abort{std::chrono::steady_clock::time_point::max(), &cancelled};
        statim::SgemmOpSync sync;
        sync.workspace = workspace.data();
        sync.abort = &abort;
        ggml_tensor* args[] = {w, x};
        ggml_tensor* y = ggml_custom_4d(ctx, GGML_TYPE_F32, CM, CN, 1, 1, args, 2,
                                        statim::sgemm_custom_op, GGML_N_TASKS_MAX, &sync);
        std::vector<std::thread> threads;
        for (int ith = 0; ith < CTHREADS - 1; ++ith)
            threads.emplace_back([&, ith] { statim::sgemm_custom_op(y, ith, CTHREADS, &sync); });
        const auto wait_until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (sync.arrived.load(std::memory_order_acquire) != CTHREADS - 1 &&
               std::chrono::steady_clock::now() < wait_until) std::this_thread::yield();
        ok &= sync.arrived.load(std::memory_order_acquire) == CTHREADS - 1;
        cancelled.store(true, std::memory_order_relaxed);
        threads.emplace_back([&] { statim::sgemm_custom_op(y, CTHREADS - 1, CTHREADS, &sync); });
        for (auto& thread : threads) thread.join();
        ok &= sync.generation.load() == 1;
        ggml_free(ctx);
    }

    std::printf("packed SGEMM random max |error| %.3e, max relative %.3e; NaN/Inf and ggml graph (x2) %s\n",
                worst_abs, worst_rel, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
