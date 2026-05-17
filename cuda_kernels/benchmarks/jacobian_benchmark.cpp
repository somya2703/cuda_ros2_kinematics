// benchmarks/jacobian_benchmark.cpp
// GPU is only faster for large N (>1000)
// overhead dominates small batches.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kinematics_lib/kinematics_solver.hpp"
#include "cuda_kinematics.hpp"

using namespace kinematics;
using Clock = std::chrono::high_resolution_clock;

static std::vector<double> flat_dh() {
    std::vector<double> v;
    for (const auto& p : DEFAULT_DH_PARAMS)
        v.insert(v.end(), {p.a, p.d, p.alpha, p.theta});
    return v;
}

static void gpu_warmup(const std::vector<double>& dh) {
    auto configs = std::vector<double>(64 * NUM_JOINTS, 0.1);
    std::vector<double> jacs(64 * 36);
    cuda_batch_jacobian(dh.data(), configs.data(), jacs.data(), 64, 1e-5);
}

void run(int N, const std::vector<double>& dh) {
    KinematicsSolver solver;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);
    std::vector<double> configs(N * NUM_JOINTS);
    for (auto& x : configs) x = dist(rng);

    double cpu_ms = 1e9;
    for (int r = 0; r < 3; ++r) {
        auto t = Clock::now();
        for (int i = 0; i < N; ++i) {
            std::vector<double> q(configs.begin()+i*NUM_JOINTS,
                                   configs.begin()+(i+1)*NUM_JOINTS);
            solver.jacobian(q);
        }
        double ms = std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        if (ms < cpu_ms) cpu_ms = ms;
    }

    std::vector<double> jacs(N * 36);
    double gpu_ms = 1e9;
    for (int r = 0; r < 3; ++r) {
        auto t = Clock::now();
        cuda_batch_jacobian(dh.data(), configs.data(), jacs.data(), N, 1e-5);
        double ms = std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        if (ms < gpu_ms) gpu_ms = ms;
    }

    printf("N = %6d | CPU: %8.3f ms | GPU: %8.3f ms | Speedup: %5.1fx\n",
           N, cpu_ms, gpu_ms, cpu_ms/gpu_ms);
}

int main() {
    auto dh = flat_dh();
    printf("Warming up GPU...\n");
    gpu_warmup(dh);
    printf("Done.\n\n=== Jacobian Benchmark (CPU vs CUDA) — best of 3 ===\n");
    printf("%s\n", std::string(65, '-').c_str());
    for (int N : {1000, 5000, 10000, 50000})
        run(N, dh);
}
