// benchmarks/batch_fk_benchmark.cpp
// Measures CPU vs GPU throughput for batch forward kinematics.
// A warmup pass runs first to absorb the one-time ~200ms CUDA driver init
// cost, so timed results reflect steady-state kernel throughput.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "kinematics_lib/kinematics_solver.hpp"
#include "cuda_kinematics.hpp"

using namespace kinematics;
using Clock = std::chrono::high_resolution_clock;

static std::vector<double> flat_dh_table() {
    std::vector<double> flat;
    flat.reserve(NUM_JOINTS * 4);
    for (const auto& p : DEFAULT_DH_PARAMS) {
        flat.push_back(p.a); flat.push_back(p.d);
        flat.push_back(p.alpha); flat.push_back(p.theta);
    }
    return flat;
}

static std::vector<double> random_configs(int N, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);
    std::vector<double> v(N * NUM_JOINTS);
    for (auto& x : v) x = dist(rng);
    return v;
}

static void gpu_warmup(const std::vector<double>& dh_flat) {
    auto configs = random_configs(64);
    std::vector<double> out(64 * 16);
    cuda_batch_fk(dh_flat.data(), configs.data(), out.data(), 64);
}

void run_benchmark(int N, const std::vector<double>& dh_flat) {
    KinematicsSolver solver;
    auto configs = random_configs(N);

    // Best of 3 runs — eliminates OS scheduling noise
    double cpu_ms = 1e9;
    std::vector<Transform> cpu_results;
    for (int r = 0; r < 3; ++r) {
        auto t0 = Clock::now();
        cpu_results = solver.batch_forward_kinematics(configs);
        double ms = std::chrono::duration<double, std::milli>(Clock::now()-t0).count();
        if (ms < cpu_ms) cpu_ms = ms;
    }

    std::vector<double> gpu_transforms(N * 16);
    double gpu_ms = 1e9;
    for (int r = 0; r < 3; ++r) {
        auto t = Clock::now();
        cuda_batch_fk(dh_flat.data(), configs.data(), gpu_transforms.data(), N);
        double ms = std::chrono::duration<double, std::milli>(Clock::now()-t).count();
        if (ms < gpu_ms) gpu_ms = ms;
    }

    bool ok = true;
    for (int r = 0; r < 4 && ok; ++r)
        for (int c = 0; c < 4 && ok; ++c)
            if (std::abs(cpu_results[0](r,c) - gpu_transforms[r*4+c]) > 1e-6) ok = false;

    printf("N = %7d | CPU: %7.3f ms | GPU: %7.3f ms | Speedup: %5.1fx | %s\n",
           N, cpu_ms, gpu_ms, cpu_ms/gpu_ms, ok ? "OK" : "MISMATCH");
}

int main() {
    auto dh_flat = flat_dh_table();
    printf("Warming up GPU...\n");
    gpu_warmup(dh_flat);
    printf("Done.\n\n=== Batch FK Benchmark (CPU vs CUDA) — best of 3 ===\n");
    printf("%s\n", std::string(70, '-').c_str());
    for (int N : {1000, 5000, 10000, 50000, 100000})
        run_benchmark(N, dh_flat);
    return 0;
}
