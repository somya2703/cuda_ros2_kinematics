// benchmarks/ik_benchmark.cpp
// IK is compute-heavy per problem so GPU wins at much smaller M than FK.

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
    std::vector<double> tgt(16); tgt[0]=tgt[5]=tgt[10]=tgt[15]=1.0; tgt[3]=0.5;
    std::vector<double> res(NUM_JOINTS); std::vector<int> conv(1);
    cuda_batch_ik(dh.data(), tgt.data(), nullptr, res.data(), conv.data(),
                  1, 0.5, 1e-3, 200);
}

void run(int M, const std::vector<double>& dh) {
    KinematicsSolver solver;
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Generate M reachable targets via FK of random configs
    std::vector<double> seed_configs(M * NUM_JOINTS);
    for (auto& x : seed_configs) x = dist(rng);

    std::vector<double> targets_flat(M * 16);
    for (int i = 0; i < M; ++i) {
        std::vector<double> q(seed_configs.begin()+i*NUM_JOINTS,
                               seed_configs.begin()+(i+1)*NUM_JOINTS);
        Transform T = solver.forward_kinematics(q);
        for (int r=0;r<4;++r) for (int c=0;c<4;++c)
            targets_flat[i*16+r*4+c] = T(r,c);
    }

    // CPU
    double cpu_ms = 1e9; int cpu_conv = 0;
    for (int rep = 0; rep < 3; ++rep) {
        int conv = 0;
        auto t = Clock::now();
        for (int i = 0; i < M; ++i) {
            Transform tgt; for (int r=0;r<4;++r) for (int c=0;c<4;++c)
                tgt(r,c)=targets_flat[i*16+r*4+c];
            std::vector<double> q_out;
            if (solver.inverse_kinematics(tgt, q_out)) ++conv;
        }
        double ms = std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        if (ms < cpu_ms) { cpu_ms = ms; cpu_conv = conv; }
    }

    // GPU
    std::vector<double> gpu_res(M * NUM_JOINTS);
    std::vector<int>    gpu_conv_vec(M, 0);
    double gpu_ms = 1e9; int gpu_conv = 0;
    for (int rep = 0; rep < 3; ++rep) {
        auto t = Clock::now();
        cuda_batch_ik(dh.data(), targets_flat.data(), nullptr,
                      gpu_res.data(), gpu_conv_vec.data(),
                      M, 0.5, 1e-3, 200);
        double ms = std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        if (ms < gpu_ms) {
            gpu_ms = ms;
            gpu_conv = 0; for (int x : gpu_conv_vec) gpu_conv += x;
        }
    }

    printf("M = %5d | CPU: %8.3f ms (conv %d/%d) | GPU: %8.3f ms (conv %d/%d) | Speedup: %5.1fx\n",
           M, cpu_ms, cpu_conv, M, gpu_ms, gpu_conv, M, cpu_ms/gpu_ms);
}

int main() {
    auto dh = flat_dh();
    printf("Warming up GPU...\n");
    gpu_warmup(dh);
    printf("Done.\n\n=== IK Benchmark (CPU vs CUDA) — best of 3 ===\n");
    printf("%s\n", std::string(72, '-').c_str());
    for (int M : {50, 100, 500, 1000})
        run(M, dh);
}
