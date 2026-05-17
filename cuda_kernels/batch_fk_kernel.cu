// batch_fk_kernel.cu
// GPU-parallel forward kinematics for N joint configurations.
// Each CUDA thread computes FK for one configuration.
//
// Design: all internal arithmetic uses float (FP32) — consumer Ada/Ampere GPUs
// have 1/32 FP64 throughput vs FP32. The public API stays double for
// compatibility with the C++ library; conversion happens only at I/O.

#include "cuda_kinematics.hpp"
#include <cuda_runtime.h>
#include <cmath>

// Device helpers (float)


__device__ static void mat4_mul_f(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += A[r*4+k] * B[k*4+c];
            C[r*4+c] = s;
        }
}

__device__ static void mat4_identity_f(float* M) {
    for (int i = 0; i < 16; ++i) M[i] = 0.f;
    M[0] = M[5] = M[10] = M[15] = 1.f;
}

__device__ static void dh_transform_f(float a, float d, float alpha, float theta,
                                       float* out) {
    float ct = cosf(theta), st = sinf(theta);
    float ca = cosf(alpha), sa = sinf(alpha);
    out[0]  =  ct;   out[1]  = -st*ca;  out[2]  =  st*sa;  out[3]  = a*ct;
    out[4]  =  st;   out[5]  =  ct*ca;  out[6]  = -ct*sa;  out[7]  = a*st;
    out[8]  = 0.f;   out[9]  =  sa;     out[10] =  ca;      out[11] = d;
    out[12] = 0.f;   out[13] = 0.f;     out[14] = 0.f;      out[15] = 1.f;
}


// Kernel


__global__ void batch_fk_kernel(const float* __restrict__ dh_params,
                                 const float* __restrict__ configs,
                                 float*       __restrict__ transforms,
                                 int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* q = configs + idx * CUDA_NUM_JOINTS;
    float*        T = transforms + idx * 16;

    float acc[16], tmp[16], joint_T[16];
    mat4_identity_f(acc);

    for (int j = 0; j < CUDA_NUM_JOINTS; ++j) {
        float a     = dh_params[j * CUDA_DH_STRIDE + 0];
        float d     = dh_params[j * CUDA_DH_STRIDE + 1];
        float alpha = dh_params[j * CUDA_DH_STRIDE + 2];
        float theta = dh_params[j * CUDA_DH_STRIDE + 3] + q[j];
        dh_transform_f(a, d, alpha, theta, joint_T);
        mat4_mul_f(acc, joint_T, tmp);
        for (int i = 0; i < 16; ++i) acc[i] = tmp[i];
    }
    for (int i = 0; i < 16; ++i) T[i] = acc[i];
}


// Host launcher — double API, float internally


extern "C" void cuda_batch_fk(const double* dh_params,
                               const double* configs,
                               double*       transforms,
                               int           N) {
    // Convert inputs to float
    const int n_dh  = CUDA_NUM_JOINTS * CUDA_DH_STRIDE;
    const int n_cfg = N * CUDA_NUM_JOINTS;
    const int n_out = N * 16;

    float *h_dh_f  = new float[n_dh];
    float *h_cfg_f = new float[n_cfg];
    float *h_out_f = new float[n_out];
    for (int i = 0; i < n_dh;  ++i) h_dh_f[i]  = (float)dh_params[i];
    for (int i = 0; i < n_cfg; ++i) h_cfg_f[i]  = (float)configs[i];

    float *d_dh=nullptr, *d_cfg=nullptr, *d_out=nullptr;
    cudaMalloc(&d_dh,  n_dh  * sizeof(float));
    cudaMalloc(&d_cfg, n_cfg * sizeof(float));
    cudaMalloc(&d_out, n_out * sizeof(float));
    cudaMemcpy(d_dh,  h_dh_f,  n_dh  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cfg, h_cfg_f, n_cfg * sizeof(float), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    batch_fk_kernel<<<blocks, threads>>>(d_dh, d_cfg, d_out, N);

    cudaMemcpy(h_out_f, d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_out; ++i) transforms[i] = (double)h_out_f[i];

    delete[] h_dh_f; delete[] h_cfg_f; delete[] h_out_f;
    cudaFree(d_dh); cudaFree(d_cfg); cudaFree(d_out);
}
