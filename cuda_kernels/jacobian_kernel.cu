// jacobian_kernel.cu
// GPU-parallel geometric Jacobian using float (FP32) for throughput.
// Consumer Ada/Ampere GPUs have 1/32 FP64 : FP32 ratio.
//
// Uses the analytic geometric Jacobian (z-axis cross products) — same
// formulation as the CPU solver — rather than finite differences.
// This is both faster and more accurate.

#include "cuda_kinematics.hpp"
#include <cuda_runtime.h>
#include <cmath>


// Device helpers


__device__ static void _dh_f(float a, float d, float al, float th, float* out) {
    float ct=cosf(th), st=sinf(th), ca=cosf(al), sa=sinf(al);
    out[0]=ct; out[1]=-st*ca; out[2]= st*sa; out[3]=a*ct;
    out[4]=st; out[5]= ct*ca; out[6]=-ct*sa; out[7]=a*st;
    out[8]=0;  out[9]=sa;     out[10]=ca;     out[11]=d;
    out[12]=0; out[13]=0;     out[14]=0;      out[15]=1;
}

__device__ static void _mat4_mul_f(const float* A, const float* B, float* C) {
    for (int r=0;r<4;++r) for (int c=0;c<4;++c) {
        float s=0; for (int k=0;k<4;++k) s+=A[r*4+k]*B[k*4+c];
        C[r*4+c]=s;
    }
}

// Partial FK up to (and including) joint `upto` (0-indexed), result in T[16]
__device__ static void _partial_fk_f(const float* dh, const float* q,
                                      int upto, float* T) {
    float acc[16]={}, tmp[16], Tj[16];
    acc[0]=acc[5]=acc[10]=acc[15]=1.f;
    for (int j=0; j<=upto; ++j) {
        _dh_f(dh[j*4], dh[j*4+1], dh[j*4+2], dh[j*4+3]+q[j], Tj);
        _mat4_mul_f(acc, Tj, tmp);
        for (int i=0;i<16;++i) acc[i]=tmp[i];
    }
    for (int i=0;i<16;++i) T[i]=acc[i];
}


// Kernel: one block per config, one thread per joint
// Each thread writes one column of the 6×6 Jacobian


__global__ void jacobian_kernel(const float* __restrict__ dh_params,
                                 const float* __restrict__ configs,
                                 float*       __restrict__ jacobians,
                                 int N) {
    const int config_idx = blockIdx.x;
    const int joint_idx  = threadIdx.x;
    if (config_idx >= N || joint_idx >= CUDA_NUM_JOINTS) return;

    const float* q = configs + config_idx * CUDA_NUM_JOINTS;
    float*        J = jacobians + config_idx * 36;

    // Full FK to get end-effector position (shared across threads via __shared__)
    __shared__ float p_ee[3];
    if (joint_idx == 0) {
        float T_ee[16];
        _partial_fk_f(dh_params, q, CUDA_NUM_JOINTS-1, T_ee);
        p_ee[0]=T_ee[3]; p_ee[1]=T_ee[7]; p_ee[2]=T_ee[11];
    }
    __syncthreads();

    // Partial FK up to (but not including) this joint to get z_{i-1} and p_{i-1}
    float z[3], p_i[3];
    if (joint_idx == 0) {
        // Joint 0: z and p are world frame (identity)
        z[0]=0.f; z[1]=0.f; z[2]=1.f;
        p_i[0]=0.f; p_i[1]=0.f; p_i[2]=0.f;
    } else {
        float T_pre[16];
        _partial_fk_f(dh_params, q, joint_idx-1, T_pre);
        z[0]=T_pre[2]; z[1]=T_pre[6]; z[2]=T_pre[10];  // col 2 = z-axis
        p_i[0]=T_pre[3]; p_i[1]=T_pre[7]; p_i[2]=T_pre[11];
    }

    // Geometric Jacobian column:
    //   linear:  z × (p_ee - p_i)
    //   angular: z
    float dp[3] = { p_ee[0]-p_i[0], p_ee[1]-p_i[1], p_ee[2]-p_i[2] };

    J[0*6 + joint_idx] = z[1]*dp[2] - z[2]*dp[1];  // cross product
    J[1*6 + joint_idx] = z[2]*dp[0] - z[0]*dp[2];
    J[2*6 + joint_idx] = z[0]*dp[1] - z[1]*dp[0];
    J[3*6 + joint_idx] = z[0];
    J[4*6 + joint_idx] = z[1];
    J[5*6 + joint_idx] = z[2];
}


// Host launcher


extern "C" void cuda_batch_jacobian(const double* dh_params,
                                    const double* configs,
                                    double*       jacobians,
                                    int           N,
                                    double        /*eps — unused, analytic Jacobian*/) {
    const int n_dh  = CUDA_NUM_JOINTS * 4;
    const int n_cfg = N * CUDA_NUM_JOINTS;
    const int n_jac = N * 36;

    float *h_dh_f  = new float[n_dh];
    float *h_cfg_f = new float[n_cfg];
    float *h_jac_f = new float[n_jac];
    for (int i=0;i<n_dh; ++i) h_dh_f[i]  = (float)dh_params[i];
    for (int i=0;i<n_cfg;++i) h_cfg_f[i]  = (float)configs[i];

    float *d_dh=nullptr, *d_cfg=nullptr, *d_jac=nullptr;
    cudaMalloc(&d_dh,  n_dh  * sizeof(float));
    cudaMalloc(&d_cfg, n_cfg * sizeof(float));
    cudaMalloc(&d_jac, n_jac * sizeof(float));
    cudaMemcpy(d_dh,  h_dh_f,  n_dh  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cfg, h_cfg_f, n_cfg * sizeof(float), cudaMemcpyHostToDevice);

    // N blocks × 6 threads
    jacobian_kernel<<<N, CUDA_NUM_JOINTS>>>(d_dh, d_cfg, d_jac, N);

    cudaMemcpy(h_jac_f, d_jac, n_jac * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0;i<n_jac;++i) jacobians[i] = (double)h_jac_f[i];

    delete[] h_dh_f; delete[] h_cfg_f; delete[] h_jac_f;
    cudaFree(d_dh); cudaFree(d_cfg); cudaFree(d_jac);
}
