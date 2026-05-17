// ik_solver_kernel.cu
// GPU gradient-descent IK using float (FP32) for throughput.
// Uses DLS (damped least squares) and analytic geometric Jacobian.
// M IK problems solved in parallel, one thread per problem.

#include "cuda_kinematics.hpp"
#include <cuda_runtime.h>
#include <cmath>


// Device helpers (float)


__device__ static void _ik_dh_f(float a,float d,float al,float th, float* out) {
    float ct=cosf(th),st=sinf(th),ca=cosf(al),sa=sinf(al);
    out[0]=ct; out[1]=-st*ca; out[2]= st*sa; out[3]=a*ct;
    out[4]=st; out[5]= ct*ca; out[6]=-ct*sa; out[7]=a*st;
    out[8]=0;  out[9]=sa;     out[10]=ca;     out[11]=d;
    out[12]=0; out[13]=0;     out[14]=0;      out[15]=1;
}

__device__ static void _ik_mat4_mul_f(const float* A, const float* B, float* C) {
    for (int r=0;r<4;++r) for (int c=0;c<4;++c) {
        float s=0; for (int k=0;k<4;++k) s+=A[r*4+k]*B[k*4+c];
        C[r*4+c]=s;
    }
}

__device__ static void _ik_fk_f(const float* dh, const float* q, float* T) {
    float acc[16]={},tmp[16],Tj[16];
    acc[0]=acc[5]=acc[10]=acc[15]=1.f;
    for (int j=0;j<CUDA_NUM_JOINTS;++j) {
        _ik_dh_f(dh[j*4],dh[j*4+1],dh[j*4+2],dh[j*4+3]+q[j],Tj);
        _ik_mat4_mul_f(acc,Tj,tmp);
        for (int i=0;i<16;++i) acc[i]=tmp[i];
    }
    for (int i=0;i<16;++i) T[i]=acc[i];
}

// Partial FK up to joint `upto` inclusive
__device__ static void _ik_partial_fk_f(const float* dh, const float* q,
                                          int upto, float* T) {
    float acc[16]={},tmp[16],Tj[16];
    acc[0]=acc[5]=acc[10]=acc[15]=1.f;
    for (int j=0;j<=upto;++j) {
        _ik_dh_f(dh[j*4],dh[j*4+1],dh[j*4+2],dh[j*4+3]+q[j],Tj);
        _ik_mat4_mul_f(acc,Tj,tmp);
        for (int i=0;i<16;++i) acc[i]=tmp[i];
    }
    for (int i=0;i<16;++i) T[i]=acc[i];
}

// Analytic geometric Jacobian (float)
__device__ static void _ik_jacobian_f(const float* dh, const float* q,
                                       float J[6][CUDA_NUM_JOINTS]) {
    float T_ee[16]; _ik_fk_f(dh, q, T_ee);
    float p_ee[3] = {T_ee[3], T_ee[7], T_ee[11]};

    for (int j=0;j<CUDA_NUM_JOINTS;++j) {
        float z[3], p_i[3];
        if (j==0) {
            z[0]=0; z[1]=0; z[2]=1;
            p_i[0]=0; p_i[1]=0; p_i[2]=0;
        } else {
            float T_pre[16]; _ik_partial_fk_f(dh, q, j-1, T_pre);
            z[0]=T_pre[2]; z[1]=T_pre[6]; z[2]=T_pre[10];
            p_i[0]=T_pre[3]; p_i[1]=T_pre[7]; p_i[2]=T_pre[11];
        }
        float dp[3]={p_ee[0]-p_i[0], p_ee[1]-p_i[1], p_ee[2]-p_i[2]};
        J[0][j]=z[1]*dp[2]-z[2]*dp[1];
        J[1][j]=z[2]*dp[0]-z[0]*dp[2];
        J[2][j]=z[0]*dp[1]-z[1]*dp[0];
        J[3][j]=z[0]; J[4][j]=z[1]; J[5][j]=z[2];
    }
}

// World-frame pose error
__device__ static void _ik_pose_error_f(const float* T_cur, const float* T_tgt,
                                         float* e) {
    for (int i=0;i<3;++i) e[i]=T_tgt[i*4+3]-T_cur[i*4+3];
    // R_err = R_tgt * R_cur^T
    float R_err[9]={};
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
        for (int k=0;k<3;++k) R_err[r*3+c]+=T_tgt[r*4+k]*T_cur[c*4+k];
    e[3]=(R_err[2*3+1]-R_err[1*3+2])/2.f;
    e[4]=(R_err[0*3+2]-R_err[2*3+0])/2.f;
    e[5]=(R_err[1*3+0]-R_err[0*3+1])/2.f;
}


// Kernel


__global__ void ik_solver_kernel(const float* __restrict__ dh_params,
                                  const float* __restrict__ target_poses,
                                  const float* __restrict__ seeds,
                                  float*       __restrict__ results,
                                  int*         __restrict__ converged,
                                  int M, float alpha, float tol, int max_iter) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    const float* target = target_poses + idx * 16;
    float* result       = results + idx * CUDA_NUM_JOINTS;

    float q[CUDA_NUM_JOINTS];
    if (seeds) for (int j=0;j<CUDA_NUM_JOINTS;++j) q[j]=seeds[idx*CUDA_NUM_JOINTS+j];
    else       for (int j=0;j<CUDA_NUM_JOINTS;++j) q[j]=0.f;

    const float lam2 = 0.05f * 0.05f;
    float J[6][CUDA_NUM_JOINTS], e[6];
    bool ok = false;

    for (int iter=0; iter<max_iter; ++iter) {
        float T_cur[16]; _ik_fk_f(dh_params, q, T_cur);
        _ik_pose_error_f(T_cur, target, e);

        float norm2=0; for (int i=0;i<6;++i) norm2+=e[i]*e[i];
        if (norm2 < tol*tol) { ok=true; break; }

        _ik_jacobian_f(dh_params, q, J);

        // DLS: Δq = α · Jᵀ (J Jᵀ + λ²I)⁻¹ e  — 6×6 Gaussian elimination
        float A[36]={}, b[6];
        for (int r=0;r<6;++r) {
            b[r]=e[r];
            for (int c=0;c<6;++c) {
                float s=0; for (int k=0;k<CUDA_NUM_JOINTS;++k) s+=J[r][k]*J[c][k];
                A[r*6+c]=s+(r==c?lam2:0.f);
            }
        }
        for (int col=0;col<6;++col) {
            float piv=A[col*6+col]; if (fabsf(piv)<1e-9f) continue;
            for (int row=col+1;row<6;++row) {
                float f=A[row*6+col]/piv;
                for (int k=col;k<6;++k) A[row*6+k]-=f*A[col*6+k];
                b[row]-=f*b[col];
            }
        }
        float x[6]={};
        for (int row=5;row>=0;--row) {
            float s=b[row]; for (int k=row+1;k<6;++k) s-=A[row*6+k]*x[k];
            x[row]=(fabsf(A[row*6+row])>1e-9f)?s/A[row*6+row]:0.f;
        }
        for (int j=0;j<CUDA_NUM_JOINTS;++j) {
            float dq=0; for (int r=0;r<6;++r) dq+=J[r][j]*x[r];
            q[j]+=alpha*dq;
        }
    }

    for (int j=0;j<CUDA_NUM_JOINTS;++j) result[j]=q[j];
    converged[idx]=ok?1:0;
}


// Host launcher


extern "C" void cuda_batch_ik(const double* dh_params,
                               const double* target_poses,
                               const double* seeds,
                               double*       results,
                               int*          converged,
                               int M, double alpha, double tol, int max_iter) {
    const int n_dh  = CUDA_NUM_JOINTS*4;
    const int n_tgt = M*16;
    const int n_res = M*CUDA_NUM_JOINTS;

    float *h_dh_f  = new float[n_dh];
    float *h_tgt_f = new float[n_tgt];
    float *h_res_f = new float[n_res];
    for (int i=0;i<n_dh; ++i) h_dh_f[i]  = (float)dh_params[i];
    for (int i=0;i<n_tgt;++i) h_tgt_f[i]  = (float)target_poses[i];

    float *d_dh=nullptr, *d_tgt=nullptr, *d_seed_f=nullptr, *d_res=nullptr;
    int   *d_conv=nullptr;
    cudaMalloc(&d_dh,   n_dh  * sizeof(float));
    cudaMalloc(&d_tgt,  n_tgt * sizeof(float));
    cudaMalloc(&d_res,  n_res * sizeof(float));
    cudaMalloc(&d_conv, M     * sizeof(int));
    cudaMemcpy(d_dh,  h_dh_f,  n_dh  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, h_tgt_f, n_tgt * sizeof(float), cudaMemcpyHostToDevice);

    if (seeds) {
        float *h_seed_f = new float[n_res];
        for (int i=0;i<n_res;++i) h_seed_f[i]=(float)seeds[i];
        cudaMalloc(&d_seed_f, n_res * sizeof(float));
        cudaMemcpy(d_seed_f, h_seed_f, n_res*sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_seed_f;
    }

    const int threads=128, blocks=(M+threads-1)/threads;
    ik_solver_kernel<<<blocks,threads>>>(d_dh, d_tgt, d_seed_f, d_res, d_conv,
                                          M, (float)alpha, (float)tol, max_iter);

    cudaMemcpy(h_res_f, d_res,  n_res * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(converged, d_conv, M   * sizeof(int),   cudaMemcpyDeviceToHost);
    for (int i=0;i<n_res;++i) results[i]=(double)h_res_f[i];

    delete[] h_dh_f; delete[] h_tgt_f; delete[] h_res_f;
    cudaFree(d_dh); cudaFree(d_tgt); cudaFree(d_res); cudaFree(d_conv);
    if (d_seed_f) cudaFree(d_seed_f);
}
