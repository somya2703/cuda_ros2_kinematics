#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif


// CUDA kernel launchers 


/// Number of joints (must match kinematics_lib).
#define CUDA_NUM_JOINTS 6

/// Size of one DH parameter set (a, d, alpha, theta) in doubles.
#define CUDA_DH_STRIDE 4

// ─────────────────────────────────────────────────────────────────────────────
// Batch FK
// ─────────────────────────────────────────────────────────────────────────────

/// GPU batch forward kinematics.
///
/// @param dh_params   Flat DH table [a0,d0,alpha0,th0, a1,…]  length 6×4
/// @param configs     Flat joint configs  [q0_j0,…,q0_j5, q1_j0,…]  length N×6
/// @param transforms  Output 4×4 matrices (row-major)  length N×16
/// @param N           Number of configurations
void cuda_batch_fk(const double* dh_params,
                   const double* configs,
                   double*       transforms,
                   int           N);


// Jacobian


/// GPU Jacobian computation (central finite differences).
///
/// @param dh_params   Flat DH table  length 6×4
/// @param configs     Flat joint configs  length N×6
/// @param jacobians   Output 6×6 Jacobians (row-major)  length N×36
/// @param N           Number of configurations
/// @param eps         Finite-difference step (default 1e-5)
void cuda_batch_jacobian(const double* dh_params,
                         const double* configs,
                         double*       jacobians,
                         int           N,
                         double        eps);


// IK solver


/// GPU gradient-descent IK.
///
/// @param dh_params    Flat DH table  length 6×4
/// @param target_poses Flat 4×4 target transforms (row-major)  length M×16
/// @param seeds        Initial joint configs  length M×6  (may be NULL → zeros)
/// @param results      Output joint configs   length M×6
/// @param converged    Output convergence flag per problem  length M (bool-as-int)
/// @param M            Number of IK problems
/// @param alpha        Gradient-descent step size
/// @param tol          Convergence tolerance (‖e‖₂)
/// @param max_iter     Maximum iterations
void cuda_batch_ik(const double* dh_params,
                   const double* target_poses,
                   const double* seeds,
                   double*       results,
                   int*          converged,
                   int           M,
                   double        alpha,
                   double        tol,
                   int           max_iter);

#ifdef __cplusplus
}
#endif
