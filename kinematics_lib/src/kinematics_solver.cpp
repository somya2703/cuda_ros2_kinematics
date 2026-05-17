#include "kinematics_lib/kinematics_solver.hpp"

#include <cmath>
#include <stdexcept>

namespace kinematics {

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

static Transform make_dh(double a, double d, double alpha, double theta) {
    const double ct = std::cos(theta), st = std::sin(theta);
    const double ca = std::cos(alpha), sa = std::sin(alpha);

    Transform T;
    T << ct, -st * ca,  st * sa, a * ct,
         st,  ct * ca, -ct * sa, a * st,
         0.0,       sa,       ca,      d,
         0.0,      0.0,      0.0,    1.0;
    return T;
}

// ────────────────────────────────────────────────────────────────────────────
// Construction
// ────────────────────────────────────────────────────────────────────────────

KinematicsSolver::KinematicsSolver()
    : dh_(DEFAULT_DH_PARAMS) {}

KinematicsSolver::KinematicsSolver(const std::array<DHParams, NUM_JOINTS>& dh)
    : dh_(dh) {}

// ────────────────────────────────────────────────────────────────────────────
// DH transform for one joint
// ────────────────────────────────────────────────────────────────────────────

Transform KinematicsSolver::dh_transform(std::size_t i, double theta) const {
    return make_dh(dh_[i].a, dh_[i].d, dh_[i].alpha, theta + dh_[i].theta);
}

// ────────────────────────────────────────────────────────────────────────────
// Forward kinematics
// ────────────────────────────────────────────────────────────────────────────

Transform KinematicsSolver::forward_kinematics(
    const std::vector<double>& joints) const {
    if (joints.size() != NUM_JOINTS) {
        throw std::invalid_argument("Expected " + std::to_string(NUM_JOINTS) +
                                    " joint angles, got " +
                                    std::to_string(joints.size()));
    }
    Transform T = Transform::Identity();
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        T *= dh_transform(i, joints[i]);
    }
    return T;
}

// ────────────────────────────────────────────────────────────────────────────
// Pose error
// ────────────────────────────────────────────────────────────────────────────

Eigen::Matrix<double, 6, 1> KinematicsSolver::pose_error(
    const Transform& current, const Transform& target) {
    Eigen::Matrix<double, 6, 1> e;

    // Position error (world frame)
    e.head<3>() = target.block<3, 1>(0, 3) - current.block<3, 1>(0, 3);

    // Rotation error in world frame: R_err = R_tgt * R_cur^T
    // Rotation vector = skew-symmetric part: (R_err - R_err^T) / 2
    Eigen::Matrix3d R_err =
        target.block<3, 3>(0, 0) * current.block<3, 3>(0, 0).transpose();

    e(3) = (R_err(2, 1) - R_err(1, 2)) / 2.0;
    e(4) = (R_err(0, 2) - R_err(2, 0)) / 2.0;
    e(5) = (R_err(1, 0) - R_err(0, 1)) / 2.0;

    return e;
}

// ────────────────────────────────────────────────────────────────────────────
// Jacobian (numerical, central differences)
// ────────────────────────────────────────────────────────────────────────────

Jacobian KinematicsSolver::jacobian(const std::vector<double>& joints) const {
    Jacobian J;

    // Geometric Jacobian:
    //   position col i = z_{i-1} × (p_ee - p_{i-1})
    //   rotation col i = z_{i-1}
    // where z_{i-1} and p_{i-1} come from the partial FK up to (but not including) joint i.

    // Pre-compute partial transforms T_0..i for i = 0..NUM_JOINTS
    std::array<Transform, NUM_JOINTS + 1> T_partial;
    T_partial[0] = Transform::Identity();
    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        T_partial[i + 1] = T_partial[i] * dh_transform(i, joints[i]);
    }

    // End-effector position
    const Eigen::Vector3d p_ee = T_partial[NUM_JOINTS].block<3, 1>(0, 3);

    for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
        // z-axis of frame i (the axis joint i+1 revolves around)
        Eigen::Vector3d z_i = T_partial[i].block<3, 1>(0, 2);
        Eigen::Vector3d p_i = T_partial[i].block<3, 1>(0, 3);

        // Revolute joint: linear vel = z × (p_ee - p_i), angular vel = z
        J.col(i).head<3>() = z_i.cross(p_ee - p_i);
        J.col(i).tail<3>() = z_i;
    }
    return J;
}

// ────────────────────────────────────────────────────────────────────────────
// Inverse kinematics (gradient descent)
// ────────────────────────────────────────────────────────────────────────────

bool KinematicsSolver::inverse_kinematics(const Transform& target_pose,
                                          std::vector<double>& joints_out,
                                          const std::vector<double>& seed) const {
    // Initialise from seed or zeros
    if (seed.size() == NUM_JOINTS) {
        joints_out = seed;
    } else {
        joints_out.assign(NUM_JOINTS, 0.0);
    }

    // Damped Least-Squares (Levenberg-Marquardt style):
    //   Δq = α · Jᵀ (J Jᵀ + λ²I)⁻¹ e
    // Much more robust than plain gradient descent near singularities.
    const double lambda2 = kIKDamping * kIKDamping;

    for (int iter = 0; iter < kIKMaxIter; ++iter) {
        Transform T = forward_kinematics(joints_out);
        auto e = pose_error(T, target_pose);

        if (e.norm() < kIKTolerance) {
            return true;
        }

        Jacobian J = jacobian(joints_out);

        // DLS: Δq = α · Jᵀ (J Jᵀ + λ²I)⁻¹ e
        Eigen::Matrix<double, 6, 6> JJt =
            J * J.transpose() + lambda2 * Eigen::Matrix<double, 6, 6>::Identity();

        Eigen::Matrix<double, NUM_JOINTS, 1> dq =
            kIKAlpha * J.transpose() * JJt.ldlt().solve(e);

        for (std::size_t i = 0; i < NUM_JOINTS; ++i) {
            joints_out[i] += dq(i);
        }
    }
    return false;
}

// ────────────────────────────────────────────────────────────────────────────
// Batch FK (CPU fallback; CUDA path in cuda_kernels)
// ────────────────────────────────────────────────────────────────────────────

std::vector<Transform> KinematicsSolver::batch_forward_kinematics(
    const std::vector<double>& configs) const {
    if (configs.size() % NUM_JOINTS != 0) {
        throw std::invalid_argument("configs size must be a multiple of NUM_JOINTS");
    }
    const std::size_t N = configs.size() / NUM_JOINTS;
    std::vector<Transform> results(N);

    for (std::size_t i = 0; i < N; ++i) {
        std::vector<double> q(configs.begin() + i * NUM_JOINTS,
                               configs.begin() + (i + 1) * NUM_JOINTS);
        results[i] = forward_kinematics(q);
    }
    return results;
}

}  // namespace kinematics
