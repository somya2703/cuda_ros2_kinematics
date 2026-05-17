#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

#include "dh_parameters.hpp"

namespace kinematics {

/// 4×4 homogeneous transform alias.
using Transform = Eigen::Matrix4d;

/// 6×6 Jacobian alias (rows: vx vy vz wx wy wz, cols: joints).
using Jacobian = Eigen::Matrix<double, 6, NUM_JOINTS>;

/// -------------------------------------------------------------------
/// KinematicsSolver
///
/// Pure C++ class — no ROS dependency.
/// Optionally delegates batch / GPU-heavy operations to CUDA kernels
/// when compiled with CUDA support (BUILD_CUDA=ON).
/// -------------------------------------------------------------------
class KinematicsSolver {
public:
    /// Construct with default UR5-like DH parameters.
    KinematicsSolver();

    /// Construct with custom DH table.
    explicit KinematicsSolver(const std::array<DHParams, NUM_JOINTS>& dh);

    ~KinematicsSolver() = default;

    // ----------------------------------------------------------------
    // Single-configuration FK / IK
    // ----------------------------------------------------------------

    /// Forward kinematics: joint angles → end-effector pose (4×4).
    /// @param joints  Joint angles in radians, size == NUM_JOINTS.
    Transform forward_kinematics(const std::vector<double>& joints) const;

    /// Inverse kinematics: target pose → joint angles.
    /// Uses gradient-descent (CPU). Returns false if not converged.
    /// @param target_pose  Desired 4×4 end-effector transform.
    /// @param joints_out   Output joint angles (initialised to seed).
    /// @param seed         Starting configuration (default: all zeros).
    bool inverse_kinematics(const Transform& target_pose,
                            std::vector<double>& joints_out,
                            const std::vector<double>& seed = {}) const;

    // ----------------------------------------------------------------
    // Jacobian
    // ----------------------------------------------------------------

    /// Geometric Jacobian at the given configuration (numerical, CPU).
    Jacobian jacobian(const std::vector<double>& joints) const;

    // ----------------------------------------------------------------
    // Batch operations (CPU; CUDA path used when available)
    // ----------------------------------------------------------------

    /// Batch FK: compute end-effector transforms for N configurations.
    /// @param configs  Flat vector of joint angles, size N × NUM_JOINTS.
    /// @returns        Vector of N Transform matrices.
    std::vector<Transform> batch_forward_kinematics(
        const std::vector<double>& configs) const;

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    /// DH transform for joint i given angle theta_i.
    Transform dh_transform(std::size_t joint, double theta) const;

    /// Pose error vector (6-DOF): [pos_err; rot_err].
    static Eigen::Matrix<double, 6, 1> pose_error(const Transform& current,
                                                   const Transform& target);

private:
    std::array<DHParams, NUM_JOINTS> dh_;

    static constexpr double kIKAlpha       = 0.5;   // step scale for DLS
    static constexpr double kIKDamping     = 0.05;  // Levenberg-Marquardt λ
    static constexpr double kIKTolerance   = 1e-3;  // convergence threshold ‖e‖
    static constexpr int    kIKMaxIter     = 200;   // converges in <30 iters typically
};

}  // namespace kinematics
