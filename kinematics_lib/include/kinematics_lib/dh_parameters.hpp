#pragma once

#include <array>
#include <cstddef>

namespace kinematics {

/// Denavit–Hartenberg parameters for a single joint.
struct DHParams {
    double a;      ///< Link length (metres)
    double d;      ///< Link offset  (metres)
    double alpha;  ///< Link twist   (radians)
    double theta;  ///< Joint angle offset (radians); runtime angle is added on top
};

/// Number of joints in the default 6-DOF arm (UR5-like).
constexpr std::size_t NUM_JOINTS = 6;

/// Default DH table for a UR5-like 6-DOF arm.
/// Values: a (m), d (m), alpha (rad), theta_offset (rad)
inline constexpr std::array<DHParams, NUM_JOINTS> DEFAULT_DH_PARAMS = {{
    {0.000,  0.0892,  M_PI / 2.0,  0.0},
    {0.425,  0.000,   0.0,         0.0},
    {0.392,  0.000,   0.0,         0.0},
    {0.000,  0.1093,  M_PI / 2.0,  0.0},
    {0.000,  0.0950, -M_PI / 2.0,  0.0},
    {0.000,  0.0820,  0.0,         0.0},
}};

}  // namespace kinematics
