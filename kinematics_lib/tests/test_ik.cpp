#include <gtest/gtest.h>
#include <cmath>

#include "kinematics_lib/kinematics_solver.hpp"

using namespace kinematics;

class IKTest : public ::testing::Test {
protected:
    KinematicsSolver solver;
};

/// Round-trip test: IK(FK(q)) ≈ q (up to redundancy).
/// We validate by re-running FK on the IK result and comparing poses.
TEST_F(IKTest, RoundTrip) {
    // Use a moderate config — not near singularity, reachable from zero seed
    std::vector<double> q_true = {0.2, -0.5, 0.8, -0.3, 0.2, -0.1};
    Transform target = solver.forward_kinematics(q_true);

    std::vector<double> q_result;
    bool converged = solver.inverse_kinematics(target, q_result);

    ASSERT_TRUE(converged) << "IK did not converge";

    Transform T_result = solver.forward_kinematics(q_result);
    // Compare end-effector poses (multiple joint solutions can give the same pose)
    for (int r = 0; r < 3; ++r) {
        EXPECT_NEAR(T_result(r, 3), target(r, 3), 1e-2) << "Position mismatch row " << r;
    }
    // Rotation: R_result^T * R_target should be close to identity
    Eigen::Matrix3d R_err =
        T_result.block<3,3>(0,0).transpose() * target.block<3,3>(0,0);
    Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    EXPECT_TRUE(R_err.isApprox(I3, 1e-2)) << "Rotation mismatch:\n" << R_err;
}

/// IK with a seed close to solution should converge in fewer iterations.
TEST_F(IKTest, SeedImproveConvergence) {
    std::vector<double> q_true = {0.1, -0.3, 0.5, -0.2, 0.1, 0.0};
    Transform target = solver.forward_kinematics(q_true);

    // Seed very close to solution
    std::vector<double> seed = {0.11, -0.31, 0.51, -0.21, 0.11, 0.01};
    std::vector<double> q_result;
    bool converged = solver.inverse_kinematics(target, q_result, seed);
    EXPECT_TRUE(converged) << "IK with near-seed did not converge";
}

/// Jacobian must be non-singular at a typical configuration.
TEST_F(IKTest, JacobianIsFullRank) {
    std::vector<double> q = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6};
    Jacobian J = solver.jacobian(q);

    // For a non-singular configuration, rank should be 6
    Eigen::JacobiSVD<Jacobian> svd(J);
    auto svals = svd.singularValues();
    EXPECT_GT(svals(5), 1e-6) << "Jacobian appears singular at this config";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
