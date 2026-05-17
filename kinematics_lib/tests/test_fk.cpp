#include <gtest/gtest.h>
#include <cmath>

#include "kinematics_lib/kinematics_solver.hpp"

using namespace kinematics;


// Fixture


class FKTest : public ::testing::Test {
protected:
    KinematicsSolver solver;
};


// Tests


/// FK at all-zero configuration should return a rigid body transform.
TEST_F(FKTest, ZeroConfigIsRigidBodyTransform) {
    std::vector<double> zeros(NUM_JOINTS, 0.0);
    auto T = solver.forward_kinematics(zeros);

    // Bottom row must be [0 0 0 1]
    EXPECT_NEAR(T(3, 0), 0.0, 1e-10);
    EXPECT_NEAR(T(3, 1), 0.0, 1e-10);
    EXPECT_NEAR(T(3, 2), 0.0, 1e-10);
    EXPECT_NEAR(T(3, 3), 1.0, 1e-10);

    // Rotation sub-block must be orthonormal
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Matrix3d I = R * R.transpose();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(I(i, j), (i == j ? 1.0 : 0.0), 1e-10);
}

/// Wrong number of joints should throw.
TEST_F(FKTest, WrongJointCountThrows) {
    std::vector<double> bad = {0.1, 0.2};
    EXPECT_THROW(solver.forward_kinematics(bad), std::invalid_argument);
}

/// FK must be consistent: same input → same output.
TEST_F(FKTest, DeterministicResult) {
    std::vector<double> q = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6};
    auto T1 = solver.forward_kinematics(q);
    auto T2 = solver.forward_kinematics(q);
    EXPECT_TRUE(T1.isApprox(T2, 1e-14));
}

/// FK with one non-zero joint must differ from all-zero.
TEST_F(FKTest, NonZeroJointChangesPose) {
    std::vector<double> zeros(NUM_JOINTS, 0.0);
    std::vector<double> q(NUM_JOINTS, 0.0);
    q[1] = M_PI / 4.0;

    auto T0 = solver.forward_kinematics(zeros);
    auto Tq = solver.forward_kinematics(q);

    EXPECT_FALSE(T0.isApprox(Tq, 1e-6));
}

/// Batch FK must match single FK for every configuration.
TEST_F(FKTest, BatchMatchesSingle) {
    const int N = 10;
    std::vector<double> configs;
    configs.reserve(N * NUM_JOINTS);
    for (int i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < NUM_JOINTS; ++j) {
            configs.push_back(0.1 * i - 0.3 * j);
        }
    }

    auto batch = solver.batch_forward_kinematics(configs);
    ASSERT_EQ(batch.size(), static_cast<std::size_t>(N));

    for (int i = 0; i < N; ++i) {
        std::vector<double> q(configs.begin() + i * NUM_JOINTS,
                               configs.begin() + (i + 1) * NUM_JOINTS);
        auto T_single = solver.forward_kinematics(q);
        EXPECT_TRUE(batch[i].isApprox(T_single, 1e-12))
            << "Mismatch at config " << i;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
