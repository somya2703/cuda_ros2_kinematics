// trajectory_planner.cpp
// Subscribes to /joint_states, linearly interpolates between waypoints, and
// republishes a smooth trajectory on /trajectory.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <cmath>
#include <vector>

class TrajectoryPlanner : public rclcpp::Node {
public:
    TrajectoryPlanner() : Node("trajectory_planner") {
        declare_parameter<int>("interp_steps", 10);
        declare_parameter<double>("dt_seconds", 0.05);
        declare_parameter<std::vector<std::string>>(
            "joint_names",
            {"shoulder_pan", "shoulder_lift", "elbow",
             "wrist_1", "wrist_2", "wrist_3"});

        steps_       = get_parameter("interp_steps").as_int();
        dt_          = get_parameter("dt_seconds").as_double();
        joint_names_ = get_parameter("joint_names").as_string_array();

        sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) {
                on_joint_state(msg);
            });

        pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/trajectory", 10);

        RCLCPP_INFO(get_logger(),
                    "TrajectoryPlanner ready — interpolating %d steps between waypoints",
                    steps_);
    }

private:
    void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr& msg) {
        if (prev_q_.empty()) {
            prev_q_ = msg->position;
            return;
        }

        const auto& q0 = prev_q_;
        const auto& q1 = msg->position;
        const int   n  = static_cast<int>(q0.size());

        trajectory_msgs::msg::JointTrajectory traj;
        traj.header.stamp     = now();
        traj.header.frame_id  = "base_link";
        traj.joint_names      = joint_names_;

        for (int step = 0; step <= steps_; ++step) {
            double alpha = static_cast<double>(step) / steps_;
            trajectory_msgs::msg::JointTrajectoryPoint pt;
            pt.positions.resize(n);
            for (int j = 0; j < n; ++j)
                pt.positions[j] = (1.0 - alpha) * q0[j] + alpha * q1[j];

            // Velocity: finite difference (constant across segment)
            pt.velocities.resize(n);
            for (int j = 0; j < n; ++j)
                pt.velocities[j] = (q1[j] - q0[j]) / (steps_ * dt_);

            // Time from start
            double t_sec = step * dt_;
            pt.time_from_start.sec  = static_cast<int32_t>(t_sec);
            pt.time_from_start.nanosec =
                static_cast<uint32_t>((t_sec - pt.time_from_start.sec) * 1e9);

            traj.points.push_back(pt);
        }

        pub_->publish(traj);
        prev_q_ = msg->position;
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_;

    std::vector<std::string> joint_names_;
    std::vector<double>      prev_q_;
    int                      steps_;
    double                   dt_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryPlanner>());
    rclcpp::shutdown();
    return 0;
}
