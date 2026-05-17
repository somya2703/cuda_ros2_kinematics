// joint_command_publisher.cpp
// ROS 2 node that publishes joint states from keyboard input or a trajectory
// file to the /joint_states topic.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std::chrono_literals;

class JointCommandPublisher : public rclcpp::Node {
public:
    JointCommandPublisher() : Node("joint_command_publisher"), tick_(0) {
        declare_parameter<std::string>("trajectory_file", "");
        declare_parameter<double>("publish_rate_hz", 10.0);
        declare_parameter<std::vector<std::string>>(
            "joint_names",
            {"shoulder_pan", "shoulder_lift", "elbow",
             "wrist_1", "wrist_2", "wrist_3"});

        joint_names_ = get_parameter("joint_names")
                           .as_string_array();
        double hz = get_parameter("publish_rate_hz").as_double();
        std::string traj_file = get_parameter("trajectory_file").as_string();

        if (!traj_file.empty()) {
            load_trajectory(traj_file);
            RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from '%s'",
                        trajectory_.size(), traj_file.c_str());
        } else {
            // Default: sinusoidal sweep
            generate_sine_trajectory(100);
            RCLCPP_INFO(get_logger(),
                        "No trajectory file — using sinusoidal sweep (%zu waypoints)",
                        trajectory_.size());
        }

        publisher_ = create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states", 10);

        auto period = std::chrono::duration<double>(1.0 / hz);
        timer_ = create_wall_timer(period,
                                   [this]() { publish_next(); });
    }

private:
    void publish_next() {
        if (trajectory_.empty()) return;
        const auto& q = trajectory_[tick_ % trajectory_.size()];
        ++tick_;

        sensor_msgs::msg::JointState msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "base_link";
        msg.name            = joint_names_;
        msg.position        = q;
        publisher_->publish(msg);
    }

    void generate_sine_trajectory(int steps) {
        trajectory_.reserve(steps);
        for (int i = 0; i < steps; ++i) {
            double t = 2.0 * M_PI * i / steps;
            trajectory_.push_back({
                0.5 * std::sin(t),
                0.3 * std::sin(t + 1.0),
                0.4 * std::sin(t + 2.0),
                0.2 * std::sin(t + 3.0),
                0.3 * std::sin(t + 4.0),
                0.1 * std::sin(t + 5.0),
            });
        }
    }

    void load_trajectory(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            RCLCPP_ERROR(get_logger(), "Cannot open trajectory file: %s", path.c_str());
            return;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::vector<double> q;
            double v;
            while (ss >> v) q.push_back(v);
            if (q.size() == joint_names_.size()) trajectory_.push_back(q);
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::string> joint_names_;
    std::vector<std::vector<double>> trajectory_;
    std::size_t tick_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointCommandPublisher>());
    rclcpp::shutdown();
    return 0;
}
