// state_publisher.cpp
// Subscribes to /joint_states, runs FK via KinematicsSolver, and publishes
// the end-effector pose on /ee_pose.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "kinematics_lib/kinematics_solver.hpp"

class StatePublisher : public rclcpp::Node {
public:
    StatePublisher() : Node("state_publisher") {
        sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) {
                on_joint_state(msg);
            });

        pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/ee_pose", 10);

        RCLCPP_INFO(get_logger(),
                    "StatePublisher ready. Listening on /joint_states → publishing /ee_pose");
    }

private:
    void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr& msg) {
        if (msg->position.size() < kinematics::NUM_JOINTS) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                  "Received fewer than %zu joint positions — skipping",
                                  kinematics::NUM_JOINTS);
            return;
        }

        std::vector<double> q(msg->position.begin(),
                               msg->position.begin() + kinematics::NUM_JOINTS);

        kinematics::Transform T = solver_.forward_kinematics(q);

        // Extract position
        Eigen::Vector3d pos = T.block<3,1>(0,3);

        // Extract quaternion from rotation matrix
        Eigen::Matrix3d rot = T.block<3,3>(0,0);
        Eigen::Quaterniond quat(rot);

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp    = msg->header.stamp;
        pose_msg.header.frame_id = "base_link";

        pose_msg.pose.position.x = pos.x();
        pose_msg.pose.position.y = pos.y();
        pose_msg.pose.position.z = pos.z();

        pose_msg.pose.orientation.x = quat.x();
        pose_msg.pose.orientation.y = quat.y();
        pose_msg.pose.orientation.z = quat.z();
        pose_msg.pose.orientation.w = quat.w();

        pub_->publish(pose_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;
    kinematics::KinematicsSolver solver_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StatePublisher>());
    rclcpp::shutdown();
    return 0;
}
