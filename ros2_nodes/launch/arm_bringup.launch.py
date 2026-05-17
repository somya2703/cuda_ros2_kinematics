"""arm_bringup.launch.py"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('cuda_ros2_kinematics')
    urdf_path = os.path.join(pkg_share, 'urdf', 'cuda_arm.urdf')
    rviz_path = os.path.join(pkg_share, 'rviz', 'arm_view.rviz')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # Read directly from env vars 
    
    trajectory_file  = os.environ.get('TRAJECTORY_FILE', '')
    publish_rate_hz  = float(os.environ.get('PUBLISH_RATE_HZ', '10.0'))
    show_rviz        = os.environ.get('SHOW_RVIZ', 'true').lower() == 'true'

    return LaunchDescription([

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description,
                         'publish_frequency': 30.0}],
        ),

        Node(
            package='cuda_ros2_kinematics',
            executable='joint_command_publisher',
            name='joint_command_publisher',
            output='screen',
            parameters=[{
                'trajectory_file': trajectory_file,
                'publish_rate_hz':  publish_rate_hz,
                'joint_names': [
                    'shoulder_pan', 'shoulder_lift', 'elbow',
                    'wrist_1', 'wrist_2', 'wrist_3',
                ],
            }],
        ),

        Node(
            package='cuda_ros2_kinematics',
            executable='trajectory_planner',
            name='trajectory_planner',
            output='screen',
            parameters=[{'interp_steps': 10, 'dt_seconds': 0.05}],
        ),

        Node(
            package='cuda_ros2_kinematics',
            executable='state_publisher',
            name='state_publisher',
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_path],
            output='screen',
            condition=IfCondition('true' if show_rviz else 'false'),
        ),
    ])
