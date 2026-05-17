cmake_minimum_required(VERSION 3.18)
project(cuda_ros2_kinematics LANGUAGES CXX CUDA)


# Options

option(BUILD_CUDA    "Build CUDA kernels and benchmarks" ON)
option(BUILD_ROS2    "Build ROS 2 nodes"                 ON)
option(BUILD_TESTS   "Build Google Test unit tests"      ON)


# Global compiler flags

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra)


# Dependencies

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
endif()

if(BUILD_ROS2)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(trajectory_msgs REQUIRED)
    find_package(robot_state_publisher REQUIRED)
endif()


# kinematics_lib — pure C++ library

add_library(kinematics_lib STATIC
    kinematics_lib/src/kinematics_solver.cpp
)

target_include_directories(kinematics_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kinematics_lib/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(kinematics_lib PUBLIC Eigen3::Eigen)


# CUDA kernels library

if(BUILD_CUDA)
    enable_language(CUDA)

    add_library(cuda_kinematics STATIC
        cuda_kernels/batch_fk_kernel.cu
        cuda_kernels/jacobian_kernel.cu
        cuda_kernels/ik_solver_kernel.cu
    )

    target_include_directories(cuda_kinematics
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kinematics_lib/include>
    )

    set_target_properties(cuda_kinematics PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;86;89"   # Turing, Ampere, Ada — adjust as needed
    )

    target_link_libraries(cuda_kinematics PUBLIC kinematics_lib)

    #Benchmarks 
    foreach(bench batch_fk jacobian ik)
        add_executable(${bench}_benchmark
            cuda_kernels/benchmarks/${bench}_benchmark.cpp)
        target_link_libraries(${bench}_benchmark
            PRIVATE kinematics_lib cuda_kinematics)
        target_include_directories(${bench}_benchmark PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}              # so "cuda_kernels/cuda_kinematics.hpp" resolves
            ${CMAKE_CURRENT_SOURCE_DIR}/kinematics_lib/include)
    endforeach()
endif()


# Unit tests

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_fk kinematics_lib/tests/test_fk.cpp)
    target_link_libraries(test_fk PRIVATE kinematics_lib GTest::GTest GTest::Main)
    add_test(NAME test_fk COMMAND test_fk)

    add_executable(test_ik kinematics_lib/tests/test_ik.cpp)
    target_link_libraries(test_ik PRIVATE kinematics_lib GTest::GTest GTest::Main)
    add_test(NAME test_ik COMMAND test_ik)
endif()


# ROS 2 nodes

if(BUILD_ROS2)
    # joint_command_publisher
    add_executable(joint_command_publisher
        ros2_nodes/src/joint_command_publisher.cpp)
    ament_target_dependencies(joint_command_publisher
        rclcpp sensor_msgs)
    target_link_libraries(joint_command_publisher kinematics_lib)

    # trajectory_planner
    add_executable(trajectory_planner
        ros2_nodes/src/trajectory_planner.cpp)
    ament_target_dependencies(trajectory_planner
        rclcpp sensor_msgs trajectory_msgs)

    # state_publisher
    add_executable(state_publisher
        ros2_nodes/src/state_publisher.cpp)
    ament_target_dependencies(state_publisher
        rclcpp sensor_msgs geometry_msgs)
    target_link_libraries(state_publisher kinematics_lib Eigen3::Eigen)

    install(TARGETS
        joint_command_publisher
        trajectory_planner
        state_publisher
        DESTINATION lib/${PROJECT_NAME}
    )

    install(DIRECTORY ros2_nodes/launch/
        DESTINATION share/${PROJECT_NAME}/launch
    )

    install(DIRECTORY ros2_nodes/urdf/
        DESTINATION share/${PROJECT_NAME}/urdf
    )

    install(DIRECTORY ros2_nodes/rviz/
        DESTINATION share/${PROJECT_NAME}/rviz
    )

    ament_package()
endif()
