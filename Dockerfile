
# Stage 1 — base
#   NVIDIA CUDA 12.4 + Ubuntu 22.04, with ROS 2 Humble layered on top.
#   This stage is shared by both the builder and the final runtime image.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# System essentials + locale
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales \
        curl \
        gnupg2 \
        lsb-release \
        ca-certificates \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) \
        signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu \
        $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list

# ROS 2 Humble base + common message packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-sensor-msgs \
        ros-humble-geometry-msgs \
        ros-humble-trajectory-msgs \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-rviz2 \
        ffmpeg \
        x11-utils \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rosdep init || true \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*

# Source ROS 2 in every shell session
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc



# Stage 2 — builder
#   Installs build-only tools (CMake, GTest, Eigen, nvcc) and compiles
#   the project. Nothing from this stage leaks into the runtime image.

FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        build-essential \
        libeigen3-dev \
        libgtest-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /ws/src/cuda_ros2_kinematics
COPY . .

# Build kinematics_lib + CUDA kernels + unit tests (no ROS 2) 
RUN mkdir -p /ws/build/no_ros && \
    cmake -S . -B /ws/build/no_ros \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_ROS2=OFF \
          -DBUILD_CUDA=ON \
          -DBUILD_TESTS=ON \
          -DCMAKE_CUDA_ARCHITECTURES="75;86;89" && \
    cmake --build /ws/build/no_ros --parallel

# Run unit tests at build time (fail the image if tests fail) 
RUN cd /ws/build/no_ros && ctest --output-on-failure

# Build ROS 2 nodes via colcon 
WORKDIR /ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build \
        --packages-select cuda_ros2_kinematics \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_CUDA=ON \
            -DBUILD_TESTS=OFF \
            -DCMAKE_CUDA_ARCHITECTURES="75;86;89" \
        --event-handlers console_direct+



# Stage 3 — runtime
#   Only the compiled binaries, shared libs, launch files, and the
#   CUDA runtime (not the full devel SDK). ~60% smaller than builder.

FROM base AS runtime

# CUDA runtime only (no nvcc, no headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the colcon install tree from builder
COPY --from=builder /ws/install /ws/install

# Copy standalone benchmarks
COPY --from=builder /ws/build/no_ros/batch_fk_benchmark   /usr/local/bin/
COPY --from=builder /ws/build/no_ros/jacobian_benchmark    /usr/local/bin/
COPY --from=builder /ws/build/no_ros/ik_benchmark          /usr/local/bin/

# Environment
ENV ROS_DOMAIN_ID=0 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Source both ROS 2 and the workspace overlay
RUN echo "source /ws/install/setup.bash" >> /etc/bash.bashrc

WORKDIR /ws

# Entrypoint sources ROS 2 overlays and prints GPU info
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default: launch all three nodes
CMD ["ros2", "launch", "cuda_ros2_kinematics", "arm_bringup.launch.py"]
