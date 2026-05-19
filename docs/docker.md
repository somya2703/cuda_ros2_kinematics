# Docker Guide
## Image Architecture

```
nvidia/cuda:12.4.1-devel-ubuntu22.04
           │
           ▼
        [base]
   ROS 2 Humble + message pkgs
   locale, rosdep
           │
    ┌──────┴──────┐
    ▼             ▼
[builder]      (same base in runtime)
cmake, gtest    libeigen3
nvcc, eigen     CUDA runtime only
    │
    │  compiles & tests
    │
    ▼
[runtime]  ← final image (~40% smaller)
  /ws/install  (colcon overlay)
  benchmarks in /usr/local/bin
```

The multi-stage build means the final runtime image does **not** contain nvcc, CMake, GTest sources, or the full CUDA devel headers.

---

## Quick Start

### Run the full arm stack

```bash
# Build and start all three ROS 2 nodes
docker compose up kinematics

# With a custom trajectory file
TRAJECTORY_FILE=/trajectories/my_traj.csv \
PUBLISH_RATE_HZ=20 \
docker compose up kinematics
```

### Run the CUDA benchmarks

```bash
docker compose run --rm benchmark
```

### Open a development shell

```bash
docker compose run --rm dev bash
# Inside the container — source tree is live-mounted at /ws/src/cuda_ros2_kinematics
# Rebuild just the library:
cd /ws && colcon build --packages-select cuda_ros2_kinematics
```

---

## Building Manually

```bash
# Build the runtime image
docker build --target runtime -t cuda_ros2_kinematics:latest .

# Build the dev image (includes all build tools)
docker build --target builder -t cuda_ros2_kinematics:dev .

# Run with GPU access
docker run --rm --gpus all cuda_ros2_kinematics:latest
```

---

## RViz (GUI forwarding)

On Linux with X11:

```bash
xhost +local:docker
docker compose up kinematics
# RViz will open on your display
xhost -local:docker   # revoke when done
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `TRAJECTORY_FILE` | *(empty)* | Path inside container to a trajectory CSV |
| `PUBLISH_RATE_HZ` | `10.0` | Joint state publish rate |
| `ROS_DOMAIN_ID` | `0` | Isolate ROS 2 DDS traffic |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` | DDS middleware |
| `DISPLAY` | `:0` | X11 display for RViz |

---

## Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `./trajectories/` | `/trajectories` | Trajectory CSV files |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | RViz X11 socket |
| `dev-build-cache` (named) | `/ws/build` | Persistent incremental builds in dev shell |

---

## Multi-GPU Systems

To restrict to a specific GPU:

```bash
NVIDIA_VISIBLE_DEVICES=1 docker compose up kinematics
```

Or pin in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["1"]
          capabilities: [gpu]
```
colcon build --packages-select cuda_ros2_kinematics \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON
```
