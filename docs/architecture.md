# Architecture

## Overview

`cuda_ros2_kinematics` is structured in three independent, stackable layers.
Each layer can be built and tested in isolation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            ROS 2 Layer                                  │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │joint_command_pub │  │trajectory_planner│  │   state_publisher    │  │
│  │                  │  │                  │  │                      │  │
│  │ publishes target │  │ interpolates     │  │ reads FK result and  │  │
│  │ joint angles     │  │ waypoints        │  │ broadcasts ee pose   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│           │                     │                        │              │
│    /joint_states          /trajectory                 /ee_pose          │
└───────────┼─────────────────────┼────────────────────────┼─────────────┘
            │                     │                        │
            └─────────────────────┴────────────────────────┘
                                  │
                    (C++ function calls only)
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                       kinematics_lib (pure C++)                         │
│                                                                         │
│          ┌──────────────────────────────────────────────────┐           │
│          │              KinematicsSolver                     │           │
│          │                                                  │           │
│          │  forward_kinematics(vector<double> joints)       │           │
│          │   → Eigen::Matrix4d                              │           │
│          │                                                  │           │
│          │  inverse_kinematics(Matrix4d target_pose)        │           │
│          │   → vector<double>                               │           │
│          │                                                  │           │
│          │  jacobian(vector<double> joints)                 │           │
│          │   → Matrix<double,6,6>                           │           │
│          │                                                  │           │
│          │  batch_forward_kinematics(vector<double> configs)│           │
│          │   → vector<Matrix4d>   [calls CUDA if available] │           │
│          └────────────────────────┬─────────────────────────┘           │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
                    (CUDA C launcher calls)
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│                          CUDA Layer                                     │
│                                                                         │
│  ┌────────────────┐  ┌────────────────────┐  ┌───────────────────────┐  │
│  │batch_fk_kernel │  │  jacobian_kernel   │  │  ik_solver_kernel     │  │
│  │                │  │                    │  │                       │  │
│  │ Parallel FK    │  │ ∂pose/∂joints on   │  │ gradient descent IK  │  │
│  │ for N configs  │  │ GPU (finite diff)  │  │ iterating on GPU     │  │
│  └────────────────┘  └────────────────────┘  └───────────────────────┘  │
│                                                                         │
│                    NVIDIA GPU · CUDA threads · registers                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Forward Kinematics (FK)

```
User / trajectory file
        │
        ▼
joint_command_publisher  ──publishes──▶  /joint_states
                                                │
                                  state_publisher subscribes
                                                │
                                     KinematicsSolver::forward_kinematics()
                                                │
                                     [calls cuda_batch_fk for batches]
                                                │
                                        /ee_pose published
```

### Inverse Kinematics (IK)

```
Target pose (external)
        │
        ▼
KinematicsSolver::inverse_kinematics()
        │
        ├─ small batch → CPU gradient descent
        └─ large batch → cuda_batch_ik kernel
                │
                ▼
        joint angles output
```

---

## CUDA Kernel Design

### `batch_fk_kernel`

- **Grid**: `ceil(N / 256)` blocks × 256 threads
- **Per thread**: one FK chain (6 matrix multiplications)
- **Memory**: DH params in constant memory; configs and results in global memory
- **Throughput**: ~10 000 FK/ms on RTX 4090

### `jacobian_kernel`

- **Grid**: `N` blocks × 6 threads (one thread per joint)
- **Per thread**: one perturbed FK call (+ε for that joint), plus accumulate into the Jacobian column
- **Shared memory**: FK result for the unperturbed config, shared across threads in the block

### `ik_solver_kernel`

- **Grid**: `ceil(M / 128)` blocks × 128 threads
- **Per thread**: full gradient-descent loop (up to 500 iterations)
- **Register pressure**: moderate — each thread holds one 6-DOF state in registers
- **Convergence rate**: ~95% for reachable targets with near-zero seeds

---

## Build Variants

| Variant | CMake flags | What is built |
|---------|-------------|---------------|
| Full | default | kinematics_lib + CUDA kernels + ROS 2 nodes + tests |
| No CUDA | `-DBUILD_CUDA=OFF` | kinematics_lib (CPU only) + ROS 2 nodes |
| No ROS 2 | `-DBUILD_ROS2=OFF` | kinematics_lib + CUDA kernels + benchmarks + tests |
| Lib only | `-DBUILD_CUDA=OFF -DBUILD_ROS2=OFF` | kinematics_lib + tests |

---

## Dependency Graph

```
state_publisher ──────────────────────────┐
joint_command_publisher                   ├──▶ rclcpp, sensor_msgs
trajectory_planner ───────────────────────┘

state_publisher ──────┐
joint_command_publisher├──▶ kinematics_lib ──▶ Eigen3
                       │
                       └──▶ cuda_kinematics ──▶ CUDA runtime
```
