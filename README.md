# cuda_ros2_kinematics

**GPU-Accelerated Robot Arm Kinematics with ROS 2**

A 6-DOF robot arm stack where joint states flow over ROS 2 topics, a pure C++ kinematics library handles FK/IK math, and the heavy computation runs on CUDA kernels. Each layer is independently testable and clearly separated.

---


## Benchmark Results
 
Tested on **NVIDIA GeForce RTX 4050 Laptop GPU** (6 GB VRAM, SM 8.9) · Ubuntu 24.04 · CUDA driver 595.58.03
 
### Batch Forward Kinematics (CPU vs CUDA)
 
| Configurations (N) | CPU | GPU | Speedup |
|---|---|---|---|
| 1,000 | 0.200 ms | 0.110 ms | **1.8×** |
| 5,000 | 1.018 ms | 0.187 ms | **5.4×** |
| 10,000 | 2.055 ms | 0.289 ms | **7.1×** |
| 50,000 | 11.599 ms | 1.135 ms | **10.2×** |
| 100,000 | 23.571 ms | 2.252 ms | **10.5×** |
 
> FK scales well because each configuration is fully independent. The kernel launches N threads, one per configuration, with no inter-thread communication.
 
### Jacobian Computation (CPU vs CUDA)
 
| Configurations (N) | CPU | GPU | Speedup |
|---|---|---|---|
| 1,000 | 0.206 ms | 0.138 ms | **1.5×** |
| 5,000 | 1.052 ms | 0.359 ms | **2.9×** |
| 10,000 | 2.108 ms | 0.595 ms | **3.6×** |
| 50,000 | 10.586 ms | 2.656 ms | **4.0×** |

> Jacobian speedup is lower than FK because each evaluation requires 6 finite-difference FK calls per configuration, increasing per-thread register pressure and limiting occupancy.
 
### Inverse Kinematics Solver (CPU vs CUDA, gradient descent)
 
| Queries (M) | CPU | GPU | Speedup | Convergence |
|---|---|---|---|---|
| 50 | 0.905 ms | 1.892 ms | 0.5× | 45/50 |
| 100 | 1.792 ms | 1.894 ms | ~1.0× | 90/100 |
| 500 | 8.814 ms | 1.917 ms | **4.6×** | 458/500 |
| 1,000 | 17.648 ms | 1.927 ms | **9.2×** | 914/1,000 |
 
> IK is the most interesting case: GPU is *slower* at M=50 and breaks even at M=100 because the PCIe transfer cost dominates. Above M≈200 the parallelism pays off and by M=1000 it is 9× faster. Each IK problem runs gradient descent independently on its own thread, so latency is nearly constant on the GPU regardless of M.
 
---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      ROS 2 Layer                        │
│  joint_command_pub  trajectory_planner  state_publisher │
│   /joint_states        /trajectory        /ee_pose      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│               KinematicsSolver (C++ lib)                │
│       forward_kinematics()   inverse_kinematics()       │
└──────────┬────────────────────────────────┬─────────────┘
           │                                │
┌──────────▼────────────────────────────────▼─────────────┐
│                     CUDA Layer                          │
│  batch_fk_kernel   jacobian_kernel   ik_solver_kernel   │
│              NVIDIA GPU · CUDA threads                  │
└─────────────────────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for full details.

---
## Why These Design Choices?
 
### Batch size matters: PCIe is the bottleneck at small N
 
Every GPU call has a fixed overhead: data transfer over PCIe (host → device and device → host), kernel launch latency (~5–10 µs), and driver scheduling. At N=1000, FK takes only 0.11 ms total on the GPU but the transfer cost is a significant fraction of that. At N=100,000 the compute work dominates and the transfer is amortized, hence the 10.8× speedup. This is why the benchmark sweeps N across two orders of magnitude. A single-point benchmark at small N would give a misleading picture of GPU utility.
 
### FK kernel: one thread per configuration
 
Each configuration is a 6-joint DH chain evaluated independently via `batch_fk_kernel`. There is zero cross-thread dependency, which means:
- No shared memory synchronisation needed
- No warp divergence from conditional paths
- Thread count scales linearly with N (occupancy stays high)
This makes FK the cleanest GPU win in the codebase.
 
### Jacobian via finite differences, not analytical
 
The Jacobian is computed numerically (6 FK evaluations per joint, perturbed by ε). An analytical Jacobian would be faster on CPU but the numerical approach lets the GPU reuse the already verified FK kernel directly, keeping the two implementations consistent and independently testable. The cost is ~6× more kernel work per configuration, which shows up in the lower Jacobian speedups compared to FK.
 
### IK solver: gradient descent on GPU
 
The IK solver runs per-target gradient descent iterations fully on the GPU, avoiding round trips per iteration. CPU IK is faster for a single target (less than ~100 ms) because serial scalar code has near-zero overhead. The GPU version is designed for motion planning samplers that need to solve hundreds of IK problems in parallel — the use case where it is 9× faster.
 
### Pure C++ kinematics library (no ROS dependency)
 
`kinematics_lib/` has no ROS headers. This was intentional: it can be unit tested with Google Test, benchmarked standalone, and linked into non-ROS tools without a ROS workspace. ROS 2 nodes in `ros2_nodes/` call into the library through a clean API (`forward_kinematics(joints)`, `inverse_kinematics(target_pose)`). The separation also means the CUDA kernels are exercised directly in benchmarks rather than through the full ROS stack.
 
---

## The Math

### DH Parameters (6-DOF arm)

Forward kinematics is computed by chaining Denavit–Hartenberg transforms:

```
T_0^n = T_0^1 · T_1^2 · ... · T_(n-1)^n
```

Each DH matrix:

```
T_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)
    = | cos θ_i  -sin θ_i·cos α_i   sin θ_i·sin α_i   a_i·cos θ_i |
      | sin θ_i   cos θ_i·cos α_i  -cos θ_i·sin α_i   a_i·sin θ_i |
      |    0          sin α_i           cos α_i             d_i     |
      |    0              0                 0                1      |
```

Default DH table (UR5-like, all values in metres/radians):

| Joint | a (m) | d (m)  | α (rad) | θ offset |
|-------|-------|--------|---------|---------|
| 1     | 0.000 | 0.0892 | π/2     | 0       |
| 2     | 0.425 | 0.000  | 0       | 0       |
| 3     | 0.392 | 0.000  | 0       | 0       |
| 4     | 0.000 | 0.1093 | π/2     | 0       |
| 5     | 0.000 | 0.0950 | −π/2    | 0       |
| 6     | 0.000 | 0.0820 | 0       | 0       |

### Jacobian

The geometric Jacobian J(q) ∈ ℝ^(6×n) is computed numerically via central finite differences on the GPU:

```
∂p/∂q_i ≈ (FK(q + ε·e_i) − FK(q − ε·e_i)) / (2ε)
```

with ε = 1×10⁻⁵ rad.

### Gradient-Descent IK

```
q_{k+1} = q_k − α · J^T(q_k) · e(q_k)
```

where e(q) is the 6-DOF pose error (position + rotation vector). Convergence threshold: ‖e‖ < 1×10⁻⁴, max 500 iterations.

---

## ROS 2 Topics
 
| Topic | Type | Rate (default) | Rate (custom) |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | 10 Hz | 5 Hz |
| `/ee_pose` | `geometry_msgs/msg/PoseStamped` | 10 Hz | 5 Hz |
| `/trajectory` | `trajectory_msgs/msg/JointTrajectory` | 10 Hz | 5 Hz |
 
Joints: `shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_2`, `wrist_3`
 
---

## Quick Start

### Docker (recommended)

The fastest way to run the project — no local CUDA or ROS 2 install needed.

```bash
# Prerequisites: Docker + NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Run all three ROS 2 nodes
docker compose up kinematics

# Print CPU vs GPU benchmark table
docker compose run --rm benchmark

# Interactive dev shell (source live-mounted)
docker compose run --rm dev bash
```

See [`docs/docker.md`](docs/docker.md) for full details, RViz setup, CI integration, and troubleshooting.

---

### Build from source

#### Prerequisites

- CMake ≥ 3.18
- CUDA Toolkit ≥ 12.0
- Eigen3
- Google Test (for unit tests)
- ROS 2 Humble or later (for the ROS 2 layer only)

### Build without ROS 2

```bash
mkdir build && cd build
cmake .. -DBUILD_ROS2=OFF
make -j$(nproc)
# Run unit tests
ctest --output-on-failure
# Run CUDA benchmarks
./cuda_kernels/benchmarks/batch_fk_benchmark
```

### Build with ROS 2

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select cuda_ros2_kinematics
source install/setup.bash
ros2 launch ros2_nodes arm_bringup.launch.py
```

---

## Project Structure

```
cuda_ros2_kinematics/
├── kinematics_lib/          # Pure C++ lib ( no ROS dependency )
│   ├── include/
│   │   └── kinematics_lib/
│   │       ├── kinematics_solver.hpp
│   │       └── dh_parameters.hpp
│   ├── src/
│   │   ├── kinematics_solver.cpp
│   │   └── dh_parameters.cpp
│   └── tests/
│       ├── test_fk.cpp
│       └── test_ik.cpp
├── cuda_kernels/            # .cu files + benchmarks
│   ├── batch_fk_kernel.cu
│   ├── jacobian_kernel.cu
│   ├── ik_solver_kernel.cu
│   ├── cuda_kinematics.hpp
│   └── benchmarks/
│       ├── batch_fk_benchmark.cpp
│       ├── jacobian_benchmark.cpp
│       └── ik_benchmark.cpp
├── ros2_nodes/              # ROS 2 package
│   ├── src/
│   │   ├── joint_command_publisher.cpp
│   │   ├── trajectory_planner.cpp
│   │   └── state_publisher.cpp
│   ├── launch/
│       └── arm_bringup.launch.py
│   
├── docs/
│   └── architecture.md
|    └── Docker.md
├── CMakeLists.txt
├── docker-compose.yml
├── Dockerfile
├── package.xml
├── record_sim.sh
├── run_guide.md
└── README.md
```

---

## Hardware Tested On
 
```
Host:   Linux 6.8.0-111-generic (Ubuntu) x86_64
GPU:    NVIDIA GeForce RTX 4050 Laptop GPU
VRAM:   6141 MiB
Driver: 595.58.03
SM:     8.9
```
 
---
