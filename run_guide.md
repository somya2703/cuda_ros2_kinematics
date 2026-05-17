# Run Guide

GPU-accelerated 6-DOF robot arm kinematics with ROS 2, CUDA, and RViz2 simulation recording.

---

## Setup — Prerequisites

### Step 1 — Install host dependencies

Docker, NVIDIA Container Toolkit, ffmpeg (for screen recording), and X11 utils.

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # then log out and back in

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ct-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Screen recording + X11 tools
sudo apt-get install -y ffmpeg x11-utils
```

### Step 2 — Verify GPU access inside Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

> **Expected:** Your GPU name and driver version. If this fails, fix the toolkit before continuing.

### Step 3 — Extract the project

```bash
unzip cuda_ros2_kinematics.zip
cd cuda_ros2_kinematics
chmod +x record_sim.sh
```

---

## Setup — Clean Slate

### Step 4 — Remove any previous images and create result folders

```bash
docker compose down --remove-orphans
docker rmi cuda_ros2_kinematics:latest cuda_ros2_kinematics:dev 2>/dev/null || true

mkdir -p results/default results/custom trajectories
```

---

## Phase 1 — Build

### Step 5 — Build both images and save the log

Takes 5–15 min first time. Unit tests run automatically inside the builder — the build fails if any test fails.

```bash
docker compose build 2>&1 | tee results/build_$(date +%Y%m%d_%H%M%S).log
```

> **Look for:** `100% tests passed, 0 tests failed` — confirms the kinematics math is correct before anything runs.

---

## Phase 2 — GPU vs CPU Benchmarks

### Step 6 — Capture system info

```bash
{ echo "=== Date ===" && date
  echo "=== Host ===" && uname -a
  echo "=== GPU ===" && nvidia-smi \
    --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader
} | tee results/system_info.txt
```

### Step 7 — Run benchmarks 3× — report the best run

Three runs cancel out OS scheduling noise. Each run prints Batch FK, Jacobian, and IK results.

```bash
for i in 1 2 3; do
  echo "--- run $i ---"
  docker compose run --rm benchmark \
    | tee results/benchmark_run${i}_$(date +%Y%m%d_%H%M%S).txt
done
```

---

## Phase 3a — Simulation: Default Trajectory (sinusoidal sweep)

The arm sweeps all six joints through a sinusoidal pattern. RViz2 opens automatically showing the 3D model. Screen + bag are recorded simultaneously.


### Step 8 — Run the recording script

One command starts RViz2, records the screen to MP4, and records the bag. Let it run **30–60 seconds** then Ctrl+C.

**Terminal 1**
```bash
./record_sim.sh
```

> **Output:** `results/sim_default_<timestamp>.mp4` and `trajectories/default_session/`

### Step 9 — Confirm the bag is recording mid-session

While Terminal 1 is still running, open a second terminal and check the bag folder is growing.

**Terminal 2**
```bash
ls -lh trajectories/default_session/
```

> **Expected:** a growing `.db3` file and a `metadata.yaml`.
>
> **If the folder is missing:** check whether bag recording reached the container:
> `docker compose logs simulate | grep -i "bag\|record"`

### Step 10 — Capture topic rates and a sample message per topic

`ros2 topic hz` only accepts one topic at a time — the loop below handles each separately. First find your container name.

**Terminal 2**
```bash
# Find the running container name
docker compose ps

# Then capture snapshots (replace container name if different)
CONTAINER=$(docker compose ps -q simulate | xargs docker inspect --format '{{.Name}}' | sed 's/\///')
for topic in /joint_states /ee_pose /trajectory; do
  echo "=== $topic ===" >> results/default/topic_snapshot.txt
  docker exec "$CONTAINER" bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /ws/install/setup.bash && \
     ros2 topic hz $topic --window 20 2>&1 | head -5" \
    >> results/default/topic_snapshot.txt
  docker exec "$CONTAINER" bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /ws/install/setup.bash && \
     ros2 topic echo $topic --once" \
    >> results/default/topic_snapshot.txt
done
```

---

## Phase 3b — Simulation: Custom Trajectory

### Step 11 — Create the trajectory CSV

One row per waypoint. Six joint angles in radians. The arm loops through these continuously. Lines starting with `#` are ignored.

```bash
cat > trajectories/custom.csv << 'EOF'
# shoulder_pan  shoulder_lift  elbow   wrist_1  wrist_2  wrist_3
 0.0    0.0     0.0    0.0     0.0    0.0
 0.8   -0.4     0.9   -0.5     0.4   -0.2
 1.2   -0.7     1.3   -0.9     0.6   -0.4
 0.8   -0.4     0.9   -0.5     0.4   -0.2
 0.0    0.0     0.0    0.0     0.0    0.0
-0.8   -0.4     0.9   -0.5    -0.4    0.2
 0.0    0.0     0.0    0.0     0.0    0.0
EOF
```

> **Keep all angles within ±1.5 rad** to avoid singularities and IK divergence.

### Step 12 — Run with custom trajectory at 5 Hz

```bash
./record_sim.sh custom 5
```

> **Output:** `results/sim_custom_<timestamp>.mp4` and `trajectories/custom_session/`

### Step 13 — Capture topic snapshots (custom run)

**Terminal 2**
```bash
CONTAINER=$(docker compose ps -q simulate | xargs docker inspect --format '{{.Name}}' | sed 's/\///')
for topic in /joint_states /ee_pose /trajectory; do
  echo "=== $topic ===" >> results/custom/topic_snapshot.txt
  docker exec "$CONTAINER" bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /ws/install/setup.bash && \
     ros2 topic hz $topic --window 20 2>&1 | head -5" \
    >> results/custom/topic_snapshot.txt
  docker exec "$CONTAINER" bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /ws/install/setup.bash && \
     ros2 topic echo $topic --once" \
    >> results/custom/topic_snapshot.txt
done
```

---

## Phase 4 — Verify Bags

> **Important:** Complete both simulation runs (steps 8 and 12) and press Ctrl+C to finish each before running the verify command — the bag folders only exist after each session is stopped cleanly.

### Step 14 — Inspect both bags and save summaries

```bash
docker compose run --rm simulate_headless bash -c \
  "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && \
   echo '=== DEFAULT SESSION ==' && \
   ros2 bag info /trajectories/default_session && \
   echo '=== CUSTOM SESSION ==' && \
   ros2 bag info /trajectories/custom_session" \
  | tee results/bag_summary.txt
```

> **Look for:** Duration, message count per topic, and all three topics present: `/joint_states`, `/ee_pose`, `/trajectory`.

### Step 15 — Optional: replay and verify a bag

**Terminal 1 — replay**
```bash
docker compose run --rm simulate_headless bash -c \
  "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && \
   ros2 bag play /trajectories/default_session"
```

**Terminal 2 — watch replayed data**
```bash
docker compose exec $(docker compose ps -q simulate_headless) bash -c \
  "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && \
   ros2 topic echo /ee_pose"
```

---

## Results — Final Folder Layout

```
cuda_ros2_kinematics/
│
├── results/
│   ├── system_info.txt           ✓ hardware record
│   ├── build_<timestamp>.log     ✓ test pass proof
│   ├── benchmark_run1_<ts>.txt   ✓ GPU vs CPU
│   ├── benchmark_run2_<ts>.txt   ✓ GPU vs CPU
│   ├── benchmark_run3_<ts>.txt   ✓ GPU vs CPU
│   ├── bag_summary.txt           ✓ bag metadata
│   ├── sim_default_<ts>.mp4      ▶ screen recording
│   ├── sim_custom_<ts>.mp4       ▶ screen recording
│   ├── default/
│   │   └── topic_snapshot.txt   ✓ hz + sample msgs
│   └── custom/
│       └── topic_snapshot.txt   ✓ hz + sample msgs
│
└── trajectories/
    ├── custom.csv                ✓ waypoint file
    ├── default_session/          ⬡ ROS 2 bag
    └── custom_session/           ⬡ ROS 2 bag
```

---

