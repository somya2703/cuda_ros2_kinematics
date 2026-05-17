#!/usr/bin/env bash
# record_sim.sh
# Usage:
#   ./record_sim.sh                # default sinusoidal sweep
#   ./record_sim.sh custom         # uses trajectories/custom.csv
#   ./record_sim.sh custom 5       # custom CSV at 5 Hz

MODE="${1:-default}"
RATE="${2:-10.0}"
STAMP=$(date +%Y%m%d_%H%M%S)
BAG_NAME="${MODE}_session"

if [ "$MODE" = "custom" ]; then
    echo "Using custom trajectory at ${RATE} Hz"
else
    echo "Using default sinusoidal sweep at ${RATE} Hz"
fi

mkdir -p results trajectories

# Write .env for docker compose variable substitution
if [ "$MODE" = "custom" ]; then
    TRAJ_FILE="/trajectories/custom.csv"
else
    TRAJ_FILE=""
fi

cat > .env << EOF
PUBLISH_RATE_HZ=${RATE}
TRAJECTORY_FILE=${TRAJ_FILE}
BAG_NAME=${BAG_NAME}
RECORD=false
EOF

xhost +local:docker 2>/dev/null || true

RESOLUTION=$(xdpyinfo 2>/dev/null | grep dimensions | awk '{print $2}' || echo "1920x1080")
OUTPUT="results/sim_${MODE}_${STAMP}.mp4"

echo ""
echo "=== Starting simulation + screen recording ==="
echo "Output : ${OUTPUT}"
echo "Bag    : trajectories/${BAG_NAME}/"
echo "Press Ctrl+C to stop."
echo ""

# Screen recording
ffmpeg -loglevel warning \
    -f x11grab \
    -video_size "${RESOLUTION}" \
    -framerate 30 \
    -i "${DISPLAY:-:0}.0" \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
    "${OUTPUT}" &
FFMPEG_PID=$!

# Start simulation
docker compose up -d simulate

echo "Waiting for nodes to start (15s)..."
sleep 15

# Start bag recording — write a small wrapper script inside the container
# so backgrounding happens entirely within the container shell
docker compose exec simulate bash -c "
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
ros2 bag record --output /trajectories/${BAG_NAME} /joint_states /ee_pose /trajectory > /tmp/bag.log 2>&1 &
echo \$! > /tmp/bag.pid
sleep 1
if kill -0 \$(cat /tmp/bag.pid) 2>/dev/null; then
    echo 'Bag recording running (PID '\$(cat /tmp/bag.pid)')'
    head -5 /tmp/bag.log
else
    echo 'ERROR: bag record failed to start'
    cat /tmp/bag.log
fi
"

echo ""
echo "Ctrl+C to stop."

# Follow logs
docker compose logs -f simulate &
LOGS_PID=$!
wait $LOGS_PID

# Cleanup
echo ""
echo "Stopping bag..."
docker compose exec simulate bash -c \
    "[ -f /tmp/bag.pid ] && kill \$(cat /tmp/bag.pid) 2>/dev/null; sleep 2" 2>/dev/null || true

echo "Stopping simulation..."
docker compose stop simulate 2>/dev/null || true
docker compose rm -f simulate 2>/dev/null || true

echo "Stopping screen recording..."
kill "$FFMPEG_PID" 2>/dev/null || true
wait "$FFMPEG_PID" 2>/dev/null || true

xhost -local:docker 2>/dev/null || true

echo ""
echo "=== Done ==="
ls -lh "${OUTPUT}" 2>/dev/null || true
ls -lh "trajectories/${BAG_NAME}/" 2>/dev/null || true
