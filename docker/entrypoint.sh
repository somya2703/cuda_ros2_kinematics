#!/usr/bin/env bash
# entrypoint.sh
# Sources ROS 2 and the workspace overlay, then exec's whatever command
# was passed (default CMD from Dockerfile, or an override from compose/docker run).

set -e

# Source ROS 2 base
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash

# Source workspace overlay if it exists (build stage may not have it)
if [ -f /ws/install/setup.bash ]; then
    # shellcheck source=/dev/null
    source /ws/install/setup.bash
fi

# Print GPU info on startup (informational, non-fatal)
if command -v nvidia-smi &>/dev/null; then
    echo "── GPU ──────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
               --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{printf "  %s | driver %s | %s MiB VRAM\n", $1,$2,$3}'
    echo "─────────────────────────────────────────────────────────────"
fi

exec "$@"
