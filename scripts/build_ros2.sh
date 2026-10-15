#!/usr/bin/env bash
# Build the ROS executable with the same Python interpreter as the inference models.
set -euo pipefail

vlnav_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vlnav_python="${1:-${VLNAV_PYTHON:-python3}}"
vlnav_ros_distro="${ROS_DISTRO:-humble}"
set +u  # ROS environment scripts reference optional unset variables.
source "/opt/ros/${vlnav_ros_distro}/setup.bash"
set -u
cd "$vlnav_repo_dir"

"$vlnav_python" - <<'PY'
import sys
try:
    import rclpy
    import numpy
    import pydantic
except ImportError as exc:
    raise SystemExit(
        f"{sys.executable} cannot load the ROS/model dependencies: {exc}\n"
        "Pass your model environment's Python to this script (Python 3.10 for Humble)."
    )
print(f"Building ROS 2 workspace with {sys.executable}", flush=True)
PY

# Ubuntu's colcon is installed outside Conda. Keep the model environment's modules
# ahead of Ubuntu modules so its numpy/typing_extensions versions are preserved.
vlnav_python_site="$("$vlnav_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
vlnav_package_roots=(ros2)
if [[ -f vr_common_msgs/package.xml ]]; then
    vlnav_package_roots+=(vr_common_msgs)
fi
PYTHONPATH="$vlnav_python_site:${PYTHONPATH:-}:/usr/lib/python3/dist-packages" \
    "$vlnav_python" /usr/bin/colcon build --base-paths "${vlnav_package_roots[@]}" \
    --cmake-clean-cache --cmake-args \
    "-DPYTHON_EXECUTABLE=$vlnav_python" "-DPython3_EXECUTABLE=$vlnav_python"

printf '\nBuild complete. In your zsh terminal run:\n'
printf '  source /opt/ros/%s/setup.zsh\n' "$vlnav_ros_distro"
printf '  source "%s/install/setup.zsh"\n' "$vlnav_repo_dir"
printf '  PYTHONPATH="%s:${PYTHONPATH:-}" ros2 launch vlnav_ros vlnav.launch.py instruction:="Find a sofa"\n' "$vlnav_repo_dir"
