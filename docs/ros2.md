# ROS 2 camera/LiDAR goal node

`ros2/vlnav_ros/vlnav_ros/node.py` runs the existing perception → occupancy grid → unified
memory → Qwen task planner/acquisition → exploration/path planner on live ROS topics. It
sends selected goals to the robot's `vr_common_msgs/action/NavigateToPose` server and keeps
`/exploration/goal` for debugging. It does not publish `/cmd_vel`.

## Install and launch (ROS 2 Humble, Python 3.10)

Run from the repository root. Install the model checkpoints described in the main README first.
Use an editable core install: the core resolves `configs/` and `checkpoint/` relative to this
checkout. The ROS package is a separate ament Python package.

```bash
source /opt/ros/humble/setup.bash  # use setup.zsh in zsh
/usr/bin/python3 -m venv --system-site-packages .venv-ros
source .venv-ros/bin/activate
python -m pip install -e '.[io,perception,vlm,dev]'
bash scripts/build_ros2.sh "$(command -v python)"
source install/setup.bash        # setup.zsh in zsh
PYTHONPATH=.:$PYTHONPATH ros2 launch vlnav_ros vlnav.launch.py instruction:='Find a sofa'
```

An existing Python 3.10 environment containing the model dependencies and ROS Python bindings
can also be used. Build with that environment's Python so the installed `vlnav_node` executable
uses the same interpreter. ROS Humble's compiled Python bindings cannot run in Python 3.13.
For the existing `fastvggt` environment on this machine:

```zsh
bash scripts/build_ros2.sh /home/lenguyen1/miniconda3/envs/fastvggt/bin/python
source /opt/ros/humble/setup.zsh
source install/setup.zsh
ros2 pkg prefix vlnav_ros
PYTHONPATH=.:$PYTHONPATH ros2 launch vlnav_ros vlnav.launch.py instruction:='Find a sofa'
```

The build script handles Ubuntu's colcon modules when using Conda and puts the package in
`install/`. Source that overlay in **each new terminal**. If the error says it searched only
`['/opt/ros/humble']`, the project overlay is not sourced. The earlier verification build under
`runs/ros2_build_check/` is separate from this normal installation.

For development, you can also run directly from the checkout in your existing Python 3.10
model environment without building the ament package:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH=.:ros2/vlnav_ros:$PYTHONPATH python -m vlnav_ros.node \
  --ros-args -p instruction:='Find a sofa'
# In another sourced terminal:
rviz2 -d ros2/vlnav_ros/rviz/vlnav.rviz
```

The default model/device settings still come from `configs/perception.yaml`,
`configs/acquisition.yaml`, `configs/checkpoints.yaml` and `configs/vlm.yaml`. In particular,
the default local Qwen configuration uses GPU **1**; override `vlm.local.devices: [0]` in your
run config for a single GPU. The live overrides are in `configs/runs/vr_h3_2_ros2.yaml`.

Wait for the `Ready` log before starting sensors or replay. Without `instruction`, the node
builds the map and memory and waits for a task. Submit or replace the task at any time:

```bash
ros2 topic pub --once /task_instruction std_msgs/msg/String "{data: 'Find a sofa'}"
# Empty instruction cancels the task and clears the goal.
ros2 topic pub --once /task_instruction std_msgs/msg/String "{data: ''}"
```

To run without opening RViz, add `rviz:=false`. This does not disable publishing the map/markers.
Direct entry, with all node parameters and ROS remapping available:

```bash
ros2 run vlnav_ros vlnav_node --ros-args -p instruction:='Find a sofa'
```

## Inputs matching vr_h3_2_calib_1

Defaults are read from `configs/bags/vr_h3_2_calib_1.yaml`; its `bag_dir` is not opened by the
live node. A connected robot must publish the same inputs (including odometry TF):

| Topic | Type | Purpose |
|---|---|---|
| `/camera/waist_front_zed_onboard/left/color/rect/image` | `sensor_msgs/msg/Image` | RGB image |
| `/camera/waist_front_zed_onboard/left/color/rect/image/camera_info` | `sensor_msgs/msg/CameraInfo` | Camera intrinsics |
| `/livox/lidar` | `sensor_msgs/msg/PointCloud2` | XYZ points, in the message's sensor frame |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | `odom_lidar` to `base_link`, camera and LiDAR transforms |
| `/task_instruction` | `std_msgs/msg/String` | Natural-language task; optional if supplied at launch |

This adapter supports the bag's raw image and PointCloud2 messages, not Livox CustomMsg or a
depth stream. Camera/LiDAR subscriptions use sensor-data QoS (best effort, volatile), compatible
with the recorded publishers. For different topic names, use ROS remapping or a copy of the
sensor YAML. TF topic names are changed with `-r /tf:=... -r /tf_static:=...`.

The bag lacks the camera optical frame. By default, the node broadcasts the configured static
transform from `waist_front_zed_onboard_left_camera_frame` to
`waist_front_zed_onboard_left_camera_frame_optical`: roll/pitch/yaw `[-90, 0, -90]` degrees.
Set `publish_extra_static_tf:=false` as a node parameter if your robot already publishes it.

Images are sampled at 2 Hz, with a 0.3 s arrival wait and a 1 s past LiDAR window, as in the bag
config. Each scan is transformed at **its own** timestamp. Camera and base transforms are
looked up at the image timestamp; there is no fallback to an unrelated latest transform.
Sensor buffers are bounded and older images are skipped when inference cannot keep up.
No frame is processed without camera calibration, usable LiDAR and TF.

## Outputs and RViz

All geometry uses `header.frame_id: odom_lidar`, in meters. Goals preserve the core's full
position and quaternion (including yaw). They are robot navigation poses, not object centroids.
For a selected object, the live node chooses a reachable free map cell 0.6–1.2 m from its
centroid and points the goal heading toward the object. The saved best-view camera pose remains
evidence for object verification; it is not sent as the live navigation goal. If the map has
no reachable approach cell yet, `goal_active` stays false while mapping continues.

| Topic | Type | Contents |
|---|---|---|
| `/exploration/goal` | `geometry_msgs/msg/PoseStamped` | Selected robot goal `(x, y, z)` + quaternion |
| `/exploration/goal_point` | `geometry_msgs/msg/PointStamped` | The same goal as an XYZ point |
| `/exploration/goal_active` | `std_msgs/msg/Bool` | Whether a current reachable goal is available |
| `/exploration/navigation_action_status` | `std_msgs/msg/String` | JSON action acceptance, feedback and result |
| `/exploration/path` | `nav_msgs/msg/Path` | Planned XY path, with pose headings |
| `/mapping/occupancy` | `nav_msgs/msg/OccupancyGrid` | Unknown `-1`, free `0`, occupied `100` |
| `/perception/annotated_image` | `sensor_msgs/msg/Image` | Processed camera frame with detector boxes, labels, and scores (`bgr8`) |
| `/exploration/markers` | `visualization_msgs/msg/MarkerArray` | Orange goal disk, heading arrow, label |
| `/exploration/frontiers` | `visualization_msgs/msg/MarkerArray` | Cyan frontier candidate points; gray when temporarily blacklisted |
| `/exploration/robot_markers` | `visualization_msgs/msg/MarkerArray` | Blue robot heading arrow and `ROBOT` label; clears when pose is stale |
| `/planner/status` | `std_msgs/msg/String` | JSON task state, input wait state and goal validity |
| `/planner/report` | `std_msgs/msg/String` | Existing `TargetReport` serialized as JSON when reached |
| `/move/markers` | `visualization_msgs/msg/MarkerArray` | Move subtasks only: S2 camera ray, projected floor point, S2 text |

The map defaults to 1 Hz; navigation output uses `exploration.select_rate_hz` (2 Hz by default).
Map, annotated camera image, path, goal/frontier/robot markers and status use reliable transient-local QoS so RViz can join after startup.
The map's origin and dimensions update as the grid grows. The included RViz config uses
`odom_lidar` as Fixed Frame and follows `base_link` in a top-down view: free space is white,
occupied space black, unknown space gray, frontier points cyan, chosen goal orange and robot blue.
The **Annotated camera** Image display and both **Frontier points** and **Selected goal**
MarkerArray displays are enabled by default. Yellow camera boxes match the active task;
cyan boxes are other detections. The image topic publishes for every processed camera frame,
even when disk image saving is disabled.
Its narrow top panel shows the instruction, current mode and target (PLAN, REPLAN, EXPLORE or
GOTO), and the latest planner reason. Long lines are shortened to fit the image. The saved
`latest_camera.jpg` uses the same annotated frame.
For an existing RViz session, add a MarkerArray display for `/exploration/frontiers`, with
Reliable / Transient Local QoS, or reload the installed `vlnav.rviz` configuration.

Frontier points come from the exploration module's representative candidates in world XY;
they are displayed slightly above the z=0 map and update whenever the frontier pool changes.
They remain visible with no active task. An empty frontier pool clears the old dots.
`/planner/status` includes the current count as `frontier_points`.
The goal disk, arrow and label are also projected above the map for the top-down view, so a
negative robot/goal altitude cannot hide them underneath the occupancy surface. The published
navigation goal and path retain their original positions and orientations.

The node emits no goal until the planner has an active subtask and a path is available.
While waiting, replanning, cancelling, finished, or missing fresh geometry, it publishes
`goal_active=false`, an empty path and selected-goal marker deletion. Frontier markers are
independent of goal validity. `PoseStamped` has no cancellation field:
downstream consumers must check `goal_active` and message freshness instead of retaining an old
goal forever. Goal topics themselves are volatile. Marker lifetimes also remove abandoned goals.

The node sends each selected goal once to `vr_navigate_to_pose` by default. The action pose is
transformed from the exploration frame into `map` through TF; if that transform is missing, the
debug goal topic still publishes but no action is sent. A changed goal cancels the previous action,
and `goal_active=false` cancels the current action. Repeated publications of the same goal do not
create repeated action requests. `waypoint_name` defaults to `use_goal_pose`; `behavior_tree`
defaults to an empty string.
Set `nav_action_name:=/your/action/name` if your robot uses a different action endpoint, or
`nav_action_enabled:=false` for visualization-only replay. The action status topic uses the
`action_goal_response`, `action_feedback` and `action_result` JSON event names.

## Move subtasks (DualVLN routes)

Route parts of an instruction ("walk down the corridor and stop at the glass door") can run on
DualVLN instead of exploration (plans/TASK_move_subtask.md). Start the S2 server first, then the
node with a run config that sets `move.enabled: true`:

```bash
env -u LD_LIBRARY_PATH ~/miniconda3/envs/internnav/bin/python scripts/serve_dualvln_s2.py \
    --internnav ~/hoangpqn/vln/InternNav \
    --aux-ckpt-dir ~/hoangpqn/vr_vln_foca/main_baseline/InternNav/checkpoints \
    --model checkpoint/vln/InternVLA-N1-DualVLN --devices 0 --port 8001
PYTHONPATH=.:$PYTHONPATH ros2 launch vlnav_ros vlnav.launch.py \
    run_config:=configs/runs/vr_h3_2_ros2_move.yaml
```

DualVLN takes GPU 0 (~17.4 GB), so that config puts perception on GPU 1 next to Qwen3-VL. During
a `move` subtask:

- `/exploration/goal` (and the navigation action) carries the reachable goal of the last S2 pixel
  goal, with `goal_kind: move`. A goal kept by hysteresis keeps its id and is not resent.
- `/perception/annotated_image` marks the S2 pixel goal (magenta) and adds a fourth panel line with
  the S2 replies and the decision. The pixel was asked on an earlier frame, about 1 s old.
- `/move/markers` (MarkerArray, transient local) shows the camera ray to the floor and the projected
  floor point (orange) with the S2 text. Saved map PNGs draw the same ray.
- `/planner/status` gains `move`: cycles, in-flight query, consecutive failures, last decision and
  S2 texts.

S2 queries never block the pipeline thread (the node forces `move.s2.blocking: false`). DualVLN
STOP ends the route (`reached`); `nav_failed` / `timeout` make the planner replan.

**Rotation [VERIFY on the robot].** `←` / `→` replies give a goal at the robot's position with a
turned heading (`move.actions.rotate_mode: in_place`). If the robot's `NavigateToPose` server
ignores yaw within its xy tolerance, the robot will not turn. In that case set
`rotate_mode: offset`, which puts the goal `rotate_offset_m` ahead in the new heading.

## Headless map images

To inspect the map over SSH without opening RViz, enable PNG output at launch:

```zsh
ros2 launch vlnav_ros vlnav.launch.py use_sim_time:=true rviz:=false \
  save_map_images:=true instruction:='Find a sofa'
```

`save_map_images` defaults to `false`. Once a map exists, the node saves the published occupancy
map with frontier points, a blue `ROBOT` position/heading arrow, path, and active goal every second of wall time. The files are
`runs/ros2/map_images/latest.png` and successive `frame_000001.png`, `frame_000002.png`, etc.
`latest.png` is replaced each time; numbered frames accumulate for later review. Set
`map_image_dir:=/path/to/output` to change the folder or `map_image_every_s:=5.0` to save less
often. This works without RViz or a display server.

The same switch also saves the annotated camera topic image under `runs/ros2/map_images/camera/`
(or under `map_image_dir/camera/`). `camera_<stamp>.jpg` contains each processed camera frame with
all detector boxes and labels; `latest_camera.jpg` shows the newest one. When a
`goto` object is selected, `goal_<task>_<subtask>_node_<id>.jpg` and `latest_goal.jpg` show the
full stored camera view with the selected object's box and `GOAL` label. These are the detector's
labels, so the images let you inspect the evidence for a selected goal.

After stopping the node, remove its saved map and camera images with
`python3 scripts/clean_ros2_images.py`. If you launched with a custom `map_image_dir`, pass that
directory as the argument: `python3 scripts/clean_ros2_images.py /path/to/output`. The script
only removes filenames generated by the node; other files in the folder remain.

## Replay the recorded inputs through ROS

In terminal 1, after sourcing the environment and install overlay:

```bash
ros2 launch vlnav_ros vlnav.launch.py use_sim_time:=true instruction:='Find a sofa'
```

After `Ready`, in terminal 2:

```bash
source /opt/ros/humble/setup.bash
ros2 bag play sample-data/vr_h3_2_calib_lidar_rgb_20261007_093649_calib_1 --clock --rate 0.25
```

Slower replay gives inference more time. This still follows the **recorded** robot trajectory;
it does not make the bag follow the proposed goals. In this node, reached/failure events come
from exploration using the actual TF pose. The offline runner's `BagEventSim` is not used.
Restart the node before looping or seeking backwards: the planner and memory assume monotonic
time, and the node clears goals and reports an error on a backwards ROS clock jump.

## Parameters and troubleshooting

`sensor_config`, `run_config`, `run_dir`, `instruction`, `use_sim_time`, `max_frame_age_s`, `save_map_images`,
`map_image_every_s`, `map_image_dir`, `nav_action_enabled`, `nav_action_name`,
`nav_action_frame`, `nav_waypoint_name` and `nav_behavior_tree` are launch arguments.
The image age limit `max_frame_age_s` defaults to 30 seconds and can be set at launch, for example
`max_frame_age_s:=60.0`. The direct node also accepts `max_scan_age_s` (0.5), `max_pose_age_s` (30.0),
`goal_timeout_s` (2), `map_publish_hz` (1) and `publish_extra_static_tf` (true). Parameters are
read at startup. Live planning always uses `planner.blocking=false` (and `move.s2.blocking=false`).

Check `/planner/status` if there is no map or goal. All sensor timestamps must share the ROS
clock; set `use_sim_time=true` for bag replay and false for a live robot. A current base TF is
required for goal selection, even when older transforms still suffice for image processing.
For a live robot, a larger image age limit can accommodate delayed frames, but camera, LiDAR and
TF stamps must still share the same clock. The scan and pose age limits remain 0.5 seconds.
The launch default is `use_sim_time=false`; receiving camera and LiDAR messages alone does not
mean their timestamps are valid. For replay, **both** `use_sim_time:=true` on the launch command
and `--clock` on `ros2 bag play` are required. Status includes `use_sim_time`, `ros_time`,
`latest_image_stamp`, `image_age_s` and `processed_frames`; its `input` field explains missing
`/clock` or rejected timestamps. Status publication continues before simulated time starts.
Tasks wait for fresh sensor geometry rather than starting their timeout on rejected frames.

```bash
ros2 topic echo /planner/status --once --qos-durability transient_local
ros2 topic echo /mapping/occupancy --once --field info --qos-durability transient_local
ros2 topic echo /exploration/goal_active --once
```

Inference runs in a separate worker so ROS sensor/TF callbacks and output expiry keep running.
Adjust the age limits deliberately if your inference latency exceeds them.

When changing the map frame in the sensor YAML, also change RViz's Fixed Frame. The occupancy
map lies at z=0 in that gravity-aligned frame; height classification still uses the existing
mapping ground model. Memory checkpoints, image evidence and planner logs go to `run_dir`.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/
# After sourcing ROS, include adapter tests with real ROS message serialization and DDS:
PYTHONPATH=.:ros2/vlnav_ros:$PYTHONPATH python -m pytest -q tests/ros2 tests/runtime tests/io
```

The ROS tests inject lightweight perception and a scripted VLM, while exercising the real
map/planner/exploration and ROS subscriptions/publications with wall time and simulated bag
time. They also check diagnostics before `/clock` starts and rejection of historical images
by a wall-clock node, retained frontier overlays, goal clearing and marker visibility above
the map while preserving navigation coordinates. The PNG tests check image creation and clearing
of stale goals. They do not validate model accuracy or
robot motion. Without ROS bindings those tests are skipped; the synchronizer and live
runtime tests remain available without ROS.
