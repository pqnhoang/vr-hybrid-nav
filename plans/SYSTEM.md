# SYSTEM: VL-Nav re-implementation — architecture & shared contracts

Reference: VL-Nav: Neuro-Symbolic Reasoning-based Vision-Language Navigation, arXiv:2502.00931v7.
No official code exists. This file is the **single source of truth** for everything shared between
modules. Module specs only describe their own internals and must not redefine anything here.

| File | Scope |
|---|---|
| `SYSTEM.md` (this file) | Architecture, repo layout, shared models, shared packages (`common`, `io`, `perception`, `mapping`, `navigation`), interface contracts, ROS 2 topics, run modes, global milestone order |
| `TASK_nesy_task_planner.md` | Unified Memory (`memory/`), planner (`planner/`), bag event simulator |
| `TASK_target_acquisition.md` | Coarse-to-fine verification (`acquisition/`), used by the planner |
| `TASK_nesy_exploration.md` | NeSy exploration system (`exploration/`), Habitat closed-loop harness |
| `AUDIT_AND_PITFALLS.md` | Paper ↔ spec traceability, corrections, implementation pitfalls — **read before coding** |
| `checkpoints/README.md` | Model weights: list, folder layout, download script |

Rule for the coding agent: if a module spec conflicts with `SYSTEM.md`, `SYSTEM.md` wins; report the
conflict in `docs/design_choices.md`. Choices the paper leaves open are marked **[DESIGN]** and must
stay configurable.

---

## 1. Architecture

```
                         ┌──────────── PLANNER PROCESS (GPU server, async, event-driven) ────────────┐
 /task_instruction ────► │ PlannerCore FSM ─► Decomposer/Replanner (VLM) ─► TargetAcquirer (VLM)      │
                         │        ▲ events                      ▲ MemoryView (direct or mirror)      │
                         └────────┼──────────────────────────────┼───────────────┬────────────────────┘
                                  │ PlannerEvent                 │ scene graph   │ Subtask, detector prompts
                         ┌────────┼──────────────────────────────┼───────────────▼────────────────────┐
                         │        │      CORE PROCESS (onboard, real-time)                           │
 RGB / depth / LiDAR ──► │ FrameSource ─► PerceptionPipeline (one detector pass / frame, union vocab) │
 pose (LIO or GT)        │                 ├─► Mapping: OccupancyGrid2D ─────────┬──────────────┐     │
                         │                 ├─► Memory: SceneGraph + ImageStore ◄──┘ (rooms)      │     │
                         │                 └─► Exploration: frontiers, instances, NeSy scoring ◄─┘     │
                         │                              │ goal                                         │
                         │                    Navigation: path planner + controller ─► robot / sim     │
                         └────────────────────────────────────────────────────────────────────────────┘
```

Key alignment decisions (apply everywhere):

1. **One perception pass per frame.** The detector runs once with vocabulary =
   `generic_vocab ∪ active detector_prompts`. Detections are routed: *all* go to Memory; those whose
   label ∈ active `detector_prompts` also go to Exploration.
2. **One occupancy grid.** `mapping.OccupancyGrid2D` is owned by the core process and shared by
   Exploration (frontiers, curiosity, path planning) and Memory (room segmentation).
3. **One pose source.** All components receive explicit `t` and poses; no component reads wall time.
4. **Planner ↔ Exploration contract** is only `Subtask` (down) and `PlannerEvent` (up)
   (Section 5). Neither imports the other.
5. **Memory lives in the core process**; the planner reads it via the `MemoryView` protocol, either
   directly (single process) or through `RemoteMemoryMirror` (ROS 2, Section 7).

## 2. Repository layout

```
vlnav/
├── common/        models.py, geometry.py, config.py, logging.py         (shared, Section 3)
├── io/            sources.py (BagSource, HabitatSource, LiveSource), trajectory.py
├── perception/    lidar_depth.py, depth_utils.py, detector.py, vocabulary.py, lifting.py, pipeline.py
├── mapping/       occupancy.py, ground.py
├── navigation/    path_planner.py (GridAStar), controller.py (interfaces + Habitat follower)
├── memory/        → TASK_nesy_task_planner.md
├── acquisition/   → TASK_target_acquisition.md
├── planner/       → TASK_nesy_task_planner.md
├── exploration/   → TASK_nesy_exploration.md
├── runtime/       core_runtime.py (wires core process), system_runtime.py (single-process full system)
├── sim/           bag_event_sim.py (planner spec), habitat_env.py, habitat_runner.py (exploration spec)
├── eval/          shared metric utils; module evals live in their specs
├── configs/       system.yaml + one YAML per package (Section 8)
├── docs/          design_choices.md (agent-maintained)
└── tests/
vlnav_ros/
├── vlnav_msgs/    msg + srv (Section 7)
└── vlnav_ros/     core_node.py, planner_node.py, bag_event_sim_node.py, launch/
```

Rules: Python 3.10+, type hints, pydantic v2, `pytest`. Only `vlnav_ros/` imports ROS 2. Unit tests
run without GPU/network (GPU tests marked `@pytest.mark.gpu`, Habitat tests `@pytest.mark.habitat`).
No magic numbers in code. Streaming APIs (`update(...)` per message/frame) everywhere; batch scripts
wrap them.

## 3. Shared models (`common/models.py`)

```python
class Pose(BaseModel):            # SE(3), map frame
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    stamp: float
    def yaw(self) -> float: ...
    def matrix(self) -> np.ndarray: ...

class Frame(BaseModel):           # one camera image + geometry at its stamp
    stamp: float
    camera_id: str
    rgb: np.ndarray                         # HxWx3 uint8
    K: np.ndarray                           # 3x3
    hfov: float                             # rad, derived from K and width
    T_map_cam: np.ndarray                   # 4x4 (optical frame: z forward, x right, y down)
    T_map_base: np.ndarray                  # 4x4 robot base pose
    depth: np.ndarray | None = None         # HxW meters (depth camera / sim)
    lidar_points_map: np.ndarray | None = None   # Nx3 accumulated window (LiDAR robots)
    sensor_origin_map: tuple[float, float, float]  # for ray casting

class Detection2D(BaseModel):
    label: str; score: float
    bbox: tuple[int, int, int, int]         # x1, y1, x2, y2
    mask: np.ndarray                        # HxW bool
    mask_centroid_uv: tuple[float, float]

class Observation(BaseModel):     # 3D-lifted detection
    stamp: float; camera_id: str; label: str; score: float
    centroid_map: tuple[float, float, float]
    points_map: np.ndarray                  # Mx3 voxelized
    bbox: tuple[int, int, int, int]
    T_map_cam: np.ndarray; T_map_base: np.ndarray
    frame_ref: str                          # key into ImageStore / frame cache

class FrameResult(BaseModel):     # output of PerceptionPipeline for one frame
    frame: Frame
    detections: list[Detection2D]
    observations: list[Observation]         # subset of detections that could be lifted
    relevant_idx: list[int]                 # indices of detections whose label ∈ active prompts

class ObjectNode(BaseModel):      # owned by memory/, consumed by acquisition/ and planner/
    id: int; label: str; label_votes: dict[str, float]
    centroid: tuple[float, float, float]
    points: np.ndarray
    best_score: float
    best_view_pose: Pose                    # base pose at best detection (navigation goal)
    best_view_cam_pose: Pose
    best_view_image_path: str
    best_view_bbox: tuple[int, int, int, int]
    room_id: int | None
    n_obs: int; first_seen_t: float; last_seen_t: float

class RoomNode(BaseModel):
    id: int; label: str; label_confidence: float
    mask_cells: np.ndarray                  # Kx2 grid indices
    centroid_xy: tuple[float, float]; area_m2: float
    object_ids: list[int]; adjacent_room_ids: list[int]

class Subtask(BaseModel):         # planner -> exploration
    id: int
    task_id: int
    type: Literal["explore", "goto", "stop"]
    target: str
    detector_prompts: list[str] = []
    room_hint: str | None = None
    node_id: int | None = None
    goal_pose: Pose | None = None           # required for goto

class PlannerEvent(BaseModel):    # exploration / sim / planner-internal -> planner FSM
    type: Literal["new_task", "reached", "no_frontier", "nav_failed",
                  "timeout", "candidate_found", "abort"]
    stamp: float
    task_id: int | None = None
    subtask_id: int | None = None
    payload: dict = {}

class TargetReport(BaseModel):
    task_id: int; target: str; node_id: int; label: str
    position: tuple[float, float, float]; image_path: str; stamp: float
```

`TargetEntry`, `PlannerState`, `Candidate`, `AcquisitionResult` are planner/acquisition-internal and
defined in their specs. `TASK_target_acquisition.md` must import `Pose`, `ObjectNode`, `RoomNode`
from `common.models` instead of defining its own.

## 4. Shared packages

### 4.1 `io/` — frame sources

```python
class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...      # time-ordered, possibly several cameras
```

- `BagSource`: ROS 2 bag via `rosbags` (no ROS install needed). Poses from a TUM trajectory (LIO
  output) with SE(3) interpolation (slerp + lerp); drop frames with pose gap > `max_pose_gap_s`.
  LiDAR: accumulate deskewed `cloud_registered` in `[t - lidar_window_s, t]`. `T_cam_lidar`,
  intrinsics from config. Emits at `frame_rate_hz`.
- `HabitatSource`: wraps the Habitat env (exploration spec); RGB + depth + GT pose; multiple cameras
  allowed.
- `LiveSource`: adapter fed by `core_node` callbacks (ROS 2).

Prerequisites outside the codebase: camera intrinsics, `T_cam_lidar` (4x4), LIO run on bags
(FAST-LIO2 / Super Odometry) producing TUM trajectory + `cloud_registered`. The map frame **must be
gravity-aligned** (z up, floor at constant height); with a tilted LiDAR FAST-LIO's default frame is
not — `BagSource` applies `T_gravity_map` from config, estimated from static IMU data at the start of
the bag by `io/gravity_align.py` (AUDIT pitfall 1). Livox `CustomMsg` types must be registered with
`rosbags` (AUDIT pitfall 8).

### 4.2 `perception/`

- `lidar_depth.py`: project window points into the image with z-buffer → sparse depth + per-pixel map
  points. `depth_utils.py`: dense depth → points (sim / depth cameras).
- `detector.py`: `OpenVocabDetector` protocol (`set_vocabulary`, `detect`), `YoloWorldFastSam`
  (paper's YOLO-World + FastSAM), `MockDetector`. Detection cache on disk keyed by
  (frame stamp, camera_id, vocab hash) for offline experiments.
- `vocabulary.py`: `VocabularyManager` = `generic_vocab` (file) ∪ `active_prompts`; calls
  `set_vocabulary` only on change; exposes `is_relevant(label)`.
- `lifting.py`: mask erosion → depth samples (LiDAR or dense) → MAD outlier rejection → median
  centroid; reject if `< min_points` or beyond `max_obs_range_m`.
- `pipeline.py`:
  ```python
  class PerceptionPipeline:
      def set_active_prompts(self, prompts: list[str]) -> None
      def process(self, frame: Frame) -> FrameResult
  ```

### 4.3 `mapping/`

```python
class OccupancyGrid2D:
    resolution: float; origin_xy: tuple[float, float]
    def update(self, points_map: np.ndarray, sensor_origin: tuple, t: float) -> UpdateInfo
    def value(self, cells) -> np.ndarray          # -1 unknown, 0 free, 1 occupied
    def as_array(self) -> np.ndarray; def world_to_cell(...); def cell_to_world(...)
    def inflated(self, radius_m: float) -> np.ndarray   # cached per version
    version: int

class UpdateInfo(BaseModel):
    t: float; changed_bbox_cells: tuple[int, int, int, int]; version: int
```

- Growable grid, log-odds internally. Ground estimation (`ground.py`): per-cell min z with
  smoothing, or constant floor height for sim. Obstacle if height above ground in
  `[obstacle_z_min, obstacle_z_max]`; also negative obstacles optional. Free space by 2D ray casting
  from sensor origin. This is a simplified stand-in for the paper's terrain analysis.
- Adapter `ExternalGrid` wraps a `nav_msgs/OccupancyGrid` (e.g. from the CMU autonomy stack).

### 4.4 `navigation/`

```python
class PathPlanner(Protocol):
    def plan(self, grid: OccupancyGrid2D, start_xy, goal_xy) -> list[tuple[float, float]] | None

class Controller(Protocol):        # follows a path; sim- or robot-specific
    def step(self, path, robot_pose: Pose) -> object   # action / velocity command
```

- `GridAStar` (default, ROS-free): 8-connected A* on inflated grid (`robot_radius_m`); **[DESIGN]**
  unknown cells traversable with cost `unknown_cost` (needed to reach frontiers); occupied = blocked.
  Goal snapping to nearest free cell within `goal_snap_radius_m`.
- `FarPlannerAdapter` (ROS 2 only, in `vlnav_ros/`): publishes the goal to the FAR planner topic
  (configurable, default `/goal_point`); reached-detection is done by distance in our code.
- Habitat controller: pure-pursuit on the path → discrete actions (exploration spec).

## 5. Planner ↔ Exploration contract

Down (planner → core):

| Message | Semantics |
|---|---|
| `Subtask(type="explore", detector_prompts, room_hint)` | Exploration resets semantic state, sets active prompts, explores with NeSy scoring. |
| `Subtask(type="goto", goal_pose, node_id)` | Exploration bypasses scoring and drives directly to `goal_pose`. |
| `Subtask(type="stop")` | Hold position; clear active prompts except generic vocab. |
| detector prompts | Always sent together with the subtask; perception switches vocab **before** exploration starts using the new subtask. |

Up (core/sim → planner):

| Event | Emitted by | When |
|---|---|---|
| `reached` | exploration | `goto` goal reached within `goto_reach_radius_m` |
| `nav_failed` | exploration | `goto` goal unreachable after `max_plan_failures` |
| `no_frontier` | exploration | `explore`: no frontier and no active instance point for `no_frontier_patience_s` |
| `candidate_found`, `timeout` | planner (internal) | see planner spec |
| `new_task`, `abort` | user / launcher | — |

Every event carries `task_id` and `subtask_id`; the planner discards events whose ids are stale.
In bag open-loop mode `reached`/`no_frontier` come from `sim/bag_event_sim.py` instead of
exploration (Section 6).

## 6. Run modes

| Mode | Data | Closed loop? | Exploration role | Event source |
|---|---|---|---|---|
| **A. Bag replay** | ROS 2 bag + LIO output | No | *Shadow*: computes frontiers, scores, goals; visualized/logged, not executed | `bag_event_sim` |
| **B. Habitat** | HM3D / HSSD scenes | Yes | Drives the agent | exploration |
| **C. Robot** | live sensors, ROS 2 | Yes | Drives via FAR planner | exploration |

`runtime/system_runtime.py` runs modes A and B in one Python process (no ROS):

```
for frame in source:                                   # source = BagSource | HabitatSource
    res  = perception.process(frame)
    info = grid.update(points_of(frame), frame.sensor_origin_map, frame.stamp)
    memory.update(res, info)                           # rate-limited internally
    exploration.on_frame(res, info)
    out  = exploration.step(robot_pose(frame), frame.stamp)    # goal + events
    events = out.events if mode == "B" else bag_sim.step(robot_pose, t)
    for e in events + planner.poll_internal(t): planner.on_event(e)
    for st in planner.pop_dispatched():                # new subtask(s)
        perception.set_active_prompts(st.detector_prompts); exploration.set_subtask(st)
    if mode == "B": env.act(controller.step(path_planner.plan(...), pose))
```

Planner VLM calls run in a worker thread; in offline replay the runtime may block on them
(`planner.blocking: true`) to keep replays deterministic.

## 7. ROS 2 contract (`vlnav_ros/`)

`vlnav_msgs`:

```
Subtask.msg:       int32 id, int32 task_id, string type, string target, string[] detector_prompts,
                   string room_hint, int32 node_id (-1 none), geometry_msgs/PoseStamped goal
PlannerEvent.msg:  string type, int32 task_id, int32 subtask_id, builtin_interfaces/Time stamp,
                   string payload_json
TargetReport.msg:  int32 task_id, string target, int32 node_id, string label,
                   geometry_msgs/Point position, string image_path, builtin_interfaces/Time stamp
SceneGraphJson.msg: builtin_interfaces/Time stamp, int64 version, string json
```

| Topic | Type | Publisher → Subscriber |
|---|---|---|
| `/task_instruction` | std_msgs/String | user → planner_node |
| `/planner/subtask` | Subtask | planner_node → core_node |
| `/planner/event` | PlannerEvent | core_node / bag_event_sim_node → planner_node |
| `/planner/report` | TargetReport | planner_node → user |
| `/memory/scene_graph` | SceneGraphJson (0.5 Hz, no images) | core_node → planner_node |
| `/memory/best_view` | sensor_msgs/CompressedImage, `header.frame_id = "node:<id>"` | core_node → planner_node (on best-view change) |
| `/mapping/occupancy` | nav_msgs/OccupancyGrid | core_node → RViz / planner (rooms viz) |
| `/exploration/markers`, `/memory/markers` | visualization_msgs/MarkerArray | core_node → RViz |
| `/exploration/goal` | geometry_msgs/PoseStamped | core_node → FAR planner adapter |

- `core_node`: hosts io (LiveSource), perception, mapping, memory, exploration, navigation adapter.
- `planner_node`: hosts planner + acquisition, reads memory through `RemoteMemoryMirror`
  (implements `MemoryView` from the scene-graph JSON + mirrored best-view images).
- Launch files: `replay.launch.py` (mode A), `robot.launch.py` (mode C).

## 8. Configuration

`configs/system.yaml`:

```yaml
mode: bag            # bag | habitat | robot
run_dir: runs/${now}
includes: [checkpoints.yaml, io.yaml, perception.yaml, mapping.yaml, navigation.yaml,
           memory.yaml, acquisition.yaml, planner.yaml, exploration.yaml, sim.yaml]
frames: {map: map, base: base_link}
seed: 0
```

Each package reads only its own YAML section; cross-package values (e.g. `robot_radius_m`) live in
the owning package and are passed explicitly by `runtime/`.

## 9. Global milestone order

IDs refer to milestones inside each spec (S = this file, E = exploration, P = planner,
A = acquisition).

| # | Milestone | Depends on |
|---|---|---|
| 1 | S1 `common` + `io` (BagSource, TUM interpolation) | — |
| 2 | S2 `perception` (lidar depth, lifting, detector wrapper, vocab, pipeline, cache) | 1 |
| 3 | S3 `mapping` (OccupancyGrid2D, ground) | 1 |
| 4 | E1–E4 exploration core, tested on synthetic grids and in bag shadow mode | 2, 3 |
| 5 | S4 `navigation` + E5–E6 Habitat harness, ObjectNav eval vs baselines | 4 |
| 6 | P1–P3 memory (object map, rooms, scene graph) on bags | 2, 3 |
| 7 | A1–A5 acquisition on scene-graph fixtures | 6 |
| 8 | P4–P6 planner (serializer, decomposer/replanner, FSM) + bag event sim | 6, 7 |
| 9 | S5 `runtime/system_runtime.py`, mode A full-system replay + P7 eval | 4, 8 |
| 10 | S6 mode B full system in Habitat (multi-target tasks) | 5, 9 |
| 11 | S7 ROS 2 package, `RemoteMemoryMirror`, mode C on robot | 10 |

S-milestone acceptance:
- **S1**: interpolation error tests; frames dropped on pose gaps; a 10 s bag segment iterates.
- **S2**: synthetic box scene → centroid within 2 cm; MAD removes injected background; vocab
  switch triggers exactly one `set_vocabulary`; cache round-trip.
- **S3**: synthetic room scan → free/occupied/unknown correct; ray casting clears free space;
  `UpdateInfo.changed_bbox_cells` covers all changed cells.
- **S4**: A* finds paths through unknown with cost, never through occupied/inflated cells.
- **S5–S7**: as described in Sections 6–7; end-to-end smoke tests.

## 10. Global acceptance criteria

- `pytest -q` passes without GPU/network/Habitat.
- Full mode-A replay of one bag produces: occupancy map, scene graph checkpoints, exploration shadow
  log, planner decision log, reports, and the planner eval table.
- Mode B runs ObjectNav episodes (exploration only) and multi-target tasks (full system) end to end.
- `docs/design_choices.md` documents every **[DESIGN]** decision, one paragraph each.

## 11. Model checkpoints

All weights live in one folder, `checkpoints/` at the repo root (git-ignored except README, manifest
and scripts). Code never downloads weights implicitly at runtime; it loads paths from
`configs/checkpoints.yaml`, which mirrors `checkpoints/manifest.yaml`:

```yaml
checkpoints_root: checkpoints
detector:
  yolo_world: checkpoints/detector/yolov8x-worldv2.pt     # paper: YOLO-World
  yolo_world_small: checkpoints/detector/yolov8s-worldv2.pt   # Jetson / fast iteration
  fastsam: checkpoints/segmenter/FastSAM-x.pt              # paper: FastSAM
  fastsam_small: checkpoints/segmenter/FastSAM-s.pt
  clip_text: checkpoints/text_encoder/clip/ViT-B-32.pt     # YOLO-World text encoder
vlm:
  qwen3_vl_8b: checkpoints/vlm/Qwen3-VL-8B-Instruct        # paper backbone
  qwen3_vl_8b_fp8: checkpoints/vlm/Qwen3-VL-8B-Instruct-FP8
  qwen3_vl_4b: checkpoints/vlm/Qwen3-VL-4B-Instruct        # ablation
text_encoder:
  siglip: checkpoints/text_encoder/siglip-base-patch16-224 # label matching / association
optional:
  depth_anything_v2_small: checkpoints/optional/Depth-Anything-V2-Small-hf
  owlv2: checkpoints/optional/owlv2-base-patch16-ensemble  # detector upper bound
baselines:
  vlfm_dir: checkpoints/baselines/vlfm                     # mobile_sam.pt, groundingdino_swint_ogc.pth, yolov7-e6e.pt
```

`perception/detector.py` must point the YOLO-World text encoder at `detector.clip_text` (Ultralytics
loads it from `~/.cache/clip` by default; `checkpoints/download.py` creates that symlink, but the
loader should also verify the file exists and fail with a clear message). A startup check
`python -m vlnav.common.check_checkpoints` verifies every path in `checkpoints.yaml` required by the
selected run mode.
