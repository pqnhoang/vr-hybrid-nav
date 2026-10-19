# TASK: `move` subtask — DualVLN pixel goals projected onto the occupancy map

**Read `SYSTEM.md` first**: shared models (`Frame`, `Subtask`, `PlannerEvent`, `Pose`), mapping,
navigation, run modes and the planner ↔ exploration contract are defined there and are only
*extended* here. Choices not fixed by DualVLN / InternNav are marked **[DESIGN]**. Facts about
DualVLN that must be checked on real outputs before relying on them are marked **[VERIFY]**.

Model: `checkpoint/vln/InternVLA-N1-DualVLN` (symlink to
`vr_vln_foca/main_baseline/InternNav/checkpoints/base_model/InternVLA-N1-DualVLN`, ~19 GB bf16).
InternNav source: `vr_vln_foca/main_baseline/InternNav` (below: `InternNav/`).

---

## 1. What the move subtask does

The current subtasks are about **objects**: `explore` (find a target) and `goto` (drive to a remembered
object). Instructions that describe a **route** ("go out of the room and turn left", "walk to the end
of the corridor") cannot be expressed by them. `move` fills that gap:

- The planner emits `Subtask(type="move", instruction="walk to the end of the corridor")`.
- `MoveCore` asks DualVLN's System 2 (S2) what to do next, given the instruction, the current image and a short
  image history.
- S2 answers either a **pixel goal** `(x, y)` (a waypoint on the floor in the image) or a
  **discrete action** (`STOP`, `↑`, `←`, `→`, `↓`).
- A pixel goal is **projected from the image onto the 2D occupancy grid**, snapped to a reachable
  free cell, and sent as a normal navigation goal (`GridAStar` offline, Nav2 `NavigateToPose` on the robot).
- The loop repeats until S2 says `STOP` (→ `reached`) or a limit is hit (→ `timeout` / `nav_failed`).

**Non-goals (first version):**
- No DualVLN System 1 (diffusion trajectory). Nav2 / A* replaces it.
- No camera tilt for `↓` (see §6.4).
- No fine-tuning.
- No change to explore/goto behaviour.

## 2. DualVLN facts (from InternNav source)

| Fact | Where |
|---|---|
| S2 prompt: "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task." | `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py:65` |
| Output parsing: if the text has digits → pixel goal, `coord = re.findall(r'\d+')`, stored as `[coord[1], coord[0]]` = `[row, col]`; else actions `STOP→0, ↑→1, ←→2, →→3, ↓→5` | `internvla_n1_policy.py:183-196`, `:79-85` |
| Normal frames are resized to `resize_w × resize_h` = **384 × 384** (aspect *not* kept) before S2 | `internvla_n1_policy.py:113`, `scripts/realworld/http_internvla_server.py:85` |
| `look_down` frames are **not** resized and are appended to the same conversation | `internvla_n1_policy.py:109-115, 139-147` |
| History: `num_history = 8` past frames picked by `np.linspace(0, episode_idx-1, 8)` over all frames since the episode start | `internvla_n1_agent_realworld.py:184`, `http_internvla_server.py:87` |
| S2 runs every `PLAN_STEP_GAP` steps, or after `look_down`, or when there is no output | `internvla_n1_agent_realworld.py:130` |
| Training pixel goals are **floor points** of the future trajectory projected into the image | `internnav/dataset/navdp_lerobot_dataset.py:224-235` |
| The real-world HTTP server needs RGB **and depth** and also runs S1 | `scripts/realworld/http_internvla_server.py:19-77` |
| Habitat: 640×480, hfov 79°; `↑` = 0.25 m forward, `←/→` = 15° turn | `scripts/eval/configs/*.yaml` [VERIFY step sizes] |

**V1: coordinate frame of the pixel goal — settled in M0** (`docs/design_choices.md`, "Move subtask").
- In Habitat a pixel goal is only produced on the `look_down` image: the raw 640×480 frame (hfov 79°)
  of a camera pitched 30° down. Pixel coordinates are in that image (`x ∈ [0, 640)`, `y ∈ [0, 480)`).
- Sending the raw 1920×1080 ZED frame gives pixels in a ~640×480 range, i.e. wrong. Fix: show the model
  a **virtual Habitat view** (`move/view.py`): crop around the principal point to hfov 79° and 4:3,
  resize to 640×480; map a reply back with `view.to_camera`. The ZED is already pitched ~29° down, so
  the same view image is used as the `look_down` image (`look_down_mode: same_frame`).
- A pixel on a forward (non-look-down) query is assumed to be in the 384×384 resized image
  (`main_pixel_ref: resized`): scale by 640/384, 480/384 into the view first. M0 saw none (`↓` 73%,
  `STOP` 16%, turns otherwise), so this branch is unverified.
- M0 on vr_h3_2 (27 look-down replies): `view` hypothesis median 117 px from the recorded future path
  (59% < 150 px), `processed` 114 px (≈ same), `view_from_384` 336 px.

## 3. Data flow

```
                  Subtask(type="move", instruction)
planner ─────────────────────────────────────────────► MoveCore
                                                          │ frame buffer (K, T_map_cam, depth/LiDAR, rgb)
                                                          │ history sampler (§6.2)
                                                          ▼
                              S2Client ──HTTP──► scripts/serve_dualvln_s2.py (InternNav env, GPU)
                                                          │ text → S2Reply(pixel | actions)
                                                          ▼
              pixel (model image) ─► camera pixel ─► ray ∩ floor ─► depth check ─► range clamp
                                                          ▼
                                     grid cell ─► snap / walk back along ray ─► reachable goal
                                                          ▼
                     ExplorationOutput(goal, goal_kind="move")  ──► GridAStar (bag) / Nav2 (robot)
                                                          │
            events: reached (STOP) / nav_failed / timeout ──► planner
```

`MoveCore` does not import `planner/`, `memory/` or `acquisition/`, the same rule as `ExplorationCore`.

## 4. Shared model changes (`common/models.py`, SYSTEM §3, §5)

```python
class Subtask(_Model):
    type: Literal["explore", "goto", "stop", "move"]      # + "move"
    instruction: str | None = None                        # move: text sent to DualVLN (required)
    ...                                                   # target = short label, e.g. "corridor end"
```

- `PlannerEvent` types are reused; no new type.
  - `reached`: S2 said `STOP`. Payload: `{"reason": "vln_stop", "n_queries": k}`.
  - `nav_failed`: `max_unreachable` consecutive pixel goals could not be made reachable.
  - `timeout`: `timeout_s` or `max_queries` exceeded.
- `ExplorationOutput.goal_kind` gains `"move"`, so runtimes and the ROS node handle it like any other goal.
- Update SYSTEM.md §3 and the §5 tables in the same change.

## 5. Package layout (`vlnav/move/`)

```
move/
├── config.py        # MoveConfig (pydantic) + load_move_config(), like exploration/config.py
├── s2_client.py     # S2Client protocol; HttpS2Client; CachedS2Client (disk cache by request hash); MockS2Client
├── reply.py         # parse_s2_text(text) -> S2Reply (same rules as InternNav, §2)
├── projection.py    # pixel_to_camera(), ray_floor_intersection(), depth_check(), pixel_to_map()
├── goal.py          # make_reachable(grid, planner, robot_xy, p_map) -> GoalResult
├── history.py       # FrameBuffer: keeps frames since subtask start, motion-based sampling
├── core.py          # MoveCore: set_subtask / on_frame / step / snapshot
└── viz.py           # pixel overlay on the image, ray + goal on the top-down map (ROS-free)
scripts/
├── serve_dualvln_s2.py   # S2-only HTTP server (runs in the InternNav env)
├── run_move_bag.py       # bag shadow mode for one fixed move subtask (like run_exploration_bag.py)
└── inspect_move.py       # readable view of move_log.jsonl
configs/move.yaml
eval/move/                # cases + metrics (§13)
tests/move/               # unit tests, no GPU / network (§14)
```

## 6. DualVLN S2 server and client

### 6.1 Server (`scripts/serve_dualvln_s2.py`) **[DESIGN]**

- InternNav's server needs depth and also runs S1, so we don't use it. We write our own S2-only server.
- It runs in the **InternNav conda env**, so vlnav never imports InternNav and their `transformers`
  versions never mix. Communication is HTTP only.
- It loads the model with InternNav's own classes (the same code path as `InternVLAN1Policy`), so the
  prompt and image processing stay identical. It does not call S1 or `generate_latents`.
- **Stateless:** each request carries everything needed, so replies are cacheable and replayable,
  like `vlm.cache_dir`.

```
POST /s2
  form: image (current, original resolution, PNG), history_0..history_7 (PNG, already resized),
        json = {"instruction": str, "look_down": bool, "prev_output": str | null}
  -> {"text": "<raw LLM output>", "input_size": [w, h], "processed_size": [w, h], "latency_ms": float}
GET  /health -> {"model": path, "device_map": ..., "dtype": "bfloat16"}
```

- The server applies InternNav's resize, conjunction phrase and history placeholder exactly as
  `s2_step` does. It returns `input_size` / `processed_size` so the client can convert coordinates (§7.1).
- **GPU placement** [DESIGN, settle in M0]:
  - Perception runs on `cuda:0`; Qwen3-VL-8B runs on GPU 1 (~17 GB). DualVLN (~19 GB) does not fit next to Qwen3-VL.
  - Measure first: DualVLN on GPU 0 next to perception (YOLO-World + FastSAM + CLIP).
  - Fallback: split DualVLN over both GPUs (`device_map="auto"`, `max_memory`), or move Qwen3-VL to the vLLM backend.
  - Record the result in `docs/design_choices.md`.

### 6.2 History (`history.py`) **[DESIGN]**

In Habitat, one "step" is one action (0.25 m or 15°), and InternNav keeps every step's image. On the
real robot, frames arrive at camera rate, so keeping all of them would make the 8 linspace-sampled
history images very close in time. `FrameBuffer`:

- appends a frame (resized to 384×384, plus its pose) when the robot has moved `history_step_m`
  (0.25) or turned `history_step_rad` (15°) since the last kept frame. This mimics Habitat steps;
- picks history ids with InternNav's rule `np.unique(np.linspace(0, n-1, num_history))`;
- is reset by `set_subtask` (one DualVLN "episode" = one move subtask).

### 6.3 Client (`s2_client.py`)

- `HttpS2Client(base_url, timeout_s)`, `CachedS2Client(inner, cache_dir)` (key = sha256 of request
  bytes), and `MockS2Client(script)` for tests.
- Calls run in a **worker thread**, like the planner's VLM jobs. In bag replay, `move.blocking: true`
  waits for the reply to keep runs deterministic.

### 6.4 Reply handling (`reply.py`, `core.py`)

| S2 reply | Action |
|---|---|
| pixel `(x, y)` | §7 projection → goal (§8) |
| `STOP` | emit `reached`, hold |
| `↑` (×n) | goal `n · forward_step_m` ahead along the robot heading (default 0.25 m per arrow) [DESIGN] |
| `←` / `→` (×n) | rotate in place by `n · turn_step_rad` (15°) [DESIGN, see §9 rotation] |
| `↓` | **[DESIGN]** the waist camera cannot tilt. Default `look_down_mode: same_frame`: re-query with the same frame as the look-down image, continuing the conversation with `prev_output`. Alternative `ignore`: treat it as no output and query again on the next frame. Decide in M0 from what S2 does next. |
| unparsable | count as failure; re-query on the next frame |

## 7. Pixel → map projection (`projection.py`)

All inputs come from the **frame the request was built from**, not the latest frame. S2 takes
~0.5–1 s and the robot keeps moving. The goal is computed in the map frame, so it stays valid after
the robot moves.

### 7.1 Model pixel → camera pixel

```
(x_m, y_m) = first two integers of the reply             # parse_s2_text keeps (x, y) = (col, row)
look-down reply:            (x_v, y_v) = (x_m, y_m)                           # 640×480 view image
forward reply (resized):    (x_v, y_v) = (x_m · 640 / 384, y_m · 480 / 384)
(u, v) = view.to_camera(x_v, y_v)                         # crop offset + scale back to the camera image
```

`model_view.mode: full` makes the view the whole camera image (old behaviour, for other cameras).
Reject a pixel outside `[0, W) × [0, H)` of the camera image (`out_of_image`).

### 7.2 Ray ∩ floor (primary; works without depth)

```
d_cam = K⁻¹ [u, v, 1]ᵀ                 # optical frame: z fwd, x right, y down
d     = R_map_cam · d_cam / |·|        # map frame
c     = t_map_cam                      # camera centre
z_f   = floor(robot_xy)                # floor_fn, below
if d_z > −min_down_dz:  → "above_horizon" (no floor hit; go to §7.4)
t     = (z_f − c_z) / d_z
p     = c + t · d
refine once: z_f ← floor(p_xy); recompute t, p               # sloped / uneven floor
```

`floor_fn` (added in M3). `CellMinGround` is wrong next to the robot: the waist LiDAR does not see the floor
there, so early in a run the cells under the robot hold body points at about base height. Ray
hits then came out ~1 m short. So:
- the floor at the robot is `base z − base_height_m` when set, else the median grid ground on the
  `ground_ring_m` [1, 3] m ring;
- `floor(xy)` is that value within `ground_min_dist_m` (1 m) of the robot;
- farther away it is the grid ground, if that is within `max_floor_step_m` (0.5 m) of the robot's floor.

### 7.3 Depth check (when depth is available)

- **Depth source:**
  - `Frame.depth` (D455 on `bkhn_round1`);
  - otherwise `sparse_depth()` from `vlnav/perception/lidar_depth.py` on `Frame.lidar_points_map` (vr_h3_2).
- **Measured range:** `r_meas` = median of valid depths in a `depth_window_px` (default 7 px) square
  around `(u, v)`. For sparse LiDAR, use the nearest `depth_k` (5) valid pixels within `depth_search_px` (15).
- **Floor range:** `r_floor` = `t` along the ray.
- If `r_meas < r_floor − max(depth_margin_m, depth_margin_rel · r_floor)` (0.3 m, 10%; the relative term
  added after M4: at 6–8 m the sparse LiDAR window picks up nearer floor returns), something stands between the camera and the floor
  point (wall, furniture, a person). Set `p ← c + (r_meas − obstacle_backoff_m) · d` and flag `clipped_by_depth`.
- If there is no valid depth, keep `p` and flag `no_depth`. This is normal on sparse LiDAR.

### 7.4 Range limits

| Case | Rule |
|---|---|
| `|p − robot|_xy < min_range_m` (0.3) | `too_close`: treated as "already there"; re-query |
| `> max_range_m` (5.0) | clamp along the ray to `max_range_m`, flag `clamped`. Floor projection error grows ~ d²/h · δθ: at h ≈ 1 m and 10 px error that is ~0.5 m at 5 m. |
| `above_horizon` | if depth is valid, use `c + r_meas · d` (projected to z_f); else direction-only goal at `fallback_range_m` (2.0) on the floor, flag `direction_only` |

### 7.5 Output

```python
class Projection(BaseModel):
    ok: bool
    reason: str | None                 # out_of_image | above_horizon | too_close | ...
    uv: tuple[float, float]            # camera pixel
    p_map: tuple[float, float, float] | None
    flags: list[str]                   # clipped_by_depth | no_depth | clamped | direction_only
    r_floor: float | None; r_meas: float | None
```

## 8. Making the goal reachable (`goal.py`)

1. Look up cell `world_to_cell(p_xy)`: free / unknown / occupied / out of bounds.
2. Try `GridAStar.snap_goal(grid, p_xy)` (existing, `goal_snap_radius_m` = 0.5) and `plan(robot, goal)`.
3. If that fails, **walk back along the ray** from `p` toward the robot in `walkback_step_m` (0.2)
   steps and repeat step 2 at each point. The first success is the goal (flag `walked_back`). This
   handles goals in unknown space behind a frontier and goals inside obstacles (the open
   `nav_failed`-on-unknown problem).
4. Goal yaw = heading from robot to goal.
5. If nothing succeeds: `unreachable`. After `max_unreachable` (3) in a row → `nav_failed`.

```python
class GoalResult(BaseModel):
    goal: Pose | None; path: list[tuple[float, float]] | None
    cell_state: str; flags: list[str]; reason: str | None
```

## 9. `MoveCore` (`core.py`)

```python
class MoveCore:
    def __init__(self, cfg: MoveConfig, grid: OccupancyGrid2D, planner: PathPlanner, s2: S2Client): ...
    def set_subtask(self, st: Subtask) -> None        # reset buffer, counters, conversation
    def on_frame(self, frame: Frame) -> None           # history buffer + latest frame cache
    def step(self, robot_pose: Pose, t: float) -> ExplorationOutput   # goal_kind="move"
    def snapshot(self) -> dict                         # for logging / viz
```

**When to query S2** (at most one request in flight):
- no current goal; or
- robot within `requery_radius_m` (0.5) of the goal; or
- goal age > `requery_period_s` (2.0) [DESIGN: continuous re-query while moving, like InternNav's `PLAN_STEP_GAP`]; or
- no progress (< 0.1 m in `stall_s` = 5 s); or
- the last reply was a rotation and the rotation is done.

**Hysteresis** [DESIGN]: a new pixel goal replaces the current goal only if it moves the goal by more
than `switch_dist_m` (0.4). This avoids Nav2 goal churn from small pixel jitter.

**Rotation** (`←/→`): output a goal at the current xy with yaw ± n·15°.
- **[VERIFY]** Does the `vr_common_msgs/NavigateToPose` server execute pure in-place rotations?
  Some servers ignore yaw inside the xy tolerance.
- If not, `rotate_mode: offset` puts the goal `rotate_offset_m` (0.3) ahead in the new heading.

**Limits:** `timeout_s` (120), `max_queries` (60) → `timeout`; `max_unreachable` → `nav_failed`.

## 10. Integration

### 10.1 Runtimes

- `SystemRuntime` (bag / sim) and `LiveRuntime` (ROS) route by subtask type:
  - `move` → `MoveCore.step`;
  - everything else → `ExplorationCore.step`.
- Both cores receive every frame:
  - `ExplorationCore.on_frame` keeps frontiers and VL scores fresh for later subtasks;
  - `MoveCore.on_frame` is cheap (buffer only).
- Perception keeps the generic vocabulary during `move`, so memory keeps building.
- `build_move(rc, grid, planner)` in `runtime/builder.py` creates the core. `move.enabled: false`
  (default) means no move subtasks are planned and no server is needed.
- ROS node (`ros2/vlnav_ros/`):
  - no change to `NavigationAction`, since `goal_kind="move"` is a normal pose goal;
  - `camera_images.py` draws the pixel goal and the S2 text;
  - `map_images.py` / RViz markers draw the ray and the snapped goal.

### 10.2 Planner (after M3; standalone first)

- `decomposer.py`: schema `type` enum gains `"move"` and the field `instruction`.
- `decompose.j2` / `replan.j2`, new rule: *use `{"type": "move", "target": "<short label>",
  "instruction": "<route in imperative form>"}` for parts of the instruction that describe a route
  or a place relative to the robot (corridor, door, turn left), not an object to find.*
- `validation.py`: `move` needs a non-empty `instruction`, no `node_id`; invalid → dropped with a warning.
- `core.py` FSM: `move` + `reached` → next subtask; `move` + `nav_failed` / `timeout` → replan with
  the outcome in the prompt. As built (M5): every move end goes through REPLANNING, like every
  other subtask end, and the replanner picks the next subtask from the history (outcome and
  route instruction); see docs/design_choices.md "Planner FSM (M5)". A move subtask is **not** a target: it never creates a `TargetEntry` and
  never produces a `TargetReport` [DESIGN].
- `move.enabled: false` keeps the old schema (no `"move"` in the enum), so existing evals are unchanged.

## 11. Configuration (`configs/move.yaml`)

```yaml
enabled: false
s2:
  base_url: http://localhost:8001
  timeout_s: 10.0
  cache_dir: null              # reproducible bag replays
  blocking: false              # true in bag replay
model_input: {resize_w: 384, resize_h: 384, num_history: 8}
model_view: {mode: habitat, width: 640, height: 480, hfov_deg: 79.0}   # habitat | full  (V1, M0)
main_pixel_ref: resized        # resized | view: frame of a pixel on a forward query (unseen in M0)
look_down_mode: same_frame     # same_frame | ignore
max_look_down: 2               # look-down queries per cycle
history: {step_m: 0.25, step_rad: 0.2618}
actions: {forward_step_m: 0.25, turn_step_rad: 0.2618, rotate_mode: in_place, rotate_offset_m: 0.3}
projection:
  min_down_dz: 0.02
  base_height_m: null          # floor at the robot = base z − this; null: median grid ground on the ring
  ground_ring_m: [1.0, 3.0]
  ground_min_dist_m: 1.0
  max_floor_step_m: 0.5
  depth_window_px: 7
  depth_search_px: 15
  depth_k: 5
  depth_margin_m: 0.3
  depth_margin_rel: 0.1
  obstacle_backoff_m: 0.4
  min_range_m: 0.3
  max_range_m: 5.0
  fallback_range_m: 2.0
goal: {walkback_step_m: 0.2, max_unreachable: 3, switch_dist_m: 0.4}
loop: {requery_radius_m: 0.5, requery_period_s: 2.0, stall_s: 5.0, timeout_s: 120.0, max_queries: 60}
```

Loaded through the same three layers as other packages (SYSTEM §8): `configs/move.yaml` →
`configs/runs/*.yaml` key `move:` → command-line flags.

## 12. Milestones

| ID | What | Done when |
|---|---|---|
| **M0** | **Model check (no vlnav code).** Start `serve_dualvln_s2.py`; run S2 on ~30 frames from `vr_h3_2` with 3–4 hand-written instructions; save raw text + images | V1 settled (`model_view`, `main_pixel_ref`); `↓` / action frequency noted; GPU placement + latency (p50/p95) recorded in `docs/design_choices.md` |
| **M1** | `reply.py`, `projection.py`, `goal.py` + unit tests (§14) | tests pass; synthetic camera round-trip error < 1 cm |
| **M2** | Projection check on bags: script draws N floor pixels → map points on the top-down map; on `bkhn_round1` compare floor-plane vs depth projection | median floor-vs-depth error < 0.15 m within 4 m; visual overlay looks right on `vr_h3_2` |
| **M3** | `MoveCore`, `history.py`, `run_move_bag.py` (shadow mode, fixed instruction), logging + viz | runs end to end on `vr_h3_2`; `move/shadow.mp4` shows pixel, ray and goal |
| **M4** | Shadow-mode eval (§13.1) on hand-written cases | metrics reported for both bags |
| **M5** | Planner integration (§10.2) + `run_system_bag.py --enable-move` | decomposition of route + object instructions verified on the open-loop planner eval |
| **M6** | ROS node: `move` goals to Nav2 on the robot; rotation behaviour (§9 VERIFY) | closed-loop runs (§13.2) |

## 13. Evaluation

### 13.1 Bag shadow mode (no closed loop)

- **Cases:** `eval/move/cases_<bag>.yaml`. Each case has `start_offset_s`, `end_offset_s` and an
  `instruction` that describes what the recorded robot actually did in that segment (written by hand
  from the video).
- **Waypoint–path distance:** distance from each projected goal to the recorded future trajectory
  over the next `T` s (default 10 s). A good waypoint lies near where the robot actually went.
  Report median and the fraction < 0.5 m.
- **Projection validity:** fraction of pixel replies that become a reachable goal, with the share of
  each flag / reason (`clipped_by_depth`, `walked_back`, `above_horizon`, ...).
- **STOP timing:** time between S2's first `STOP` and `end_offset_s`.
- **Latency:** p50 / p95 of S2 replies and of projection + goal (projection must be < 5 ms).

### 13.2 Robot (closed loop, M6)

- 10–20 route instructions in the office. Each has a goal region marked on the saved map.
- Metrics: success (stops inside the region), path length vs. shortest, time, number of queries, `nav_failed` / `timeout` count.
- Compare against InternNav's own real-world client (S1 trajectories) on the same instructions, if feasible.

## 14. Tests (`tests/move/`, no GPU / network)

- `parse_s2_text`: `"(123, 45)"`, `"123,45"`, `"STOP"`, `"←←"`, `"↓"`, garbage.
- `pixel_to_map` with a synthetic camera (known `K`, height 1 m, pitch −15°):
  - round-trip a known floor point;
  - pixel above the horizon;
  - out of image;
  - virtual view crop / round trip and the 384 → view scaling.
- `depth_check`: a wall at 2 m with a floor point at 4 m → clipped to 1.6 m. Sparse depth with no valid pixels → `no_depth`.
- `make_reachable`: goal in a free cell; in an unknown cell behind a frontier (walk-back); inside an inflated obstacle (snap); fully blocked (`unreachable`).
- `FrameBuffer`: sampling by distance / angle; linspace ids match InternNav for n = 1, 5, 30.
- `MoveCore` with `MockS2Client`:
  - pixel → goal;
  - `STOP` → `reached`;
  - 3 unreachable → `nav_failed`;
  - timeout;
  - hysteresis keeps the goal for small jitter;
  - a stale reply after `set_subtask` is dropped.
- Planner: validation of `move` subtasks; FSM transitions (`move` + `reached` → next subtask).

## 15. Acceptance criteria

1. With `move.enabled: false`, all existing tests and evals give the same results as before.
2. `pixel_to_map` + `make_reachable` unit tests pass. Synthetic round-trip error < 1 cm.
3. On `bkhn_round1`, floor-plane vs depth projection median error < 0.15 m within 4 m (M2).
4. `run_move_bag.py` runs on both bags. It writes `move_log.jsonl` with one record per S2 query:
   - request frame stamp;
   - raw text;
   - parsed reply;
   - `uv`, `p_map`, flags;
   - goal;
   - latency.

   It also writes `shadow.mp4` (image with the pixel + top-down map with ray and goal) and `summary.json` with the §13.1 metrics.
5. Every S2 reply that does not become a goal is logged with a reason. No silent drops.
6. The planner can emit `move` subtasks. Invalid ones are dropped with a logged warning, and
   `move` + `reached` advances to the next subtask.
7. Docs:
   - `docs/design_choices.md` records V1, `look_down_mode`, GPU placement and latency;
   - README gains a run-table row;
   - SYSTEM.md §3 / §5 are updated.

## 16. Open decisions and risks

| Item | Default | Risk / note |
|---|---|---|
| V1 pixel coordinate frame | `model_view: habitat` | Settled in M0 for look-down replies; `main_pixel_ref` unverified (no forward pixels seen). |
| Camera mismatch | — | DualVLN was trained on Habitat images (640×480, hfov 79°, camera at ~1.25 m?). The ZED waist camera has a different height, FoV and pitch. Expect worse pixel goals; measure in M4 before deciding on fine-tuning. |
| `↓` without a tilting camera | `same_frame` | S2 may loop on `↓`. Cap consecutive `↓` at 2, then `ignore`. |
| Re-query while moving vs. stop-and-go | re-query every 2 s | Motion blur / pose latency on the humanoid. Stop-and-go (`requery_period_s: null`) is the fallback. |
| Rotation through Nav2 | `in_place` | Server may ignore yaw (§9). |
| Model memory / placement | GPU 0 next to perception | Split or vLLM if OOM (§6.1). As built: the server's allocator cache grew to 19.6 GiB, so it now frees the cache after each request (17.4 GB). Full-system runs put perception on GPU 1 next to Qwen3-VL; `detector.device` now really takes effect (it did not before, an ultralytics quirk). |
| History sampling by motion | 0.25 m / 15° | Differs from InternNav's real-world client (every call). Compare both in M4. |
| Move vs. explore in planner prompts | rule in §10.2 | The LLM may use `move` for object search. Validation can't catch this; check in the open-loop eval. Found in M5: route landmarks listed as targets (validation drops them, docs/design_choices.md), and routes after the last target lost (the replanner now runs after it). |
