# TASK: Implement the NeSy Exploration System (VL-Nav re-implementation)

Reference: arXiv:2502.00931v7, Sec. III-B ("NeSy Exploration System") and III-C ("Goal Selection &
Path Planning"). **Read `SYSTEM.md` first**: shared models (`Frame`, `FrameResult`, `Subtask`,
`PlannerEvent`, `Pose`), perception, mapping, navigation, run modes and ROS 2 topics are defined
there and must not be redefined here. Paper-unspecified choices are marked **[DESIGN]**.

---

## 1. What exploration does

Consumes the current `Subtask` from the planner and, every step, outputs one navigation goal:

- `explore`: gather candidate goal points — **frontier-based target points** (geometry) and
  **instance-based target points (IBTP)** (open-vocabulary detections of the active prompts) — score
  them with the **NeSy score** (neural VL score × symbolic curiosity heuristics) and select a goal.
- `goto`: bypass scoring and drive to `Subtask.goal_pose` (the best-view pose chosen by the planner).
- `stop`: hold.

It emits `reached`, `nav_failed`, `no_frontier` events (contract in `SYSTEM.md` §5). It does **not**
decide whether a target is found — that is the planner's job (via memory + acquisition). Instance
points only bring the robot closer so perception can produce a confident, close-range observation.

## 2. Package layout (`vlnav/exploration/`)

```
exploration/
├── frontiers.py      # FoV-gated frontier detection, BFS clustering, representative points, pool
├── instances.py      # instance-based target points pool
├── vl_score.py       # GMM over FoV + view confidence (Eq. 3–4)
├── curiosity.py      # S_dist, unknown ratio BFS, S_unknown (Eq. 5–7)
├── scoring.py        # S_NeSy combination (Eq. 8)
├── goal_selector.py  # instance-first selection, hysteresis, reachability, hold
├── explorer.py       # ExplorationCore: public API, subtask handling, events
└── viz.py            # marker / image overlays for debugging (ROS-free data, rendered by vlnav_ros)
sim/
├── habitat_env.py    # Habitat wrapper -> HabitatSource + action interface
└── habitat_runner.py # ObjectNav + multi-target episode loops, metrics
```

## 3. Public API (`explorer.py`)

```python
class ExplorationOutput(BaseModel):
    goal: Pose | None
    goal_kind: Literal["instance", "frontier", "goto", "hold"]
    goal_id: int | None
    path: list[tuple[float, float]] | None
    events: list[PlannerEvent]

class ExplorationCore:
    def __init__(self, cfg, grid: OccupancyGrid2D, planner: PathPlanner): ...
    def set_subtask(self, st: Subtask) -> None
    def on_frame(self, res: FrameResult, info: UpdateInfo) -> None   # update pools & VL scores
    def step(self, robot_pose: Pose, t: float) -> ExplorationOutput  # select goal, emit events
    def snapshot(self) -> dict                                         # for logging / viz
```

`on_frame` is called for every processed frame (all cameras); `step` at `select_rate_hz`.
`ExplorationCore` never imports `planner/`, `memory/` or `acquisition/`.

## 4. Frontier-based target points (`frontiers.py`)

### 4.1 Detection (paper Eq. 1)

A cell $(m_x, m_y)$ is a frontier if $G=0$ and at least one neighbor has $G=-1$.
After every grid update, only test cells inside the FoV wedge of each camera of the frame:

$$\big|\mathrm{wrap}\big(\mathrm{atan2}(m_y-y_c,\, m_x-x_c) - \psi_c\big)\big| \le \tfrac{h_{fov}}{2},\qquad \|(m_x,m_y)-(x_c,y_c)\| \le R$$

Reference pose $(x_c,y_c,\psi_c)$ is set by `angle_reference` (AUDIT 2.1):
- `robot` (**paper default**, Eq. 1–2 use robot position $(x_r,y_r)$ and heading $\theta_r$):
  $(x_c,y_c,\psi_c) = (x_r,y_r,\theta_r)$ from `Frame.T_map_base`.
- `camera` (required for multiple cameras or a yawed mount): camera position and heading = optical
  z-axis projected on the ground plane (from `Frame.T_map_cam`).
Config validation rejects `robot` if more than one camera is configured or a camera yaw offset
exceeds 2°. $h_{fov}$ = `Frame.hfov`.

- **[DESIGN]** neighbor test 8-connected (`frontier_connectivity: 8`).
- Additionally re-validate existing frontier cells inside `UpdateInfo.changed_bbox_cells` (a cell may
  stop being a frontier outside the current wedge, e.g. seen by LiDAR).
- Maintain the frontier cell set incrementally (set of cell indices).
- **[DESIGN] ➕ `outside_fov`** (AUDIT pitfall 17): `ignore` (paper) | `zero_score` — also create
  frontiers outside the wedge with $S_{VL}=0$, so space revealed only by a 360° LiDAR still
  produces goals. Plus optional `initial_spin: true` — rotate 360° in place at the start of each
  explore subtask (implemented in `goal_selector.py` as a special goal).

### 4.2 Clustering and representative points

- BFS (8-connected) over frontier cells → clusters; drop clusters with fewer than
  `min_cluster_cells`.
- Paper: small cluster → single centroid; large cluster → multiple sampled points. **[DESIGN]**:
  - size ≤ `small_cluster_cells`: centroid snapped to the nearest cell of the cluster;
  - otherwise farthest-point sampling over cluster cells with
    `n = ceil(size * resolution / sample_spacing_m)` points.
- Re-cluster only the clusters touched by the update (cells within changed bbox ∪ wedge); keep others.

### 4.3 Persistent frontier pool **[DESIGN]**

Clusters change every update, but VL scores are assigned at creation time (paper), so points need
identity:

```python
class FrontierPoint(BaseModel):
    id: int; xy: tuple[float, float]; cluster_id: int; created_t: float
    s_vl: float = 0.0
    s_vl_subtask_id: int | None = None     # subtask for which s_vl is valid
    n_vl_updates: int = 0
    blacklisted_until: float = 0.0
```

- After re-clustering, each new representative point inherits `s_vl`/`s_vl_subtask_id` from the
  nearest old point within `inherit_radius_m` (take max if several); otherwise it is new.
- Points whose cell is no longer a frontier are removed.

## 5. VL module and VL score (`vl_score.py`, paper Eq. 3–4)

Inputs per frame: `FrameResult.detections[relevant_idx]` (labels ∈ active prompts).

### 5.1 Mixture components

For each relevant detection $k$ in camera $c$:

- $\mu_k = -\arctan\!\big((u_k - c_x)/f_x\big)$ — angle of the mask centroid column relative to the
  optical axis, **CCW-positive** (left of image center = positive), consistent with $\Delta\theta$.
- $\alpha_k$ = detection score.
- $\sigma_k$ = `sigma_rad` (paper: constant 0.1).

### 5.2 Score of a point $g$ seen from camera $c$

$$\Delta\theta = \mathrm{wrap}\big(\mathrm{atan2}(g_y-y_c,\, g_x-x_c) - \psi_c\big)\quad\text{(paper Eq. 2, reference pose per §4.1)}$$

If `angle_reference: robot` and the single camera has no yaw offset, $\mu_k$ (camera-relative) is
directly comparable with $\Delta\theta$ (robot-relative). With `camera`, both are camera-relative.

$$S_{VL}(g) = \mathrm{clip}_{[0,1]}\Big(\sum_k \alpha_k \exp\!\Big[-\tfrac12\Big(\tfrac{\Delta\theta-\mu_k}{\sigma_k}\Big)^2\Big]\cdot C(\Delta\theta)\Big),\qquad C(\Delta\theta)=\cos^2\!\Big(\tfrac{\Delta\theta}{h_{fov}/2}\cdot\tfrac{\pi}{2}\Big)$$

$S_{VL}=0$ if $|\Delta\theta| > h_{fov}/2$ or the point is beyond $R$.

### 5.3 When scores are assigned **[DESIGN]**

Paper: the VL score is assigned when a point is initially detected. Because subtasks change the
prompts, define validity per subtask:

- A frontier point gets a score the first time it lies in a camera wedge **during the current
  subtask** (`s_vl_subtask_id != current` → compute and set). A frame with zero relevant detections
  still assigns `0.0` (point counts as "scored").
- Update policy `vl_update_mode`:
  - `first` (paper default): never update after assignment within a subtask;
  - `max`: `s_vl = max(old, new)`;
  - `conf_weighted` (VLFM-style): weighted average with weights $C(\Delta\theta)$.
- On `set_subtask(explore)` all frontier `s_vl` become invalid (lazily, via `s_vl_subtask_id`).
- **➕ `rescore_on_new_subtask`** (AUDIT 2.5): `true` (default) = the per-subtask rule above;
  `false` = strict paper behavior: a point is scored only in the frame where it is created, scores
  are never recomputed (old frontiers keep a score computed for a previous prompt, reset to 0 on
  subtask change).

## 6. Instance-based target points (`instances.py`)

Input: `FrameResult.observations` whose detection index ∈ `relevant_idx`.

- Keep observations with score > `tau_det`; take $(q_x, q_y)$ = centroid xy, confidence = score.
- Voxel-grid merge in 2D with `inst_voxel_m`: one point per voxel, `score = max`, position = score-
  weighted mean.

```python
class InstancePoint(BaseModel):
    id: int; xy: tuple[float, float]; score: float; label: str
    subtask_id: int; created_t: float; last_seen_t: float
    status: Literal["active", "visited", "unreachable"]
    approach_xy: tuple[float, float] | None
```

- **[DESIGN] Approach point**: the instance xy lies on an occupied object; navigation goal = nearest
  free, reachable cell within `[inst_standoff_min_m, inst_standoff_max_m]` from it, preferring cells
  closer to the robot. Recomputed when the grid version changes.
- **[DESIGN] Visit rule**: when the robot comes within `delta_reached_m` of `approach_xy` (or within
  `visit_radius_m` of `xy`), mark `visited`; visited points are never selected again in this subtask.
  If the close-range view confirms the object, memory gets a high-score node and the planner ends the
  subtask (`candidate_found`); if not, exploration simply continues — exactly the paper's
  "approach and verify, otherwise keep exploring" behavior.
- Pool is cleared on every `set_subtask`.
- **Ranking score** (AUDIT 2.2; paper: "highest $S_{VL}$"), `instance_score`:
  - `vl_eq3` (default): Eq. 3 evaluated at the instance's own bearing in the frame where it was
    detected (all relevant detections of that frame contribute), max over frames;
  - `confidence`: raw detection confidence (ablation).
- **No curiosity terms** for instance points (paper).

## 7. Curiosity cues (`curiosity.py`, paper Eq. 5–7)

- $S_{dist}(g) = 1/(1 + d(x_r, g))$, $d$ Euclidean (paper). **[DESIGN]** option `dist_metric: path`
  (A* length, more expensive).
- Unknown ratio: BFS from $g$'s cell over cells with $G \ne 1$ (not occupied), up to radius
  `unknown_bfs_radius_m`; `ratio = #unknown / #visited` (visited = all cells reached by the BFS).
- $S_{unknown}(g) = 1 - \exp(-k \cdot \mathrm{ratio}(g))$, $k$ = `unknown_k`.
- Curiosity terms are computed at **selection time** (paper), cached per (point id, grid version).

## 8. NeSy score (`scoring.py`, paper Eq. 8)

For frontier points:

$$S_{NeSy}(g) = w_{dist}\,S_{dist}(g) + w_{VL}\,\big(S_{VL}(g)\,S_{unknown}(g)\big)$$

⚠️ Grouping read from text-extracted PDF (AUDIT 2.3). `scoring.form`:
`additive_dist` (above, default) | `multiplicative_unknown`:
$(w_{dist}S_{dist} + w_{VL}S_{VL})\cdot S_{unknown}$. Note: with the default form and no relevant
detections the policy reduces exactly to nearest-frontier.

**[DESIGN] Extension (off by default)** `room_bonus`: if the subtask has `room_hint` and a room
label lookup is provided (callable injected by `runtime/`, so exploration does not import memory),
add `w_room` to frontiers inside or adjacent to rooms with that label. Keep disabled for paper-
faithful runs; evaluate as an ablation.

## 9. Goal selection (`goal_selector.py`, paper Sec. III-C)

At each `step(robot_pose, t)`:

1. **goto**: goal = `Subtask.goal_pose`. If within `goto_reach_radius_m` (and yaw within
   `goto_yaw_tol_rad` if enabled) → emit `reached` once, then hold. If planning fails
   `max_plan_failures` times consecutively → emit `nav_failed`, hold.
2. **explore — instances first**: among `active` instance points with distance >
   `delta_reached_m`, pick the highest ranking score (§6, `instance_score`) whose approach point is
   reachable.
3. **explore — frontiers**: else among frontier points with distance > `delta_reached_m` and not
   blacklisted, sort by $S_{NeSy}$; take the first reachable (check at most `max_reach_checks`).
4. Else **hold**. If no frontier points and no active instance points exist continuously for
   `no_frontier_patience_s` → emit `no_frontier` once per subtask.

Robustness additions **[DESIGN]**:
- **Hysteresis**: keep the current goal unless it became invalid (removed / visited / reached /
  unreachable) or a new candidate's score exceeds the current goal's by `switch_margin`
  (instance candidates always preempt frontier goals).
- **Unreachable goals**: blacklist frontier points for `blacklist_s` after `max_plan_failures`.
- **Stuck detection**: if robot displacement < `stuck_dist_m` over `stuck_window_s` while
  driving, blacklist current goal.
- **Replan path** when grid version changes or every `replan_period_s`.

## 10. Habitat closed-loop harness (`sim/habitat_env.py`, `sim/habitat_runner.py`)

- habitat-sim + habitat-lab, HM3D v0.2 scenes (user must obtain Matterport access) with
  ObjectNav HM3D **v1** episodes (same as VLFM; do not mix v1/v2 — AUDIT pitfall 30).
- Coordinate conversion Habitat (Y-up, agent looks along −Z) → system (Z-up, X-forward) happens
  once in `HabitatSource`, with a unit test using a known agent rotation (AUDIT pitfall 10).
- Depth sensor with normalization disabled, or rescaled to meters (AUDIT pitfall 11). Agent: RGB-D camera
  (`hfov` 79°, height 0.88 m, 640×480 by default, VLFM-like), GT pose. Optional extra cameras
  to mimic the 5-camera Spot setup of the paper.
- `HabitatSource` (SYSTEM §4.1) produces `Frame` with `depth`; mapping uses depth points.
- Controller: pure pursuit on the A* path → discrete actions (`move_forward 0.25 m`,
  `turn 30°`, configurable) or velocity control if enabled.
- **ObjectNav stop policy** (eval-only, `habitat_runner.py`; the full system uses the planner
  instead): call STOP when an observation of the target category with score ≥ `stop_score` is within
  `stop_dist_m` of the robot (Habitat success: within 1 m of a goal object).
- Multi-target mode (full system, SYSTEM run mode B): the runner instantiates
  `runtime/system_runtime.py` with planner + memory; episode ends when the planner reaches DONE /
  FAILED or time limit.

## 11. Configuration (`configs/exploration.yaml`)

```yaml
select_rate_hz: 2.0
angle_reference: robot          # robot (paper) | camera (multi-camera / yawed mount)
frontier:
  sensor_range_m: 5.0           # R
  outside_fov: ignore           # ignore (paper) | zero_score
  initial_spin: false
  frontier_connectivity: 8
  min_cluster_cells: 5
  small_cluster_cells: 30
  sample_spacing_m: 1.5
  inherit_radius_m: 0.5
vl:
  sigma_rad: 0.1
  vl_update_mode: first          # first | max | conf_weighted
  rescore_on_new_subtask: true   # false = strict paper behavior
instance:
  tau_det: 0.3                   # YOLO-World scores are low; tune (AUDIT pitfall 12)
  instance_score: vl_eq3         # vl_eq3 (paper: highest S_VL) | confidence
  inst_voxel_m: 0.4
  inst_standoff_min_m: 0.6
  inst_standoff_max_m: 1.2
  visit_radius_m: 1.0
curiosity:
  unknown_bfs_radius_m: 2.5
  unknown_k: 4.0
  dist_metric: euclidean         # euclidean | path
scoring:
  form: additive_dist            # additive_dist | multiplicative_unknown (AUDIT 2.3)
  w_dist: 1.0
  w_vl: 3.0
  room_bonus: {enabled: false, w_room: 0.2}
selection:
  delta_reached_m: 0.8
  switch_margin: 0.1
  max_reach_checks: 10
  max_plan_failures: 3
  blacklist_s: 60
  stuck_dist_m: 0.3
  stuck_window_s: 10
  replan_period_s: 2.0
  no_frontier_patience_s: 5.0
  goto_reach_radius_m: 0.5
  goto_yaw_tol_rad: null
habitat:
  scenes: hm3d_v0.2/val
  hfov_deg: 79
  cam_height_m: 0.88
  forward_m: 0.25
  turn_deg: 30
  max_steps: 500
  stop_score: 0.5
  stop_dist_m: 1.0
```

Ablation switches (CLI and config): `--no-ibtp` (instance pool disabled), `--no-curiosity`
($S_{NeSy} = S_{VL}$ for frontiers, i.e. $w_{dist}=0$ and $S_{unknown}\equiv 1$),
`--nearest-frontier` ($w_{VL}=0$, no IBTP), `--vl-update-mode`, `--room-bonus`.

## 12. Milestones (E-IDs referenced in SYSTEM §9)

1. **E1 Frontiers**: synthetic grids — wedge gating, 8-connectivity, cluster sampling counts,
   incremental updates equal to full recomputation, pool identity/inheritance across updates.
2. **E2 VL score**: unit tests of Eq. 3–4 (zero at FoV edge, peak at μ, clipping, sign convention
   with a synthetic detection left of center → positive μ), assignment rules per subtask and all
   three update modes.
3. **E3 Instances + curiosity + scoring**: voxel merge, approach point selection, visit rule,
   unknown-ratio BFS on hand-made grids, Eq. 5–8 values.
4. **E4 Goal selection + ExplorationCore**: instance-first priority, δ_reached filter, hysteresis,
   blacklist, stuck, hold, event emission (`reached`, `nav_failed`, `no_frontier` exactly once).
   Then **bag shadow mode** (SYSTEM run mode A): run on a real bag, render per-frame overlays
   (frontier points colored by $S_{NeSy}$, instance points, chosen goal, FoV wedges) to video —
   manual inspection checkpoint.
5. **E5 Habitat harness**: env wrapper, controller, ObjectNav loop with stop policy; smoke test on
   one scene.
6. **E6 ObjectNav evaluation**: see §13.

## 13. Evaluation

**ObjectNav (exploration only, mode B)** — HM3D v0.2 val (iterate on a fixed 300-episode subset,
report on full val):

| Method | Config |
|---|---|
| Nearest frontier | `--nearest-frontier` |
| VLFM | official VLFM code (BLIP-2), run separately; plus VLFM-style with our YOLO-World scores (`vl_update_mode: conf_weighted`, no IBTP, no curiosity) |
| Ours w/o IBTP | `--no-ibtp` |
| Ours w/o curiosity | `--no-curiosity` |
| Ours full | default |

Metrics: SR, SPL, distance-to-goal at stop, steps; plus MTUR-equivalent = steps / max_steps.
Report per category and overall. Expected qualitative pattern from the paper: IBTP matters most in
cluttered scenes, curiosity in large open scenes — check by splitting results by scene area.

**Bag shadow mode (mode A)**: qualitative videos + logged statistics (number of frontiers, fraction
with nonzero $S_{VL}$, selection switches per minute) to tune thresholds before closed loop.

**Full system (mode B, multi-target)** — owned by SYSTEM milestone S6: GOAT-Bench episodes and/or
custom abstract-instruction tasks on HM3D/HSSD; metrics SR, SPL, MTUR (avg time / time limit).

## 14. Acceptance criteria

- All E1–E4 unit tests pass without GPU/Habitat.
- No imports of `planner/`, `memory/`, `acquisition/` from `exploration/`.
- Bag shadow-mode video produced for at least one bag.
- ObjectNav table produced for all rows in §13 on the 300-episode subset.
- Every **[DESIGN]** choice documented in `docs/design_choices.md`.

## 15. Open decisions (keep configurable)

See also `AUDIT_AND_PITFALLS.md` Parts 2–3 (angle reference, instance score, Eq. 8 grouping,
frontier-outside-FoV).

- VL score update mode (`first` vs `max` vs `conf_weighted`).
- Euclidean vs path distance in $S_{dist}$.
- Unknown cells cost in A* (`navigation.unknown_cost`).
- Instance approach standoff and visit radius.
- Room-hint bonus (paper does not use room hints in exploration).
