# TASK: Implement the NeSy Task Planner (VL-Nav re-implementation)

Reference: arXiv:2502.00931v7, Sec. III-A ("NeSy Task Planner"). **Read `SYSTEM.md` first**: shared
models (`Pose`, `Frame`, `FrameResult`, `Observation`, `ObjectNode`, `RoomNode`, `Subtask`,
`PlannerEvent`, `TargetReport`), perception, mapping (`OccupancyGrid2D`), run modes and ROS 2 topics
are defined there and must not be redefined here. Paper-unspecified choices are marked **[DESIGN]**.

Sub-component: `TASK_target_acquisition.md` (`acquisition/` package, coarse-to-fine verification).

---

## 1. What the planner does

Input: an abstract, possibly multi-target instruction, e.g.
`"Today's weather report indicates rain. Help Rob find an umbrella, an appropriate jacket, and shoes."`

1. Maintains the **Unified Memory** = 3D scene graph (object nodes + room nodes) + object-centric
   best-view image memory, updated from `FrameResult`s produced by the shared perception pipeline.
2. Uses a VLM (Qwen3-VL-8B) to **decompose** the instruction into a dynamic target list and atomic
   subtasks: `explore(target, detector_prompts, room_hint)` and `goto(node_id → goal_pose)`.
3. Selects `goto` goals via **coarse-to-fine target acquisition** (symbolic top-k → VLM verification).
4. **Replans** after every subtask terminates, using the updated memory.
5. Determines **task completion** and publishes `TargetReport`s.

Only the first subtask of the current plan is executed; the plan is regenerated after each subtask
(receding horizon). The planner talks to exploration **only** through `Subtask` / `PlannerEvent`
(SYSTEM §5).

## 2. Scope

In scope: `memory/`, `planner/`, `sim/bag_event_sim.py`, planner evaluation, `RemoteMemoryMirror`.
Out of scope (other specs / SYSTEM): io, perception, occupancy grid, exploration, navigation, ROS 2
node wiring (SYSTEM S7), LIO, calibration.

## 3. Package layout

```
memory/
├── object_map.py      # association, node update, best-view selection, dedup, pruning
├── image_store.py     # best-view frames on disk, ref-counted
├── rooms.py           # morphological room segmentation on mapping.OccupancyGrid2D + id tracking
├── room_labeler.py    # LLM room labeling
├── scene_graph.py     # SceneGraph: assembly, edges, snapshot(t), save/load, implements MemoryView
└── remote_mirror.py   # RemoteMemoryMirror: MemoryView from scene-graph JSON + mirrored images
planner/
├── vlm.py             # re-export acquisition.vlm_client (one VLM client for the whole planner)
├── serializer.py      # scene graph -> compact text under a token budget
├── decomposer.py
├── replanner.py
├── validation.py      # plan repair rules
├── state.py           # TargetEntry, PlannerState, history
├── core.py            # PlannerCore FSM
└── prompts/  decompose.j2, replan.j2, room_label.j2   (verify_target.j2 lives in acquisition/)
sim/bag_event_sim.py
eval/planner/  annotate_gt.py, run_openloop.py, metrics.py
```

## 4. Unified Memory

### 4.1 `MemoryView` (shared with acquisition)

Defined in `TASK_target_acquisition.md` §6; `SceneGraph` and `RemoteMemoryMirror` implement it and
add:

```python
def rooms(self) -> list[RoomNode]
def snapshot(self, t: float) -> "MemoryView"
def version(self) -> int
```

### 4.2 Update entry point

```python
class SceneGraph:
    def __init__(self, cfg, grid: OccupancyGrid2D, image_store: ImageStore, labeler: RoomLabeler): ...
    def update(self, res: FrameResult, info: UpdateInfo) -> None   # called by runtime per frame
```

- Observations are consumed at `memory_rate_hz` (frames between are skipped for memory only).
- Rooms are recomputed every `room_update_every_s` from the **shared** grid (no own occupancy map).
- Room labeling runs asynchronously (worker thread); the graph is usable with `"unknown"` labels.

### 4.3 ObjectMap (`object_map.py`)

Paper: object nodes come from the open-vocabulary detector and store centroid, highest detection
score, robot pose at that detection, and best-view RGB image.

**[DESIGN] Association** (not specified in the paper):
- Candidate nodes: centroid within `assoc_radius_m`.
- Semantic gate: same label OR text-embedding cosine ≥ `assoc_label_sim` (embedder shared with
  `acquisition.LabelMatcher`).
- Geometric gate: point overlap (fraction of observation points within `assoc_voxel_m` of node
  points) ≥ `assoc_overlap_min`, or centroid distance < `assoc_close_m`. Best passing node wins;
  otherwise create a node.
- Merge: append voxelized points (cap `max_node_points`), centroid = median, `label_votes[label] +=
  score`, `label = argmax(label_votes)`, `n_obs += 1`, `last_seen_t = t`.
- Best view: replace when `obs.score > best_score` (optional `best_view_border_penalty` for bboxes
  touching the image border). Store frame in ImageStore, update bbox, `best_view_pose`
  (= `T_map_base`) and `best_view_cam_pose`.
- `merge_duplicates()` every `dedup_every_s`; prune nodes with `n_obs < min_obs` and
  `best_score < prune_score` older than `prune_age_s`.

### 4.4 ImageStore (`image_store.py`)

JPEG (quality configurable) under `run_dir/images/{camera_id}_{stamp:.3f}.jpg`, ref-counted; frames no
longer referenced are deleted at checkpoint time.

### 4.5 Rooms (`rooms.py`)

Paper: room masks via morphological operations. **[DESIGN]**:

1. `free = grid == 0`, optional closing (`closing_kernel_m`).
2. Erode with disk radius `door_width_m / 2` → seeds = connected components with area ≥
   `min_seed_area_m2`.
3. Watershed on the distance transform (markers = seeds) restricted to `free`.
4. Merge regions < `min_room_area_m2` into the neighbor with the longest shared border.
5. Adjacency = regions touching (4-connected).

Room id stability: Hungarian matching on IoU with previous rooms (`room_iou_min`); matched keep id and
label; unmatched → new id, label `"unknown"`; IoU < `relabel_iou` → mark for relabel.
`rooms.enabled: false` disables rooms (e.g. outdoors) — all objects get `room_id = None`.

### 4.6 Room labeling (`room_labeler.py`)

Paper: an LLM infers room labels from connected object nodes. Input: object labels with counts and
mean score (top `room_label_max_objects`); output JSON `{"label", "confidence"}` constrained to
`room_label_set` + `"other"`. Relabel only when ≥ `relabel_min_new_objects` new objects or region
changed. Uses the planner VLM endpoint in text-only mode.

### 4.7 SceneGraph (`scene_graph.py`)

- Edge rule (paper): object–room edge iff object centroid xy falls in the room mask.
  **[DESIGN]** fallback: nearest room within `room_assign_fallback_m`, else unassigned.
- `snapshot(t)`: read-only view of nodes with `first_seen_t ≤ t` (rooms as of the latest update ≤ t).
- `save(dir)` / `load(dir)`: `graph.json` + `grid.npz` + `images/`; checkpoint every
  `checkpoint_every_s`.
- `to_json(include_points=False)`: format used by `/memory/scene_graph` (SYSTEM §7).

### 4.8 RemoteMemoryMirror (`remote_mirror.py`)

ROS-free class: `apply_scene_graph_json(json, version)`, `apply_best_view(node_id, jpeg_bytes)`;
implements `MemoryView`. Images written to a local mirror store. Used by `planner_node` (S7).

## 5. Planner

### 5.1 State (`state.py`)

```python
class TargetEntry(BaseModel):
    name: str
    detector_prompts: list[str]
    room_priors: list[str] = []
    status: Literal["pending", "exploring", "verified", "reported", "failed"]
    node_id: int | None = None
    attempts: int = 0
    notes: str = ""

class PlannerState(BaseModel):
    task_id: int; instruction: str
    fsm: Literal["IDLE", "DECOMPOSING", "EXECUTING", "REPLANNING", "DONE", "FAILED"]
    targets: list[TargetEntry]
    current: Subtask | None
    history: list[dict]          # finished subtasks with outcome + event
    started_t: float
```

### 5.2 Serializer (`serializer.py`)

Compact text under `max_graph_tokens` (≈ chars/4):

```
ROOMS
R0 bedroom (18.2 m², adj: R1) objects: bed, wardrobe, jacket[#17 .62], chair[#9 .71], ...
R1 hallway (9.0 m², adj: R0,R2) objects: shoe rack[#4 .55], umbrella stand[#21 .40], ...
UNASSIGNED: box[#30 .33]
EXPLORATION: known free area 142 m²; frontiers remaining: yes
```

Objects relevant to current targets first (embedding similarity), then by score; `+N more` when
truncated. Node ids appear as `#id`; every id coming back from the VLM is validated (§5.5).

### 5.3 Decomposer (`decomposer.py`)

On `new_task`. Inputs: instruction, serialized graph, and — because the paper uses the scene graph
**and image memory** for decomposition and replanning (AUDIT 2.4) — best-view crops of up to
`planning_images` nodes most relevant to the instruction (text-embedding similarity between node
labels and instruction/targets), each tagged with its `#id`. Optional `n_context_images` recent
frames. Guided JSON output:

```json
{
  "reasoning": "string",
  "targets": [
    {"name": "umbrella", "detector_prompts": ["umbrella"], "room_priors": ["hallway", "entryway"]},
    {"name": "rain jacket", "detector_prompts": ["raincoat", "jacket"], "room_priors": ["bedroom"]},
    {"name": "rain boots", "detector_prompts": ["rain boots", "boots"], "room_priors": ["hallway"]}
  ],
  "subtasks": [
    {"type": "explore", "target": "umbrella", "room_hint": "hallway"},
    {"type": "goto", "target": "rain jacket", "node_id": 17}
  ]
}
```

Prompt (`decompose.j2`) requirements: resolve implicit semantics and disambiguate (rain → waterproof
items; "appropriate jacket" → rain jacket); targets are concrete detectable categories;
`detector_prompts` are 1–3-word noun phrases for an open-vocabulary detector incl. one broader
fallback class; for room-type goals ("find the living room") emit object proxies (sofa, TV) because
the open-vocabulary detector cannot detect rooms (AUDIT 2.6); `goto` only when a plausible node
exists; order subtasks to reduce travel; JSON only.

### 5.4 Replanner (`replanner.py`)

After every subtask termination. Inputs: instruction, target list with statuses/attempts, history,
last event, updated serialized graph, best-view crops of up to `planning_images` relevant nodes.
Output: same schema plus `"task_done": bool` and optional
`"target_updates": [{"name", "status": "failed", "reason"}]`.

### 5.5 Validation (`validation.py`)

Applied to every decomposer/replanner output before dispatch:
- Unknown `node_id` or `goto` without node → `explore` for that target.
- Subtasks for targets not in the list → dropped (except during decomposition).
- Subtasks for `reported`/`failed` targets → dropped.
- Empty list while targets pending → synthesize `explore` for the first pending target.
- Invalid JSON → one repair retry; second failure → deterministic fallback (explore pending targets
  in order).
- `detector_prompts` of a subtask = union of the target's prompts (subtask JSON need not repeat them).

### 5.6 Target acquisition

Every `goto` goes through `TargetAcquirer.acquire(instruction, target, memory)` (acquisition spec),
**even if the VLM proposed a node id**: a valid proposed id is force-included in the candidate set,
verification decides.
- `selected` → `Subtask(type="goto", goal_pose=node.best_view_pose, node_id)`, target `verified`.
- `rejected_all` / `no_candidates` → `Subtask(type="explore", ...)`, target `exploring`.

### 5.7 PlannerCore FSM (`core.py`)

Public API (used by `runtime/` and `planner_node`):

```python
class PlannerCore:
    def __init__(self, cfg, memory: MemoryView, acquirer: TargetAcquirer, vlm: VLMClient): ...
    def on_event(self, e: PlannerEvent) -> None
    def poll_internal(self, t: float) -> list[PlannerEvent]   # timeouts, candidate checks
    def pop_dispatched(self) -> list[Subtask]                 # subtasks to send down
    def pop_reports(self) -> list[TargetReport]
    def state(self) -> PlannerState
```

Transitions:

```
new_task              → DECOMPOSING → validate → acquire (for goto) → EXECUTING(first subtask)
EXECUTING explore:
  poll_internal every check_period_s: SymbolicFilter(target) on live memory (no VLM);
     candidate with rank_score ≥ explore_candidate_score → candidate_found
  events: candidate_found | no_frontier | timeout(explore_timeout_s) → end subtask
EXECUTING goto:
  reached   → publish TargetReport, target reported → end subtask
  nav_failed | timeout(goto_timeout_s) → acquirer blacklists node for target → end subtask
end subtask           → REPLANNING (Replanner) → validate → DONE if task_done or all targets
                        reported/failed, else dispatch first subtask
global                → task_timeout_s → FAILED (reports already sent stay valid)
abort                 → dispatch Subtask(type="stop"), FAILED
```

- `max_attempts_per_target` exceeded → target `failed`.
- Stale events (`task_id`/`subtask_id` mismatch) are ignored.
- VLM calls run in a worker thread; responses for an outdated `task_id` are discarded. With
  `planner.blocking: true` (offline replay), calls block to keep runs deterministic.
- When dispatching a subtask, `detector_prompts` are attached so the runtime switches perception
  vocabulary before exploration uses the subtask (SYSTEM §5).

### 5.8 Decision log

`run_dir/decisions.jsonl`: timestamp, FSM state, event, prompt hash, serialized graph excerpt, raw
response, parsed plan, validation fixes, acquisition results.

## 6. Bag event simulator (`sim/bag_event_sim.py`)

Used in SYSTEM run mode A (exploration runs in shadow mode and does not emit events).

```python
class BagEventSim:
    def set_subtask(self, st: Subtask) -> None
    def step(self, robot_pose: Pose, t: float) -> list[PlannerEvent]
```

- `oracle`: `goto` → `reached` after `oracle_delay_s`; `explore` → no events (termination by the
  planner's candidate check, or `no_frontier` at bag end).
- `trajectory`: `goto` → `reached` the first time the bag trajectory comes within
  `reach_radius_m` of the goal after dispatch; `nav_failed` at bag end otherwise.

## 7. Configuration

`configs/memory.yaml`:

```yaml
memory_rate_hz: 2.0
assoc_radius_m: 0.5
assoc_label_sim: 0.85
assoc_voxel_m: 0.05
assoc_overlap_min: 0.2
assoc_close_m: 0.25
max_node_points: 2000
best_view_border_penalty: 0.0
dedup_every_s: 10
min_obs: 2
prune_score: 0.2
prune_age_s: 60
rooms:
  enabled: true
  closing_kernel_m: 0.2
  door_width_m: 0.9
  min_seed_area_m2: 2.0
  min_room_area_m2: 3.0
  room_update_every_s: 10
  room_iou_min: 0.3
  relabel_iou: 0.6
  room_assign_fallback_m: 0.5
  room_label_set: [bedroom, bathroom, kitchen, living room, office, meeting room, hallway,
                   entryway, storage, garage, outdoor]
  room_label_max_objects: 20
  relabel_min_new_objects: 3
checkpoint_every_s: 10
image_jpeg_quality: 90
```

`configs/planner.yaml`:

```yaml
vlm: {base_url: http://localhost:8000/v1,
      model: checkpoints/vlm/Qwen3-VL-8B-Instruct,   # served name = path passed to `vllm serve`
      timeout_s: 60,
      structured_output: response_format}         # response_format | guided_json (AUDIT 23)
blocking: false
max_graph_tokens: 3000
planning_images: 4         # best-view crops attached to decompose/replan (0 = text only ablation)
n_context_images: 0
check_period_s: 2.0
explore_candidate_score: 0.35
explore_timeout_s: 180
goto_timeout_s: 120
task_timeout_s: 900
max_attempts_per_target: 3
```

`configs/sim.yaml`: `bag_event_sim: {mode: trajectory, reach_radius_m: 1.0, oracle_delay_s: 5.0}`.

## 8. Milestones (P-IDs referenced in SYSTEM §9)

1. **P1 ObjectMap + ImageStore**: synthetic observation streams — merge vs new, label voting,
   best-view replacement, dedup, pruning.
2. **P2 Rooms + labeler** (mock LLM): synthetic floor plans (2–4 rooms with doors) on
   `OccupancyGrid2D` → room count, adjacency, stable ids under incremental growth.
3. **P3 SceneGraph + RemoteMemoryMirror**: edges, fallback assignment, `snapshot(t)`, save/load
   round-trip, JSON → mirror equivalence, `MemoryView` conformance tests shared with acquisition.
   Then run on a real bag (mode A without planner) and inspect `graph.json` + RViz/rendered top-down
   map — manual checkpoint.
4. **P4 Serializer + Decomposer + Replanner + validation** (mock VLM): token budget, id repair,
   JSON repair retry, fallback plan.
5. **P5 PlannerCore FSM** with scripted events: full multi-target task to DONE; goto failure →
   blacklist; max attempts → failed; stale events/responses ignored; abort.
6. **P6 BagEventSim** both modes.
7. **P7 Evaluation** (after SYSTEM S5): §9.

## 9. Evaluation (mode A, open-loop on bags)

GT per bag (`gt_objects.yaml`), created with `annotate_gt.py` (pick nodes from a saved graph, correct
positions):

```yaml
objects:
  - {gt_id: umbrella_1, category: umbrella, position: [1.2, 3.4, 0.9], room: hallway}
  - {gt_id: jacket_rain, category: rain jacket, position: [3.1, 4.2, 1.0], room: bedroom}
  - {gt_id: jacket_plain, category: jacket, position: [5.0, 1.1, 1.0], room: office}   # distractor
tasks:
  - id: t01
    instruction: "Today's weather report indicates rain. Help Rob find an umbrella, an appropriate jacket, and shoes."
    gt_targets: {umbrella: [umbrella_1], rain jacket: [jacket_rain], rain boots: [boots_1]}
    category: implicit_multi
    start_t: 0.0
```

Categories: `implicit`, `attribute`, `spatial_relation`, `hierarchical_room`, `implicit_multi`,
`absent_target`.

| Metric | Definition |
|---|---|
| Target set P/R | decomposed targets vs `gt_targets` keys (embedding ≥ 0.8 or alias table) |
| Report precision | reports within `report_radius_m` (1.0) of a GT instance of that target |
| Task SR | all GT targets reported, no wrong reports |
| False commit rate | `goto` to a node not matching any GT instance of the target |
| Time-to-correct-report | bag time from `start_t` to correct report |
| Symbolic recall@k, verification acc | from acquisition logs |
| VLM calls / task, latency mean/p95 | decision log |
| Memory quality | GT object matched by a node; duplicates per GT object; room assignment accuracy |

Ablations: `--no-rooms`, `--no-images` (verification on text only), `--no-planning-images`, `--no-symbolic-filter`,
`--k {1,3,5}`, `--rank-strategy`, `--model Qwen3-VL-4B|8B`, `--no-replan`.

Full-system closed-loop evaluation in Habitat is SYSTEM milestone S6.

## 10. Acceptance criteria

- `pytest -q` passes without GPU/network.
- `memory/` and `planner/` import nothing from `exploration/` or ROS 2.
- No subtask dispatched with an unvalidated node id or goal pose.
- VLM never called per frame: only on new task, subtask end, room relabel, acquisition.
- Mode-A replay produces checkpoints, `graph.json`, decision log, reports, metrics table.
- Every **[DESIGN]** choice documented in `docs/design_choices.md`.

## 11. Open decisions (keep configurable)

- Association criteria and thresholds (§4.3).
- Context images for decomposition (`n_context_images`).
- Explore-termination threshold vs replanning on any new candidate.
- Rooms disabled outdoors.
- Forced inclusion of VLM-proposed node ids in acquisition (§5.6).
- See `AUDIT_AND_PITFALLS.md` (planning images, Instruct vs Thinking model, structured outputs).
