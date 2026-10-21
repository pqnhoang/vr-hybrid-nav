# Design choices

Agent-maintained (SYSTEM.md §1, §10). One paragraph per **[DESIGN]** decision; spec conflicts are
listed first. Status: Unified Memory first build (M1–M5 + JSON part of M6).

## Spec conflicts found

**Which memory spec.** SYSTEM.md's file table assigns `memory/` to `TASK_nesy_task_planner.md`,
whose §4/§7 differ from the newer `TASK_unified_memory.md`: `max_node_points` vs
`max_node_voxels`, a `prune_score` rule, best-view replacement on raw score vs on view quality `q`,
and `snapshot(t)` as a `first_seen_t` filter vs checkpoint reconstruction. The implementation
follows `TASK_unified_memory.md`. Suggest pointing the SYSTEM.md table there and removing planner
spec §4 / the memory part of §7.

**`until_t` semantics.** Acquisition spec §6 defines `all_objects(until_t)` as "nodes with
`first_seen_t <= until_t`"; memory spec §10 forbids that for real memory (it leaks later centroids,
labels and best views). Followed the memory spec: `until_t` delegates to `snapshot(until_t)`.
`MemoryView` in `acquisition/memory_api.py` already has the extended signature
(`include_tentative`, planner extensions).

**Missing AUDIT entries.** The memory spec cites AUDIT pitfalls 33 (snapshot leakage) and 34
(co-located duplicates); `AUDIT_AND_PITFALLS.md` ends at 32. Their content is covered by memory
spec §10 and §6.2 respectively.

**`Observation` has no mask.** Consolidation needs mask IoU but SYSTEM `Observation` carries no mask
and no detection index. Masks are looked up from `FrameResult.detections` by `(label, bbox)`, with
bbox IoU as fallback. Suggest adding `det_idx: int` to `Observation` in SYSTEM §3.

**Other gaps.** `/memory/get_best_view` (memory §11.2) is not in SYSTEM §7's topic table; the
`floor_id` the memory spec wants reserved does not exist on SYSTEM `ObjectNode` (not added); the
`TextEmbedder` signature is unspecified — chosen as `encode(texts) -> (N, D)` L2-normalized.

## Objects

**Rate limiting and gating.** At most `memory_rate_hz` frames per camera (`t - last < 1/rate` →
skip). An observation is kept if `score >= min_obs_score`, `#points >= min_points`, centroid
distance to the camera `<= max_obs_range_m`, bbox area `>= min_bbox_area_px`, and its label is not
in `ignore_labels` (case-insensitive).

**Intra-frame consolidation.** Union-find over pairs with mask IoU `>= same_object_mask_iou` (so
A~B, B~C groups A, B, C). Group: union of points, `label_votes[label] = max member score`,
score / bbox / label / poses of the top member; centroid = member centroid for single members, else
median of the union.

**Association.** As specified: KD-tree candidates within `assoc_radius_m` (3D centroid distance,
tentative nodes included so they can be confirmed), semantic gate (shared label in the vote sets or
top-label cosine `>= assoc_label_sim`; without an embedder only shared labels pass), geometric gate
(voxel overlap fraction of the observation `>= assoc_overlap_min` or distance `< assoc_close_m`),
cost `1 - overlap + w_dist·dist/assoc_radius_m`, Hungarian matching with gated-out pairs excluded.

**Node geometry.** Voxels at `assoc_voxel_m` kept as an insertion-ordered set; over
`max_node_voxels` the oldest voxels are dropped (the spec's "random-drop oldest" is
contradictory; dropping oldest is deterministic). `points` = voxel centers, centroid = their median.
Label = argmax of accumulated votes, ties keep the current label, then alphabetical.

**Confirmed vs tentative.** Confirmed once `n_obs >= min_obs` or `best_score >= confirm_score`, and
never un-confirmed. `MemoryView` hides tentative nodes by default; `get_object` still returns them.

**Best view.** Quality `q = score - border_penalty·[bbox touches border] -
small_penalty·[area < best_view_min_area_px]`; replace iff `q > best_q + best_view_min_gain`
(defaults = paper's highest confidence). `best_view_require_free` checks the base-pose cell for
*occupied* (not inflated: robot radius belongs to navigation). When the grid changes, a current view
whose cell became occupied gets `best_q = -inf` so the next observation replaces it; this check runs
before integration in `update`.

**Dedup.** Every `dedup_every_s` (first run one period after start). Candidate pairs of confirmed
nodes within `assoc_radius_m`; merge if (`dist < assoc_close_m` and semantic gate) or voxel overlap
relative to the smaller node `>= dedup_overlap_min`, literally as specified. Kept id = smaller (=
older, ids are monotonic); alias `merged → kept`; the better best view (by `q`) survives and the
other image reference is released.

**Prune.** Every `prune_every_s`: tentative nodes with `t - created_t > prune_age_s`. Confirmed
nodes are never pruned.

## Images

Frames are written synchronously (the async writer is deferred) as
`images/{camera_id}_{stamp:.3f}.jpg`, RGB→BGR for OpenCV. Reference counts include live nodes and
every retained in-memory checkpoint, so an image used by `snapshot(t)` is never deleted; frames at
zero references are deleted by `gc()` at checkpoint time. Crops: bbox grown by `margin·size`,
clamped, long side `<= max_px` (INTER_AREA), LRU-cached and read-only.

## Rooms

**Segmentation.** The grid is reduced to `room_grid_res_m` blocks (block free iff `>= 50 %` free
and no occupied cell, so thin walls survive). The closing fills unknown speckle only: cells occupied
in the grid are never turned free (a plain closing of the free mask erases walls up to the kernel
size). Erosion radius = `ceil(door_width_m / 2 / res)`, so a door of exactly `door_width_m` is
closed. Free components with no seed are left unassigned. Regions below `min_room_area_m2` merge
into the neighbor with the longest 4-connected border; isolated ones are dropped. Labels are mapped
back to grid resolution and restricted to grid-free cells. Polygon = largest external contour,
`approxPolyDP` with 1-cell tolerance. Re-segmentation when the grid changed and either
`room_update_every_s` passed or free area grew by `room_update_min_new_m2`.

**Room ids.** Hungarian on region IoU (old id image re-expressed in the new grid frame, so a growing
grid is handled), threshold `room_iou_min`. An unmatched old room that is covered by a new region
for at least `room_alias_overlap_min` (new config key) of its area becomes an alias of that region's
id (merge); otherwise it is removed. Room ids start at 0; `-1` marks "no room" in the id image.

**Object–room edges.** Paper rule (centroid cell inside the mask) with fallback to the nearest
room cell within `room_assign_fallback_m`. `RoomNode.object_ids` lists confirmed nodes only and is
rebuilt from `ObjectNode.room_id` (single source of truth).

**Labeling.** One worker thread (`labeling.worker_thread`, `false` = inline for tests), at most one
request per room in flight; `tick` applies finished results. A room needs a label when new, when
matched with IoU `< relabel_iou`, or when its confirmed object set differs from the set used for
the last label by `>= relabel_min_new_objects`. A result is applied if the room's objects changed by
fewer than `relabel_min_new_objects` meanwhile (**[DESIGN]**: an exact-match rule never labels a busy
room while mapping), else the room is re-queued. Rooms below `room_label_min_objects` are
`"unknown"` without an LLM call — also a room that was labeled and later lost its objects (split,
dedup, prune); a stale label there misled its neighbors' prompts. The JSON label is lowercased and
mapped to `"other"` if outside the enum. On LLM/parse failure the room is set to `"unknown"` and not
retried until its objects change (avoids calling a failing endpoint every tick). Cache key: sorted
(label, count) multiset + `floor(area / cache_area_bucket_m2)`. Runners call `wait_labels()` at
bag end so the final graph has no request in flight.

**Room prompt.** Adjacent labels come after the object list and are marked as weak context that may
share the room's type: with Qwen3-VL-8B and a plain "Adjacent rooms: office" line, an open-plan
office (45 chairs, 22 desks, 9 laptops) was labeled "living room" — the model avoided repeating the
neighbor's label. With the current wording the label no longer depends on the neighbors for rooms
with distinctive objects; a room with only chairs still follows its neighbor (acceptable tie-break).

## Time, snapshots, checkpoints

**Snapshots.** `view()` returns a `MemorySnapshot` sharing node objects with the writer
(copy-on-write: nodes are replaced, never mutated, and arrays are read-only), cached per
(version, time). Version increases on every visible change.

**`snapshot(t)`.** Latest checkpoint with stamp `<= t`, else an empty snapshot; `t >= now` returns
the live view. Checkpoints are kept in memory every `checkpoint_every_s` (the first one at the first
tick) with geometry stripped (`points`, `mask_cells`); stripped node copies are shared between
checkpoints while unchanged. `max_checkpoints = 0` keeps all. With a `checkpoint_dir`, each
checkpoint is also written as `ckpt_{stamp}/graph.json` + `objects.npz` + `rooms.npz` + `grid.npz`.

**`update` runs `tick`.** SYSTEM §6's loop only calls `memory.update(res, info)`, so `update` runs
the periodic jobs itself; an explicit `tick(t)` is harmless.

## Serialization

JSON as in memory spec §11.1, plus `"confirmed"` per object and `"room_aliases"`. Images are
referenced by file name; the reader resolves them against its own image root.

## Deferred (not in the first build)

Async image writer, `RemoteMemoryMirror` (M6), optional room-segmentation worker (§12),
`exact_snapshots` and `negative_evidence` (config validation rejects `true`).

# Real-bag run: io, calibration, perception (bkhn_round1)

**Bag reading.** `io/mcap_bag.py` reads the decompressed `.mcap` files directly with `mcap` +
`rosbags` deserialization (the metadata lists `.mcap.zstd`; `rosbags` would decompress 9 × 2 GB).
Odometry `odometry_bkhn_round1_point2plane.txt` = blocks of stamp + 4×4 `T_odom_livox`; stamps are
`/livox/lidar` header stamps; it is a keyframe log (gaps up to 7.4 s), so camera frames inside gaps
> `max_pose_gap_s` (0.5 s) are dropped (114 of 1722 at 2 Hz).

**Base frame.** The bag has no `base_link`; `T_map_base` is the LiDAR pose. `best_view_pose` is
therefore a LiDAR pose (fine as a navigation goal position; its height is the sensor height).

**Gravity.** `T_gravity_odom` from the mean accelerometer direction over the whole trajectory
(motion averages out, no static period needed): 0.49° tilt. The 3° seen at the start is the slope
of the starting area; the 12.8 m height gain is real (stairwell at the end).

**Camera ↔ LiDAR extrinsic.** `/tf_static` is empty. Prior = CAD translation
`livox_frame → camera_link` (0.148, 0, −0.056) m with identity rotation (user's CAD investigation) ∘
`camera_link → camera_color_optical_frame` from the bag `/tf`. Refined by joint point-to-plane ICP
of D455 depth clouds against 1 s LiDAR windows on 60 slow-rotation frames
(`scripts/calibrate_lidar_camera.py`): +9.7 cm / 3.8° from CAD, `camera_link` at
(0.116, 0.060, −0.121) m, rpy (−2.2, −1.8, 2.5)°. The ICP residual did not drop (2.58 cm both:
the MID-360 barely sees the ground the camera sees, ~30 % inliers), but pixel-wise depth agreement
favours the refined transform in 31/43 frames (indoor median |Δz| 16.5 vs 22.6 cm). Weakly
constrained; a target-based or edge-based calibration is still recommended.

**Undistortion.** `Frame` carries only `K`, so `BagSource` undistorts RGB (linear) and depth
(nearest) with the camera_info plumb_bob coefficients, keeping `K`.

**Lifting from depth.** The D455 aligned depth is dense, so observations are lifted from
`Frame.depth` (mask erosion, depth in [0.3, 6] m, MAD rejection, voxelization) instead of
projecting LiDAR points; the LiDAR path of SYSTEM §4.2 is not implemented yet.

**Masks.** FastSAM's built-in box prompt returns a de-duplicated mask set and loses the
box ↔ mask pairing; the detector runs FastSAM "everything" and gives each YOLO-World box the
mask with the best box/mask IoU (same rule), intersected with the box; below `mask_min_iou` the box
itself is the mask.

**Dynamic objects (➕).** Students' backpacks produced ~40 nodes in one minute. `person` is added
to the detector vocabulary but never lifted, and detections whose box lies > 50 % inside a person
box are dropped (`perception.dynamic`). Ridden motorbikes still pass (the bike box is not inside
the rider box).

**Association overlap normalisation (deviation from spec §6.3).** On real data a partial first
view (107 voxels) followed by a close view (2755 voxels) gave `|o ∩ n| / |o|` = 0.04 and created a
duplicate. `association.overlap_norm: min` divides by `min(|o|, |n|)` (spec value `observation`
kept as an option).

**Known gap: multi-view duplicates of large objects.** 31 of the 40 closest same-label pairs
(< 0.7 m) are the same object seen from different sides (mostly benches): centroids move
0.3–0.7 m and the seen surfaces differ (voxel overlap median 0.08). Geometry alone cannot fix it:
a 2D-footprint gate would also merge tightly parked motorbikes (their footprints overlap 0.86–1.0,
more than the duplicates). Candidate fix: an appearance cue (CLIP embedding of best-view crops) in
association and dedup.

# Second bag: vr_h3_2_calib_1 (humanoid, TF poses, no depth camera)

**Bag format.** sqlite3 `.db3` read through `rosbags` (`io/rosbag2_reader.py`); `open_bag` picks
`McapBag` for directories with decompressed `.mcap` files. Raw `sensor_msgs/Image` (`bgra8`,
row padding via `step`) is decoded alongside `CompressedImage`.

**Poses from TF (`poses.kind: tf`).** The robot publishes `base_link -> odom_lidar` (LiDAR
odometry, inverted tree), `odom_lidar -> odom -> map` (identity) and the joint chain
`pelvis -> waist_yaw -> waist_roll -> head_yaw -> livox_frame` (LiDAR on the head, upside down).
`io/tf_buffer.py` builds the tree from `/tf_static` + `/tf` and looks up `T_map_frame(t)` along
the chain, interpolating dynamic edges (joints were static here: ≤ 0.05° change, but heads may
move in other bags). `odom_lidar` is used as map frame: it is gravity aligned (base z within
0.02–0.20 m over 22 m), so no `T_gravity`.

**Missing optical frame.** Images are stamped `waist_front_zed_onboard_left_camera_frame_optical`,
which the TF tree does not contain; the standard ROS body→optical rotation (rpy −90, 0, −90) is
added from the bag config (`extra_static_tf`). Verified with LiDAR-on-image overlays
(`scripts/check_extrinsics.py`): floor, sofa, cabinet and plant edges line up.

**Lifting without depth.** With no depth topic, the LiDAR window (1 s) is projected into the
camera with a z-buffer (`perception/lidar_depth.py`) and the sparse depth goes through the same
mask lifting (erosion, MAD, median); `max_depth_m` 10 m for LiDAR depth. Head-LiDAR / waist-camera
parallax is left to the MAD filter. The MID-360 does not see the floor right in front of the robot.

**Consolidation cost.** On 1920×1080 frames with ~26 detections, full-image mask IoU for every
pair took 79 % of memory time (46 ms/frame). Masks lie inside their boxes, so IoU is 0 for
disjoint boxes and computed inside the union box otherwise (identical result, tested): memory
update 9.9 ms p50 / 26 ms p95.

**Observations.** 201 confirmed objects (chair 76, desk 31, potted plant 25, laptop 14, sofa 9…),
most re-observed 10–50 times. YOLO-World labels the pool table "desk"/"sofa" although "pool table"
is in the vocabulary. Multi-view duplicates of large objects remain (pool table ×3, sofa ×2–3,
round table ×2); in a dense office many close same-label pairs are genuinely different objects
(stools in a row, chairs around a table), so geometry alone cannot separate them.

# Target acquisition (TASK_target_acquisition.md)

**One VLM client, shared config (deviation).** The acquisition and planner specs each carry a
`vlm:` block; one client is shared (planner spec §3), so the client settings live in
`configs/vlm.yaml` and only verifier-specific values stay in `configs/acquisition.yaml`.
Backends: `local` (Qwen3-VL-8B-Instruct in-process with transformers, `local.devices: [1]` or
`[0, 1]` to split over both GPUs with accelerate), `openai` (vLLM / any OpenAI-compatible server,
`response_format` or `guided_json`, AUDIT 23) and `MockVLMClient` for tests. vLLM is not installed
on this machine, so `local` is the default: no constrained decoding, the JSON schema is appended to
the prompt and replies are parsed robustly (code fences, prose, `<think>` blocks), with one repair
retry. Bf16 8B takes 17 GB on one 4090; replies take 0.8–3.5 s. Replies can be cached by prompt
hash (`cache_dir`, AUDIT 27); the key includes image hashes.

**Text encoder and thresholds.** No SigLIP weights are complete on this machine; the default text
encoder is CLIP ViT-B/32 (`clip` package, the same weights YOLO-World uses). CLIP text similarities
are compressed: synonyms 0.88–0.99, *different* objects 0.78–0.91 (chair~table 0.86, tv~monitor
0.90, lamp~street lamp 0.91), so the spec default 0.75 would match almost everything. Matching:
`sim >= label_sim_threshold` (0.88) **and** `sim >= best - label_rel_margin` (0.06) over the labels
present (an exact / synonym label suppresses near-misses). Synonym table entries score 1.0. When
this embedder is injected into memory association, `assoc_label_sim` must be ≥ 0.92.

**InMemoryMemory.** Fixture-backed `MemoryView`; filters by `first_seen_t` (allowed for static
fixtures, memory spec §10); fixture objects need only the acquisition fields; rooms' `object_ids`
are derived from the objects when absent. It passes the shared `MemoryView` conformance suite
(5 implementations). `make_fixture.py --from-scene-graph` converts a memory checkpoint.

**Verifier prompt.** Rendered from three Jinja macros (header / candidate / footer) so the crop and
the full frame of each candidate are interleaved with its text context. Letters A, B, C… only; the
schema enum is generated per call; a choice outside the offered letters is an error, so the VLM can
never select a non-candidate.

**Acquirer details.** Rejected candidates are added to the per-target blacklist (so the next
round and later calls move on); force-included ids are validated against the confirmed objects
visible at `until_t` (hallucinated ids are dropped); `--no-vlm` returns the symbolic top-1.

**Eval on the office bag** (`eval/acquisition/cases_vr_h3_2.yaml`, 9 hand-labelled cases, Qwen3-VL):
VLM verification brings false commits from 0.11 (symbolic top-1) to 0.00. `rank_strategy:
confidence` (paper) misses "armchair" because many "chair" bar stools outscore the armchairs;
`confidence_x_similarity` reaches symbolic recall@3 0.71 and acceptable end-to-end 0.78 with no
false commit (paper default kept, switch documented). Objects stored under a wrong label ("pool
table" / "round table" as "desk") are unreachable by label matching (AUDIT 25); an image-text
retrieval fallback (CLIP crop vs target text) is the next step.

# NeSy task planner (TASK_nesy_task_planner.md)

**Serializer.** Relevance (label similarity to the targets, threshold 0.88) is a gate; inside each
group objects are ordered by detection score — CLIP similarities are too compressed to rank on.
The budget is `max_graph_tokens × chars_per_token`; objects are added by priority and every room
line ends with `+N more` when truncated. Without rooms everything is one `OBJECTS:` line.

**Planning images.** The top `planning_images` objects of the same priority order (relevance gate,
then score) are attached as best-view crops, each preceded by `Image of #id (label):`.

**Decomposition prompt (➕ affordance hint).** On the office bag "help me get rid of the empty cup"
was first decomposed into finding the cup; the prompt now asks for the objects needed to act
(throw away → trash can, rest → sofa / armchair, print → printer). After the change Qwen3-VL
plans trash can + sofa + armchair.

**Validation additions (➕).** `task_done` from the replanner is ignored while any target is still
active: a target ends only when reported or explicitly failed in `target_updates`. On the bag the
VLM had confused "candidate seen" with "reported" and closed the task with two targets open. A
decomposition without a single valid target fails the task (no deterministic fallback possible).

**Target notes.** The FSM keeps `TargetEntry.notes` in sync (candidate seen / verified, robot on
the way / reported / goto failed) and the replan prompt explains the statuses; stale notes had
misled the replanner.

**Explore candidate check (➕).** Every `check_period_s` the symbolic filter runs on the live
memory, excluding blacklisted nodes and nodes that already triggered `candidate_found` in this
task. After acquisition rejects all candidates of a target, only a candidate scoring above the
best rejected one can re-trigger (`_reject_floor`); otherwise "armchair" re-triggered on each of
76 "chair" nodes.

**Threading.** Decomposition / replanning + acquisition of the first subtask run as one job in a
single worker thread; results for an outdated task are discarded; `blocking: true` runs them
inline (offline replays). Acquirer access is serialized with a lock.

**Bag end.** `BagEventSim.finish` closes the open subtask (`nav_failed` / `no_frontier`); the
resulting replan may dispatch one more subtask that has no data left to run on.

**Trajectory mode caveat.** The goal of a `goto` is the best-view pose, i.e. where the robot was
when it saw the object; on a fixed bag path a goto dispatched right after the detection is
"reached" immediately. `oracle` mode (reached after `oracle_delay_s`) exercises the same flow
without that shortcut.

**Run mode A runtime.** `runtime/system_runtime.py` + `runtime/builder.py` implement SYSTEM §6 for
bags (no exploration yet); `scripts/run_system_bag.py` runs one task with optional Rerun recording
(planner log, goal and reported objects in 3D); `eval/planner/run_openloop.py` runs one replay per
GT task and `eval/planner/metrics.py` computes the §9 metrics.

# Occupancy grid (SYSTEM S3, `vlnav/mapping/`)

One `OccupancyGrid2D` is built from `configs/mapping.yaml` (+ run-config key `mapping:`) and shared by
the runtime (writer), memory (rooms, best-view free check) and later exploration. Per frame
(SYSTEM §6): `info = grid.update(points_of(frame), frame.sensor_origin_map, t)`, then
`memory.update(res, info)`. `points_of` = the LiDAR window in the map frame if present, else dense
depth via `perception/depth_utils.depth_to_points` (stride, depth range from `depth_points`).

**Ground (`ground.py`).** Height above ground decides floor / obstacle / overhead. `cell_min`
(default): per-cell minimum z seen so far, smoothed by a minimum filter of `smooth_radius_m`, so
cells whose floor is hidden (under a table) take the floor of their neighborhood. Measured on
vr_h3_2: the floor z drifts from −0.88 to −0.75 m along the walk (odometry), there are no
below-floor returns, and the LiDAR sees the floor from ~1.2 m — a constant floor height would mark
ramps / drift as obstacles (unit test: a 10 % ramp is free with `cell_min`, occupied with
`constant`). The min filter lags on slopes by at most `slope × smooth_radius_m`. `constant` is kept
for flat sim scenes. `negative_obstacles: true` (stairs down / holes) is rejected by validation:
with `cell_min`, a hole lowers its own ground.

**Classification.** Height `h = z − ground`: `obstacle_z_min ≤ h ≤ obstacle_z_max` → hit (occupied
evidence); `h < obstacle_z_min` → floor (free evidence for its own cell); above → ignored (ceiling,
no ray). Points beyond `max_range_m` (horizontal) are dropped; hits are subsampled to `max_points`.

**Log-odds.** `hit +0.85`, `miss −0.4`, clamped to [−2, 3.5]; ternary state from thresholds
(`occupied ≥ 0.7`, `free ≤ −0.7`), so one scan marks walls occupied but free space needs two
(one stray ray does not free a cell). A cell hit in this update gets no miss in the same update.
The ternary `int8` array is the source of truth for readers (`as_array`, rooms, planner);
`UpdateInfo.changed_bbox_cells` = bbox of cells whose ternary state changed (`(0, 0, -1, -1)` = none).

**Ray casting [DESIGN].** All endpoints share the sensor origin, so they are binned by azimuth
(`ray_angle_bins = 1440`, 0.25°; narrower than a 5 cm cell up to ~11 m) and one ray per bin is
rasterized (`ray_step_factor × resolution` steps) to the farthest evidence in the bin: a floor
point frees its own cell, an obstacle frees up to one cell before it. Per-point rays over 90 k
points cost 180 ms/frame on vr_h3_2; binned rays 18 ms (p95 20 ms) with the same maps in the unit
tests. Trade-off: in a bin, a near obstacle behind which the LiDAR also sees farther floor (through
a gap narrower than the bin) gets a miss; the hit in the same update wins, so it stays occupied.

**Growth.** The grid starts empty (`shape (0, 0)`); the first update allocates `initial_size_m`
around the sensor; points outside grow it by `grow_margin_m`, keeping the origin on the same cell
lattice (content and ground are pasted, never resampled). Rooms' id image handles origin shifts.

**`ExternalGrid`.** Wraps a `nav_msgs/OccupancyGrid` (`-1` unknown, `≥ external.occupied_threshold`
occupied, `≤ free_threshold` free); `update()` raises (read-only, filled by `set_from_occupancy`).

**vr_h3_2 result.** 808 × 833 cells (40 × 42 m), 536 m² free; 4 rooms: open office 295 m²
(129 objects, "office"), lounge 154 m² (sofas / armchairs, "living room"), a 26 m² side room with
5 chairs, a 12 m² empty alcove ("unknown"). Room segmentation runs on about half the frames (free
area grows ≥ 2 m² per frame while mapping), 116 ms p95, inside the 300 ms budget of memory §12;
`memory.update` itself stays at 11 ms p50. The full system on the coffee task gives the same
reports as without rooms; the replanner now attaches `room_hint: "living room"` to the armchair.


# Navigation (SYSTEM S4, `vlnav/navigation/`)

**GridAStar = Dijkstra in compiled code [DESIGN].** 8-connected shortest paths on the shared grid,
occupied cells inflated by `robot_radius_m` (0.3 m), unknown cells traversable at `unknown_cost`
(3 × free) so frontiers are reachable. Solved with scikit-image `MCP_Geometric` from the start cell
instead of a per-goal Python A*: same optimal paths, and one search per (grid version, start cell)
answers every goal of a selection step — 114 ms for the full 833 × 808 office grid, 0.25 ms per
extracted path; `reach_costs` returns path costs for many cells at once (instance approach cells).
The goal snaps to the nearest free, non-inflated cell within `goal_snap_radius_m`; the start snaps
to the nearest traversable cell within `start_snap_radius_m` (new key: a walking robot brushes the
inflation zone). The controller is only an interface until the Habitat harness (E5).

# NeSy exploration (TASK_nesy_exploration.md, `vlnav/exploration/`)

Built: E1–E4 + bag shadow mode (run mode A) + the top-down debug view; Habitat harness and
ObjectNav evaluation (E5–E6) are not built (no habitat-sim in `fastvggt`, no HM3D data checked).
Exploration imports only `common`, `mapping`, `navigation` (tested).

**Angle reference.** `robot` (paper Eq. 1–2) is validated on the first frame of each camera: more
than one camera, or a camera whose optical axis is yawed more than `max_camera_yaw_offset_deg` (new
key, 2°) from the base heading, raises (AUDIT 2.1). vr_h3_2: ZED yaw offset 1.0°, robot reference OK.

**Frontier cells.** A bool mask aligned with the grid (re-padded when the grid grows); cells outside
the grid count as unknown. Wedge cells are tested every frame; frontier cells in
`changed_bbox_cells` + 1 cell are re-validated (a neighbor change can end a frontier just outside
the bbox). With `outside_fov: zero_score` every new frontier cell of the changed bbox is added too.

**Clusters and points.** Components (8-connected) are re-labeled every update (≈ 3 ms); only
components within 2 cells of the touched region are re-sampled (a removed cell can split a
neighbor). **Points on frontier cells survive re-sampling [DESIGN]:** old points still lying on a
re-sampled component are kept (id, score) and farthest-point sampling only fills up to
n = ceil(size · res / spacing), starting at the cell nearest the centroid when there is no old
point. Re-sampling from scratch replaced most goals between updates. With this rule the
incremental update is exactly equal (points and ids) to re-clustering every component (tested).
New points take the id of the nearest old point within `inherit_radius_m` (greedy, each id once)
and the max blacklist of nearby old points.

**VL score inheritance anchored where measured [DESIGN].** A point inherits a VL score only from an
old score measured (`vl_xy`) within `inherit_radius_m` — scores valid for the current subtask
first, then the highest. Inheriting by the old point's current position let one measurement creep
0.5 m per re-clustering along a cluster: on vr_h3_2, 20 of 51 points carried the same 0.276 after
20 frames; anchored, 3.

**`min_cluster_cells` per resolution.** The spec keys are in cells; the grid is 5 cm (S3 choice). On
vr_h3_2, 61 of 70 frontier clusters were shorter than 1 m — mostly LiDAR shadows behind furniture.
`configs/exploration.yaml` keeps the spec default (5); the vr_h3_2 run configs use 10 (0.5 m).

**VL score.** σ in radians; S_VL = 0 outside the wedge / beyond R; a frame with no relevant
detection still assigns 0.0 (the point counts as scored). `conf_weighted` keeps the sum of C(Δθ)
per point. Strict paper mode (`rescore_on_new_subtask: false`): on a new explore subtask every
point is set to 0 and marked scored; only points created in a frame are scored, never re-scored.
"Has VL evidence" in the debug view and stats means S_VL ≥ 0.01: the Gaussian tails make almost
every point in a wedge with a detection slightly positive.

**Instances.** An observation is relevant when its (label, bbox) matches a relevant detection
(`Observation` has no detection index). `vl_eq3` ranking evaluates Eq. 3 at the instance's own
bearing without the frontier range gate R (instances are lifted detections up to 10 m). Approach
point: free, non-inflated cells in [`inst_standoff_min_m`, `inst_standoff_max_m`], nearest to the
robot among those *reachable* (path cost field), and one plan per instance and step. Trying
approach cells in distance order let one instance whose nearest cells lay behind a wall use the
whole `max_reach_checks` budget every step, so no frontier was ever checked (8-step hold on
vr_h3_2). An instance is `unreachable` after `max_plan_failures` steps without any reachable
approach cell.

**Curiosity.** The unknown-ratio BFS = the 8-connected component of non-occupied cells containing
g inside a disk of `unknown_bfs_radius_m`; cached per (point id, grid version). `dist_metric: path`
uses `GridAStar.cost_to`.

**Selection.** Reach checks are counted per plan; a candidate whose options were not all tried
because the budget ran out gets no failure. Hysteresis: the current goal is re-planned (or its path
reused while the grid version is unchanged and `replan_period_s` has not passed) and kept unless a
challenger of the same kind beats it by `switch_margin`; an instance always preempts a frontier
goal, a frontier never preempts an instance goal. Unreachable frontiers are blacklisted for
`blacklist_s` after `max_plan_failures` steps; the stuck clock starts when a goal is first set.
**`no_frontier` on hold [DESIGN]:** emitted once per subtask after `no_frontier_patience_s` of
holding (nothing selectable), not only when no point exists — AUDIT pitfall 20 names this event as
the escape from the δ_reached / all-blacklisted deadlock. Goal pose yaw: towards the goal from the
robot (frontier), towards the instance from the approach point (instance). `initial_spin: true` is
rejected until a controller exists.

**Bag shadow mode.** Mode A runs exploration on the same grid and subtasks; its outputs and events
are logged (`RunLog.exploration`, `summary.json` → `exploration_shadow`), the planner keeps getting
`reached` / `nav_failed` / `no_frontier` from `BagEventSim`. `stuck_detection: false` (new key) in
the bag configs: the recorded robot does not follow our goals. In the coffee task the shadow
`reached` events coincide with the bag simulator's.

**vr_h3_2 shadow results** (`scripts/run_exploration_bag.py`, 184 steps, 98 s): exploration step
110 ms p50 (the path search; `on_frame` 6 ms). Target "sofa": instance goals on 183 steps (sofas
are visible most of the walk), 25 instance points, 9 visited by the recorded path. Target
"printer": frontier goals until the printer is detected at ≈ 60 s, then instance goals; 28 goal
switches per minute, 39 of 46 because the goal frontier disappeared — replayed and checked: 22
cells became interior free space, 15 clusters shrank below `min_cluster_cells`, 2 cells turned
unknown (people). The walking robot's 360° LiDAR consumes the nearby frontiers that S_dist favors,
so in shadow mode switching reflects the map, not oscillation.

## Live ROS 2 adapter [DESIGN]

`ros2/vlnav_ros` implements a single-process sensor-to-goal node using the same core builder.
This is a first live adapter, not the split core/planner deployment or RemoteMemoryMirror in
SYSTEM §7. It publishes standard ROS messages rather than introducing custom message packages;
planner status and TargetReport use JSON in std_msgs/String. See [ros2.md](ros2.md).

LiveRuntime feeds exploration's actual TF-based arrival/failure events to PlannerCore and
clears selections during replanning. One worker owns mutable pipeline state; ROS callbacks only
buffer bounded sensor inputs and publish snapshots. Images are dropped when inference falls
behind. Each LiDAR scan is transformed at its acquisition timestamp before window accumulation,
matching BagSource. Selected goals without a reachable path are withheld. An active flag,
empty paths and marker deletion express cancellation because PoseStamped cannot represent it.
No velocity commands or autonomous motion controller are added.

## Move subtask (DualVLN S2) [DESIGN]

Plan: [plans/TASK_move_subtask.md](../plans/TASK_move_subtask.md). M0 was run on vr_h3_2_calib_1:
4 hand-written cases (`eval/move/cases_vr_h3_2.yaml`), a query every 2 s, 37 cycles
(`scripts/check_dualvln_s2.py`, `runs/m0/`).

- **Server.** `scripts/serve_dualvln_s2.py` runs only S2 (no S1, no depth) in the `internnav` conda env
  with the upstream InternNav checkout. The FOCA fork needs a `common` module that is not installed.
  It needs two shims: a `pkg_resources` stub for LongCLIP, and `--aux-ckpt-dir`, because
  `MODEL_PATH_TO` points at a depth checkpoint that only exists in the FOCA checkout. The server is
  stateless. The client sends the full conversation for every query, so replies can be cached for bag
  replays. `pixel_goal_grounding_weight` is newly initialized at load; it belongs to S1 and is unused.
  A period is appended to instructions that lack one (InternNav's datasets end with one).
- **Model image (V1).** In Habitat, DualVLN outputs a pixel only on the `↓` image: the raw 640×480,
  hfov 79° frame of a camera pitched 30° down. The ZED waist camera (1920×1080, hfov ≈ 104°,
  pitched ≈ 29° down) already looks like that view. Sending the full frame gave pixels in a
  640×480 range, so they landed in the wrong place. `move/view.py` crops the camera image around the
  principal point to hfov 79° and 4:3, and resizes it to 640×480. The model sees this view for the
  current image, the look-down image and history (history is resized to 384×384 as in InternNav).
  Replies are mapped back with `view.to_camera`. Median distance from the recorded future path:
  117 px for the view frame, 336 px when read as 384×384 coordinates.
- **Look-down.** The main query never returned a pixel: `↓` 73%, STOP 16%, `←` ×3–4 otherwise.
  The look-down query on the same view image always returned a pixel (27/27). So
  `look_down_mode: same_frame` with `max_look_down: 2`, and one S2 cycle is usually 2 queries.
  The frame of a pixel on a forward query is unknown (`main_pixel_ref: resized`, unverified).
- **Placement and latency.** DualVLN uses 15.6 GiB (bf16, flash-attn 2) on GPU 0, next to perception:
  peak 20.9 of 24.5 GB with both running. Qwen3-VL stays on GPU 1. Measured p50/p95:
  forward query 390/439 ms, look-down 594/701 ms, so about 1 s per cycle. Cycles run in a worker
  thread. The reply is projected with the geometry of the frame it was asked on, and replies from
  an older subtask are dropped.
  Later runs showed two problems:
  - The server's allocator cache grew to 19.6 GiB after a few hundred requests. Perception then
    ran out of memory on GPU 0. The server now calls `torch.cuda.empty_cache()` after every
    request, which keeps it at 17.4 GB. Latency did not get worse: forward 420/465 ms, look-down
    634/659 ms on the vr_h3_2 cases.
  - `detector.device: cuda:1` was silently ignored. Ultralytics' `select_device("cuda:1")` sets
    `CUDA_VISIBLE_DEVICES`, which has no effect once CUDA is initialised, and returns `cuda:0`.
    The detector now passes a `torch.device`.
  Full-system runs with move use `configs/runs/vr_h3_2_system_move.yaml`: DualVLN on GPU 0;
  perception and Qwen3-VL on GPU 1.
- **Projection (M2).** Floor-plane projection (ray ∩ floor, no depth) was checked against sensor
  floor points (`scripts/check_move_projection.py`). Median horizontal error within 4 m: 0.076 m on
  vr_h3_2 (LiDAR reference) and 0.070 m on bkhn_round1 (D455 depth reference). Depth only clips
  goals behind obstacles. The floor comes from `projection.floor_fn`, not the raw grid ground: the
  min-z ground next to the robot holds body points (LiDAR blind zone). With the raw ground, the
  first cycles of a run hit the "floor" ~1 m short, and bkhn's 0.5–2 m bin had a 0.78 m median
  (now 0.066 m). The floor is assumed locally flat (one refinement step; steps over 0.5 m are
  ignored). In bkhn's stairwell p90 stays above 1 m.
- **Depth margin.** At 6–8 m range the 15 px sparse-LiDAR window picks up nearer floor returns
  (measured range 0.4–0.9 m short), so a fixed 0.3 m margin falsely clipped 12 of 29 vr_h3_2 goals.
  The margin is `max(0.3 m, 10% of the floor range)`; 1 clip is left.
- **Shadow eval (M4).** `scripts/run_move_bag.py`, 8 hand-written cases (`eval/move/cases_*.yaml`),
  S2 blocking, replies cached in `runs/s2_cache_*`. A terminal event restarts the same instruction,
  so each whole segment is scored.

  | bag | cycles | pixel → goal | waypoint–path median | < 0.5 m | events |
  |---|---|---|---|---|---|
  | vr_h3_2 (4 cases) | 39 | 27/29 (93%) | 0.60 m | 44% | reached 2 |
  | bkhn_round1 (4 cases) | 104 | 67/83 (81%) | 0.46 m | 52% | reached 4, nav_failed 3 |

  Per-case medians range from 0.17 m (straight corridor, plaza) to 1.3–1.8 m (turns). Turns are
  the weak spot: after a turn S2 often keeps pointing straight ahead, or answers `←←←←`.
  Projection takes 0.8–2.8 ms. The goal step (projection + reachability A*) takes 80–90 ms p50.
  bkhn's 14 `unreachable` replies have one cause: the outdoor grid is speckled with single-cell
  obstacles (LiDAR noise, people). With 0.3 m inflation every walked-back point is blocked. After
  a failure the next query is sent on the next frame, so 3 failures happen within 1.5 s and give
  `nav_failed`. Premature STOPs: the restart hides them in the metrics, but `stop.first_t_rel` shows
  them (e.g. sofa_to_window stops at 7.2 s of 11 s).
- **Planner FSM (M5).** A `move` subtask is a route, not a target, so it never creates a
  `TargetEntry` or a `TargetReport`; its label is free text. The plan's "reached → next subtask" is
  implemented the way every other subtask ends: the planner replans, and the move's outcome and
  instruction are in the replan history. The replanner sees what is left of the instruction, so
  "find X, then walk to Y" works without a subtask queue. Other rules:
  - The decomposer plans one move per route, in instruction order.
  - With move enabled, finishing the last target also goes to the replanner (one more VLM call)
    instead of ending the task: "find a printer, then walk to the dining tables" has a route
    after its only target. Without this, the planner never dispatched that route (open-loop
    eval r04).
  - Landmarks: Qwen3-VL-8B lists objects that only describe a route ("stop near the plant",
    "walk past the beige sofa") as targets, even when the prompt tells it not to. Validation drops a
    target whose head noun occurs in the instruction only within 4 words after a landmark word
    (past, near, along, by, at, next, ...). Implicit targets, which are not in the text, are kept.
    The rule is English-only and conservative: "walk past the bar chairs and find an office chair"
    keeps both chairs.
  - Object search planned as a route ("take me to an armchair" as a `move`): when a move's
    instruction names an active target outside a landmark phrase, validation turns it into
    `explore` of that target, or drops it if that target already has a subtask. Searching for
    objects stays with the planner, which has memory and verification; DualVLN has neither.
  - A task that is only a route has no targets. It is done when the replanner says so after the
    move ends. It fails when the last move ended with `nav_failed` / `timeout` and nothing else
    is left.
  - A route that failed `max_attempts_per_target` times is dropped by validation (like a target).
  - With `move.enabled: false` the schema, prompts and validation are byte-identical to before;
    a `move` a VLM still returns is dropped with a fix note.
  - In bag replay S2 blocks, so the replay is deterministic and S2 replies can be cached
    (`move.s2.cache_dir`).
- **Open-loop planner eval (M5).** `eval/planner/run_openloop.py --enable-move` on vr_h3_2.
  `eval/planner/gt_vr_h3_2_move.yaml` has 4 route tasks (route only, route→object,
  landmark-heavy route→object, object→route) and 2 object-only controls.

  | Version | Success | Routes ok | Note |
  |---|---|---|---|
  | First prompt | 2/6 | 4/6 | landmarks became targets; "boarding pass" → wallet; route after the printer lost |
  | Final | 6/6 | 6/6 | report precision 1.0, target recall 1.0 |

  Final means: prompt with the landmark note and a move item in the JSON example, plus the
  validation rules above and a replan after the last target. The original 5 object tasks with
  move enabled: success 0.8, the same as without move, and no move was planned. Target precision
  rose from 0.8 to 0.9.
  Caveats:
  - The eval is small, the routes are hand-written, and replies are cached (temperature 0).
  - Route success is DualVLN STOP on the recorded path, not a check of the goal region; that
    needs the closed-loop runs (§13.2).
- **Planner prompts without few-shot examples.** `prompt/planner_prompt_v1.py` sends the rules as a
  system turn, then the request with its images. Few-shot examples (5 decompose + 5 replan, made-up
  scenes) were tried and removed to keep the prompt simple:
  - They cost about 0.9k tokens per call with move off, 1.8k with move on.
  - Open-loop eval on vr_h3_2, zero-shot → few-shot, live VLM:

    | Eval | Success | Target precision | Order ok | VLM calls/task |
    |---|---|---|---|---|
    | Objects (5 tasks) | 0.8 → 0.6 | 0.8 → 1.0 | — | 5.2 → 4.4 |
    | Move (6 tasks) | 1.0 → 0.83 | 0.92 → 1.0 | 0.83 → 1.0 | 5.0 → 5.5 |

  - Precision and order improved; both success drops were the TV task, decided by the fragile
    object check, not the planner. The eval is small (11 tasks): one task moves success by 0.17–0.2.

