# Visualization (Rerun)

`scripts/run_memory_bag.py --rerun <sinks>` streams the pipeline to [Rerun](https://rerun.io)
(`vlnav/viz/rerun_logger.py`, layout + options in `configs/viz.yaml`). Env: `fastvggt`
(rerun-sdk 0.30.2). Sinks, comma-separated:

| Sink | Use |
|---|---|
| `save` | record `<run-dir>/recording.rrd`; open anywhere with `rerun recording.rrd` (same rerun version) — for reports / sharing |
| `serve` | live web viewer for a browser on another machine (prints the URL); a detached `rerun --serve-web` server keeps serving after the replay |
| `spawn` | native viewer window on this machine's display |

```bash
cd ~/hoangpqn/vr-hybrid-vln
PY=~/miniconda3/envs/fastvggt/bin/python
# live demo at real bag speed, watched from a laptop browser, also recorded
PYTHONPATH=. $PY scripts/run_memory_bag.py --bag-config configs/bags/bkhn_round1.yaml \
  --run-config configs/runs/bkhn_round1_memory.yaml --run-dir runs/demo \
  --rerun serve,save --realtime 1.0            # --start-offset/--end-offset for a segment
# -> open http://<server-ip>:9090/?url=rerun+http://<server-ip>:9876/proxy
# stop the web server afterwards:  pkill -f "rerun --serve-web"
```

What is shown (timelines `bag_time`, `frame`; scrub with the time panel):

- **3D view** (follows the robot): LiDAR scan (height colors), accumulated map (gray, grows),
  trajectory, camera frustum with the image, memory objects (`#id label score xN_obs`; gray `?` =
  tentative), best-view rays (gray lines robot pose → object), this frame's observations, and
  highlights of nodes created (green) / merged (yellow) / best view updated (blue) in this frame.
- **Camera view**: masks + boxes by status — label color = lifted to 3D, red `[no_3d]` = no
  valid depth (too far / sky), gray `[on_dynamic]` = dropped because inside a person box, thin
  white = person; plus memory objects projected into the image (`#id label`) to check association.
- **Occupancy + rooms** (2D, every `map_every_frames` frames): occupancy grid (light = free,
  black = occupied, gray = unknown, north up), free cells tinted by room, `R<id> <label>` at each
  room centroid, robot position (blue). The 3D view shows room outlines at floor height with the
  same labels (`world/rooms`).
- **Exploration (top-down)** (exploration and system runs; first tab): the exploration debug view of
  every step — occupancy (white free, dark occupied, gray unknown), frontier cells (light blue), FoV
  wedge (yellow), robot (blue disc + heading) and its path so far, frontier points colored by S_NeSy
  (turbo: blue low → red high; green ring = S_VL ≥ 0.01; small gray = not selectable, × =
  blacklisted), the top candidates numbered, instance points (diamonds: orange active, green
  visited, red unreachable; small square = approach point), the goal (magenta ring) with its A*
  path; text panel with the task (instruction, planner state, plan: `[x]` reported, `[>]` current
  target, `[ ]` pending, `[!]` failed — in exploration-only runs the fixed subtask, text via
  `--instruction`), the subtask, goal, counts, the VL detections of the frame (label
  confidence@bearing) and the candidate table S_NeSy = w_dist·S_dist + w_VL·S_VL·S_unknown. Hover a
  point for its breakdown. The 3D view shows the selectable frontier points, the goal and the path
  at floor height (`world/exploration`); goal changes, events and notes go to the log tab
  "exploration". The same frames are in `<run-dir>/exploration/shadow.mp4`.
- **Plots / log**: memory size (confirmed / tentative), timing (perception, mapping, memory update),
  detections / observations per frame, memory event log, latest best-view crop.

Debug tips: click any object to see its components; select an entity in the stream tree and the
time panel shows when it changed; set the 3D view's visible time range to "all" on
`world/observations` to see every observation ever made.
