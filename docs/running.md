# Running the system

Env: `~/miniconda3/envs/fastvggt` (Python 3.10, torch 2.11 + CUDA, ultralytics, transformers 5.5
with Qwen3-VL, clip, rerun-sdk 0.30). All commands from the repo root with `PYTHONPATH=.`.

```bash
PY=~/miniconda3/envs/fastvggt/bin/python
```

| What | Command |
|---|---|
| Unit tests (no GPU / network) | `PYTHONPATH=. $PY -m pytest -q tests/` |
| Memory only on a bag (occupancy grid + rooms; `--label-rooms` loads Qwen3-VL to name them) | `PYTHONPATH=. $PY scripts/run_memory_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_memory.yaml --run-dir runs/mem --label-rooms --rerun save` |
| Exploration bag shadow mode (E4): fixed explore subtask, top-down debug video (+ ablations `--no-ibtp --no-curiosity --nearest-frontier --vl-update-mode`) | `PYTHONPATH=. $PY scripts/run_exploration_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_exploration.yaml --run-dir runs/expl --target sofa --prompts sofa,armchair --rerun save` |
| Full system, mode A (memory + exploration shadow + planner + acquisition + Qwen3-VL) | `PYTHONPATH=. $PY scripts/run_system_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_system.yaml --run-dir runs/sys --rerun save [--instruction "..."]` |
| Acquisition fixture from a memory run | `PYTHONPATH=. $PY eval/acquisition/make_fixture.py --from-scene-graph runs/memory_vr_h3_2/memory/final --image-root runs/memory_vr_h3_2/memory/images --out runs/fixtures/vr_h3_2 --confirmed-only --link` |
| Acquisition eval (+ ablations `--no-vlm --k N --rank-strategy confidence_x_similarity --no-full-frame --no-neighbors`) | `PYTHONPATH=. $PY eval/acquisition/eval_acquisition.py --fixture runs/fixtures/vr_h3_2/memory.json --cases eval/acquisition/cases_vr_h3_2.yaml --out runs/eval_acq/default.csv` |
| Planner open-loop eval (+ `--no-planning-images --k --rank-strategy`) | `PYTHONPATH=. $PY eval/planner/run_openloop.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --run-config configs/runs/vr_h3_2_system.yaml --gt eval/planner/gt_vr_h3_2.yaml --out runs/openloop_vr_h3_2` |
| Recompute planner metrics | `PYTHONPATH=. $PY eval/planner/metrics.py --gt eval/planner/gt_vr_h3_2.yaml --runs runs/openloop_vr_h3_2` |
| Occupancy map files from a saved run (runs also write them to `<run-dir>/map/`) | `PYTHONPATH=. $PY scripts/export_map.py --run-dir runs/mem` |
| Extrinsic check on a bag | `PYTHONPATH=. $PY scripts/check_extrinsics.py --bag-config configs/bags/vr_h3_2_calib_1.yaml --out-dir runs/extr` |
| Live viewer from a laptop | add `--rerun serve,save --realtime 1.0`, open the printed URL (see docs/visualization.md) |

## VLM

`configs/vlm.yaml`: `backend: local` loads `checkpoint/vlm/Qwen3-VL-8B-Instruct` in-process on GPU 1
(17 GB bf16); `local.devices: [0, 1]` splits it over both 4090s. With a vLLM server instead:
`vllm serve checkpoint/vlm/Qwen3-VL-8B-Instruct --port 8000` and `backend: openai`
(`model` = the served name, i.e. the same path). `cache_dir` makes reruns with identical prompts
reproducible and free. Perception (YOLO-World + FastSAM) runs on GPU 0.

## Outputs of a system run

`decisions.jsonl` (planner decision log: prompts hash, graph excerpt, raw VLM reply, parsed plan,
validation fixes, acquisition result, dispatched subtask), `acquisition_debug.jsonl` (verifier
prompts without images, candidates + context, raw replies), `reports.json`, `summary.json`,
`memory/` (checkpoints with `grid.npz`, best-view images, final graph), `map/` (see below),
`recording.rrd` (Rerun:
planner tab, occupancy + rooms tab, goal and reported objects in 3D). `summary.json` also has
`mapping_ms` and the final rooms (id, label, area, number of objects).

## Occupancy grid and rooms

Top-down map files in `<run-dir>/map/` (memory and system runs):

| File | Content |
|---|---|
| `occupancy.png` | plain 2D occupancy map, north up, 1 px = 1 cell (5 cm): white free, black occupied, gray unknown |
| `occupancy.pgm` + `occupancy.yaml` | same map in ROS `map_server` / Nav2 format (`origin` = map-frame corner of the lower-left pixel); `vlnav.mapping.export.load_map` reads it back |
| `rooms.png` | map with rooms tinted, `R<id> <label>` boxes, robot path (blue), 5 m scale bar; no object dots (`topdown/final.png` has those) |
| `trajectory.csv` | robot path (x, y), reused by `scripts/export_map.py` |

`configs/mapping.yaml` (resolution, ground model, obstacle height band, log-odds, ray bins); override
per run with a `mapping:` key in the run config. Rooms are on in the vr_h3_2 configs (indoor, one
floor) and off for bkhn_round1 (outdoor campus). Room labels come from the same VLM as the planner,
text only, in a worker thread; the final graph waits for labels still in flight. For a fully
reproducible replay set `memory: {labeling: {worker_thread: false}}` (labels inline: about 140 label
calls instead of 30 on vr_h3_2, roughly twice the wall time without the VLM cache).


## Exploration (shadow mode on bags)

The robot follows the recorded path; exploration computes what it *would* do every step (frontiers,
VL scores, instance points, goal, A* path) and emits its events into the log only. Outputs in
`<run-dir>/exploration/`:

| File | Content |
|---|---|
| `shadow.mp4` | one frame per step: top-down occupancy map + exploration overlay + text panel (legend in docs/visualization.md) |
| `frames/frame_NNNNN.png`, `final.png` | every `viz.keyframe_every`-th step, last step |
| `shadow_log.jsonl` | per step: subtask, goal (kind, id, score), path length, every frontier point with its score breakdown (`s_nesy`, `s_dist`, `dist`, `s_vl`, `s_unknown`), instances (rank, status, approach point), VL mixture of the frame, events, notes (visited / blacklisted / unreachable), timing |
| `stats.json` | spec §13: frontier points / cells, fraction of scored points with S_VL ≥ 0.01, goal switches per minute, goal kinds, events, timing |

In a full system run the subtasks come from the planner (`scripts/run_system_bag.py` writes the same
folder). Debug a decision: find the step in `shadow_log.jsonl` (or scrub the Rerun tab "exploration
(top-down)" and hover a point for its score breakdown), then compare the goal with the ranked
candidates. Tuning keys: `configs/exploration.yaml` (`scoring.w_vl` vs `w_dist` first, AUDIT
pitfall 18), per run under `exploration:` in the run config.

## Debugging exploration

Work from the outside in: watch, then query the log, then stop the replay and inspect live state.

1. **Watch.** `<run-dir>/exploration/shadow.mp4` (or `frames/frame_NNNNN.png`, NNNNN = step k), or
   `rerun <run-dir>/recording.rrd` → tab "exploration (top-down)": scrub the time line, hover a
   frontier point / instance for its score breakdown; tab "exploration" (bottom) lists goal changes,
   events and notes (visited / blacklisted / unreachable).
2. **Query the log** (`scripts/inspect_exploration.py`, reads `shadow_log.jsonl`):

   ```bash
   PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl                       # every goal change + why
   PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --step 40              # goal, ranked candidates, instances
   PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --step 40 --why 110    # why F110 was not chosen
   PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --point 202            # one frontier point over time
   PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --instance 1           # one instance over time
   ```

   Switch causes: *removed* (its cells were observed or the cluster shrank below
   `min_cluster_cells`), *not reachable this step* (plan failed), *outscored* (by more than
   `switch_margin`), *instance preempts frontier*, *blacklisted*, *within delta_reached*.
3. **Re-run a short segment with other settings.** `--start-offset 15 --end-offset 30` replays only
   that part (perception comes from the detection cache, so it is fast); change scoring under
   `exploration:` in a copy of the run config (e.g. `scoring: {w_vl: 6.0}`), or use the ablation
   flags `--no-ibtp --no-curiosity --nearest-frontier --vl-update-mode max`, and compare the two
   runs' `stats.json` and timelines.
4. **Stop at a step and inspect live state.** `--stop-at 40` opens the Python debugger after step
   40 with `expl` (ExplorationCore), `grid`, `frame`, `res`, `out`, `snap` in scope, e.g.

   ```python
   expl.frontiers.points[202]                        # FrontierPoint: s_vl, vl_xy, blacklisted_until, ...
   expl.frontiers.mask.sum(); grid.as_array()        # frontier cells; ternary grid (-1/0/1)
   xy = frame.T_map_base[:2, 3]                      # robot position
   expl.planner.plan(grid, xy, (9.0, 4.3))           # A* path (None = unreachable) to any point
   expl.planner.cost_to(grid, xy, (9.0, 4.3))        # path cost in m (inf = unreachable)
   [(d.label, round(d.score, 2)) for d in res.detections]; res.relevant_idx
   expl._mix                                          # this frame's VL mixture (mu rad, alpha)
   ```
5. **Reproduce in a synthetic scene.** `tests/exploration/expl_helpers.py` builds frames,
   detections and hand-made grids (`corridor_grid`, `frame`, `det`, `result`); the E1–E4 tests in
   `tests/exploration/` are small worked examples. Render any state with
   `vlnav.exploration.viz.render(grid, core.snapshot(), core.cfg.viz)`.

## Debugging VLM decomposition / replanning

Every system run writes, next to `decisions.jsonl`, one folder per VLM planning call
(`planner.debug_dumps: true`):

```
<run-dir>/planner_debug/003_replan/
  prompt.txt        full text sent to the VLM (images are interleaved after it)
  images/node_N.jpg best-view crops sent as planning images
  reply_0.txt       raw VLM reply (reply_1.txt = answer to the JSON repair retry)
  input.json        planner input: instruction, targets (status, attempts, notes), history, last event
  result.json       parsed JSON, validated plan + repair fixes, chosen subtask, acquisition result
  memory/           exact memory snapshot the VLM saw (graph.json + npz)
```

| Goal | Command |
|---|---|
| Timeline of a run (decisions, reasoning, events, reports) | `PYTHONPATH=. $PY scripts/inspect_decisions.py runs/sys` |
| One decision in full (input state, replies, fixes, acquisition) | `PYTHONPATH=. $PY scripts/inspect_decisions.py runs/sys --show 5 [--prompt]` |
| Re-run that decision with the current prompts / config, no bag replay | `PYTHONPATH=. $PY scripts/replay_planner_query.py runs/sys/planner_debug/006_replan [--show-prompt] [--with-acquisition] [--planning-images 0] [--instruction "..."]` |
| Target verification prompts / replies | `<run-dir>/acquisition_debug.jsonl` |
| Watch it live | Rerun recording, tab "planner" (VLM reasoning, plan, fixes, acquisition, debug folder path) |

Prompt templates: `vlnav/planner/prompts/decompose.j2`, `replan.j2`,
`vlnav/acquisition/prompts/verify_target.j2`; repair rules: `vlnav/planner/validation.py`;
FSM: `vlnav/planner/core.py`. Edit a template, then replay a dump to see the effect in ~3 s.
`vlm.cache_dir` returns cached replies for identical prompts; the replay script disables the cache.

## Different instructions -> different recordings

```bash
PYTHONPATH=. $PY scripts/run_system_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
  --run-config configs/runs/vr_h3_2_system.yaml --rerun save \
  --run-dir runs/sys_print_tv --instruction "I need to print a document, then sit somewhere and watch the news on TV."
# compare several runs in one viewer (each recording is named after its instruction, left panel):
rerun runs/sys_print_tv/recording.rrd runs/system_vr_h3_2_oracle/recording.rrd
# or watch a new one live from a laptop browser, paced at bag speed:
#   --rerun serve,save --realtime 1.0
```

The "task" tab shows the instruction, planner state, current subtask and a table of targets
(status, detector prompts, found object, notes) over time; the "planner" tab the VLM reasoning.
`sim: {bag_event_sim: {mode: oracle}}` (e.g. `runs/vr_h3_2_system_oracle.yaml`) makes gotos end with
`reached` after a delay; `trajectory` needs the recorded path to pass the goal.
