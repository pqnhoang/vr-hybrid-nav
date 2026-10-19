# vr-hybrid-vln

A re-implementation of **VL-Nav** ([arXiv:2502.00931](https://arxiv.org/abs/2502.00931)). Three parts:

- **Task planner**: Qwen3-VL splits an instruction into `explore` / `goto` subtasks, looks up
  memory, and checks the target in images before reporting it.
- **Unified memory**: a scene graph of objects and rooms, with a best-view image per object.
- **Exploration**: scores frontier points and object instances (paper Eq. 1–8) and picks the next goal.

It runs offline on recorded ROS 2 bags (no ROS install needed), or live as a ROS 2 node
([docs/ros2.md](docs/ros2.md)).

**Status:** bag replay and the live ROS 2 node work. Not built yet: the Habitat simulator
(closed-loop eval), `RemoteMemoryMirror`, and a robot motion controller.

## Repository layout

```
vlnav/        source: io, perception, mapping, navigation, memory, acquisition, planner, exploration, runtime, viz
configs/      defaults per package; bags/ = one file per bag; runs/ = per-run overrides
scripts/      run on a bag, inspect logs, export maps, check calibration
ros2/         ROS 2 package (vlnav_ros)
eval/         planner and acquisition evaluation
tests/        unit tests (no GPU or network needed)
docs/         detailed guides
plans/        the spec (plans/SYSTEM.md is the main one)
checkpoint/   model weights (not in git)
sample-data/  bags (not in git)
runs/         outputs (not in git)
```

## 1. Setup

You need Linux, an NVIDIA GPU (2 × 24 GB recommended: one for perception, one for Qwen3-VL-8B),
and Python 3.10. `ffmpeg` is optional (it writes the debug videos as H.264).

```bash
conda create -n vlnav python=3.10 -y && conda activate vlnav
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # match your CUDA driver
pip install -e ".[dev,io,perception,vlm,viz]"
export PY=$(which python)          # all commands below use $PY
```

On the lab machine `vrb-rtx5-pd`, the `fastvggt` env is ready: `export PY=~/miniconda3/envs/fastvggt/bin/python`.

> `rerun-sdk` must be exactly **0.30.2**: the viewer only opens `.rrd` files from the same version.

### Model weights

Paths are set in `configs/checkpoints.yaml` and `configs/vlm.yaml`.

```bash
mkdir -p checkpoint/{detector,segmenter,text_encoder/clip,vlm}
(cd checkpoint/detector  && $PY -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')")
(cd checkpoint/segmenter && $PY -c "from ultralytics import FastSAM; FastSAM('FastSAM-x.pt')")
$PY -c "import clip; clip.load('ViT-B/32', device='cpu', download_root='checkpoint/text_encoder/clip')"
mkdir -p ~/.cache/clip && ln -sf "$PWD/checkpoint/text_encoder/clip/ViT-B-32.pt" ~/.cache/clip/ViT-B-32.pt
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir checkpoint/vlm/Qwen3-VL-8B-Instruct   # Instruct, not Thinking (~17 GB)
```

### Data

Put bags in `sample-data/`. Each bag has a config in `configs/bags/`:

| Bag | Config | Notes |
|---|---|---|
| `vr_h3_2_calib_lidar_rgb_..._calib_1` (`.db3`) | `vr_h3_2_calib_1.yaml` | humanoid, indoor office, 98 s, ZED RGB + Livox MID-360, poses from TF |
| `bkhn_round1` (`.mcap`) | `bkhn_round1.yaml` | outdoor campus, RealSense D455 (with depth) + Livox, poses from an odometry file |

### Check the install

Run every command from the repo root.

```bash
PYTHONPATH=. $PY -m pytest -q tests/       # ~210 tests, a few seconds, no GPU
# GPU smoke test: first 10 s of a bag, no VLM needed
PYTHONPATH=. $PY scripts/run_exploration_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
  --run-config configs/runs/vr_h3_2_exploration.yaml --run-dir runs/smoke --target sofa --end-offset 10
```

## 2. Run

Prefix every command with `PYTHONPATH=. $PY`. Set `BAG=configs/bags/vr_h3_2_calib_1.yaml` first.

| Goal | Command | VLM? |
|---|---|---|
| Exploration only | `scripts/run_exploration_bag.py --bag-config $BAG --run-config configs/runs/vr_h3_2_exploration.yaml --run-dir runs/expl --target sofa --prompts sofa,armchair --rerun save` | no |
| Memory only (objects + rooms) | `scripts/run_memory_bag.py --bag-config $BAG --run-config configs/runs/vr_h3_2_memory.yaml --run-dir runs/mem --label-rooms --rerun save` | only with `--label-rooms` |
| Full system | `scripts/run_system_bag.py --bag-config $BAG --run-config configs/runs/vr_h3_2_system.yaml --run-dir runs/sys --rerun save --instruction "I need to print a document, then sit somewhere and watch TV."` | yes |
| Planner eval (5 tasks with ground truth) | `eval/planner/run_openloop.py --bag-config $BAG --run-config configs/runs/vr_h3_2_system.yaml --gt eval/planner/gt_vr_h3_2.yaml --out runs/openloop` | yes |
| Acquisition eval | `eval/acquisition/make_fixture.py --from-scene-graph runs/mem/memory/final --image-root runs/mem/memory/images --out runs/fixtures/mem --confirmed-only --link`, then `eval/acquisition/eval_acquisition.py --fixture runs/fixtures/mem/memory.json --cases eval/acquisition/cases_vr_h3_2.yaml --out runs/eval_acq/default.csv` | yes (or `--no-vlm`) |
| Export a map | `scripts/export_map.py --run-dir runs/mem` | no |
| Check LiDAR ↔ camera calibration | `scripts/check_extrinsics.py --bag-config $BAG --out-dir runs/extr` | no |

Useful flags:
- `--start-offset 15 --end-offset 30`: play only seconds 15–30 of the bag.
- `--realtime 1.0`: play at real bag speed (for demos).
- Exploration ablations: `--no-ibtp`, `--no-curiosity`, `--nearest-frontier`, `--vl-update-mode {first,max,conf_weighted}`.

Detections and VLM replies are cached on disk, so later runs are faster and give the same results.

### Outputs (in `--run-dir`)

| Path | What it holds |
|---|---|
| `summary.json`, `stats.json` | timings, rooms, reports, statistics |
| `recording.rrd` | Rerun recording (with `--rerun save`) |
| `exploration/shadow.mp4` | top-down debug video (legend in [docs/visualization.md](docs/visualization.md)) |
| `map/` | occupancy map (`.png`, plus `.pgm` + `.yaml` for ROS map_server / Nav2), rooms, trajectory |
| `memory/` | scene-graph checkpoints and best-view images |
| `decisions.jsonl`, `planner_debug/`, `reports.json` | every planner / VLM decision |

### View in Rerun

```bash
rerun runs/expl/recording.rrd                        # open a recording
rerun runs/sys_a/recording.rrd runs/sys_b/recording.rrd   # compare runs
```

Use `--rerun spawn` to open a viewer on this machine. Use `--rerun serve,save --realtime 1.0` to
watch live from another machine's browser at the printed URL. Stop it with `pkill -f "rerun --serve-web"`.

## 3. Configuration

Settings come from three layers. Each one overrides the one before it:

1. `configs/<package>.yaml`: defaults.
2. `configs/runs/*.yaml` (`--run-config`): per-run overrides, keyed by package name, e.g.
   `exploration: {scoring: {w_vl: 6.0}}`.
3. Command-line flags.

GPUs: perception runs on `cuda:0` (`configs/perception.yaml`) and Qwen3-VL on GPU 1
(`configs/vlm.yaml` → `local.devices`). Set `devices: [0, 1]` to split the model across both GPUs.
To use a vLLM server instead, run `vllm serve checkpoint/vlm/Qwen3-VL-8B-Instruct --port 8000` and set `backend: openai`.

## 4. Debug

```bash
PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl                # every goal change and why
PYTHONPATH=. $PY scripts/inspect_exploration.py runs/expl --step 40 --why 110   # why point 110 lost
PYTHONPATH=. $PY scripts/inspect_decisions.py runs/sys                   # planner decision timeline
PYTHONPATH=. $PY scripts/replay_planner_query.py runs/sys/planner_debug/003_replan --show-prompt   # re-run one VLM call
```

Prompts are in `vlnav/planner/prompts/`, `vlnav/acquisition/prompts/` and `vlnav/memory/prompts/`.
For the full debug workflow, see [docs/running.md](docs/running.md).

In bag replay the robot follows the recorded path, not the chosen goal. The goal often changes
when the LiDAR sees into a frontier. This is expected.

### Common problems

| Problem | Fix |
|---|---|
| `zsh: permission denied: scripts/...py` | `$PY` is not set |
| `checkpoint missing` / CLIP not found in `~/.cache/clip` | download the weights and create the symlink (see [Model weights](#model-weights)) |
| CUDA out of memory loading Qwen3-VL | free the GPU (`nvidia-smi`) or set `vlm: {local: {devices: [0, 1]}}` |
| `angle_reference 'robot'` error | set `exploration: {angle_reference: camera}` |
| Rerun tabs empty or old layout | click "Reset to default blueprint"; check `rerun --version` is 0.30.2 |
| Planner gives different results each run | set `vlm.cache_dir`, and `memory: {labeling: {worker_thread: false}}` |

## 5. Add a new bag

1. Copy a file in `configs/bags/` and edit the bag path, topics, pose source and frame rate.
2. Check calibration with `scripts/check_extrinsics.py`. The LiDAR points should line up with
   object edges in the image. If they don't, see `scripts/calibrate_lidar_camera.py`.
3. Create a run config in `configs/runs/`. Use the `office.txt` vocabulary indoors or `campus.txt`
   outdoors, set `detector.cache_dir`, and set `memory.rooms.enabled: false` outdoors.
4. Run memory first and check `map/` and `topdown/`. Then run exploration and the full system.

## 6. More docs

- [docs/ros2.md](docs/ros2.md): build and launch the live ROS 2 node.
- [docs/running.md](docs/running.md): all commands, outputs and debugging.
- [docs/visualization.md](docs/visualization.md): Rerun views and the map legend.
- [docs/design_choices.md](docs/design_choices.md): every design decision not in the paper, with measurements.
- [plans/](plans/): the spec.
