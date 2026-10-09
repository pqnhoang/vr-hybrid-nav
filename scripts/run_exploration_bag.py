"""Bag shadow mode of NeSy exploration (TASK_nesy_exploration.md E4, SYSTEM run mode A).

bag -> perception (generic vocab + the explore prompts) -> occupancy grid -> ExplorationCore with one
fixed explore subtask. Every step it computes frontiers, VL scores, instance points and a goal + A*
path; nothing is executed (the robot follows the recorded path). No memory, planner or VLM.

    PYTHONPATH=. python scripts/run_exploration_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
        --run-config configs/runs/vr_h3_2_exploration.yaml --run-dir runs/expl_sofa \
        --target sofa --prompts sofa,armchair [--rerun save] [--no-curiosity ...]

Outputs in --run-dir/exploration: shadow.mp4 (top-down debug view per step), frames/ (every N-th
step as PNG), final.png, shadow_log.jsonl (per step: goal, candidates with score breakdown,
instances, events), stats.json (§13: frontiers, fraction with S_VL > 0, switches per minute,
timing); summary.json; recording.rrd with --rerun (tab "exploration (top-down)").
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml

from vlnav.common.config import deep_merge, load_yaml
from vlnav.common.models import Pose, Subtask
from vlnav.exploration.config import ablation_overrides, add_ablation_args
from vlnav.exploration.viz import ShadowRecorder
from vlnav.io.sources import BagSource, load_bag_config, resolve
from vlnav.mapping.config import load_mapping_config
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.runtime.builder import build_exploration, build_perception
from vlnav.runtime.system_runtime import points_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag-config", required=True)
    ap.add_argument("--run-config", required=True, help="YAML with bag/perception/mapping/navigation/exploration")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", required=True, help="explore subtask target, e.g. 'sofa'")
    ap.add_argument("--prompts", default=None, help="comma list of detector prompts (default: the target)")
    ap.add_argument("--instruction", default=None, help="text shown as the task in the debug view (display only)")
    ap.add_argument("--start-offset", type=float, default=None)
    ap.add_argument("--end-offset", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--rerun", type=str, default="", help="comma list of save,serve,spawn")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9876)
    ap.add_argument("--realtime", type=float, default=0.0, help="pace at this multiple of bag time")
    ap.add_argument("--stop-at", type=int, action="append", default=[],
                    help="open the Python debugger after step k (repeatable); see docs/running.md")
    add_ablation_args(ap)
    args = ap.parse_args()

    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rc = load_yaml(resolve(args.run_config))
    rc["exploration"] = deep_merge(rc.get("exploration", {}), ablation_overrides(args))
    prompts = [p.strip() for p in (args.prompts or args.target).split(",") if p.strip()]
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump({**rc, "args": vars(args)}))

    bag_over = {k: v for k, v in (("start_offset_s", args.start_offset), ("end_offset_s", args.end_offset))
                if v is not None}
    src = BagSource(load_bag_config(resolve(args.bag_config), {**rc.get("bag", {}), **bag_over}))
    perception = build_perception(rc)
    mapping = load_mapping_config(overrides=rc.get("mapping"))
    grid = OccupancyGrid2D.from_config(mapping)
    expl = build_exploration(rc, grid)
    st = Subtask(id=1, task_id=0, type="explore", target=args.target, detector_prompts=prompts)
    perception.set_active_prompts(prompts)          # vocab switch before exploration uses the subtask
    expl.set_subtask(st)
    recorder = ShadowRecorder(run_dir / "exploration", expl.cfg.viz)
    task = {"instruction": args.instruction, "state": "no planner (fixed explore subtask)",
            "targets": [{"name": args.target, "status": "exploring", "node_id": None}],
            "current": {"type": "explore", "target": args.target}}

    viz = None
    if args.rerun:
        from vlnav.viz.rerun_logger import RerunLogger, lan_ip, load_viz_config, start
        url = start("vlnav_exploration", args.rerun.split(","), str(run_dir / "recording.rrd"), args.web_port,
                    args.grpc_port, lan_ip(), recording_name=f"explore '{args.target}' {prompts}",
                    memory=False, planner=False, exploration=True)
        if url:
            print(f"Rerun web viewer: {url}", flush=True)
        viz = RerunLogger(load_viz_config())

    traj, t_perc, t_map, t_expl = [], [], [], []
    last_select, t_first, t_wall0 = -np.inf, None, time.time()
    for k, frame in enumerate(src):
        t = frame.stamp
        t_first = t if t_first is None else t_first
        if args.realtime > 0 and (wait := (t - t_first) / args.realtime - (time.time() - t_wall0)) > 0:
            time.sleep(wait)
        a = time.perf_counter()
        res = perception.process(frame)
        b = time.perf_counter()
        info = grid.update(points_of(frame, mapping.depth_points), frame.sensor_origin_map, t)
        c = time.perf_counter()
        expl.on_frame(res, info)
        traj.append(frame.T_map_base[:2, 3].copy())
        if t - last_select < 1.0 / expl.cfg.select_rate_hz - 1e-6:
            continue
        last_select = t
        out = expl.step(Pose.from_matrix(frame.T_map_base, t), t)
        d = time.perf_counter()
        t_perc.append(1e3 * (b - a))
        t_map.append(1e3 * (c - b))
        t_expl.append(1e3 * (d - c))
        snap = expl.snapshot()
        img, view = recorder.add(k, grid, snap, np.asarray(traj), task=task)
        if viz is not None:
            viz.log_frame(k, frame, res, perception.last_status, None, [],
                          {"perception": t_perc[-1], "mapping": t_map[-1], "exploration": t_expl[-1]})
            viz.log_exploration(img, view, snap, grid)
        for e in out.events:
            print(f"[t={t - t_first:6.1f}s] EVENT {e.type} {e.payload}", flush=True)
        if k in args.stop_at:                  # live state: expl, grid, frame, res, out, snap, img
            print(f"stopped after step {k}: inspect expl / grid / frame / res / out / snap, 'c' continues", flush=True)
            breakpoint()
        if k % args.log_every == 0:
            g = snap["goal"]
            print(f"[{k:5d}] t={t - t_first:6.1f}s goal={out.goal_kind}" + (f" #{g['id']}" if g else "")
                  + f" frontier_pts={len(snap['points'])} inst={len(snap['instances'])}"
                  f" perc={t_perc[-1]:.0f}ms map={t_map[-1]:.0f}ms expl={t_expl[-1]:.0f}ms"
                  f" wall={time.time() - t_wall0:.0f}s", flush=True)

    stats = recorder.close()

    def pct(v, q):
        return round(float(np.percentile(v, q)), 1) if v else None
    summary = {"target": args.target, "prompts": prompts, "frames": len(traj), "wall_s": round(time.time() - t_wall0, 1),
               "camera_yaw_offsets_deg": {k: round(float(np.degrees(v)), 2) for k, v in expl.camera_yaw_offsets.items()},
               "exploration": stats,
               "timing_ms": {n: {"p50": pct(v, 50), "p95": pct(v, 95)}
                             for n, v in (("perception", t_perc), ("mapping", t_map), ("exploration", t_expl))}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
