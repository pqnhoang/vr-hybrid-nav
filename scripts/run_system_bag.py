"""Run mode A end to end (SYSTEM.md §6, milestone S5): bag -> perception -> grid -> memory ->
exploration (shadow) -> planner (+ target acquisition with the VLM); BagEventSim produces the
planner's navigation events (the robot follows the bag).

  python scripts/run_system_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
      --run-config configs/runs/vr_h3_2_system.yaml --run-dir runs/system_vr_h3_2 --rerun save

Outputs in --run-dir: decisions.jsonl (planner decision log), acquisition_debug.jsonl (VLM
verification prompts / replies), reports.json, summary.json, memory/ (checkpoints, images),
map/ (occupancy.png, occupancy.pgm + .yaml for ROS map_server, rooms.png), exploration/ (shadow
mode: shadow.mp4 top-down debug video, frames/, shadow_log.jsonl, stats.json), recording.rrd (Rerun,
with --rerun; tab "exploration (top-down)").
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml

from vlnav.common.config import deep_merge, load_yaml
from vlnav.exploration.config import ablation_overrides, add_ablation_args
from vlnav.exploration.viz import ShadowRecorder
from vlnav.io.sources import resolve
from vlnav.memory.checkpoint import save_checkpoint
from vlnav.runtime.builder import build_system
from vlnav.runtime.system_runtime import TaskSpec, task_context
from vlnav.viz.map_export import export_maps


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-config", required=True)
    ap.add_argument("--run-config", required=True,
                    help="YAML with memory/perception/bag/planner/acquisition/vlm/sim overrides + tasks")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--instruction", action="append", default=None, help="overrides the config tasks")
    ap.add_argument("--task-offset", type=float, default=0.0)
    ap.add_argument("--start-offset", type=float, default=None)
    ap.add_argument("--end-offset", type=float, default=None)
    ap.add_argument("--rerun", type=str, default="", help="comma list of save,serve,spawn")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9876)
    ap.add_argument("--realtime", type=float, default=0.0,
                    help="pace the replay at this multiple of bag time (0 = as fast as possible)")
    add_ablation_args(ap)
    args = ap.parse_args()

    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rc = load_yaml(resolve(args.run_config))
    rc["exploration"] = deep_merge(rc.get("exploration", {}), ablation_overrides(args))
    tasks = ([TaskSpec(i, args.task_offset) for i in args.instruction] if args.instruction
             else [TaskSpec(**t) for t in rc.get("tasks", [])])
    if not tasks:
        raise SystemExit("no task: pass --instruction or list `tasks` in the run config")
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump({**rc, "tasks": [vars(t) for t in tasks],
                                                             "args": vars(args)}))

    bag_over = {}
    for key, val in (("start_offset_s", args.start_offset), ("end_offset_s", args.end_offset)):
        if val is not None:
            bag_over[key] = val
    t0 = time.time()
    sysm = build_system(resolve(args.bag_config), rc, run_dir, bag_over)
    print(f"system ready in {time.time() - t0:.1f} s", flush=True)
    src, perception, memory, planner, vlm = sysm.source, sysm.perception, sysm.memory, sysm.planner, sysm.vlm
    sim = sysm.sim

    hook = None
    recorder = ShadowRecorder(run_dir / "exploration", sysm.exploration.cfg.viz)   # exploration shadow (mode A)
    rt = None
    if args.rerun:
        from vlnav.viz.rerun_logger import RerunLogger, lan_ip, load_viz_config, start
        name = " | ".join(t.instruction for t in tasks)
        url = start("vlnav_system", args.rerun.split(","), str(run_dir / "recording.rrd"),
                    args.web_port, args.grpc_port, lan_ip(),
                    recording_name=(name[:120] + "…") if len(name) > 120 else name, exploration=True)
        if url:
            print(f"Rerun web viewer: {url}", flush=True)
        viz = RerunLogger(load_viz_config())
        last_v = [memory.version()]

        def hook(k, frame, res, events, dispatched, reports, decisions):
            viz.log_frame(k, frame, res, perception.last_status, memory.view(), memory.events(last_v[0]), {})
            viz.log_planner(planner.state(), dispatched, events, reports, memory.view(), decisions)
            viz.log_map(k, sysm.grid, memory.view())
            last_v[0] = memory.version()

    n_steps = [0]

    def progress(k, frame, res, events, dispatched, reports):
        decisions = planner.pop_decisions()
        if hook is not None:
            hook(k, frame, res, events, dispatched, reports, decisions)
        if len(rt.log.exploration) > n_steps[0]:       # exploration stepped this frame
            n_steps[0] = len(rt.log.exploration)
            snap = sysm.exploration.snapshot()
            img, view = recorder.add(k, sysm.grid, snap, np.asarray(rt.log.trajectory_xy),
                                     task=task_context(planner.state()))
            if hook is not None:
                viz.log_exploration(img, view, snap, sysm.grid)
            for e in rt.log.exploration[-1][1].events:
                print(f"[t={frame.stamp - src.t_start:6.1f}s] exploration (shadow) {e.type} {e.payload}", flush=True)
        for dec in decisions:
            print(f"[t={frame.stamp - src.t_start:6.1f}s] VLM {dec['kind']}: {dec.get('reasoning', '')[:160]}"
                  + (f"  [fixes: {dec['fixes']}]" if dec.get("fixes") else ""), flush=True)
        for st in dispatched:
            print(f"[t={frame.stamp - src.t_start:6.1f}s] dispatch {st.type} '{st.target}'"
                  + (f" -> node #{st.node_id}" if st.node_id is not None else "")
                  + (f" prompts={st.detector_prompts}" if st.detector_prompts else ""), flush=True)
        for e in events:
            print(f"[t={frame.stamp - src.t_start:6.1f}s] event {e.type} {e.payload}", flush=True)
        for r in reports:
            print(f"[t={frame.stamp - src.t_start:6.1f}s] REPORT {r.target}: #{r.node_id} {r.label}", flush=True)

    t_wall = time.time()
    pace0: list = []

    def pace(frame):                                   # --realtime: wait for bag time
        if args.realtime <= 0:
            return
        if not pace0:
            pace0.extend([frame.stamp, time.time()])
        wait = (frame.stamp - pace0[0]) / args.realtime - (time.time() - pace0[1])
        if wait > 0:
            time.sleep(wait)

    rt = sysm.runtime(on_frame=progress, before_frame=pace)
    log = rt.run(tasks)
    expl_stats = recorder.close()
    planner.close()
    memory.wait_labels(memory.cfg.labeling.timeout_s)
    memory.close()
    save_checkpoint(memory.view(), run_dir / "memory" / "final", sysm.grid)
    export_maps(sysm.grid, memory.view(), run_dir / "map", np.asarray(log.trajectory_xy))
    final = planner.state()
    (run_dir / "reports.json").write_text(json.dumps([r.model_dump() for r in log.reports], indent=2))
    summary = {
        "frames": log.frames, "wall_s": time.time() - t_wall, "source": src.stats,
        "tasks": [vars(t) for t in tasks],
        "final_state": {"fsm": final.fsm, "reason": final.done_reason,
                        "targets": [t.model_dump() for t in final.targets], "history": final.history},
        "dispatched": [{"t": t - src.t_start, **st.model_dump(mode="json")} for t, st in log.dispatched],
        "events": [e.model_dump() for e in log.events],
        "reports": [r.model_dump() for r in log.reports],
        "vlm": {"calls": getattr(vlm, "n_calls", None), "cache_hits": getattr(vlm, "n_cache_hits", None)},
        "perception_ms": {"p50": pct(log.perception_ms, 50), "p95": pct(log.perception_ms, 95)},
        "mapping_ms": {"p50": pct(log.mapping_ms, 50), "p95": pct(log.mapping_ms, 95)},
        "memory_ms": {"p50": pct(log.memory_ms, 50), "p95": pct(log.memory_ms, 95)},
        "exploration_ms": {"p50": pct(log.exploration_ms, 50), "p95": pct(log.exploration_ms, 95)},
        "exploration_shadow": {**expl_stats, "events": [e.model_dump() for e in log.exploration_events]},
        "rooms": [{"id": r.id, "label": r.label, "area_m2": round(r.area_m2, 1), "objects": len(r.object_ids)}
                  for r in memory.rooms()],
        "memory_confirmed": memory.stats().n_confirmed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("frames", "wall_s", "final_state", "reports", "vlm")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
