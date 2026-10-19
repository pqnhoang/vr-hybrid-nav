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

--enable-move (TASK_move_subtask.md §10): the planner may plan `move` route subtasks, run on
MoveCore with the DualVLN S2 server (scripts/serve_dualvln_s2.py, `move.s2.base_url` or --s2-url);
S2 blocks in the replay (deterministic). Outputs in move/: move_log.jsonl (one record per S2 cycle,
read with scripts/inspect_move.py), shadow.mp4 (DualVLN view with the pixel goal | map with the
projection ray, goal and path; one frame per frame of a move subtask) and frames/ (one JPEG per S2
cycle); with --rerun, tab "move (DualVLN)".
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
from vlnav.move.viz import MoveRecorder, render as render_move
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
    ap.add_argument("--enable-move", action="store_true", help="move subtasks via DualVLN (move.enabled)")
    ap.add_argument("--s2-url", default=None, help="DualVLN S2 server (default: move.s2.base_url)")
    ap.add_argument("--s2-cache-dir", default=None, help="S2 reply cache (reproducible replays)")
    add_ablation_args(ap)
    args = ap.parse_args()

    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rc = load_yaml(resolve(args.run_config))
    rc["exploration"] = deep_merge(rc.get("exploration", {}), ablation_overrides(args))
    if args.enable_move:
        s2 = {"blocking": True, **({"base_url": args.s2_url} if args.s2_url else {}),
              **({"cache_dir": args.s2_cache_dir} if args.s2_cache_dir else {})}
        rc["move"] = deep_merge(rc.get("move", {}), {"enabled": True, "s2": s2})
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
                    recording_name=(name[:120] + "…") if len(name) > 120 else name, exploration=True,
                    move=sysm.move is not None)
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
    move_rec = MoveRecorder(run_dir / "move", sysm.move.cfg.viz) if sysm.move is not None else None
    move_seen, move_events = [0, 0], []                 # rt.log.move / move_records already shown

    def show_move(frame):
        """Move subtasks: print the S2 cycles, add a frame to move/shadow.mp4 (+ Rerun)."""
        log, t_rel = rt.log, frame.stamp - src.t_start
        for r in log.move_records[move_seen[1]:]:
            print(f"[t={t_rel:6.1f}s] S2 cycle {r['k']}: {[e['text'] for e in r['exchanges']]} -> {r['decision']}"
                  + (f" ({r['reason']})" if r["reason"] else "") + (f" goal={r['goal']}" if r.get("goal") else ""),
                  flush=True)
        move_seen[1] = len(log.move_records)
        if len(log.move) == move_seen[0]:                # no move step this frame
            return
        move_seen[0] = len(log.move)
        out = log.move[-1][1]
        move_events.extend(f"{e.type}@{t_rel:.1f}s" for e in out.events)
        snap, image = rt.move_view
        yaw = float(np.arctan2(frame.T_map_base[1, 0], frame.T_map_base[0, 0]))
        img = render_move(sysm.grid, snap, image, [*frame.T_map_base[:2, 3], yaw],
                          np.asarray(log.trajectory_xy), out.path, sysm.move.cfg.viz, t_rel, move_events)
        sub = snap["subtask"]["id"] if snap["subtask"] else 0
        move_rec.add(img, f"sub{sub:02d}_cycle{snap['last']['k']:03d}" if snap["last"] else None)
        if hook is not None:
            viz.log_move(img)

    def progress(k, frame, res, events, dispatched, reports):
        decisions = planner.pop_decisions()
        if hook is not None:
            hook(k, frame, res, events, dispatched, reports, decisions)
        if move_rec is not None:
            show_move(frame)
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
                  + (f" prompts={st.detector_prompts}" if st.detector_prompts else "")
                  + (f" instruction={st.instruction!r}" if st.instruction else ""), flush=True)
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
    if sysm.move is not None:
        sysm.move.close()
        move_rec.close()
        (run_dir / "move").mkdir(exist_ok=True)
        with open(run_dir / "move" / "move_log.jsonl", "w") as f:
            for r in log.move_records:
                f.write(json.dumps(r) + "\n")
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
        "move": None if sysm.move is None else {
            "cycles": len(log.move_records),
            "decisions": {d: sum(r["decision"] == d for r in log.move_records)
                          for d in sorted({r["decision"] for r in log.move_records})}},
        "rooms": [{"id": r.id, "label": r.label, "area_m2": round(r.area_m2, 1), "objects": len(r.object_ids)}
                  for r in memory.rooms()],
        "memory_confirmed": memory.stats().n_confirmed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("frames", "wall_s", "final_state", "reports", "vlm")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
