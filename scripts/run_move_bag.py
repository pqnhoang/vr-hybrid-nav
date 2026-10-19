"""Bag shadow mode of the move subtask (TASK_move_subtask.md §12 M3–M4, §13.1).

bag -> occupancy grid -> MoveCore with one fixed move subtask -> DualVLN S2 (scripts/serve_dualvln_s2.py).
Every frame it runs the S2 cycles, projects pixel goals onto the map and makes them reachable; nothing
is executed (the robot follows the recorded path). No perception, memory or planner. A terminal event
(STOP -> reached, nav_failed, timeout) is logged and the same instruction restarts as a new subtask
(--no-restart ends the case there), so the whole segment is evaluated.

    PYTHONPATH=. python scripts/run_move_bag.py --cases eval/move/cases_vr_h3_2.yaml --run-dir runs/move_vr_h3_2 \
        [--only sofa_to_window] [--cache-dir runs/s2_cache_vr_h3_2]
    PYTHONPATH=. python scripts/run_move_bag.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
        --instruction "walk to the window" --start-offset 0 --end-offset 11 --run-dir runs/move_try

Outputs per case in --run-dir/<case>: move_log.jsonl (one record per S2 cycle: request frame stamp, raw
texts, parsed reply, uv / p_map / flags, goal, decision / reason, latency, waypoint-path distance),
shadow.mp4 + frames/ (move/viz.py), summary.json (§13.1 metrics). --run-dir/summary.json pools all cases.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import yaml

from vlnav.common.config import deep_merge, load_yaml
from vlnav.common.models import Pose, Subtask
from vlnav.io.sources import BagSource, load_bag_config, resolve
from vlnav.mapping.config import load_mapping_config
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.move import CachedS2Client, HttpS2Client, S2Client, load_move_config
from vlnav.move.viz import MoveRecorder, render
from vlnav.runtime.builder import build_move
from vlnav.runtime.system_runtime import points_of


def dist_to_polyline(p, line: np.ndarray) -> float | None:
    if not len(line):
        return None
    if len(line) == 1:
        return float(np.linalg.norm(line[0] - p))
    a, b = line[:-1], line[1:]
    ab = b - a
    s = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(a + s[:, None] * ab - p, axis=1)))


def pct(v, q):
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def metrics(records: list[dict], events: list[dict], end_rel: float | None) -> dict:
    """§13.1: waypoint-path distance, projection validity, STOP timing, latency."""
    final = Counter(r["reply"]["kind"] if r["reply"] else "error" for r in records)
    pixel = [r for r in records if r["reply"] and r["reply"]["kind"] == "pixel"]
    goals = [r for r in pixel if r["decision"] in ("new_goal", "kept")]
    d_goal = [r["waypoint_dist_m"] for r in goals if r.get("waypoint_dist_m") is not None]
    d_proj = [r["proj_dist_m"] for r in pixel if r.get("proj_dist_m") is not None]
    flags = Counter(f for r in pixel for f in (r.get("projection") or {}).get("flags", [])
                    + (r.get("goal_result") or {}).get("flags", []))
    ex = [e for r in records for e in r["exchanges"]]          # cached replies keep their measured server latency
    lat_f = [e["latency_ms"] for e in ex if not e["look_down"]]
    lat_d = [e["latency_ms"] for e in ex if e["look_down"]]
    stops = [e["t_rel"] for e in events if e["type"] == "reached"]
    return {
        "cycles": len(records), "queries": sum(len(r["exchanges"]) for r in records),
        "final_reply_kinds": dict(final), "decisions": dict(Counter(r["decision"] for r in records)),
        "reasons": dict(Counter(r["reason"] for r in records if r["reason"])),
        "waypoint_path": {"n": len(d_goal), "median_m": pct(d_goal, 50), "frac_lt_0.5m":
                          round(float(np.mean(np.array(d_goal) < 0.5)), 3) if d_goal else None},
        "projected_point_path": {"n": len(d_proj), "median_m": pct(d_proj, 50)},
        "projection_validity": {"pixel_replies": len(pixel), "became_goal": len(goals),
                                "frac": round(len(goals) / len(pixel), 3) if pixel else None, "flags": dict(flags)},
        "stop": {"n": len(stops), "first_t_rel": stops[0] if stops else None,
                 "before_end_s": round(end_rel - stops[0], 2) if stops and end_rel is not None else None},
        "events": dict(Counter(e["type"] for e in events)),
        "latency_ms": {"cached_replies": sum(e["cached"] for e in ex), "forward": {"p50": pct(lat_f, 50), "p95": pct(lat_f, 95), "n": len(lat_f)},
                       "look_down": {"p50": pct(lat_d, 50), "p95": pct(lat_d, 95), "n": len(lat_d)},
                       "projection": {"p50": pct([r["projection_ms"] for r in records if "projection_ms" in r], 50),
                                      "p95": pct([r["projection_ms"] for r in records if "projection_ms" in r], 95)},
                       "handle": {"p50": pct([r["handle_ms"] for r in records], 50),
                                  "p95": pct([r["handle_ms"] for r in records], 95)}},
    }


def run_case(bag_config: str, rc: dict, s2: S2Client, case: dict, out_dir, args) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    over = {k: case[k] for k in ("start_offset_s", "end_offset_s") if case.get(k) is not None}
    src = BagSource(load_bag_config(resolve(bag_config), {**rc.get("bag", {}), **over}))
    mapping = load_mapping_config(overrides=rc.get("mapping"))
    grid = OccupancyGrid2D.from_config(mapping)
    move = build_move(rc, grid, s2)
    n_sub = 1

    def subtask():
        return Subtask(id=n_sub, task_id=0, type="move", target=case["id"], instruction=case["instruction"])
    move.set_subtask(subtask())
    recorder = None if args.no_video else MoveRecorder(out_dir, move.cfg.viz)
    traj_t, traj_xy, records, events, ev_text = [], [], [], [], []
    t_wall0 = time.time()
    for frame in src:
        t, t_rel = frame.stamp, frame.stamp - src.t_start
        grid.update(points_of(frame, mapping.depth_points), frame.sensor_origin_map, t)
        robot = Pose.from_matrix(frame.T_map_base, t)
        traj_t.append(t)
        traj_xy.append(np.asarray(robot.position[:2]))
        move.on_frame(frame)
        out = move.step(robot, t)
        for r in move.pop_records():
            r["subtask_id"] = n_sub
            records.append(r)
            print(f"[t={t_rel:6.1f}s] cycle {r['k']}: {[e['text'] for e in r['exchanges']]} -> {r['decision']}"
                  + (f" ({r['reason']})" if r["reason"] else "") + (f" goal={r.get('goal')}" if r.get("goal") else ""),
                  flush=True)
        if recorder is not None:
            snap = move.snapshot()
            img = render(grid, snap, move.last_image, [*robot.position[:2], robot.yaw()], np.asarray(traj_xy),
                         out.path, move.cfg.viz, t_rel, ev_text)
            recorder.add(img, f"sub{n_sub:02d}_cycle{snap['last']['k']:03d}" if snap["last"] else None)
        for e in out.events:
            events.append({"type": e.type, "t_rel": round(t_rel, 2), "subtask_id": e.subtask_id, **e.payload})
            ev_text.append(f"{e.type}@{t_rel:.1f}s")
            print(f"[t={t_rel:6.1f}s] EVENT {e.type} {e.payload}", flush=True)
        if move.done:
            if args.no_restart:
                break
            n_sub += 1
            move.set_subtask(subtask())
    move.close()
    if recorder is not None:
        recorder.close()

    tt, xy = np.asarray(traj_t), np.asarray(traj_xy).reshape(-1, 2)
    for r in records:                  # distance to where the recorded robot went next (§13.1)
        fut = xy[(tt >= r["frame_stamp"]) & (tt <= r["frame_stamp"] + args.future_s)]
        if r.get("goal") and r["decision"] in ("new_goal", "kept"):
            r["waypoint_dist_m"] = round(dist_to_polyline(np.asarray(r["goal"][:2]), fut), 3)
        p = (r.get("projection") or {}).get("p_map")
        if p is not None:
            r["proj_dist_m"] = round(dist_to_polyline(np.asarray(p[:2]), fut), 3)
    with open(out_dir / "move_log.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    end_rel = case.get("end_offset_s") - case.get("start_offset_s", 0.0) if case.get("end_offset_s") is not None else None
    summary = {"case": case["id"], "instruction": case["instruction"], "frames": len(traj_t),
               "wall_s": round(time.time() - t_wall0, 1), "future_s": args.future_s,
               **metrics(records, events, end_rel), "event_log": events}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None, help="eval/move/cases_<bag>.yaml (bag_config + cases)")
    ap.add_argument("--only", default=None, help="comma list of case ids")
    ap.add_argument("--bag-config", default=None, help="single run (with --instruction)")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--start-offset", type=float, default=None)
    ap.add_argument("--end-offset", type=float, default=None)
    ap.add_argument("--run-config", default=None, help="YAML with bag/mapping/navigation/move overrides")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--s2-url", default=None)
    ap.add_argument("--cache-dir", default=None, help="S2 reply cache (reproducible replays)")
    ap.add_argument("--future-s", type=float, default=10.0, help="recorded future path used for the waypoint metric")
    ap.add_argument("--async-s2", action="store_true", help="S2 in the worker thread (default: blocking replay)")
    ap.add_argument("--no-restart", action="store_true", help="end a case at its first terminal event")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    rc = load_yaml(resolve(args.run_config)) if args.run_config else {}
    rc["move"] = deep_merge(rc.get("move", {}), {"s2": {"blocking": not args.async_s2}})
    if args.cases:
        spec = load_yaml(resolve(args.cases))
        bag_config, cases = spec["bag_config"], spec["cases"]
        if args.only:
            keep = set(args.only.split(","))
            cases = [c for c in cases if c["id"] in keep]
    elif args.bag_config and args.instruction:
        bag_config = args.bag_config
        cases = [{"id": "case", "instruction": args.instruction, "start_offset_s": args.start_offset,
                  "end_offset_s": args.end_offset}]
    else:
        ap.error("--cases or --bag-config + --instruction")
    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump({**rc, "bag_config": bag_config, "cases": cases,
                                                             "args": vars(args)}))
    mcfg = load_move_config(overrides=rc["move"])
    http = HttpS2Client(args.s2_url or mcfg.s2.base_url, mcfg.s2.timeout_s)
    print(f"S2 server: {http.health()}", flush=True)
    cache = args.cache_dir or mcfg.s2.cache_dir
    s2 = CachedS2Client(http, resolve(cache)) if cache else http

    summaries, all_records = [], []
    for case in cases:
        print(f"=== {case['id']}: {case['instruction']}", flush=True)
        summaries.append(run_case(bag_config, rc, s2, case, run_dir / case["id"], args))
        with open(run_dir / case["id"] / "move_log.jsonl") as f:
            all_records += [json.loads(line) for line in f]
    events = [e for s in summaries for e in s["event_log"]]
    pooled = metrics(all_records, events, None)
    pooled.pop("stop")
    out = {"bag_config": bag_config, "pooled": pooled,
           "cases": {s["case"]: {k: v for k, v in s.items() if k != "event_log"} for s in summaries}}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pooled"], indent=2))
    for s in summaries:
        w = s["waypoint_path"]
        print(f"{s['case']:24s} cycles={s['cycles']:3d} goals={s['projection_validity']['became_goal']:3d} "
              f"wp_med={w['median_m']} frac<0.5={w['frac_lt_0.5m']} stop={s['stop']['first_t_rel']} "
              f"events={s['events']}")


if __name__ == "__main__":
    main()
