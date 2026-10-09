"""Mode A, memory only (TASK_unified_memory.md M7): bag -> perception -> UnifiedMemory.

Outputs in --run-dir:
  memory/checkpoints/ckpt_*/   graph.json + npz (every checkpoint_every_s)
  memory/images/               best-view frames
  topdown/                     top-down renders per checkpoint + final (+ zoomed final)
  map/                         occupancy.png, occupancy.pgm + .yaml (ROS map_server), rooms.png
  gallery.jpg                  best-view crops of confirmed nodes (highest score first)
  debug/                       first frames annotated with detections / lifted observations
  stats.json, events.jsonl, nodes_over_time.csv

The occupancy grid (mapping/, SYSTEM S3) is updated from the LiDAR window (or dense depth) before
memory, which segments rooms on it; `--label-rooms` loads the VLM to label them. Top-down renders
use the grid as background.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from vlnav.common.config import load_yaml
from vlnav.io.sources import BagSource, load_bag_config, resolve
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.mapping.config import load_mapping_config
from vlnav.memory import ImageStore, RoomLabeler, UnifiedMemory, load_memory_config
from vlnav.memory.checkpoint import save_checkpoint
from vlnav.memory.serialize import snapshot_to_json
from vlnav.memory.viz import (best_view_gallery, grid_background, label_color, points_background,
                              render_topdown)
from vlnav.perception.config import load_checkpoint_paths, load_perception_config
from vlnav.perception.detector import YoloWorldFastSam
from vlnav.perception.pipeline import PerceptionPipeline, VocabularyManager, load_vocab_file
from vlnav.runtime.system_runtime import points_of
from vlnav.viz.map_export import export_maps


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def annotate(res, path: Path) -> None:
    img = cv2.cvtColor(res.frame.rgb, cv2.COLOR_RGB2BGR)
    lifted = {(o.label, tuple(o.bbox)) for o in res.observations}
    for d in res.detections:
        col = label_color(d.label)[::-1]
        overlay = img.copy()
        overlay[d.mask] = col
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        x1, y1, x2, y2 = d.bbox
        ok = (d.label, tuple(d.bbox)) in lifted
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if ok else 1)
        cv2.putText(img, f"{d.label} {d.score:.2f}{'' if ok else ' (no 3D)'}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imwrite(str(path), cv2.resize(img, None, fx=0.5, fy=0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-config", required=True)
    ap.add_argument("--run-config", required=True, help="YAML with memory/perception/bag overrides")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--start-offset", type=float, default=None)
    ap.add_argument("--end-offset", type=float, default=None)
    ap.add_argument("--debug-frames", type=int, default=12)
    ap.add_argument("--topdown-m-per-px", type=float, default=0.2)
    ap.add_argument("--bg-points-per-frame", type=int, default=3000)
    ap.add_argument("--bg-height-band", type=str, default="-0.3:1.5",
                    help="LiDAR points kept for the background, relative to the sensor height")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dup-radius-m", type=float, default=1.0)
    ap.add_argument("--label-rooms", action="store_true", help="load the VLM to label rooms")
    ap.add_argument("--rerun", type=str, default="",
                    help="comma list of save,serve,spawn (Rerun live / recorded visualization)")
    ap.add_argument("--rerun-save", type=str, default=None, help="default: <run-dir>/recording.rrd")
    ap.add_argument("--rerun-host", type=str, default=None, help="address browsers use (default: LAN IP)")
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9876)
    ap.add_argument("--realtime", type=float, default=0.0,
                    help="pace the replay at this multiple of bag time (0 = as fast as possible)")
    args = ap.parse_args()

    run_dir = resolve(args.run_dir)
    for sub in ("topdown", "debug", "memory"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    run_cfg = load_yaml(resolve(args.run_config))
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump({**run_cfg, "args": vars(args)}))

    bag_over = dict(run_cfg.get("bag", {}))
    if args.start_offset is not None:
        bag_over["start_offset_s"] = args.start_offset
    if args.end_offset is not None:
        bag_over["end_offset_s"] = args.end_offset
    src = BagSource(load_bag_config(resolve(args.bag_config), bag_over))

    pcfg = load_perception_config(overrides=run_cfg.get("perception"))
    detector = YoloWorldFastSam(pcfg.detector, load_checkpoint_paths())
    vocab = VocabularyManager(load_vocab_file(resolve(pcfg.vocabulary.generic_vocab_file)),
                              dynamic=pcfg.dynamic.labels)
    perception = PerceptionPipeline(detector, vocab, pcfg.lifting, pcfg.dynamic)

    mcfg = load_memory_config(overrides=run_cfg.get("memory"))
    mapping = load_mapping_config(overrides=run_cfg.get("mapping"))
    grid = OccupancyGrid2D.from_config(mapping)
    labeler = None
    if mcfg.rooms.enabled and args.label_rooms:
        from vlnav.acquisition.vlm_client import load_vlm_config, make_vlm_client
        labeler = RoomLabeler(mcfg.labeling, make_vlm_client(load_vlm_config(overrides=run_cfg.get("vlm"))))
    store = ImageStore(run_dir / "memory" / "images", mcfg.images.image_jpeg_quality,
                       mcfg.images.crop_cache_size)
    memory = UnifiedMemory(mcfg, grid, store, labeler=labeler,
                           checkpoint_dir=run_dir / "memory" / "checkpoints")

    viz = None
    if args.rerun:
        from vlnav.viz.rerun_logger import RerunLogger, lan_ip, load_viz_config, start
        host = args.rerun_host or lan_ip()
        url = start("vlnav_memory", args.rerun.split(","), args.rerun_save or str(run_dir / "recording.rrd"),
                    args.web_port, args.grpc_port, host, planner=False)
        if url:
            print(f"Rerun web viewer: {url}", flush=True)
        viz = RerunLogger(load_viz_config())

    lo, hi = (float(v) for v in args.bg_height_band.split(":"))
    rng = np.random.default_rng(0)
    bg_pts, traj, series = [], [], []
    t_perc, t_mem, t_map, n_det, n_obs = [], [], [], [], []
    t_wall0, t_first = time.time(), None
    for k, frame in enumerate(src):
        t_first = frame.stamp if t_first is None else t_first
        if args.realtime > 0:                       # pace to bag time for live demos
            wait = (frame.stamp - t_first) / args.realtime - (time.time() - t_wall0)
            if wait > 0:
                time.sleep(wait)
        v_before = memory.version()
        a = time.perf_counter()
        res = perception.process(frame)
        b = time.perf_counter()
        info = grid.update(points_of(frame, mapping.depth_points), frame.sensor_origin_map, frame.stamp)
        m = time.perf_counter()
        memory.update(res, info)
        c = time.perf_counter()
        timings = {"perception": 1e3 * (b - a), "mapping": 1e3 * (m - b), "memory_update": 1e3 * (c - m)}
        if viz is not None:
            viz.log_frame(k, frame, res, perception.last_status, memory.view(), memory.events(v_before), timings)
            viz.log_map(k, grid, memory.view())
        t_perc.append(timings["perception"])
        t_map.append(timings["mapping"])
        t_mem.append(timings["memory_update"])
        n_det.append(len(res.detections))
        n_obs.append(len(res.observations))
        traj.append(frame.T_map_base[:2, 3].copy())
        p = frame.lidar_points_map
        if len(p):
            z0 = frame.sensor_origin_map[2]
            p = p[(p[:, 2] > z0 + lo) & (p[:, 2] < z0 + hi)]
            if len(p) > args.bg_points_per_frame:
                p = p[rng.choice(len(p), args.bg_points_per_frame, replace=False)]
            bg_pts.append(p[:, :2])
        if k < args.debug_frames:
            annotate(res, run_dir / "debug" / f"frame_{k:03d}.jpg")
        st = memory.stats()
        series.append((frame.stamp - t_first, st.n_confirmed, st.n_tentative))
        if k % args.log_every == 0:
            print(f"[{k:5d}] t={frame.stamp - t_first:7.1f}s dets={len(res.detections):2d} "
                  f"obs={len(res.observations):2d} nodes={st.n_confirmed}+{st.n_tentative} "
                  f"perc={t_perc[-1]:.0f}ms map={t_map[-1]:.0f}ms mem={t_mem[-1]:.1f}ms rooms={len(memory.rooms())} "
                  f"wall={time.time() - t_wall0:.0f}s",
                  flush=True)

    memory.wait_labels(mcfg.labeling.timeout_s)      # room labels still in flight at bag end
    memory.close()
    final = memory.view()
    save_checkpoint(final, run_dir / "memory" / "final", grid)
    (run_dir / "memory" / "final" / "scene_graph.json").write_text(snapshot_to_json(final))

    bg_xy = np.concatenate(bg_pts) if bg_pts else np.zeros((0, 2))
    traj_xy = np.asarray(traj)
    allxy = np.concatenate([bg_xy, traj_xy]) if len(bg_xy) else traj_xy
    bounds = (*(np.percentile(allxy, 0.5, axis=0) - 5), *(np.percentile(allxy, 99.5, axis=0) + 5))
    bg = grid_background(grid) if grid.as_array().size else points_background(bg_xy, bounds, args.topdown_m_per_px)
    for ck in sorted((run_dir / "memory" / "checkpoints").glob("ckpt_*")):
        t = float(ck.name.split("_")[1])
        snap = memory.snapshot(t)
        img = render_topdown(snap, bg, show_text=False, title=f"t = {t - t_first:.0f} s, "
                             f"{len(snap.all_objects())} confirmed objects")
        cv2.imwrite(str(run_dir / "topdown" / f"{ck.name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img = render_topdown(final, bg, trajectory_xy=traj_xy, show_text=False,
                         title=f"final: {len(final.all_objects())} confirmed objects")
    cv2.imwrite(str(run_dir / "topdown" / "final.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img = render_topdown(final, bg, trajectory_xy=traj_xy, m_per_px=args.topdown_m_per_px / 2,
                         show_text=True, title="final (labels)")
    cv2.imwrite(str(run_dir / "topdown" / "final_labels.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    export_maps(grid, final, run_dir / "map", traj_xy)

    ranked = sorted(final.all_objects(), key=lambda n: -n.best_score)
    gal = best_view_gallery(final, node_ids=[n.id for n in ranked], max_items=96, cols=12)
    cv2.imwrite(str(run_dir / "gallery.jpg"), cv2.cvtColor(gal, cv2.COLOR_RGB2BGR))

    near_dups = []                     # duplicate proxy without GT: same label, centroids close
    objs = final.all_objects()
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            d = float(np.linalg.norm(np.subtract(a.centroid, b.centroid)))
            if a.label == b.label and d < args.dup_radius_m:
                near_dups.append((a.id, b.id, a.label, round(d, 2)))
    by_label: dict[str, int] = {}
    for n in final.all_objects():
        by_label[n.label] = by_label.get(n.label, 0) + 1
    with open(run_dir / "events.jsonl", "w") as f:
        for e in memory.events(0):
            f.write(e.model_dump_json() + "\n")
    with open(run_dir / "nodes_over_time.csv", "w") as f:
        f.write("t_s,confirmed,tentative\n")
        f.writelines(f"{t:.2f},{c},{n}\n" for t, c, n in series)
    stats = {
        "source": src.stats, "frames": len(t_perc), "bag_span_s": series[-1][0] if series else 0,
        "wall_s": time.time() - t_wall0, "memory": memory.stats().model_dump(),
        "perception_ms": {"p50": pct(t_perc, 50), "p95": pct(t_perc, 95)},
        "mapping_ms": {"p50": pct(t_map, 50), "p95": pct(t_map, 95)},
        "memory_update_ms": {"p50": pct(t_mem, 50), "p95": pct(t_mem, 95)},
        "grid": {"shape": list(grid.shape), "resolution": grid.resolution,
                 "free_m2": float((grid.as_array() == 0).sum() * grid.resolution ** 2),
                 "occupied_m2": float((grid.as_array() == 1).sum() * grid.resolution ** 2)},
        "rooms": [{"id": r.id, "label": r.label, "label_confidence": r.label_confidence,
                   "area_m2": round(r.area_m2, 1), "objects": len(r.object_ids),
                   "adjacent": r.adjacent_room_ids} for r in final.rooms()],
        "dynamic_dropped": perception.n_dynamic_dropped,
        "detections_per_frame": float(np.mean(n_det)) if n_det else 0,
        "observations_per_frame": float(np.mean(n_obs)) if n_obs else 0,
        "confirmed_by_label": dict(sorted(by_label.items(), key=lambda kv: -kv[1])),
        "aliases": len(final.aliases),
        f"same_label_pairs_within_{args.dup_radius_m}m": len(near_dups),
        "same_label_pair_examples": near_dups[:20],
        "event_counts": {t: sum(e.type == t for e in memory.events(0))
                         for t in sorted({e.type for e in memory.events(0)})},
    }
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "memory"}, indent=2))


if __name__ == "__main__":
    main()
