"""M2 projection check for the move subtask (TASK_move_subtask.md §12 M2, acceptance 3).

Plays a bag, builds the occupancy grid online and, every --every-s seconds, takes reference floor
points measured by the sensors (dense depth back-projected, else the LiDAR window), keeps those
within --floor-tol of the grid's ground height, projects them into the camera and sends their
pixels back through the floor-plane projection without depth (`pixel_to_map`, §7.2). The error is
the horizontal distance between the measured point and the projected one, reported per distance
bin. Overlays (reference points colored by error) are saved for visual checks.

    PYTHONPATH=. $PY scripts/check_move_projection.py --bag-config configs/bags/bkhn_round1.yaml --out runs/m2_bkhn
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from vlnav.common.config import load_yaml
from vlnav.common.models import Pose
from vlnav.io.sources import BagSource, load_bag_config, resolve
from vlnav.mapping.config import load_mapping_config
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.move.config import load_move_config
from vlnav.move.projection import floor_fn, pixel_to_map
from vlnav.perception.depth_utils import depth_to_points
from vlnav.runtime.system_runtime import points_of

BINS = [(0.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag-config", required=True)
    ap.add_argument("--run-config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=2.0)
    ap.add_argument("--start-offset", type=float, default=None)
    ap.add_argument("--end-offset", type=float, default=None)
    ap.add_argument("--floor-tol", type=float, default=0.05)
    ap.add_argument("--per-frame", type=int, default=300)
    ap.add_argument("--overlay-every", type=int, default=5)
    args = ap.parse_args()

    rc = load_yaml(resolve(args.run_config)) if args.run_config else {}
    out = resolve(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    over = {k: v for k, v in (("start_offset_s", args.start_offset), ("end_offset_s", args.end_offset)) if v is not None}
    src = BagSource(load_bag_config(resolve(args.bag_config), {**rc.get("bag", {}), **over}))
    mapping = load_mapping_config(overrides=rc.get("mapping"))
    grid = OccupancyGrid2D.from_config(mapping)
    pc = load_move_config(overrides=rc.get("move")).projection.model_copy(update={"max_range_m": 100.0})
    rng = np.random.default_rng(0)
    rows, last, n_frames = [], -np.inf, 0
    for frame in src:
        grid.update(points_of(frame, mapping.depth_points), frame.sensor_origin_map, frame.stamp)
        if frame.stamp - last < args.every_s:
            continue
        last = frame.stamp
        if frame.depth is not None:
            ref, source = depth_to_points(frame.depth, frame.K, frame.T_map_cam, 4, 0.3, 8.0), "depth"
        else:
            ref, source = (frame.lidar_points_map if frame.lidar_points_map is not None else np.empty((0, 3))), "lidar"
        robot = Pose.from_matrix(frame.T_map_base, frame.stamp)
        rxy = np.asarray(robot.position[:2])
        floor = floor_fn(grid.ground_heights, robot.position, pc)
        T = frame.T_map_cam
        cam = (ref - T[:3, 3]) @ T[:3, :3]
        h, w = frame.rgb.shape[:2]
        ok = cam[:, 2] > 0.3
        u = cam[:, 0] / np.where(ok, cam[:, 2], 1) * frame.K[0, 0] + frame.K[0, 2]
        v = cam[:, 1] / np.where(ok, cam[:, 2], 1) * frame.K[1, 1] + frame.K[1, 2]
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.linalg.norm(ref[:, :2] - rxy, axis=1)
        ok &= (d > BINS[0][0]) & (d < BINS[-1][1])
        idx = np.nonzero(ok)[0]
        gz = np.array([grid.ground_height(ref[i, :2]) for i in idx])
        idx = idx[np.isfinite(gz) & (np.abs(ref[idx, 2] - gz) < args.floor_tol)]
        if not len(idx):
            continue
        idx = rng.choice(idx, min(args.per_frame, len(idx)), replace=False)
        n_frames += 1
        img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR) if n_frames % args.overlay_every == 0 else None
        for i in idx:
            pr = pixel_to_map((u[i], v[i]), frame.K, T, (w, h), rxy, floor, None, pc)
            if pr.p_map is None:
                continue
            err = float(np.linalg.norm(np.asarray(pr.p_map[:2]) - ref[i, :2]))
            rows.append((frame.stamp, float(d[i]), err, source, float(ref[i, 2] - robot.position[2]),
                         int(u[i]), int(v[i])))
            if img is not None:
                c = (0, 200, 0) if err < 0.15 else (0, 200, 255) if err < 0.3 else (0, 0, 255)
                cv2.circle(img, (int(u[i]), int(v[i])), 4, c, -1)
        if img is not None:
            cv2.putText(img, f"t={frame.stamp - src.t_start:.1f}s green<0.15m orange<0.3m red", (20, 40), 0, 1.0,
                        (255, 255, 255), 2)
            cv2.imwrite(str(out / "overlays" / f"{frame.stamp - src.t_start:06.1f}.jpg"),
                        cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    arr = np.array([(r[1], r[2]) for r in rows]) if rows else np.empty((0, 2))
    summary = {"bag": args.bag_config, "frames": n_frames, "points": len(rows),
               "source": rows[0][3] if rows else None, "floor_tol_m": args.floor_tol, "bins": {}}
    for lo, hi in BINS:
        e = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi), 1]
        summary["bins"][f"{lo}-{hi}m"] = None if not len(e) else {
            "n": int(len(e)), "median_m": round(float(np.median(e)), 3), "p90_m": round(float(np.percentile(e, 90)), 3)}
    within = arr[arr[:, 0] < 4.0, 1]
    summary["median_within_4m"] = round(float(np.median(within)), 3) if len(within) else None
    summary["pass_lt_0.15m"] = summary["median_within_4m"] is not None and summary["median_within_4m"] < 0.15
    np.savetxt(out / "points.csv", np.array([(r[0] - src.t_start, r[1], r[2], r[4], r[5], r[6]) for r in rows]),
               fmt="%.3f", delimiter=",", header="t,dist_m,err_m,dz_base_m,u,v")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
