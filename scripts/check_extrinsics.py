"""Project the LiDAR window into camera frames of a bag (extrinsic / time-sync sanity check).

  python scripts/check_extrinsics.py --bag-config configs/bags/vr_h3_2_calib_1.yaml \
      --out-dir runs/extrinsics_vr_h3_2 --n-frames 8
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from vlnav.io.sources import BagSource, load_bag_config, resolve
from vlnav.viz.overlay import project_points_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--max-range", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    probe = BagSource(load_bag_config(resolve(args.bag_config)))
    span = probe.t_end - probe.t_start
    src = BagSource(load_bag_config(resolve(args.bag_config),
                                    {"frame_rate_hz": max(args.n_frames - 1, 1) / span}))
    for k, f in enumerate(src):
        img = project_points_overlay(f.rgb, f.K, f.lidar_points_map, f.T_map_cam, args.max_range)
        cv2.putText(img, f"t={f.stamp - src.t_start:.1f}s  {len(f.lidar_points_map)} pts", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        cv2.imwrite(str(out / f"overlay_{k:02d}.jpg"),
                    cv2.cvtColor(cv2.resize(img, None, fx=args.scale, fy=args.scale), cv2.COLOR_RGB2BGR))
    print(f"{src.stats} -> {out}")


if __name__ == "__main__":
    main()
