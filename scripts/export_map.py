"""Occupancy map files from a saved run: occupancy.{png,pgm,yaml} (ROS map_server) + rooms.png.

    PYTHONPATH=. python scripts/export_map.py --run-dir runs/sys [--ckpt <checkpoint dir>] [--out <dir>]

Uses `<run-dir>/memory/final` if it holds `grid.npz`, else the latest periodic checkpoint. The robot
path comes from `<run-dir>/map/trajectory.csv` (written by the runs) if present.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vlnav.memory import load_memory_config
from vlnav.memory.checkpoint import load_checkpoint, load_grid
from vlnav.viz.map_export import export_maps


def pick_checkpoint(run_dir: Path) -> Path:
    final = run_dir / "memory" / "final"
    if (final / "grid.npz").exists():
        return final
    ckpts = sorted(p.parent for p in (run_dir / "memory" / "checkpoints").glob("ckpt_*/grid.npz"))
    if not ckpts:
        raise SystemExit(f"no checkpoint with grid.npz under {run_dir / 'memory'}")
    print(f"{final} has no grid.npz (older run): using {ckpts[-1].name}")
    return ckpts[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, help="checkpoint dir (default: final, else latest)")
    ap.add_argument("--out", type=Path, help="default: <run-dir>/map")
    ap.add_argument("--upscale", type=int, default=2, help="rooms.png pixels per grid cell")
    args = ap.parse_args()
    ckpt = args.ckpt or pick_checkpoint(args.run_dir)
    snap = load_checkpoint(ckpt, image_root=args.run_dir / "memory" / "images",
                           max_neighbors=load_memory_config().queries.max_neighbors)
    grid = load_grid(ckpt)
    out = args.out or args.run_dir / "map"
    traj_csv = args.run_dir / "map" / "trajectory.csv"           # written by the runs
    traj = np.loadtxt(traj_csv, delimiter=",", ndmin=2) if traj_csv.exists() else None
    paths = export_maps(grid, snap, out, traj, upscale=args.upscale)
    print(f"grid {grid.shape[1]} x {grid.shape[0]} cells at {grid.resolution} m, origin {grid.origin_xy}")
    for k, p in paths.items():
        print(f"  {k:6s} {p}")


if __name__ == "__main__":
    main()
