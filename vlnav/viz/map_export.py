"""End-of-run map files: the plain occupancy map (+ ROS format) and a room map (rooms, labels,
robot path, scale bar; no object dots)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vlnav.mapping.export import save_map
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.memory.snapshot import MemorySnapshot
from vlnav.memory.viz import FONT, grid_background, render_topdown


def draw_scale_bar(img: np.ndarray, m_per_px: float, length_m: float = 5.0) -> None:
    h = img.shape[0]
    x0, y0, n = 15, h - 20, int(round(length_m / m_per_px))
    cv2.rectangle(img, (x0 - 5, y0 - 22), (x0 + n + 5, y0 + 8), (255, 255, 255), -1)
    cv2.line(img, (x0, y0), (x0 + n, y0), (0, 0, 0), 3)
    for x in (x0, x0 + n):
        cv2.line(img, (x, y0 - 6), (x, y0 + 6), (0, 0, 0), 2)
    cv2.putText(img, f"{length_m:g} m", (x0 + n // 2 - 15, y0 - 9), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def export_maps(grid: OccupancyGrid2D, snap: MemorySnapshot, out_dir: str | Path,
                trajectory_xy: np.ndarray | None = None, upscale: int = 2) -> dict[str, Path]:
    """<out_dir>/occupancy.{png,pgm,yaml} + rooms.png (+ trajectory.csv, so a later re-export
    with `scripts/export_map.py` keeps the path). Nothing is written for an empty grid."""
    if 0 in grid.shape:
        return {}
    paths = save_map(grid, out_dir)
    traj_csv = Path(out_dir) / "trajectory.csv"
    if trajectory_xy is not None and len(trajectory_xy):
        np.savetxt(traj_csv, np.asarray(trajectory_xy)[:, :2], fmt="%.3f", delimiter=",", header="x,y")
        paths["trajectory"] = traj_csv
    m_per_px = grid.resolution / upscale
    rooms = snap.rooms()
    title = (f"{len(rooms)} rooms: " + ", ".join(f"R{r.id} {r.label}" for r in rooms)) if rooms else "no rooms"
    img = render_topdown(snap, grid_background(grid), m_per_px=m_per_px, trajectory_xy=trajectory_xy,
                         show_objects=False, title=title)
    draw_scale_bar(img, m_per_px)
    paths["rooms"] = Path(out_dir) / "rooms.png"
    cv2.imwrite(str(paths["rooms"]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return paths
