"""Occupancy grid -> files: a north-up PNG and the ROS `map_server` / Nav2 map format (PGM + YAML).

Pixel values follow `map_saver` (free 254, occupied 0, unknown 205) with its thresholds, so the
YAML loads back to the same ternary grid. `origin` = map-frame pose of the lower-left pixel corner,
which is `OccupancyGrid2D.origin_xy` (corner of cell (0, 0)). 1 px = 1 cell.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from vlnav.mapping.occupancy import FREE, OCCUPIED, OccupancyGrid2D

MAP_GRAY = {"unknown": 205, "free": 254, "occupied": 0}
OCCUPIED_THRESH = 0.65             # map_saver defaults: p = (255 - pixel) / 255
FREE_THRESH = 0.196


def grid_image(grid: OccupancyGrid2D) -> np.ndarray:
    """HxW uint8, row 0 = max y (north up), `MAP_GRAY` values."""
    arr = grid.as_array()
    img = np.full(arr.shape, MAP_GRAY["unknown"], np.uint8)
    img[arr == FREE] = MAP_GRAY["free"]
    img[arr == OCCUPIED] = MAP_GRAY["occupied"]
    return img[::-1].copy()


def save_map(grid: OccupancyGrid2D, out_dir: str | Path, name: str = "occupancy") -> dict[str, Path]:
    """`<name>.png` (view) + `<name>.pgm` and `<name>.yaml` (map_server / Nav2)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = grid_image(grid)
    paths = {"png": out / f"{name}.png", "pgm": out / f"{name}.pgm", "yaml": out / f"{name}.yaml"}
    cv2.imwrite(str(paths["png"]), img)
    cv2.imwrite(str(paths["pgm"]), img)
    meta = {"image": paths["pgm"].name, "mode": "trinary", "resolution": float(grid.resolution),
            "origin": [float(grid.origin_xy[0]), float(grid.origin_xy[1]), 0.0], "negate": 0,
            "occupied_thresh": OCCUPIED_THRESH, "free_thresh": FREE_THRESH}
    paths["yaml"].write_text(yaml.safe_dump(meta, sort_keys=False, default_flow_style=None))
    return paths


def load_map(yaml_path: str | Path) -> OccupancyGrid2D:
    """Inverse of `save_map` (map_server trinary rule), e.g. to reuse a saved map."""
    meta = yaml.safe_load(Path(yaml_path).read_text())
    img = cv2.imread(str(Path(yaml_path).parent / meta["image"]), cv2.IMREAD_GRAYSCALE)
    p = (255.0 - img) / 255.0 if not meta.get("negate", 0) else img / 255.0
    arr = np.full(img.shape, -1, np.int8)
    arr[p > meta["occupied_thresh"]] = OCCUPIED
    arr[p < meta["free_thresh"]] = FREE
    return OccupancyGrid2D.from_array(arr[::-1].copy(), float(meta["resolution"]),
                                      (float(meta["origin"][0]), float(meta["origin"][1])))
