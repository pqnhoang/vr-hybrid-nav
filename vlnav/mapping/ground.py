"""Ground height models for the occupancy grid (SYSTEM.md §4.3).

`CellMinGround`: per-cell minimum z of all points seen, smoothed by a minimum filter over
`smooth_radius_m` (cells whose floor is hidden, e.g. under a table, take the floor of their
neighbours); follows slopes and odometry z drift. `ConstantGround`: fixed floor height (sim).
A simplified stand-in for the paper's terrain analysis.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.ndimage import minimum_filter

from vlnav.mapping.config import GroundCfg


class GroundModel(Protocol):
    def resize(self, shape: tuple[int, int], offset: tuple[int, int]) -> None: ...

    def observe(self, iy: np.ndarray, ix: np.ndarray, z: np.ndarray) -> None: ...

    def height(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray: ...


class ConstantGround:
    def __init__(self, z: float):
        self.z = float(z)

    def resize(self, shape, offset) -> None:
        pass

    def observe(self, iy, ix, z) -> None:
        pass

    def height(self, iy, ix) -> np.ndarray:
        return np.full(np.shape(iy), self.z, dtype=np.float32)


class CellMinGround:
    def __init__(self, smooth_radius_cells: int):
        self.r = max(0, int(smooth_radius_cells))
        self.zmin = np.full((0, 0), np.inf, dtype=np.float32)

    def resize(self, shape: tuple[int, int], offset: tuple[int, int]) -> None:
        """New array of `shape`; the old one is pasted at row/col `offset`."""
        new = np.full(shape, np.inf, dtype=np.float32)
        oy, ox = offset
        h, w = self.zmin.shape
        new[oy:oy + h, ox:ox + w] = self.zmin
        self.zmin = new

    def observe(self, iy: np.ndarray, ix: np.ndarray, z: np.ndarray) -> None:
        np.minimum.at(self.zmin, (iy, ix), z.astype(np.float32))

    def height(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Smoothed ground at the given cells (inf where nothing was seen within the radius)."""
        if len(iy) == 0:
            return np.empty(0, dtype=np.float32)
        r = self.r
        y0, y1 = max(int(iy.min()) - r, 0), min(int(iy.max()) + r + 1, self.zmin.shape[0])
        x0, x1 = max(int(ix.min()) - r, 0), min(int(ix.max()) + r + 1, self.zmin.shape[1])
        sub = self.zmin[y0:y1, x0:x1]
        if r > 0:
            sub = minimum_filter(sub, size=2 * r + 1, mode="nearest")
        return sub[iy - y0, ix - x0]


def make_ground(cfg: GroundCfg, resolution: float) -> GroundModel:
    if cfg.mode == "constant":
        return ConstantGround(cfg.constant_z)
    return CellMinGround(int(round(cfg.smooth_radius_m / resolution)))
