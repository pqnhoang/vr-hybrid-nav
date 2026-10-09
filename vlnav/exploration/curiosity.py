"""Curiosity cues (TASK_nesy_exploration.md §7, paper Eq. 5–7), computed at selection time.

S_dist = 1 / (1 + d(robot, g)) (Euclidean, or A* path length with `dist_metric: path`).
ratio(g) = #unknown / #visited over a BFS from g's cell through non-occupied cells (connectivity of
the frontier test) within `unknown_bfs_radius_m`; S_unknown = 1 − exp(−k · ratio).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from vlnav.mapping.occupancy import OCCUPIED, UNKNOWN, OccupancyGrid2D


def s_dist(d: float) -> float:
    return 1.0 / (1.0 + d)


def unknown_ratio(arr: np.ndarray, cell, radius_cells: int, structure: np.ndarray) -> float:
    ix, iy = int(cell[0]), int(cell[1])
    ny, nx = arr.shape
    if not (0 <= ix < nx and 0 <= iy < ny) or arr[iy, ix] == OCCUPIED:
        return 0.0
    x0, y0 = max(ix - radius_cells, 0), max(iy - radius_cells, 0)
    sub = arr[y0:min(iy + radius_cells + 1, ny), x0:min(ix + radius_cells + 1, nx)]
    yy, xx = np.ogrid[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
    passable = (sub != OCCUPIED) & ((xx - ix) ** 2 + (yy - iy) ** 2 <= radius_cells ** 2)
    labels, _ = ndimage.label(passable, structure)
    comp = labels == labels[iy - y0, ix - x0]
    return float((comp & (sub == UNKNOWN)).sum()) / float(comp.sum())


def s_unknown(ratio: float, k: float) -> float:
    return 1.0 - math.exp(-k * ratio)


class UnknownRatioCache:
    """Per (point id, grid version), as §7 requires."""

    def __init__(self, radius_m: float, structure: np.ndarray):
        self.radius_m = radius_m
        self.structure = structure
        self._cache: dict[int, tuple[int, tuple[float, float], float]] = {}

    def get(self, grid: OccupancyGrid2D, pid: int, xy) -> float:
        hit = self._cache.get(pid)
        if hit is not None and hit[0] == grid.version and hit[1] == tuple(xy):
            return hit[2]
        r = unknown_ratio(grid.as_array(), grid.world_to_cell(np.asarray(xy, float)),
                          int(math.ceil(self.radius_m / grid.resolution)), self.structure)
        self._cache[pid] = (grid.version, tuple(xy), r)
        return r

    def prune(self, keep: set[int]) -> None:
        for pid in [k for k in self._cache if k not in keep]:
            del self._cache[pid]
