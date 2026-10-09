"""Shared 2D occupancy grid (SYSTEM.md §4.3), owned by the core process.

`update(points_map, sensor_origin, t)`: points classified by height above the ground model
(`ground.py`): obstacle if in `[obstacle_z_min, obstacle_z_max]`, ground below, overhead (ignored)
above. Obstacle cells get a log-odds hit; free space comes from 2D ray casting from the sensor
origin (rays to ground points include their end cell, rays to obstacles stop before it). The grid
grows when data leaves it. A simplified stand-in for the paper's terrain analysis.

Conventions: array indexed `[iy, ix]`; a cell is addressed as `(ix, iy)`; cell centers at
`origin_xy + (i + 0.5) * resolution`. `as_array()` is ternary: -1 unknown, 0 free, 1 occupied.
"""
from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from vlnav.mapping.config import MappingConfig
from vlnav.mapping.ground import GroundModel, make_ground

UNKNOWN, FREE, OCCUPIED = -1, 0, 1
NO_CHANGE = (0, 0, -1, -1)


class UpdateInfo(BaseModel):
    t: float
    changed_bbox_cells: tuple[int, int, int, int]     # ix_min, iy_min, ix_max, iy_max (inclusive)
    version: int


class OccupancyGrid2D:
    def __init__(self, resolution: float, origin_xy: tuple[float, float],
                 size_cells: tuple[int, int], cfg: MappingConfig | None = None):
        self.resolution = float(resolution)
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        nx, ny = size_cells
        self.cfg = cfg
        self._grid = np.full((ny, nx), UNKNOWN, dtype=np.int8)       # ternary, source of truth
        self._lo = np.zeros((ny, nx), dtype=np.float32)              # log-odds (update())
        self.ground: GroundModel | None = make_ground(cfg.ground, self.resolution) if cfg else None
        if self.ground is not None:
            self.ground.resize((ny, nx), (0, 0))
        self.version = 0
        self._inflated_cache: dict[float, tuple[int, np.ndarray]] = {}
        self._rng = np.random.default_rng(cfg.seed if cfg else 0)

    @classmethod
    def from_config(cls, cfg: MappingConfig) -> OccupancyGrid2D:
        """Empty grid; the first `update` allocates it around the sensor."""
        return cls(cfg.resolution, (0.0, 0.0), (0, 0), cfg)

    @classmethod
    def from_array(cls, arr: np.ndarray, resolution: float, origin_xy: tuple[float, float],
                   cfg: MappingConfig | None = None) -> OccupancyGrid2D:
        grid = cls(resolution, origin_xy, (arr.shape[1], arr.shape[0]), cfg)
        grid.set_array(arr, t=0.0)
        return grid

    @property
    def shape(self) -> tuple[int, int]:
        return self._grid.shape

    # ----------------------------------------------------------------------------- read API
    def as_array(self) -> np.ndarray:
        view = self._grid.view()
        view.flags.writeable = False
        return view

    def world_to_cell(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.floor((xy[..., :2] - self.origin_xy) / self.resolution).astype(np.int64)

    def cell_to_world(self, cells) -> np.ndarray:
        cells = np.asarray(cells, dtype=float)
        return np.asarray(self.origin_xy) + (cells + 0.5) * self.resolution

    def in_bounds(self, cells) -> np.ndarray:
        cells = np.asarray(cells)
        ny, nx = self._grid.shape
        return (cells[..., 0] >= 0) & (cells[..., 0] < nx) & (cells[..., 1] >= 0) & (cells[..., 1] < ny)

    def value(self, cells) -> np.ndarray:
        """-1 unknown, 0 free, 1 occupied; out-of-bounds cells are unknown."""
        cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
        out = np.full(len(cells), UNKNOWN, dtype=np.int8)
        ok = self.in_bounds(cells)
        out[ok] = self._grid[cells[ok, 1], cells[ok, 0]]
        return out

    def inflated(self, radius_m: float) -> np.ndarray:
        """Bool mask of occupied cells dilated by `radius_m`, cached per version."""
        cached = self._inflated_cache.get(radius_m)
        if cached is not None and cached[0] == self.version:
            return cached[1]
        r = int(np.ceil(radius_m / self.resolution))
        occ = (self._grid == OCCUPIED).astype(np.uint8)
        if r > 0 and occ.size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            occ = cv2.dilate(occ, kernel)
        mask = occ.astype(bool)
        self._inflated_cache[radius_m] = (self.version, mask)
        return mask

    def ground_height(self, xy) -> float:
        """Smoothed ground z at a world point (nan if unknown / no ground model)."""
        if self.ground is None:
            return float("nan")
        c = self.world_to_cell(np.asarray(xy, dtype=float)[None])
        if not self.in_bounds(c)[0]:
            return float("nan")
        h = float(self.ground.height(c[:, 1], c[:, 0])[0])
        return h if np.isfinite(h) else float("nan")

    # ---------------------------------------------------------------------------- writing
    def set_array(self, arr: np.ndarray, t: float) -> UpdateInfo:
        """Replace the whole grid (same shape). Used by tests and external-grid adapters."""
        arr = np.asarray(arr, dtype=np.int8)
        if arr.shape != self._grid.shape:
            raise ValueError(f"grid shape {arr.shape} != {self._grid.shape}")
        changed = np.argwhere(arr != self._grid)
        self._grid = arr.copy()
        if self.cfg is not None:
            lo = self.cfg.log_odds
            self._lo = np.where(arr == OCCUPIED, lo.clamp_max,
                                np.where(arr == FREE, lo.clamp_min, 0.0)).astype(np.float32)
        self.version += 1
        return UpdateInfo(t=t, changed_bbox_cells=_bbox(changed[:, 1], changed[:, 0]) if len(changed)
                          else NO_CHANGE, version=self.version)

    def _ensure(self, xy_min: np.ndarray, xy_max: np.ndarray, center: np.ndarray) -> None:
        """Grow (or first allocate) the grid so that [xy_min, xy_max] is inside."""
        res = self.resolution
        ny, nx = self._grid.shape
        if nx == 0 or ny == 0:
            half = self.cfg.initial_size_m / 2
            lo = np.floor((np.minimum(xy_min, center - half)) / res) * res
            hi = np.maximum(xy_max, center + half)
            n = np.ceil((hi - lo) / res).astype(int) + 1
            self.origin_xy = (float(lo[0]), float(lo[1]))
            self._grid = np.full((n[1], n[0]), UNKNOWN, dtype=np.int8)
            self._lo = np.zeros((n[1], n[0]), dtype=np.float32)
            self.ground.resize((n[1], n[0]), (0, 0))
            return
        c0 = self.world_to_cell(xy_min)
        c1 = self.world_to_cell(xy_max)
        if c0[0] >= 0 and c0[1] >= 0 and c1[0] < nx and c1[1] < ny:
            return
        m = int(np.ceil(self.cfg.grow_margin_m / res))
        pad_x0 = m - int(c0[0]) if c0[0] < 0 else 0
        pad_y0 = m - int(c0[1]) if c0[1] < 0 else 0
        pad_x1 = int(c1[0]) - nx + 1 + m if c1[0] >= nx else 0
        pad_y1 = int(c1[1]) - ny + 1 + m if c1[1] >= ny else 0
        shape = (ny + pad_y0 + pad_y1, nx + pad_x0 + pad_x1)
        grid = np.full(shape, UNKNOWN, dtype=np.int8)
        lo = np.zeros(shape, dtype=np.float32)
        grid[pad_y0:pad_y0 + ny, pad_x0:pad_x0 + nx] = self._grid
        lo[pad_y0:pad_y0 + ny, pad_x0:pad_x0 + nx] = self._lo
        self._grid, self._lo = grid, lo
        self.ground.resize(shape, (pad_y0, pad_x0))
        self.origin_xy = (self.origin_xy[0] - pad_x0 * res, self.origin_xy[1] - pad_y0 * res)

    def _subsample(self, n: int, k: int) -> np.ndarray:
        return np.arange(n) if n <= k else np.sort(self._rng.choice(n, k, replace=False))

    def update(self, points_map: np.ndarray, sensor_origin, t: float) -> UpdateInfo:
        if self.cfg is None:
            raise ValueError("update() needs a MappingConfig (use OccupancyGrid2D.from_config)")
        c = self.cfg
        o = np.asarray(sensor_origin, dtype=float)[:3]
        pts = np.asarray(points_map, dtype=float).reshape(-1, 3) if points_map is not None else np.empty((0, 3))
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[np.hypot(pts[:, 0] - o[0], pts[:, 1] - o[1]) <= c.max_range_m]
        if not len(pts):
            self.version += 1
            return UpdateInfo(t=t, changed_bbox_cells=NO_CHANGE, version=self.version)
        self._ensure(np.minimum(pts[:, :2].min(0), o[:2]), np.maximum(pts[:, :2].max(0), o[:2]), o[:2])

        cells = self.world_to_cell(pts[:, :2])
        ix, iy = cells[:, 0], cells[:, 1]
        self.ground.observe(iy, ix, pts[:, 2])
        h = pts[:, 2] - self.ground.height(iy, ix)
        obstacle = (h >= c.obstacle_z_min) & (h <= c.obstacle_z_max)
        ground = h < c.obstacle_z_min                       # overhead points: neither
        ny, nx = self._grid.shape

        hit_idx = np.flatnonzero(obstacle)
        hit_idx = hit_idx[self._subsample(len(hit_idx), c.max_points)]
        hit = np.zeros(ny * nx, dtype=bool)                # mark arrays: no sorting / np.unique
        hit[iy[hit_idx] * nx + ix[hit_idx]] = True
        miss = self._raycast(o[:2], pts[:, :2], ground, obstacle, (ny, nx))
        miss &= ~hit                                        # a hit in this update wins

        touched = np.flatnonzero(hit | miss)
        before = self._grid.ravel()[touched].copy()
        lo, flat_lo = c.log_odds, self._lo.ravel()
        flat_lo[hit] += lo.hit
        flat_lo[miss] += lo.miss
        flat_lo[touched] = np.clip(flat_lo[touched], lo.clamp_min, lo.clamp_max)
        v = flat_lo[touched]
        after = np.where(v >= lo.occupied_threshold, OCCUPIED,
                         np.where(v <= lo.free_threshold, FREE, UNKNOWN)).astype(np.int8)
        self._grid.ravel()[touched] = after
        changed = touched[before != after]
        self.version += 1
        bbox = _bbox(changed % nx, changed // nx) if len(changed) else NO_CHANGE
        return UpdateInfo(t=t, changed_bbox_cells=bbox, version=self.version)

    def _raycast(self, origin_xy: np.ndarray, xy: np.ndarray, ground: np.ndarray,
                 obstacle: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Free cells by 2D ray casting from the sensor origin, as a flat bool mask.

        All rays share one origin, so endpoints are grouped into `ray_angle_bins` azimuth bins and
        each bin is rasterized once up to its farthest evidence: a ground point frees its own cell,
        an obstacle frees up to one cell before it. With 0.25 deg bins a bin is narrower than a
        5 cm cell up to ~11 m (max_range_m 10 m).
        """
        ny, nx = shape
        out = np.zeros(ny * nx, dtype=bool)
        use = ground | obstacle
        if not use.any():
            return out
        res, nb = self.resolution, self.cfg.ray_angle_bins
        d = xy[use] - origin_xy
        r = np.hypot(d[:, 0], d[:, 1])
        b = (np.floor((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * nb).astype(np.int64)) % nb
        far = np.full(nb, -np.inf)
        np.maximum.at(far, b, np.where(ground[use], r, r - res))
        bins = np.flatnonzero(far > 0)
        if not len(bins):
            return out
        ang = (bins + 0.5) * (2 * np.pi / nb) - np.pi
        step = self.cfg.ray_step_factor * res
        n = np.ceil(far[bins] / step).astype(np.int64) + 1
        ray = np.repeat(np.arange(len(bins)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        dist = np.minimum(k * step, far[bins][ray])         # last sample exactly at the end
        cells = self.world_to_cell(origin_xy + dist[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)[ray])
        ok = (cells[:, 0] >= 0) & (cells[:, 0] < nx) & (cells[:, 1] >= 0) & (cells[:, 1] < ny)
        out[cells[ok, 1] * nx + cells[ok, 0]] = True
        return out


class ExternalGrid(OccupancyGrid2D):
    """Wraps an externally built occupancy grid (e.g. nav_msgs/OccupancyGrid of the CMU autonomy
    stack): values -1 unknown, 0..100 occupancy probability. Fed by `set_from_occupancy`, never by
    `update`. ROS-free: `vlnav_ros` passes the message fields."""

    def __init__(self, cfg: MappingConfig):
        super().__init__(cfg.resolution, (0.0, 0.0), (0, 0), None)
        self.ext = cfg.external

    def update(self, points_map, sensor_origin, t) -> UpdateInfo:
        raise TypeError("ExternalGrid is fed by set_from_occupancy(), not by sensor points")

    def set_from_occupancy(self, data: np.ndarray, width: int, height: int, resolution: float,
                           origin_xy: tuple[float, float], t: float) -> UpdateInfo:
        occ = np.asarray(data, dtype=np.int16).reshape(height, width)
        tern = np.full(occ.shape, UNKNOWN, dtype=np.int8)
        tern[(occ >= 0) & (occ <= self.ext.free_threshold)] = FREE
        tern[occ >= self.ext.occupied_threshold] = OCCUPIED
        if tern.shape != self._grid.shape or resolution != self.resolution or tuple(origin_xy) != self.origin_xy:
            self.resolution, self.origin_xy = float(resolution), (float(origin_xy[0]), float(origin_xy[1]))
            self._grid = tern
            self.version += 1
            return UpdateInfo(t=t, changed_bbox_cells=(0, 0, width - 1, height - 1), version=self.version)
        return self.set_array(tern, t)


def _bbox(ix: np.ndarray, iy: np.ndarray) -> tuple[int, int, int, int]:
    return (int(ix.min()), int(iy.min()), int(ix.max()), int(iy.max()))
