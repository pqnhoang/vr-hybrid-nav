"""Instance-based target points, IBTP (TASK_nesy_exploration.md §6).

Relevant observations (their detection's label ∈ active prompts; matched by label + bbox since
`Observation` has no detection index) with score > `tau_det` are merged on a 2D voxel grid
(`inst_voxel_m`): one point per voxel, score = max, position = score-weighted mean. Ranking
(`instance_score`): `vl_eq3` = Eq. 3 at the instance's own bearing in the frame that detected it,
max over frames (**[DESIGN]** without the frontier range gate R: instances come from lifted
detections at known positions) | `confidence`. Approach point: free, non-inflated cell in
[`inst_standoff_min_m`, `inst_standoff_max_m`] from the instance, closest to the robot first, the
first reachable one is used (§6 [DESIGN]); candidates recomputed when the grid version changes.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel

from vlnav.common.models import FrameResult, Observation
from vlnav.exploration.config import InstanceCfg
from vlnav.exploration.vl_score import Mixture, RefPose, s_vl
from vlnav.mapping.occupancy import FREE, OccupancyGrid2D


class InstancePoint(BaseModel):
    id: int
    xy: tuple[float, float]
    score: float
    label: str
    subtask_id: int
    created_t: float
    last_seen_t: float
    status: Literal["active", "visited", "unreachable"] = "active"
    approach_xy: tuple[float, float] | None = None
    rank: float = 0.0                       # instance_score (vl_eq3 or confidence), max over frames
    n_obs: int = 0
    w_sum: float = 0.0                      # score-weighted mean accumulators
    wx: float = 0.0
    wy: float = 0.0
    plan_failures: int = 0


def relevant_observations(res: FrameResult) -> list[Observation]:
    keys = {(res.detections[i].label, tuple(res.detections[i].bbox)) for i in res.relevant_idx}
    return [o for o in res.observations if (o.label, tuple(o.bbox)) in keys]


def approach_cells(grid: OccupancyGrid2D, blocked: np.ndarray, xy, rmin: float, rmax: float) -> np.ndarray:
    """World xy of free, non-blocked cells with rmin <= distance to `xy` <= rmax."""
    ny, nx = grid.shape
    r = int(math.ceil(rmax / grid.resolution))
    cx, cy = grid.world_to_cell(np.asarray(xy, float))
    x0, y0, x1, y1 = max(cx - r, 0), max(cy - r, 0), min(cx + r, nx - 1), min(cy + r, ny - 1)
    if x0 > x1 or y0 > y1:
        return np.zeros((0, 2))
    iy, ix = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    cells = np.stack([ix.ravel(), iy.ravel()], axis=1)
    w = grid.cell_to_world(cells)
    d = np.hypot(*(w - np.asarray(xy, float)).T)
    ok = (d >= rmin) & (d <= rmax) & (grid.as_array()[iy.ravel(), ix.ravel()] == FREE) & \
        ~blocked[iy.ravel(), ix.ravel()]
    return w[ok]


class InstancePool:
    def __init__(self, cfg: InstanceCfg):
        self.cfg = cfg
        self.points: dict[int, InstancePoint] = {}
        self._by_voxel: dict[tuple[int, int], int] = {}
        self._approach: dict[int, tuple[int, np.ndarray]] = {}     # id -> (grid version, cells)
        self._next_id = 0

    def clear(self) -> None:
        self.points.clear()
        self._by_voxel.clear()
        self._approach.clear()

    def add_frame(self, res: FrameResult, ref: RefPose, hfov: float, mix: Mixture, subtask_id: int,
                  t: float, mode: str) -> list[int]:
        new = []
        for o in relevant_observations(res):
            if o.score <= self.cfg.tau_det:
                continue
            xy = np.asarray(o.centroid_map[:2], float)
            rank = o.score if mode == "confidence" else float(s_vl(xy[None], ref, hfov, mix, None)[0][0])
            key = tuple(int(v) for v in np.floor(xy / self.cfg.inst_voxel_m))
            pid = self._by_voxel.get(key)
            if pid is None:
                pid = self._next_id
                self._next_id += 1
                self._by_voxel[key] = pid
                self.points[pid] = InstancePoint(id=pid, xy=(float(xy[0]), float(xy[1])), score=o.score,
                                                 label=o.label, subtask_id=subtask_id, created_t=t,
                                                 last_seen_t=t)
                new.append(pid)
            p = self.points[pid]
            p.w_sum += o.score
            p.wx += o.score * xy[0]
            p.wy += o.score * xy[1]
            p.xy = (p.wx / p.w_sum, p.wy / p.w_sum)
            if o.score >= p.score:
                p.score, p.label = o.score, o.label
            p.rank = max(p.rank, rank)
            p.n_obs += 1
            p.last_seen_t = t
        return new

    def update_visits(self, robot_xy, delta_reached_m: float) -> list[int]:
        """§6 visit rule; returns ids that became visited."""
        out = []
        for p in self.points.values():
            if p.status != "active":
                continue
            near_app = p.approach_xy is not None and math.dist(robot_xy, p.approach_xy) <= delta_reached_m
            if near_app or math.dist(robot_xy, p.xy) <= self.cfg.visit_radius_m:
                p.status = "visited"
                out.append(p.id)
        return out

    def approach_options(self, p: InstancePoint, grid: OccupancyGrid2D, blocked: np.ndarray,
                         robot_xy) -> np.ndarray:
        """Approach cells sorted by distance to the robot (cell set cached per grid version)."""
        cached = self._approach.get(p.id)
        if cached is None or cached[0] != grid.version:
            cells = approach_cells(grid, blocked, p.xy, self.cfg.inst_standoff_min_m, self.cfg.inst_standoff_max_m)
            self._approach[p.id] = cached = (grid.version, cells)
        cells = cached[1]
        return cells[np.argsort(np.hypot(*(cells - np.asarray(robot_xy, float)).T), kind="stable")]
