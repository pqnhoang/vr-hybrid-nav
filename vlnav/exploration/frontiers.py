"""Frontier-based target points (TASK_nesy_exploration.md §4).

Detection (paper Eq. 1): a cell is a frontier if it is free and has an unknown neighbor
(8-connected, `frontier_connectivity`; cells outside the grid count as unknown). After every grid
update only cells inside the FoV wedge of the frame (reference pose per `angle_reference`, range
`sensor_range_m`) are tested; existing frontier cells inside `UpdateInfo.changed_bbox_cells` (+1 cell:
a neighbor change can end a frontier) are re-validated. `outside_fov: zero_score` also adds the new
frontier cells of the changed bbox (they get S_VL = 0 unless later seen in a wedge).

The frontier cell set is a bool mask aligned with the grid (re-padded when the grid grows).
Clusters = connected components of the mask; only components that intersect the touched region are
re-sampled, the others keep their points (so incremental = full recomputation, tested).
Representative points: centroid snapped to the cluster for small clusters, farthest-point sampling
otherwise; points still on a frontier cell of a re-sampled cluster are kept and sampling fills the
gaps around them. Points carry identity across re-clustering (§4.3): a new point takes over the id
of the nearest old point within `inherit_radius_m`; VL scores are inherited near where they were
measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy import ndimage

from vlnav.common.geometry import wrap_angle
from vlnav.exploration.config import FrontierCfg
from vlnav.mapping.occupancy import FREE, UNKNOWN, OccupancyGrid2D, UpdateInfo

BBox = tuple[int, int, int, int]                 # ix_min, iy_min, ix_max, iy_max (inclusive)


class FrontierPoint(BaseModel):
    id: int
    xy: tuple[float, float]
    cluster_id: int
    created_t: float
    s_vl: float = 0.0
    s_vl_subtask_id: int | None = None     # subtask for which s_vl is valid
    n_vl_updates: int = 0
    blacklisted_until: float = 0.0
    vl_weight: float = 0.0                 # [DESIGN] sum of C(Δθ) for vl_update_mode conf_weighted
    vl_xy: tuple[float, float] | None = None   # [DESIGN] where s_vl was measured (inheritance anchor)
    plan_failures: int = 0                 # consecutive selection steps it was unreachable


@dataclass(frozen=True)
class Wedge:
    """FoV wedge of one camera: reference position / heading (§4.1), h_fov, range R."""
    x: float
    y: float
    psi: float
    hfov: float
    radius: float


def neighbor_structure(connectivity: int) -> np.ndarray:
    return np.ones((3, 3), bool) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)


def frontier_test(arr: np.ndarray, connectivity: int) -> np.ndarray:
    """Frontier mask of a whole ternary grid (reference for tests)."""
    return _test_region(arr, (0, 0, arr.shape[1] - 1, arr.shape[0] - 1), neighbor_structure(connectivity))


def _test_region(arr: np.ndarray, bbox: BBox, structure: np.ndarray) -> np.ndarray:
    """Frontier test of the cells in `bbox` (neighbors outside the grid count as unknown)."""
    x0, y0, x1, y1 = bbox
    ny, nx = arr.shape
    sub = np.full((y1 - y0 + 3, x1 - x0 + 3), UNKNOWN, np.int8)
    sx0, sy0, sx1, sy1 = max(x0 - 1, 0), max(y0 - 1, 0), min(x1 + 2, nx), min(y1 + 2, ny)
    sub[sy0 - (y0 - 1):sy1 - (y0 - 1), sx0 - (x0 - 1):sx1 - (x0 - 1)] = arr[sy0:sy1, sx0:sx1]
    near_unknown = ndimage.binary_dilation(sub == UNKNOWN, structure)
    return ((sub == FREE) & near_unknown)[1:-1, 1:-1]


def wedge_cells(grid: OccupancyGrid2D, w: Wedge) -> tuple[BBox, np.ndarray] | None:
    """Cells whose centers are within range R and |wrap(bearing - psi)| <= hfov / 2 (Eq. 1)."""
    ny, nx = grid.shape
    r = int(math.ceil(w.radius / grid.resolution))
    cx, cy = grid.world_to_cell(np.array([w.x, w.y]))
    x0, y0, x1, y1 = max(cx - r, 0), max(cy - r, 0), min(cx + r, nx - 1), min(cy + r, ny - 1)
    if x0 > x1 or y0 > y1:
        return None
    ox, oy = grid.origin_xy
    xs = ox + (np.arange(x0, x1 + 1) + 0.5) * grid.resolution - w.x
    ys = oy + (np.arange(y0, y1 + 1) + 0.5) * grid.resolution - w.y
    dx, dy = np.meshgrid(xs, ys)
    inside = (np.hypot(dx, dy) <= w.radius) & (np.abs(wrap_angle(np.arctan2(dy, dx) - w.psi)) <= w.hfov / 2)
    return (int(x0), int(y0), int(x1), int(y1)), inside


def n_samples(size: int, resolution: float, cfg: FrontierCfg) -> int:
    """§4.2: 1 (centroid) for small clusters, else ceil(size * resolution / sample_spacing_m)."""
    return 1 if size <= cfg.small_cluster_cells else min(size, math.ceil(size * resolution / cfg.sample_spacing_m))


def farthest_points(xy: np.ndarray, n: int, seeds: np.ndarray | None = None) -> np.ndarray:
    """`n` new points by farthest-point sampling over `xy`, away from `seeds`; without seeds the
    first point is the cell nearest the centroid."""
    if n <= 0:
        return np.zeros((0, 2))
    if seeds is not None and len(seeds):
        chosen, d = [], np.min(np.hypot(*(xy[:, None, :] - seeds[None]).transpose(2, 0, 1)), axis=1)
    else:
        i0 = int(np.argmin(np.sum((xy - xy.mean(0)) ** 2, axis=1)))
        chosen, d = [i0], np.hypot(*(xy - xy[i0]).T)
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.hypot(*(xy - xy[i]).T))
    return xy[chosen]


def representative_points(xy: np.ndarray, resolution: float, cfg: FrontierCfg) -> np.ndarray:
    """§4.2 for a new cluster: centroid snapped to the nearest cluster cell for small clusters,
    else farthest-point sampling starting at that cell."""
    return farthest_points(xy, n_samples(len(xy), resolution, cfg))


def _expand(b: BBox, m: int, shape: tuple[int, int]) -> BBox:
    return max(b[0] - m, 0), max(b[1] - m, 0), min(b[2] + m, shape[1] - 1), min(b[3] + m, shape[0] - 1)


class FrontierMap:
    def __init__(self, cfg: FrontierCfg):
        self.cfg = cfg
        self.structure = neighbor_structure(cfg.frontier_connectivity)
        self.mask = np.zeros((0, 0), bool)
        self._origin: tuple[float, float] | None = None
        self.points: dict[int, FrontierPoint] = {}
        self._next_id = 0
        self._next_cluster = 0

    # ----------------------------------------------------------------------------- grid
    def _align(self, grid: OccupancyGrid2D) -> None:
        if self.mask.shape == grid.shape and self._origin == grid.origin_xy:
            return
        new = np.zeros(grid.shape, bool)
        if self.mask.size and self._origin is not None:
            dx = int(round((self._origin[0] - grid.origin_xy[0]) / grid.resolution))
            dy = int(round((self._origin[1] - grid.origin_xy[1]) / grid.resolution))
            ny, nx = self.mask.shape
            new[dy:dy + ny, dx:dx + nx] = self.mask          # grids only grow (content kept)
        self.mask, self._origin = new, grid.origin_xy

    def cells(self) -> np.ndarray:
        """Frontier cells as (ix, iy) of the current grid."""
        return np.argwhere(self.mask)[:, ::-1]

    # --------------------------------------------------------------------------- update
    def update(self, grid: OccupancyGrid2D, info: UpdateInfo, wedges: list[Wedge], t: float,
               subtask_id: int | None, recluster_all: bool = False) -> list[int]:
        """Apply one grid update; returns the ids of points created (not inherited) in it.
        `recluster_all`: re-cluster every component (reference for the incremental path)."""
        self._align(grid)
        if 0 in grid.shape:
            return []
        arr = grid.as_array()
        touched: list[BBox] = []
        for w in wedges:
            wc = wedge_cells(grid, w)
            if wc is None:
                continue
            bbox, inside = wc
            test = _test_region(arr, bbox, self.structure)
            view = self.mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
            view[inside] = test[inside]
            touched.append(bbox)
        cb = info.changed_bbox_cells
        if cb[2] >= cb[0] and cb[3] >= cb[1]:
            bbox = _expand(cb, 1, grid.shape)
            test = _test_region(arr, bbox, self.structure)
            view = self.mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
            if self.cfg.outside_fov == "zero_score":
                view[:] = test
            else:
                view &= test
            touched.append(bbox)
        return self._recluster(grid, None if recluster_all else touched, t, subtask_id)

    def recompute_all(self, grid: OccupancyGrid2D, t: float, subtask_id: int | None = None) -> list[int]:
        """Re-cluster every component (reference for the incremental path)."""
        self._align(grid)
        return self._recluster(grid, None, t, subtask_id)

    def _recluster(self, grid: OccupancyGrid2D, touched: list[BBox] | None, t: float,
                   subtask_id: int | None) -> list[int]:
        labels, n = ndimage.label(self.mask, self.structure)
        if touched is None:
            redo = set(range(1, n + 1))
        else:                   # +2: a removed cell (bbox +1) can split a component next to it
            redo = set()
            for b in touched:
                x0, y0, x1, y1 = _expand(b, 2, grid.shape)
                redo.update(np.unique(labels[y0:y1 + 1, x0:x1 + 1]).tolist())
            redo.discard(0)
        kept, old, on_cluster = {}, [], {}
        for p in self.points.values():
            ix, iy = grid.world_to_cell(np.asarray(p.xy))
            lab = labels[iy, ix] if 0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0] else 0
            if lab and lab not in redo:
                kept[p.id] = p                 # its component is unchanged
            else:
                old.append(p)
                if lab:
                    on_cluster.setdefault(int(lab), []).append(p)
        objs = ndimage.find_objects(labels)
        fresh: list[tuple[int, np.ndarray]] = []
        for lab in sorted(redo):
            sl = objs[lab - 1]
            if sl is None:
                continue
            iy, ix = np.nonzero(labels[sl] == lab)
            if len(ix) < self.cfg.min_cluster_cells:
                continue
            xy = grid.cell_to_world(np.stack([ix + sl[1].start, iy + sl[0].start], axis=1))
            cid = self._next_cluster
            self._next_cluster += 1
            n = n_samples(len(xy), grid.resolution, self.cfg)
            seeds = self._seeds(on_cluster.get(int(lab), []), n) if n > 1 else []
            for q in seeds:                    # still on a frontier cell: the point stays (§4.3)
                q.cluster_id = cid
                kept[q.id] = q
            seed_xy = np.array([q.xy for q in seeds]).reshape(-1, 2)
            fresh += [(cid, q) for q in farthest_points(xy, n - len(seeds), seed_xy)]
        created = self._inherit(fresh, [p for p in old if p.id not in kept], t, subtask_id, kept)
        self.points = kept
        return created

    @staticmethod
    def _seeds(points: list[FrontierPoint], n: int) -> list[FrontierPoint]:
        """[DESIGN] Old points still lying on the re-clustered component are kept (identity and
        score unchanged) and sampling only fills the gaps: re-sampling from scratch moved points by
        more than inherit_radius_m on most updates (on vr_h3_2, 43 of 49 goal switches were the
        goal's point being replaced). At most n, chosen farthest-first from the oldest point."""
        if len(points) <= n:
            return points
        pts = sorted(points, key=lambda q: q.id)
        xy = np.array([q.xy for q in pts])
        chosen, d = [0], np.hypot(*(xy - xy[0]).T)
        while len(chosen) < n:
            i = int(np.argmax(d))
            chosen.append(i)
            d = np.minimum(d, np.hypot(*(xy - xy[i]).T))
        return [pts[i] for i in chosen]

    def _inherit(self, fresh: list[tuple[int, np.ndarray]], old: list[FrontierPoint], t: float,
                 subtask_id: int | None, out: dict[int, FrontierPoint]) -> list[int]:
        """§4.3: id (+ created_t, plan failures) from the nearest old point within inherit_radius_m,
        each old id used once (greedy by distance); blacklist = max of those old points. VL score:
        the best old score (valid for the current subtask first, then the highest s_vl) whose
        measurement position `vl_xy` is within inherit_radius_m — [DESIGN] anchoring: inheriting by
        the old point's current position let one score creep 0.5 m per re-clustering along a
        cluster (on vr_h3_2, 20 of 51 points carried one measurement after 20 frames)."""
        r = self.cfg.inherit_radius_m
        new_xy = np.array([p for _, p in fresh]).reshape(-1, 2)
        old_xy = np.array([p.xy for p in old]).reshape(-1, 2)
        d = np.hypot(*(new_xy[:, None, :] - old_xy[None, :, :]).transpose(2, 0, 1)) if len(old) else \
            np.zeros((len(fresh), 0))
        owner: dict[int, int] = {}
        used: set[int] = set()
        for i, j in sorted(zip(*np.nonzero(d <= r)), key=lambda ij: d[ij]):
            i, j = int(i), int(j)
            if i not in owner and j not in used:
                owner[i] = j
                used.add(j)
        created = []
        for i, (cid, xy) in enumerate(fresh):
            near = [old[j] for j in np.flatnonzero(d[i] <= r)] if len(old) else []
            p = FrontierPoint(id=-1, xy=(float(xy[0]), float(xy[1])), cluster_id=cid, created_t=t)
            if near:
                p.blacklisted_until = max(q.blacklisted_until for q in near)
            scored = [q for q in old if q.vl_xy is not None and math.dist(q.vl_xy, p.xy) <= r]
            if scored:
                best = max(scored, key=lambda q: (q.s_vl_subtask_id == subtask_id, q.s_vl))
                p.s_vl, p.s_vl_subtask_id, p.vl_xy = best.s_vl, best.s_vl_subtask_id, best.vl_xy
                p.n_vl_updates, p.vl_weight = best.n_vl_updates, best.vl_weight
            if i in owner:
                o = old[owner[i]]
                p.id, p.created_t, p.plan_failures = o.id, o.created_t, o.plan_failures
            else:
                p.id = self._next_id
                self._next_id += 1
                created.append(p.id)
            out[p.id] = p
        return created
