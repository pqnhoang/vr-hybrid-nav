"""Path planning on the shared occupancy grid (SYSTEM.md §4.4).

`GridAStar`: 8-connected shortest paths on the grid with occupied cells inflated by
`robot_radius_m`; unknown cells traversable at `unknown_cost` per meter (free = 1) **[DESIGN]**;
occupied / inflated cells blocked. The goal is snapped to the nearest free, non-inflated cell within
`goal_snap_radius_m`, the start to the nearest traversable cell within `start_snap_radius_m`.

**[DESIGN]** Solved as Dijkstra from the start with scikit-image's compiled minimum-cost-path
(`MCP_Geometric`, diagonal steps cost √2) instead of a per-goal Python A*: the paths are the same
optimal 8-connected paths, and one search per (grid version, start cell) answers every goal of a
selection step (exploration checks up to `max_reach_checks` candidates). 114 ms for a full search
of an 833 × 808 grid, 0.25 ms per extracted path.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from skimage.graph import MCP_Geometric

from vlnav.mapping.occupancy import FREE, OCCUPIED, OccupancyGrid2D
from vlnav.navigation.config import NavigationConfig

Path2D = list[tuple[float, float]]


class PathPlanner(Protocol):
    def plan(self, grid: OccupancyGrid2D, start_xy, goal_xy) -> Path2D | None: ...


class GridAStar:
    def __init__(self, cfg: NavigationConfig):
        self.cfg = cfg
        self._key: tuple | None = None            # (grid id, version, shape, origin, start cell)
        self._mcp: MCP_Geometric | None = None
        self._costs: np.ndarray | None = None
        self._cost_key: tuple | None = None
        self._step_cost: np.ndarray | None = None
        self.n_searches = 0

    # ------------------------------------------------------------------------- cost map
    def step_costs(self, grid: OccupancyGrid2D) -> np.ndarray:
        """Per-cell traversal cost: 1 free, `unknown_cost` unknown, inf occupied / inflated."""
        key = (id(grid), grid.version, grid.shape, grid.origin_xy)
        if key != self._cost_key:
            arr = grid.as_array()
            cost = np.where(arr == FREE, 1.0, self.cfg.unknown_cost)
            cost[(arr == OCCUPIED) | grid.inflated(self.cfg.robot_radius_m)] = np.inf
            self._step_cost, self._cost_key = cost, key
        return self._step_cost

    def _nearest(self, ok: np.ndarray, cell: np.ndarray, radius_m: float, res: float) -> tuple | None:
        """Nearest cell (iy, ix) with `ok` within radius_m of `cell` (ix, iy), or None."""
        r = int(np.ceil(radius_m / res))
        ny, nx = ok.shape
        x0, x1 = max(cell[0] - r, 0), min(cell[0] + r + 1, nx)
        y0, y1 = max(cell[1] - r, 0), min(cell[1] + r + 1, ny)
        if x0 >= x1 or y0 >= y1:
            return None
        iy, ix = np.nonzero(ok[y0:y1, x0:x1])
        if not len(ix):
            return None
        d2 = (ix + x0 - cell[0]) ** 2 + (iy + y0 - cell[1]) ** 2
        k = int(np.argmin(d2))
        if d2[k] > r * r:
            return None
        return int(iy[k] + y0), int(ix[k] + x0)

    def snap_start(self, grid: OccupancyGrid2D, xy) -> tuple | None:
        cost = self.step_costs(grid)
        return self._nearest(np.isfinite(cost), grid.world_to_cell(np.asarray(xy, float)[:2]),
                             self.cfg.start_snap_radius_m, grid.resolution)

    def snap_goal(self, grid: OccupancyGrid2D, xy) -> tuple | None:
        cost = self.step_costs(grid)
        free = (grid.as_array() == FREE) & np.isfinite(cost)
        return self._nearest(free, grid.world_to_cell(np.asarray(xy, float)[:2]),
                             self.cfg.goal_snap_radius_m, grid.resolution)

    # --------------------------------------------------------------------------- search
    def _field(self, grid: OccupancyGrid2D, start: tuple) -> np.ndarray:
        key = (id(grid), grid.version, grid.shape, grid.origin_xy, start)
        if key != self._key:
            self._mcp = MCP_Geometric(self.step_costs(grid), fully_connected=True)
            self._costs, _ = self._mcp.find_costs([start])
            self._key = key
            self.n_searches += 1
        return self._costs

    def cost_to(self, grid: OccupancyGrid2D, start_xy, goal_xy) -> float:
        """Path cost (≈ meters, unknown weighted) from start to goal; inf if unreachable."""
        if 0 in grid.shape:
            return float("inf")
        start, goal = self.snap_start(grid, start_xy), self.snap_goal(grid, goal_xy)
        if start is None or goal is None:
            return float("inf")
        return float(self._field(grid, start)[goal]) * grid.resolution

    def reach_costs(self, grid: OccupancyGrid2D, start_xy, goals_xy) -> np.ndarray:
        """Path costs (≈ m) from start to many goal cells at once, no goal snapping; inf where
        unreachable or not traversable (exploration filters instance approach cells with it)."""
        goals = np.asarray(goals_xy, dtype=float).reshape(-1, 2)
        out = np.full(len(goals), np.inf)
        start = self.snap_start(grid, start_xy) if 0 not in grid.shape else None
        if start is None or not len(goals):
            return out
        costs = self._field(grid, start)
        c = grid.world_to_cell(goals)
        ok = grid.in_bounds(c)
        out[ok] = costs[c[ok, 1], c[ok, 0]] * grid.resolution
        return out

    def plan(self, grid: OccupancyGrid2D, start_xy, goal_xy) -> Path2D | None:
        if 0 in grid.shape:
            return None
        start, goal = self.snap_start(grid, start_xy), self.snap_goal(grid, goal_xy)
        if start is None or goal is None:
            return None
        costs = self._field(grid, start)
        if not np.isfinite(costs[goal]):
            return None
        cells = np.asarray(self._mcp.traceback(goal))             # (iy, ix) from start to goal
        xy = grid.cell_to_world(cells[:, ::-1])
        return [(float(x), float(y)) for x, y in xy]
