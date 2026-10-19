"""Projected point -> reachable navigation goal (TASK_move_subtask.md §8).

The planner snaps the goal to the nearest free, non-inflated cell within `goal_snap_radius_m`; when
no path exists the point walks back toward the robot in `walkback_step_m` steps (goals in unknown
space behind a frontier, inside obstacles or their inflation) until a path is found.
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from vlnav.common.geometry import yaw_quat
from vlnav.common.models import Pose
from vlnav.mapping.occupancy import FREE, OCCUPIED, OccupancyGrid2D
from vlnav.navigation.path_planner import PathPlanner


class GoalResult(BaseModel):
    goal: Pose | None = None
    path: list[tuple[float, float]] | None = None
    cell_state: str                              # free | unknown | occupied | out_of_bounds (at the projected point)
    flags: list[str] = []                        # snapped | walked_back
    reason: str | None = None                    # unreachable
    walkback_m: float = 0.0


def cell_state(grid: OccupancyGrid2D, xy) -> str:
    cell = grid.world_to_cell(np.asarray(xy, float)[:2])
    if 0 in grid.shape or not grid.in_bounds(cell):
        return "out_of_bounds"
    v = int(grid.value(cell)[0])
    return "free" if v == FREE else "occupied" if v == OCCUPIED else "unknown"


def goal_pose(xy, z: float, yaw: float, t: float) -> Pose:
    return Pose(position=(float(xy[0]), float(xy[1]), float(z)), orientation_xyzw=yaw_quat(yaw), stamp=t)


def make_reachable(grid: OccupancyGrid2D, planner: PathPlanner, robot_xy, p_xy, z: float, t: float,
                   walkback_step_m: float, min_dist_m: float, yaw: float | None = None) -> GoalResult:
    """First point on the segment p -> robot (from p, every `walkback_step_m`) with a path.

    Points and snapped goals closer than `min_dist_m` to the robot are not used (except for
    rotations). Goal yaw is the heading from the robot to the goal unless `yaw` is given (rotations)."""
    r = np.asarray(robot_xy, float)[:2]
    p = np.asarray(p_xy, float)[:2]
    state = cell_state(grid, p)
    seg = p - r
    length = float(np.linalg.norm(seg))
    steps = [0.0]
    if walkback_step_m > 0:
        steps += list(np.arange(walkback_step_m, max(length - min_dist_m, 0.0) + 1e-9, walkback_step_m))
    for back in steps:
        q = p - seg / max(length, 1e-9) * back if back else p
        path = planner.plan(grid, r, q)
        if path is None:
            continue
        end = np.asarray(path[-1], float)
        if yaw is None and np.linalg.norm(end - r) < min_dist_m:     # snapped next to the robot: no progress
            continue
        flags = (["walked_back"] if back else []) + (["snapped"] if np.linalg.norm(end - q) > grid.resolution else [])
        heading = yaw if yaw is not None else (math.atan2(end[1] - r[1], end[0] - r[0])
                                               if np.linalg.norm(end - r) > 1e-6 else 0.0)
        return GoalResult(goal=goal_pose(end, z, heading, t), path=path, cell_state=state, flags=flags,
                          walkback_m=float(back))
    return GoalResult(cell_state=state, reason="unreachable")
