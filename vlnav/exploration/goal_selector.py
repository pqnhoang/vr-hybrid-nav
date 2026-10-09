"""Goal selection (TASK_nesy_exploration.md §9, paper Sec. III-C).

explore: instance points first (highest ranking score whose approach point is reachable), else
frontier points by S_NeSy (first reachable, at most `max_reach_checks` plans per step), else hold.
Robustness [DESIGN]: hysteresis (`switch_margin`; instances always preempt frontiers), plan-failure
blacklisting (callback to the pools), stuck detection, path reuse until the grid version changes or
`replan_period_s` passes. goto: drive to `Subtask.goal_pose`, `reached` once inside
`goto_reach_radius_m`, `nav_failed` after `max_plan_failures` consecutive planning failures.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from vlnav.common.geometry import wrap_angle
from vlnav.exploration.config import SelectionCfg
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.navigation.path_planner import PathPlanner

XY = tuple[float, float]


@dataclass
class Candidate:
    kind: Literal["instance", "frontier"]
    id: int
    score: float                        # instance: ranking score; frontier: S_NeSy
    goals: list[XY]                     # goal options in preference order (instance: approach cells)
    face_xy: XY | None = None           # instance position: the goal pose faces it


@dataclass
class Choice:
    cand: Candidate
    goal_xy: XY
    path: list[XY]


@dataclass
class SelectorState:
    current: Choice | None = None
    path_version: int = -1
    path_t: float = -math.inf
    hold_since: float | None = None
    no_frontier_sent: bool = False
    reached_sent: bool = False
    nav_failed_sent: bool = False
    goto_failures: int = 0
    history: deque = field(default_factory=deque)       # (t, xy) while driving (stuck detection)
    n_switches: int = 0
    reach_checks: int = 0                                # plans tried in the last step


class GoalSelector:
    def __init__(self, cfg: SelectionCfg, planner: PathPlanner):
        self.cfg = cfg
        self.planner = planner
        self.s = SelectorState()

    def reset(self) -> None:
        """New subtask: no goal, events re-armed."""
        self.s = SelectorState(n_switches=self.s.n_switches)

    # -------------------------------------------------------------------------- explore
    def _plan(self, grid: OccupancyGrid2D, robot_xy: XY, goal: XY):
        self.s.reach_checks += 1
        return self.planner.plan(grid, robot_xy, goal)

    def _try(self, c: Candidate, grid: OccupancyGrid2D, robot_xy: XY,
             on_result: Callable[[Candidate, bool], None], prefer: XY | None = None) -> Choice | None:
        goals = ([prefer] if prefer is not None and prefer in c.goals else []) + \
            [g for g in c.goals if g != prefer]
        tried = 0
        for g in goals:
            if self.s.reach_checks >= self.cfg.max_reach_checks:
                return None                            # budget spent: no verdict on this candidate
            tried += 1
            path = self._plan(grid, robot_xy, g)
            if path:
                on_result(c, True)
                return Choice(c, g, path)
        if tried == len(goals):                        # every option unreachable (or none exists)
            on_result(c, False)
        return None

    def _can_win(self, c: Candidate, keep: Choice | None) -> bool:
        if keep is None:
            return True
        k = keep.cand
        if (c.kind, c.id) == (k.kind, k.id):
            return False
        if c.kind != k.kind:
            return c.kind == "instance"                # instances always preempt frontier goals
        return c.score > k.score + self.cfg.switch_margin

    def explore(self, grid: OccupancyGrid2D, robot_xy: XY, t: float, instances: list[Candidate],
                frontiers: list[Candidate], on_result: Callable[[Candidate, bool], None]) -> Choice | None:
        """`instances` ranked, `frontiers` sorted by S_NeSy (both already filtered: δ_reached,
        blacklist, status). Returns the goal or None (hold)."""
        s = self.s
        s.reach_checks = 0
        by_key = {(c.kind, c.id): c for c in [*instances, *frontiers]}
        keep, replanned = None, False
        if s.current is not None and (c := by_key.get((s.current.cand.kind, s.current.cand.id))) is not None:
            fresh = grid.version == s.path_version and t - s.path_t < self.cfg.replan_period_s
            if fresh and s.current.goal_xy in c.goals:          # path still valid: no replan
                keep = Choice(c, s.current.goal_xy, s.current.path)
            else:
                keep = self._try(c, grid, robot_xy, on_result, prefer=s.current.goal_xy)
                replanned = keep is not None
        best = None
        for c in [*instances, *frontiers]:
            if s.reach_checks >= self.cfg.max_reach_checks:
                break
            if self._can_win(c, keep) and (best := self._try(c, grid, robot_xy, on_result)) is not None:
                break
        choice = best or keep
        if choice is not None and s.current is not None and \
                (choice.cand.kind, choice.cand.id) != (s.current.cand.kind, s.current.cand.id):
            s.n_switches += 1
        if best is not None or replanned:
            s.path_version, s.path_t = grid.version, t
        if choice is None:
            s.history.clear()
        elif self.cfg.stuck_detection and not s.history:
            s.history.append((t, robot_xy))            # driving starts: stuck clock starts
        s.current = choice
        return choice

    def hold_timeout(self, t: float, holding: bool) -> bool:
        """True once per subtask after holding for `no_frontier_patience_s` (pitfall 20 escape)."""
        s = self.s
        if not holding:
            s.hold_since = None
            return False
        s.hold_since = t if s.hold_since is None else s.hold_since
        if not s.no_frontier_sent and t - s.hold_since >= self.cfg.no_frontier_patience_s:
            s.no_frontier_sent = True
            return True
        return False

    def stuck(self, robot_xy: XY, t: float) -> bool:
        """Displacement < stuck_dist_m over stuck_window_s while driving to a goal."""
        s = self.s
        if not self.cfg.stuck_detection or s.current is None:
            s.history.clear()
            return False
        s.history.append((t, robot_xy))
        while len(s.history) > 1 and t - s.history[1][0] >= self.cfg.stuck_window_s:
            s.history.popleft()
        t0, xy0 = s.history[0]
        if t - t0 >= self.cfg.stuck_window_s and math.dist(xy0, robot_xy) < self.cfg.stuck_dist_m:
            s.history.clear()
            return True
        return False

    # ----------------------------------------------------------------------------- goto
    def goto(self, grid: OccupancyGrid2D, robot_xy: XY, robot_yaw: float, t: float, goal_xy: XY,
             goal_yaw: float) -> tuple[Literal["goto", "hold"], list[XY] | None, str | None]:
        """(goal kind, path, event type or None)."""
        s, c = self.s, self.cfg
        yaw_ok = c.goto_yaw_tol_rad is None or abs(float(wrap_angle(robot_yaw - goal_yaw))) <= c.goto_yaw_tol_rad
        if math.dist(robot_xy, goal_xy) <= c.goto_reach_radius_m and yaw_ok:
            event = None if s.reached_sent else "reached"
            s.reached_sent = True
            return "hold", None, event
        if s.reached_sent or s.nav_failed_sent:
            return "hold", None, None
        s.reach_checks = 0
        path = self._plan(grid, robot_xy, goal_xy)
        if path is None:
            s.goto_failures += 1
            if s.goto_failures >= c.max_plan_failures:
                s.nav_failed_sent = True
                return "hold", None, "nav_failed"
            return "goto", None, None
        s.goto_failures = 0
        return "goto", path, None
