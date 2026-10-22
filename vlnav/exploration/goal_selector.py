"""Goal selection (TASK_nesy_exploration.md §9, paper Sec. III-C).

explore: instance points first (highest ranking score whose approach point is reachable), else
frontier points by S_NeSy (first reachable, at most `max_reach_checks` plans per step), else hold.
Robustness [DESIGN]: hysteresis (`switch_margin`; instances preempt frontiers outside commitment), plan-failure
blacklisting (callback to the pools), stuck detection, path reuse until the grid version changes or
`replan_period_s` passes. Optional commitment pins exploration position/yaw and delays all
challengers while the destination is valid; yaw-change penalties adjust candidate ranking.
goto: drive to `Subtask.goal_pose`, `reached` once inside
`goto_reach_radius_m`, `nav_failed` after `max_plan_failures` consecutive planning failures.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from vlnav.common.geometry import wrap_angle
from vlnav.exploration.config import ScoringCfg, SelectionCfg
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
    yaw_penalty: float = 0.0


@dataclass
class Choice:
    cand: Candidate
    goal_xy: XY
    path: list[XY]
    goal_yaw: float


@dataclass
class SelectorState:
    current: Choice | None = None
    goal_since: float = -math.inf
    last_goal_yaw: float | None = None
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
    def __init__(self, cfg: SelectionCfg, planner: PathPlanner, scoring: ScoringCfg | None = None):
        self.cfg = cfg
        self.planner = planner
        self.scoring = scoring
        self.s = SelectorState()
        self.debug = False
        self.last_debug: dict = {}
        self.frontier_scores: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        """New subtask: no goal, events re-armed."""
        self.s = SelectorState(n_switches=self.s.n_switches)
        self.frontier_scores.clear()

    # -------------------------------------------------------------------------- explore
    def _plan(self, grid: OccupancyGrid2D, robot_xy: XY, goal: XY):
        self.s.reach_checks += 1
        return self.planner.plan(grid, robot_xy, goal)

    @staticmethod
    def _yaw(c: Candidate, goal: XY, robot_xy: XY) -> float:
        a, b = (goal, c.face_xy) if c.kind == "instance" else (robot_xy, goal)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _rank(self, c: Candidate, yaw: float, robot_yaw: float, t: float) -> Candidate:
        """Penalize wrapped goal-yaw changes, fading after the configured time window."""
        penalty = 0.0
        if self.scoring is not None:
            ref = self.s.last_goal_yaw if self.s.last_goal_yaw is not None else robot_yaw
            age = max(0.0, t - self.s.goal_since)
            window = self.scoring.yaw_change_window_s
            scale = min(1.0, window / max(window, age)) if self.s.last_goal_yaw is not None else 1.0
            penalty = self.scoring.w_yaw * abs(float(wrap_angle(yaw - ref))) / math.pi * scale
        return replace(c, score=c.score + c.yaw_penalty - penalty, yaw_penalty=penalty)

    def _try(self, c: Candidate, grid: OccupancyGrid2D, robot_xy: XY,
             on_result: Callable[[Candidate, bool], None], prefer: XY | None = None,
             goal_yaw: float | None = None, *, t: float, robot_yaw: float) -> Choice | None:
        goals = ([prefer] if prefer is not None and prefer in c.goals else []) + \
            [g for g in c.goals if g != prefer]
        tried = 0
        for g in goals:
            if self.s.reach_checks >= self.cfg.max_reach_checks:
                return None                            # budget spent: no verdict on this candidate
            tried += 1
            path = self._plan(grid, robot_xy, g)
            if self.debug:
                self.last_debug["plans"].append({"kind": c.kind, "id": c.id, "xy": g,
                                                 "reachable": bool(path)})
            if path:
                on_result(c, True)
                yaw = goal_yaw if goal_yaw is not None else self._yaw(c, g, robot_xy)
                return Choice(self._rank(c, yaw, robot_yaw, t), g, path, yaw)
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
                frontiers: list[Candidate], on_result: Callable[[Candidate, bool], None],
                *, robot_yaw: float = 0.0, current_valid: bool = True) -> Choice | None:
        """`instances` ranked, `frontiers` sorted by S_NeSy (both already filtered: δ_reached,
        blacklist, status). Returns the goal or None (hold)."""
        s = self.s
        previous = s.current
        can_keep = (previous is not None and current_valid and
                    math.dist(robot_xy, previous.goal_xy) > self.cfg.delta_reached_m)
        pin = can_keep and self.cfg.min_goal_hold_s > 0
        committing = pin and t - s.goal_since < self.cfg.min_goal_hold_s
        previous_key = (previous.cand.kind, previous.cand.id) if previous is not None else None

        if previous is not None and not can_keep:
            # Don't immediately reselect a destination just reported invalid or reached.
            def remaining(c: Candidate) -> Candidate:
                if (c.kind, c.id) == previous_key:
                    return replace(c, goals=[g for g in c.goals if math.dist(g, previous.goal_xy) >= 1e-6])
                return c
            instances = [c for c in map(remaining, instances) if c.goals]
            frontiers = [c for c in map(remaining, frontiers) if c.goals]

        def rank(c: Candidate) -> Candidate:
            if pin and (c.kind, c.id) == previous_key:
                c = replace(c, goals=[previous.goal_xy])
                yaw = previous.goal_yaw
            else:
                yaw = None
            if not c.goals:
                return c
            return self._rank(c, yaw if yaw is not None else self._yaw(c, c.goals[0], robot_xy), robot_yaw, t)

        instances = sorted((rank(c) for c in instances), key=lambda c: -c.score)
        frontiers = sorted((rank(c) for c in frontiers), key=lambda c: -c.score)
        self.frontier_scores = {c.id: (c.score, c.yaw_penalty) for c in frontiers}
        if self.debug:
            self.last_debug = {"mode": "explore", "previous": None if previous is None else
                               {"kind": previous.cand.kind, "id": previous.cand.id,
                                "xy": previous.goal_xy, "score": previous.cand.score},
                               "retention": "no_previous_goal", "plans": [],
                               "instance_candidates": len(instances), "frontier_candidates": len(frontiers),
                               "switch_margin": self.cfg.switch_margin,
                               "previous_path_version": s.path_version, "grid_version": grid.version,
                               "previous_path_age_s": t - s.path_t}
        s.reach_checks = 0
        by_key = {(c.kind, c.id): c for c in [*instances, *frontiers]}
        # Re-clustering can remove an id while its saved destination remains valid.
        # Retain that orphan only for the bounded commitment interval.
        if committing and previous_key not in by_key:
            by_key[previous_key] = rank(previous.cand)
        keep, replanned = None, False
        failed_key = None
        if self.debug and previous is not None:
            self.last_debug["retention"] = "previous_not_selectable"
        if can_keep and (c := by_key.get(previous_key)) is not None:
            fresh = grid.version == s.path_version and t - s.path_t < self.cfg.replan_period_s
            if self.debug:
                self.last_debug.update(previous_current_score=c.score,
                                       approach_changed=s.current.goal_xy not in c.goals,
                                       grid_changed=grid.version != s.path_version)
            if fresh and s.current.goal_xy in c.goals:          # path still valid: no replan
                yaw = previous.goal_yaw if pin else self._yaw(c, s.current.goal_xy, robot_xy)
                keep = Choice(self._rank(c, yaw, robot_yaw, t), s.current.goal_xy, s.current.path, yaw)
                if self.debug:
                    self.last_debug["retention"] = "cached_path"
            else:
                keep = self._try(c, grid, robot_xy, on_result, prefer=s.current.goal_xy,
                                 goal_yaw=previous.goal_yaw if pin else None, t=t, robot_yaw=robot_yaw)
                replanned = keep is not None
                if keep is None:
                    failed_key = previous_key
                if self.debug:
                    self.last_debug["retention"] = "replanned" if replanned else "previous_path_failed"
        best = None
        for c in [*instances, *frontiers]:
            if committing and keep is not None:
                break
            if s.reach_checks >= self.cfg.max_reach_checks:
                break
            if (c.kind, c.id) == failed_key:
                continue
            if self._can_win(c, keep):
                trial = self._try(c, grid, robot_xy, on_result, t=t, robot_yaw=robot_yaw)
                if trial is not None and self._can_win(trial.cand, keep):
                    best = trial
                    break
        choice = best or keep
        same_position = (previous is not None and choice is not None and
                         previous.cand.kind == choice.cand.kind and
                         math.dist(previous.goal_xy, choice.goal_xy) < 1e-6)
        if same_position and pin and choice.cand.kind == "frontier":
            choice.goal_yaw = previous.goal_yaw
            choice.cand = self._rank(choice.cand, choice.goal_yaw, robot_yaw, t)
        if choice is not None and choice.cand.kind == "frontier":
            self.frontier_scores[choice.cand.id] = (choice.cand.score, choice.cand.yaw_penalty)
        if self.debug:
            if choice is None:
                reason = "no_reachable_candidate"
            elif previous is None:
                reason = "initial_selection"
            elif (choice.cand.kind, choice.cand.id) == (previous.cand.kind, previous.cand.id):
                reason = "approach_changed" if choice.goal_xy != previous.goal_xy else "kept_goal"
            elif keep is None:
                reason = self.last_debug["retention"]
            elif choice.cand.kind != previous.cand.kind:
                reason = "instance_preempts_frontier"
            else:
                reason = "challenger_exceeds_switch_margin"
            self.last_debug.update(reason=reason, reach_checks=s.reach_checks,
                                   reach_budget_exhausted=s.reach_checks >= self.cfg.max_reach_checks,
                                   selected=None if choice is None else
                                   {"kind": choice.cand.kind, "id": choice.cand.id,
                                    "xy": choice.goal_xy, "score": choice.cand.score,
                                    "yaw_penalty": choice.cand.yaw_penalty, "goal_yaw_rad": choice.goal_yaw})
            if committing and keep is not None and choice is keep:
                self.last_debug["reason"] = "goal_commitment"
        if choice is not None and s.current is not None and \
                (choice.cand.kind, choice.cand.id) != (s.current.cand.kind, s.current.cand.id):
            s.n_switches += 1
        if best is not None or replanned:
            s.path_version, s.path_t = grid.version, t
        changed = (choice is not None and (not same_position or
                   abs(float(wrap_angle(choice.goal_yaw - previous.goal_yaw))) > 1e-6))
        if changed:
            s.history.clear()
            s.goal_since = t
            s.last_goal_yaw = choice.goal_yaw
        if choice is None:
            s.history.clear()
        elif self.cfg.stuck_detection and not s.history:
            s.history.append((t, robot_xy))            # driving starts: stuck clock starts
        if self.debug:
            age = t - s.goal_since if choice is not None else None
            self.last_debug.update(goal_age_s=age, commitment_remaining_s=
                                   max(0.0, self.cfg.min_goal_hold_s - age) if age is not None else 0.0)
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
        if self.debug:
            self.last_debug = {"mode": "goto", "goal_xy": goal_xy, "goal_yaw_rad": goal_yaw,
                               "distance_m": math.dist(robot_xy, goal_xy),
                               "goto_failures": s.goto_failures, "grid_version": grid.version}
        yaw_ok = c.goto_yaw_tol_rad is None or abs(float(wrap_angle(robot_yaw - goal_yaw))) <= c.goto_yaw_tol_rad
        if math.dist(robot_xy, goal_xy) <= c.goto_reach_radius_m and yaw_ok:
            if self.debug:
                self.last_debug["reason"] = "goto_reached"
            event = None if s.reached_sent else "reached"
            s.reached_sent = True
            return "hold", None, event
        if s.reached_sent or s.nav_failed_sent:
            if self.debug:
                self.last_debug["reason"] = "goto_already_finished"
            return "hold", None, None
        s.reach_checks = 0
        path = self._plan(grid, robot_xy, goal_xy)
        if path is None:
            s.goto_failures += 1
            if self.debug:
                self.last_debug.update(reason="goto_path_failed", goto_failures=s.goto_failures)
            if s.goto_failures >= c.max_plan_failures:
                s.nav_failed_sent = True
                return "hold", None, "nav_failed"
            return "goto", None, None
        s.goto_failures = 0
        if self.debug:
            self.last_debug.update(reason="goto_path_valid", goto_failures=0)
        return "goto", path, None
