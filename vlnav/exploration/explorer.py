"""ExplorationCore: public API of NeSy exploration (TASK_nesy_exploration.md §3).

    set_subtask(st)        explore: reset semantic state (instance pool, VL validity) | goto | stop
    on_frame(res, info)    every processed frame: frontier cells / pool, VL scores, instance points
    step(robot_pose, t)    at select_rate_hz: one goal + events (reached / nav_failed / no_frontier)
    snapshot()             state for logging and the top-down debug view (exploration/viz.py)

Never imports planner/, memory/ or acquisition/ (room lookup for `room_bonus` is injected).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import numpy as np
from pydantic import BaseModel

from vlnav.common.geometry import ground_heading, yaw_quat
from vlnav.common.models import FrameResult, PlannerEvent, Pose, Subtask
from vlnav.exploration.config import ExplorationConfig
from vlnav.exploration.curiosity import UnknownRatioCache, s_dist, s_unknown
from vlnav.exploration.frontiers import FrontierMap, Wedge
from vlnav.exploration.goal_selector import Candidate, Choice, GoalSelector
from vlnav.exploration.instances import InstancePool, approach_cells
from vlnav.exploration.scoring import FrontierScore, s_nesy
from vlnav.exploration.vl_score import Mixture, assign, mixture, reference_pose, s_vl
from vlnav.mapping.occupancy import OCCUPIED, OccupancyGrid2D, UpdateInfo
from vlnav.navigation.path_planner import PathPlanner

RoomLabelsNear = Callable[[tuple[float, float]], set[str]]     # labels of the room at xy + neighbors


class ExplorationOutput(BaseModel):
    goal: Pose | None
    goal_kind: Literal["instance", "frontier", "goto", "hold", "move"]      # move: set by move/core.py
    goal_id: int | None
    path: list[tuple[float, float]] | None
    events: list[PlannerEvent]


def _pose(xy, z: float, yaw: float, t: float) -> Pose:
    return Pose(position=(float(xy[0]), float(xy[1]), float(z)), orientation_xyzw=yaw_quat(yaw), stamp=t)


class ExplorationCore:
    def __init__(self, cfg: ExplorationConfig, grid: OccupancyGrid2D, planner: PathPlanner,
                 robot_radius_m: float = 0.0, room_labels_near: RoomLabelsNear | None = None):
        """`robot_radius_m` (navigation's) keeps approach points out of the inflated zone."""
        self.cfg = cfg
        self.grid = grid
        self.planner = planner
        self.robot_radius_m = robot_radius_m
        self.room_labels_near = room_labels_near
        self.frontiers = FrontierMap(cfg.frontier)
        self.instances = InstancePool(cfg.instance)
        self.selector = GoalSelector(cfg.selection, planner)
        self.unknown = UnknownRatioCache(cfg.curiosity.unknown_bfs_radius_m, self.frontiers.structure)
        self.subtask: Subtask | None = None
        self.goto_object_xy: tuple[float, float] | None = None
        self.goto_approach_required = False
        self.camera_yaw_offsets: dict[str, float] = {}
        self._wedges: list[Wedge] = []
        self._mix: Mixture | None = None
        self._scores: list[FrontierScore] = []
        self._out: ExplorationOutput | None = None
        self._robot: tuple[float, float, float] | None = None
        self._notes: list[str] = []
        self._t_frame = self._t_step = None
        self._ms = {"on_frame": 0.0, "step": 0.0}

    @property
    def robot(self) -> tuple[float, float, float] | None:
        """Robot (x, y, yaw) at the last `step`; None before the first."""
        return self._robot

    # --------------------------------------------------------------------------- subtask
    def set_subtask(self, st: Subtask, object_xy: tuple[float, float] | None = None,
                    *, require_object_approach: bool = False) -> None:
        self.subtask = st
        self.goto_object_xy = object_xy if st.type == "goto" else None
        self.goto_approach_required = (require_object_approach or object_xy is not None) and st.type == "goto"
        self.instances.clear()
        self.selector.reset()
        if st.type == "explore" and not self.cfg.vl.rescore_on_new_subtask:
            for p in self.frontiers.points.values():      # strict paper: old scores reset, never redone
                p.s_vl, p.s_vl_subtask_id, p.n_vl_updates, p.vl_weight, p.vl_xy = 0.0, st.id, 0, 0.0, p.xy

    def _explore_id(self) -> int | None:
        st = self.subtask
        return st.id if st is not None and st.type == "explore" else None

    def _event(self, kind: str, t: float, **payload) -> PlannerEvent:
        return PlannerEvent(type=kind, stamp=t, task_id=self.subtask.task_id, subtask_id=self.subtask.id,
                            payload={"source": "exploration", **payload})

    # ----------------------------------------------------------------------------- frame
    def _check_camera(self, res: FrameResult) -> None:
        """AUDIT 2.1: `angle_reference: robot` only for one camera without yaw offset."""
        f = res.frame
        if f.camera_id in self.camera_yaw_offsets:
            return
        T_base_cam = np.linalg.inv(f.T_map_base) @ f.T_map_cam
        off = ground_heading(T_base_cam[:3, :3], 2)
        self.camera_yaw_offsets[f.camera_id] = off
        if self.cfg.angle_reference != "robot":
            return
        if len(self.camera_yaw_offsets) > 1:
            raise ValueError(f"angle_reference 'robot' with several cameras {sorted(self.camera_yaw_offsets)}: "
                             "use 'camera' (AUDIT 2.1)")
        if abs(math.degrees(off)) > self.cfg.max_camera_yaw_offset_deg:
            raise ValueError(f"camera '{f.camera_id}' is yawed {math.degrees(off):.1f} deg from the base heading "
                             f"(> {self.cfg.max_camera_yaw_offset_deg}): use angle_reference 'camera' (AUDIT 2.1)")

    def on_frame(self, res: FrameResult, info: UpdateInfo) -> None:
        t0 = time.perf_counter()
        self._check_camera(res)
        f, cfg = res.frame, self.cfg
        ref = reference_pose(f, cfg.angle_reference)
        self._wedges = [Wedge(ref.x, ref.y, ref.psi, f.hfov, cfg.frontier.sensor_range_m)]
        sid = self._explore_id()
        created = self.frontiers.update(self.grid, info, self._wedges, f.stamp, sid)
        self._mix = None
        if sid is not None:
            self._mix = mix = mixture(res, cfg.vl.sigma_rad)
            pts = self.frontiers.points
            ids = list(pts) if cfg.vl.rescore_on_new_subtask else created
            if ids:
                xy = np.array([pts[i].xy for i in ids])
                s, c, inside = s_vl(xy, ref, f.hfov, mix, cfg.frontier.sensor_range_m)
                for i, si, ci, ok in zip(ids, s, c, inside):
                    if ok:
                        assign(pts[i], si, ci, sid, cfg.vl.vl_update_mode)
                    elif not cfg.vl.rescore_on_new_subtask:            # created outside the wedge
                        assign(pts[i], 0.0, 0.0, sid, cfg.vl.vl_update_mode)
            if cfg.ibtp_enabled:
                self.instances.add_frame(res, ref, f.hfov, mix, sid, f.stamp, cfg.instance.instance_score)
        self._t_frame = f.stamp
        self._ms["on_frame"] = 1e3 * (time.perf_counter() - t0)

    # ------------------------------------------------------------------------------ step
    def _frontier_scores(self, robot_xy, t: float, sid: int) -> list[FrontierScore]:
        cfg, sc = self.cfg, self.cfg.scoring
        hint = self.subtask.room_hint if self.subtask else None
        use_room = sc.room_bonus.enabled and hint and self.room_labels_near is not None
        path_metric = cfg.curiosity.dist_metric == "path" and hasattr(self.planner, "cost_to")
        out = []
        for p in self.frontiers.points.values():
            d = math.dist(robot_xy, p.xy)
            if d <= cfg.selection.delta_reached_m or p.blacklisted_until > t:
                continue
            if path_metric:
                d = self.planner.cost_to(self.grid, robot_xy, p.xy)
            sd = s_dist(d) if math.isfinite(d) else 0.0
            vl = p.s_vl if p.s_vl_subtask_id == sid else 0.0
            su = s_unknown(self.unknown.get(self.grid, p.id, p.xy), cfg.curiosity.unknown_k) \
                if cfg.curiosity_enabled else 1.0
            rb = sc.room_bonus.w_room if use_room and hint in self.room_labels_near(p.xy) else 0.0
            out.append(FrontierScore(p.id, p.xy, d, sd, vl, su, rb,
                                     s_nesy(sd, vl, su, cfg.w_dist, cfg.w_vl, sc.form) + rb))
        self.unknown.prune(set(self.frontiers.points))
        out.sort(key=lambda f: (-f.s_nesy, f.id))
        return out

    def _on_result(self, t: float):
        sel = self.cfg.selection

        def apply(c: Candidate, ok: bool) -> None:
            p = (self.frontiers.points if c.kind == "frontier" else self.instances.points).get(c.id)
            if p is None:
                return
            p.plan_failures = 0 if ok else p.plan_failures + 1
            if p.plan_failures >= sel.max_plan_failures:
                p.plan_failures = 0
                if c.kind == "frontier":
                    p.blacklisted_until = t + sel.blacklist_s
                    self._notes.append(f"frontier #{c.id} unreachable -> blacklisted {sel.blacklist_s:g} s")
                else:
                    p.status = "unreachable"
                    self._notes.append(f"instance #{c.id} unreachable")
        return apply

    def _blacklist_current(self, t: float, why: str) -> None:
        cur = self.selector.s.current
        if cur is None:
            return
        if cur.cand.kind == "frontier" and cur.cand.id in self.frontiers.points:
            self.frontiers.points[cur.cand.id].blacklisted_until = t + self.cfg.selection.blacklist_s
        elif cur.cand.kind == "instance" and cur.cand.id in self.instances.points:
            self.instances.points[cur.cand.id].status = "unreachable"
        self._notes.append(f"{why}: {cur.cand.kind} #{cur.cand.id} dropped")
        self.selector.s.current = None

    def step(self, robot_pose: Pose, t: float) -> ExplorationOutput:
        t0 = time.perf_counter()
        self._notes = []
        cfg = self.cfg
        x, y, z = robot_pose.position
        robot_xy, yaw = (x, y), robot_pose.yaw()
        self._robot = (x, y, yaw)
        self._scores = []
        st, events = self.subtask, []
        if st is None or st.type in ("stop", "move") or 0 in self.grid.shape:      # move: move/core.py drives
            out = ExplorationOutput(goal=None, goal_kind="hold", goal_id=None, path=None, events=[])
        elif st.type == "goto":
            gp = st.goal_pose
            if self.goto_approach_required:
                approach = self._goto_approach(robot_xy) if self.goto_object_xy is not None else None
                if approach is None:
                    reason = "no object position" if self.goto_object_xy is None else "no reachable approach cell"
                    self._notes.append(f"{reason} for object #{st.node_id}")
                    gp = None
                else:
                    face = math.atan2(self.goto_object_xy[1] - approach[1],
                                      self.goto_object_xy[0] - approach[0])
                    gp = _pose(approach, z, face, t)
            if gp is None:
                out = ExplorationOutput(goal=None, goal_kind="hold", goal_id=st.node_id, path=None, events=[])
            else:
                kind, path, ev = self.selector.goto(self.grid, robot_xy, yaw, t, gp.position[:2], gp.yaw())
                if ev is not None:
                    events.append(self._event(ev, t, goal=list(gp.position[:2])))
                out = ExplorationOutput(goal=gp if kind == "goto" else None, goal_kind=kind,
                                        goal_id=st.node_id, path=path, events=events)
        else:
            out = self._explore_step(robot_xy, z, t, st)
        self._t_step, self._out = t, out
        self._ms["step"] = 1e3 * (time.perf_counter() - t0)
        return out

    def _goto_approach(self, robot_xy: tuple[float, float]) -> tuple[float, float] | None:
        """Nearest reachable free standoff cell around a verified object."""
        target = self.goto_object_xy
        blocked = self.grid.inflated(self.robot_radius_m)
        cfg = self.cfg.instance
        cells = approach_cells(self.grid, blocked, target,
                               cfg.inst_standoff_min_m, cfg.inst_standoff_max_m)
        if not len(cells):
            return None
        reach = getattr(self.planner, "reach_costs", None)
        if reach is not None:
            costs = reach(self.grid, robot_xy, cells)
            if not np.isfinite(costs).any():
                return None
            i = int(np.argmin(costs))
            return float(cells[i, 0]), float(cells[i, 1])
        order = np.argsort(np.linalg.norm(cells - np.asarray(robot_xy), axis=1))
        for i in order:
            if self.planner.plan(self.grid, robot_xy, cells[i]) is not None:
                return float(cells[i, 0]), float(cells[i, 1])
        return None

    def _explore_step(self, robot_xy, z: float, t: float, st: Subtask) -> ExplorationOutput:
        cfg, sel = self.cfg, self.cfg.selection
        for pid in self.instances.update_visits(robot_xy, sel.delta_reached_m):
            self._notes.append(f"instance #{pid} visited")
        if self.selector.stuck(robot_xy, t):
            self._blacklist_current(t, "stuck")
        blocked = self.grid.inflated(self.robot_radius_m) if self.robot_radius_m > 0 else \
            self.grid.as_array() == OCCUPIED
        inst = []
        if cfg.ibtp_enabled:
            active = [p for p in self.instances.points.values()
                      if p.status == "active" and math.dist(robot_xy, p.xy) > sel.delta_reached_m]
            reach = getattr(self.planner, "reach_costs", None)
            for p in sorted(active, key=lambda q: (-q.rank, -q.score, q.id)):
                opts = self.instances.approach_options(p, self.grid, blocked, robot_xy)
                if reach is not None and len(opts):          # nearest *reachable* approach cell (§6)
                    opts = opts[np.isfinite(reach(self.grid, robot_xy, opts))]
                # one plan per instance and step: unreachable options must not use up the
                # max_reach_checks budget of the frontiers (found on vr_h3_2: 8-step hold)
                inst.append(Candidate("instance", p.id, p.rank, [(float(a), float(b)) for a, b in opts[:1]], p.xy))
        self._scores = self._frontier_scores(robot_xy, t, st.id)
        front = [Candidate("frontier", f.id, f.s_nesy, [f.xy]) for f in self._scores]
        choice = self.selector.explore(self.grid, robot_xy, t, inst, front, self._on_result(t))
        events = []
        if self.selector.hold_timeout(t, choice is None):
            events.append(self._event("no_frontier", t, frontier_points=len(self.frontiers.points),
                                      active_instances=sum(p.status == "active"
                                                           for p in self.instances.points.values())))
        if choice is None:
            return ExplorationOutput(goal=None, goal_kind="hold", goal_id=None, path=None, events=events)
        c = choice.cand
        if c.kind == "instance":
            self.instances.points[c.id].approach_xy = choice.goal_xy
            yaw = math.atan2(c.face_xy[1] - choice.goal_xy[1], c.face_xy[0] - choice.goal_xy[0])
        else:
            yaw = math.atan2(choice.goal_xy[1] - robot_xy[1], choice.goal_xy[0] - robot_xy[0])
        return ExplorationOutput(goal=_pose(choice.goal_xy, z, yaw, t), goal_kind=c.kind, goal_id=c.id,
                                 path=choice.path, events=events)

    # -------------------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """Plain data for logs and the debug view; `frontier_cells` is an (N, 2) array of (ix, iy)."""
        sid = self._explore_id()
        st = self.subtask
        scores = {f.id: f for f in self._scores}
        cur: Choice | None = self.selector.s.current if self._out and self._out.goal_kind in ("instance", "frontier") \
            else None
        out = self._out
        pts = []
        for p in sorted(self.frontiers.points.values(), key=lambda q: q.id):
            f = scores.get(p.id)
            pts.append({"id": p.id, "xy": p.xy, "cluster_id": p.cluster_id,
                        "s_vl": p.s_vl if p.s_vl_subtask_id == sid and sid is not None else 0.0,
                        "scored": sid is not None and p.s_vl_subtask_id == sid, "n_vl_updates": p.n_vl_updates,
                        "blacklisted": p.blacklisted_until > (self._t_step or 0.0),
                        "score": None if f is None else {"s_nesy": f.s_nesy, "s_dist": f.s_dist, "dist": f.dist,
                                                         "s_vl": f.s_vl, "s_unknown": f.s_unknown,
                                                         "room_bonus": f.room_bonus}})
        return {
            "t": self._t_step, "frame_t": self._t_frame,
            "subtask": None if st is None else {"id": st.id, "task_id": st.task_id, "type": st.type,
                                                "target": st.target, "prompts": list(st.detector_prompts),
                                                "room_hint": st.room_hint, "node_id": st.node_id,
                                                "goal_xy": list(st.goal_pose.position[:2]) if st.goal_pose else None},
            "angle_reference": self.cfg.angle_reference,
            "robot": self._robot,
            "wedges": [asdict(w) for w in self._wedges],
            "mixture": None if self._mix is None else {"labels": list(self._mix.labels),
                                                       "mu": self._mix.mu.tolist(), "alpha": self._mix.alpha.tolist()},
            "grid": {"version": self.grid.version, "origin": self.grid.origin_xy,
                     "resolution": self.grid.resolution, "shape": self.grid.shape},
            "frontier_cells": self.frontiers.cells(),
            "points": pts,
            "instances": [{"id": p.id, "xy": p.xy, "label": p.label, "score": p.score, "rank": p.rank,
                           "status": p.status, "approach_xy": p.approach_xy, "n_obs": p.n_obs}
                          for p in sorted(self.instances.points.values(), key=lambda q: q.id)],
            "goal_kind": None if out is None else out.goal_kind,
            "goal": None if out is None or out.goal is None else
            {"kind": out.goal_kind, "id": out.goal_id, "xy": tuple(out.goal.position[:2]),
             "score": cur.cand.score if cur else None, "face_xy": cur.cand.face_xy if cur else None},
            "path": None if out is None else out.path,
            "events": [] if out is None else [{"type": e.type, "stamp": e.stamp, "payload": e.payload}
                                              for e in out.events],
            "notes": list(self._notes),
            "n_switches": self.selector.s.n_switches,
            "reach_checks": self.selector.s.reach_checks,
            "timing_ms": dict(self._ms),
        }
