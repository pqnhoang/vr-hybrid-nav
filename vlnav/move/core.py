"""MoveCore: the `move` subtask (TASK_move_subtask.md §9).

    set_subtask(st)        move: reset history, counters, goal; other types: inactive
    on_frame(frame)        every frame: motion-sampled history (history.py), latest frame
    step(robot_pose, t)    S2 cycles, projection, goal, events (reached / nav_failed / timeout)
    snapshot()             state for logging and the debug view (move/viz.py)
    pop_records()          one record per S2 cycle (move_log.jsonl)

One S2 cycle = a forward query on the latest frame, then, when S2 answers `↓` and
`look_down_mode: same_frame`, look-down queries on the same frame (up to `max_look_down`); the camera
image is shown to the model through the virtual Habitat view (view.py). Cycles run in a worker
thread (`s2.blocking: false`) or inline. Replies are projected with the geometry of the frame they
were asked on, so a goal stays valid while the robot moves.

Never imports planner/, memory/ or acquisition/.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from vlnav.common.geometry import wrap_angle
from vlnav.common.models import Frame, PlannerEvent, Pose, Subtask
from vlnav.exploration.explorer import ExplorationOutput
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.move.config import MoveConfig
from vlnav.move.goal import GoalResult, goal_pose, make_reachable
from vlnav.move.history import FrameBuffer
from vlnav.move.projection import Projection, floor_fn, measured_depth, pixel_to_map
from vlnav.move.reply import S2Reply, parse_s2_text
from vlnav.move.s2_client import S2Client, S2Request, S2Response
from vlnav.move.view import ModelView, make_view
from vlnav.navigation.path_planner import PathPlanner


@dataclass
class _Cycle:
    gen: int
    k: int
    frame: Frame
    view: ModelView
    image: np.ndarray                      # model-view image of the frame
    history: list[np.ndarray]
    t_req: float
    exchanges: list[tuple[S2Request, S2Response]] = field(default_factory=list)
    reply: S2Reply | None = None
    error: str | None = None


@dataclass
class _Goal:
    kind: str                              # pixel | forward | rotate
    pose: Pose
    path: list[tuple[float, float]] | None
    t: float
    target_yaw: float | None = None
    id: int = 0                            # goal_id of the output: same goal (kept by hysteresis) -> same id


def _xyyaw(p: Pose) -> list[float]:
    return [round(p.position[0], 3), round(p.position[1], 3), round(p.yaw(), 3)]


class MoveCore:
    def __init__(self, cfg: MoveConfig, grid: OccupancyGrid2D, planner: PathPlanner, s2: S2Client):
        self.cfg = cfg
        self.grid = grid
        self.planner = planner
        self.s2 = s2
        mi = cfg.model_input
        self.buffer = FrameBuffer(cfg.history.step_m, cfg.history.step_rad, (mi.resize_w, mi.resize_h),
                                  mi.num_history)
        self._pool = None if cfg.s2.blocking else ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2")
        self.subtask: Subtask | None = None
        self.camera_id: str | None = None
        self._views: dict[tuple, ModelView] = {}
        self._gen = 0
        self._n_goals = 0                  # goal ids, never reset (ids stay unique across subtasks)
        self._reset()

    def _reset(self) -> None:
        self.buffer.reset()
        self.latest: Frame | None = None
        self.done = False
        self.n_cycles = 0
        self.n_failures = 0                # consecutive replies that gave no goal
        self.goal: _Goal | None = None
        self.last: dict | None = None
        self.last_image: np.ndarray | None = None      # model-view image of the last handled cycle (viz)
        self._job: Future | None = None
        self._pending: _Cycle | None = None
        self._records: list[dict] = []
        self._events: list[PlannerEvent] = []
        self._t_start: float | None = None
        self._progress: tuple[np.ndarray, float] | None = None
        self._requery = True

    # --------------------------------------------------------------------------- subtask
    @property
    def active(self) -> bool:
        return self.subtask is not None and self.subtask.type == "move" and not self.done

    def _new_goal(self, kind: str, pose: Pose, path, t: float, target_yaw: float | None = None) -> _Goal:
        self._n_goals += 1
        return _Goal(kind, pose, path, t, target_yaw, self._n_goals)

    def set_subtask(self, st: Subtask | None) -> None:
        self._gen += 1                     # replies of an older subtask are dropped
        self.subtask = st
        self._reset()

    def _event(self, kind: str, t: float, **payload) -> PlannerEvent:
        st = self.subtask
        return PlannerEvent(type=kind, stamp=t, task_id=st.task_id, subtask_id=st.id,
                            payload={"n_cycles": self.n_cycles, **payload})

    def _finish(self, kind: str, t: float, **payload) -> None:
        self._events.append(self._event(kind, t, **payload))
        self.done = True
        self.goal = None

    # ----------------------------------------------------------------------------- frames
    def view_for(self, frame: Frame) -> ModelView:
        key = (frame.camera_id, frame.rgb.shape[1], frame.rgb.shape[0], *np.round(frame.K.ravel(), 3))
        if key not in self._views:
            v = self.cfg.model_view
            self._views[key] = make_view(frame.K, (frame.rgb.shape[1], frame.rgb.shape[0]), v.mode,
                                         (v.width, v.height), v.hfov_deg)
        return self._views[key]

    def on_frame(self, frame: Frame) -> None:
        if not self.active:
            return
        self.camera_id = self.camera_id or frame.camera_id
        if frame.camera_id != self.camera_id:
            return
        pose = Pose.from_matrix(frame.T_map_base, frame.stamp)
        self.buffer.add(frame.stamp, self.view_for(frame).image(frame.rgb), pose.position[:2], pose.yaw())
        self.latest = frame

    # ------------------------------------------------------------------------------- S2
    def _run_cycle(self, c: _Cycle) -> _Cycle:
        """Forward query, then look-down queries on the same frame while S2 answers `↓`."""
        try:
            req = S2Request(instruction=self.subtask.instruction, image=c.image, history=c.history)
            res = self.s2.query(req)
            c.exchanges.append((req, res))
            reply = parse_s2_text(res.text)
            n_down = 0
            while (reply.kind == "actions" and reply.actions[0] == "look_down"
                   and self.cfg.look_down_mode == "same_frame" and n_down < self.cfg.max_look_down):
                n_down += 1
                req = S2Request(instruction=req.instruction, image=c.image, history=c.history, look_down=True,
                                prev_output=res.text, look_down_image=c.image)
                res = self.s2.query(req)
                c.exchanges.append((req, res))
                reply = parse_s2_text(res.text)
            c.reply = reply
        except Exception as e:                                    # network / server errors: logged, retried
            c.error = f"{type(e).__name__}: {e}"
        return c

    def _start_cycle(self, t: float) -> None:
        f = self.latest
        view = self.view_for(f)
        image = view.image(f.rgb)
        hist = [k.rgb for k in self.buffer.history(f.stamp)]
        self.n_cycles += 1
        c = _Cycle(self._gen, self.n_cycles, f, view, image, hist, t)
        self._requery = False
        if self._pool is None:
            self._pending = self._run_cycle(c)
        else:
            self._job = self._pool.submit(self._run_cycle, c)

    def _collect(self) -> _Cycle | None:
        if self._pending is not None:
            c, self._pending = self._pending, None
            return c
        if self._job is not None and self._job.done():
            c, self._job = self._job.result(), None
            return c
        return None

    @property
    def in_flight(self) -> bool:
        return self._job is not None

    # ------------------------------------------------------------------------- handling
    def _view_pixel(self, c: _Cycle, xy) -> tuple[float, float]:
        """Reply pixel -> pixel of the model-view image (§7.1)."""
        last_req = c.exchanges[-1][0]
        if not last_req.look_down and self.cfg.main_pixel_ref == "resized":
            mi = self.cfg.model_input
            return xy[0] * c.view.out_w / mi.resize_w, xy[1] * c.view.out_h / mi.resize_h
        return float(xy[0]), float(xy[1])

    def _reachable(self, robot: Pose, xy, t: float, yaw: float | None = None) -> GoalResult:
        return make_reachable(self.grid, self.planner, robot.position[:2], xy, robot.position[2], t,
                              self.cfg.goal.walkback_step_m, self.cfg.projection.min_range_m, yaw)

    def _handle(self, c: _Cycle, robot: Pose, t: float) -> dict:
        t0 = time.perf_counter()
        rec = self._decide(c, robot, t)
        rec["handle_ms"] = round(1e3 * (time.perf_counter() - t0), 2)
        self.last_image = c.image
        return rec

    def _decide(self, c: _Cycle, robot: Pose, t: float) -> dict:
        f = c.frame
        rec = {"k": c.k, "t_req": c.t_req, "t_done": t, "frame_stamp": f.stamp,
               "robot_req": _xyyaw(Pose.from_matrix(f.T_map_base, f.stamp)), "robot": _xyyaw(robot),
               "instruction": self.subtask.instruction, "n_history": len(c.history),
               "exchanges": [{"look_down": q.look_down, "text": r.text, "latency_ms": r.latency_ms,
                              "cached": r.cached} for q, r in c.exchanges],
               "reply": None, "decision": None, "reason": None}
        if c.error is not None:
            rec.update(decision="error", reason=c.error)
            self._requery = True
            return rec
        reply = c.reply
        rec["reply"] = reply.model_dump()
        if reply.kind == "invalid":
            return self._fail(rec, "unparsable", t)
        if reply.kind == "actions":
            return self._actions(rec, reply, robot, t)
        vxy = self._view_pixel(c, reply.pixel_xy)
        uv = c.view.to_camera(vxy)
        rec["view_xy"] = [round(vxy[0], 1), round(vxy[1], 1)]
        base, pc = Pose.from_matrix(f.T_map_base, f.stamp), self.cfg.projection
        t0 = time.perf_counter()
        proj: Projection = pixel_to_map(uv, f.K, f.T_map_cam, (f.rgb.shape[1], f.rgb.shape[0]), base.position[:2],
                                        floor_fn(self.grid.ground_heights, base.position, pc),
                                        measured_depth(f, uv, pc), pc)
        rec["projection"] = proj.model_dump()
        rec["projection_ms"] = round(1e3 * (time.perf_counter() - t0), 2)
        if not proj.ok:
            if proj.reason == "too_close":                       # already there: ask again
                rec.update(decision="too_close", reason="too_close")
                self.goal, self._requery = None, True
                return rec
            return self._fail(rec, proj.reason, t)
        res = self._reachable(robot, proj.p_map[:2], t)
        rec["goal_result"] = res.model_dump(exclude={"path", "goal"}) | {
            "goal": _xyyaw(res.goal) if res.goal else None}
        if res.goal is None:
            return self._fail(rec, res.reason, t)
        g = self.goal
        if (g is not None and g.kind == "pixel"
                and math.dist(g.pose.position[:2], res.goal.position[:2]) < self.cfg.goal.switch_dist_m):
            rec.update(decision="kept", goal=_xyyaw(g.pose))      # hysteresis: avoid goal churn
            g.t = t
        else:
            self.goal = self._new_goal("pixel", res.goal, res.path, t)
            rec.update(decision="new_goal", goal=_xyyaw(res.goal))
        self.n_failures = 0
        return rec

    def _fail(self, rec: dict, reason: str, t: float) -> dict:
        self.n_failures += 1
        self._requery = True
        rec.update(decision="no_goal", reason=reason)
        if self.n_failures >= self.cfg.goal.max_unreachable:
            self._finish("nav_failed", t, reason=reason)
            rec["decision"] = "nav_failed"
        return rec

    def _actions(self, rec: dict, reply: S2Reply, robot: Pose, t: float) -> dict:
        first = reply.actions[0]
        n = 1
        while n < len(reply.actions) and reply.actions[n] == first:
            n += 1
        a = self.cfg.actions
        yaw = robot.yaw()
        if first == "stop":
            rec["decision"] = "stop"
            self._finish("reached", t, reason="vln_stop")
            return rec
        if first == "look_down":                                  # ignore mode / too many `↓`: ask again
            rec.update(decision="look_down_ignored")
            self._requery = True
            return rec
        if first == "forward":
            xy = np.asarray(robot.position[:2]) + n * a.forward_step_m * np.array([math.cos(yaw), math.sin(yaw)])
            res = self._reachable(robot, xy, t)
            rec["goal_result"] = res.model_dump(exclude={"path", "goal"})
            if res.goal is None:
                return self._fail(rec, res.reason, t)
            self.goal = self._new_goal("forward", res.goal, res.path, t)
            self.n_failures = 0
            rec.update(decision="forward", goal=_xyyaw(res.goal))
            return rec
        target = float(wrap_angle(yaw + (1 if first == "left" else -1) * n * a.turn_step_rad))
        if a.rotate_mode == "in_place":
            pose = goal_pose(robot.position[:2], robot.position[2], target, t)
            self.goal = self._new_goal("rotate", pose, [tuple(robot.position[:2])], t, target_yaw=target)
        else:
            xy = np.asarray(robot.position[:2]) + a.rotate_offset_m * np.array([math.cos(target), math.sin(target)])
            res = self._reachable(robot, xy, t, yaw=target)
            if res.goal is None:
                return self._fail(rec, res.reason, t)
            self.goal = self._new_goal("rotate", res.goal, res.path, t, target_yaw=target)
        self.n_failures = 0
        rec.update(decision=f"rotate_{first}", goal=_xyyaw(self.goal.pose), turn_rad=round(n * a.turn_step_rad, 3))
        return rec

    # ------------------------------------------------------------------------------ step
    def _goal_done(self, robot: Pose, t: float) -> bool:
        g, lp = self.goal, self.cfg.loop
        if g is None:
            return True
        if g.kind == "rotate":
            return abs(float(wrap_angle(robot.yaw() - g.target_yaw))) < lp.rotate_tol_rad
        return math.dist(robot.position[:2], g.pose.position[:2]) < lp.requery_radius_m

    def _stalled(self, robot: Pose, t: float) -> bool:
        xy = np.asarray(robot.position[:2])
        if self._progress is None or np.linalg.norm(xy - self._progress[0]) > self.cfg.loop.stall_dist_m:
            self._progress = (xy, t)
            return False
        if t - self._progress[1] > self.cfg.loop.stall_s:
            self._progress = (xy, t)
            return True
        return False

    def step(self, robot: Pose, t: float) -> ExplorationOutput:
        if self.subtask is None or self.subtask.type != "move":
            return ExplorationOutput(goal=None, goal_kind="hold", goal_id=None, path=None, events=[])
        self._t_start = t if self._t_start is None else self._t_start
        if not self.done:
            c = self._collect()
            if c is not None and c.gen == self._gen:
                self.last = self._handle(c, robot, t)
                self._records.append(self.last)
        lp = self.cfg.loop
        if not self.done and not self.in_flight:
            if t - self._t_start > lp.timeout_s:
                self._finish("timeout", t, reason="timeout_s")
            elif self.n_cycles >= lp.max_queries:
                self._finish("timeout", t, reason="max_queries")
        if not self.done and not self.in_flight and self.latest is not None:
            period = lp.requery_period_s
            stalled = self._stalled(robot, t)
            if (self._requery or self._goal_done(robot, t) or stalled
                    or (period is not None and self.goal is not None and t - self.goal.t >= period)):
                if self._goal_done(robot, t) and self.goal is not None and self.goal.kind == "rotate":
                    self.goal = None
                self._start_cycle(t)
                if self._pool is None:                             # blocking: handle in this step
                    c = self._collect()
                    self.last = self._handle(c, robot, t)
                    self._records.append(self.last)
        events, self._events = self._events, []
        g = self.goal
        if g is None:
            return ExplorationOutput(goal=None, goal_kind="hold", goal_id=None, path=None, events=events)
        path = g.path
        if g.kind != "rotate":
            path = self.planner.plan(self.grid, robot.position[:2], g.pose.position[:2]) or g.path
        # goal_id names the goal, not the cycle: a kept goal must not be resent to the navigation stack
        return ExplorationOutput(goal=g.pose, goal_kind="move", goal_id=g.id, path=path, events=events)

    # ------------------------------------------------------------------------------- logs
    def pop_records(self) -> list[dict]:
        out, self._records = self._records, []
        return out

    def snapshot(self) -> dict:
        g = self.goal
        st = self.subtask
        return {"subtask": None if st is None else {"id": st.id, "type": st.type, "target": st.target,
                                                    "instruction": st.instruction},
                "active": self.active, "done": self.done, "n_cycles": self.n_cycles, "in_flight": self.in_flight,
                "n_failures": self.n_failures, "n_history_kept": len(self.buffer.kept),
                "goal": None if g is None else {"kind": g.kind, "id": g.id, "pose": _xyyaw(g.pose), "t": g.t},
                "last": self.last}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
