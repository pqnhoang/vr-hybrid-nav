"""Single-process full system for run mode A (SYSTEM.md §6): bag replay with planner.

    for frame in source:
        res = perception.process(frame)
        info = grid.update(points_of(frame), frame.sensor_origin_map, frame.stamp)
        memory.update(res, info)
        exploration.on_frame(res, info)
        out = exploration.step(robot_pose, t)            # at select_rate_hz
        events = bag_sim.step(robot_pose, t) + planner.poll_internal(t)
        for e in events: planner.on_event(e)
        for st in planner.pop_dispatched():
            perception.set_active_prompts(st.detector_prompts); bag_sim.set_subtask(st, t)
            exploration.set_subtask(st)

The occupancy grid (mapping/) is updated from the LiDAR window (or dense depth) before memory,
which uses it for rooms, and exploration. Mode A: exploration runs in *shadow* — it computes
frontiers, scores and goals from the same grid and subtasks, its outputs and events are logged
(`RunLog.exploration`, `exploration_events`), but the robot follows the recorded path and
`BagEventSim` produces the planner's `reached` / `nav_failed` / `no_frontier`; explore subtasks end
through the planner's candidate check (or `no_frontier` at bag end).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from vlnav.common.models import Frame, FrameResult, PlannerEvent, Pose, Subtask, TargetReport
from vlnav.exploration.explorer import ExplorationCore, ExplorationOutput
from vlnav.mapping.config import DepthPointsCfg
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.memory.scene_graph import UnifiedMemory
from vlnav.perception.depth_utils import depth_to_points
from vlnav.perception.pipeline import PerceptionPipeline
from vlnav.planner.core import PlannerCore
from vlnav.sim.bag_event_sim import BagEventSim


@dataclass
class TaskSpec:
    instruction: str
    start_offset_s: float = 0.0          # relative to the first frame


@dataclass
class RunLog:
    dispatched: list[tuple[float, Subtask]] = field(default_factory=list)
    events: list[PlannerEvent] = field(default_factory=list)
    reports: list[TargetReport] = field(default_factory=list)
    frames: int = 0
    perception_ms: list[float] = field(default_factory=list)
    mapping_ms: list[float] = field(default_factory=list)
    memory_ms: list[float] = field(default_factory=list)
    exploration_ms: list[float] = field(default_factory=list)
    trajectory_xy: list[np.ndarray] = field(default_factory=list)     # robot base, map frame
    exploration: list[tuple[float, ExplorationOutput]] = field(default_factory=list)   # shadow goals
    exploration_events: list[PlannerEvent] = field(default_factory=list)             # not sent (mode A)


def points_of(frame: Frame, depth_cfg: DepthPointsCfg) -> np.ndarray:
    """Points for the occupancy grid: the LiDAR window, else dense depth (SYSTEM §6)."""
    if frame.lidar_points_map is not None and len(frame.lidar_points_map):
        return frame.lidar_points_map
    if frame.depth is not None:
        return depth_to_points(frame.depth, frame.K, frame.T_map_cam, depth_cfg.stride,
                               depth_cfg.min_depth_m, depth_cfg.max_depth_m)
    return np.empty((0, 3))


def task_context(state) -> dict | None:
    """Planner state as plain data for displays that must not import the planner (exploration view)."""
    if state is None or state.task_id < 0:
        return None
    return {"instruction": state.instruction, "state": state.fsm, "done_reason": state.done_reason,
            "targets": [{"name": t.name, "status": t.status, "node_id": t.node_id} for t in state.targets],
            "current": None if state.current is None else {"type": state.current.type, "target": state.current.target}}


FrameHook = Callable[[int, Frame, FrameResult, list[PlannerEvent], list[Subtask], list[TargetReport]], None]


class SystemRuntime:
    def __init__(self, source: Iterable[Frame], perception: PerceptionPipeline,
                 memory: UnifiedMemory, planner: PlannerCore, bag_sim: BagEventSim,
                 grid: OccupancyGrid2D, depth_cfg: DepthPointsCfg,
                 exploration: ExplorationCore | None = None,
                 on_frame: FrameHook | None = None,
                 before_frame: Callable[[Frame], None] | None = None):
        self.source = source
        self.perception = perception
        self.memory = memory
        self.planner = planner
        self.bag_sim = bag_sim
        self.grid = grid                         # shared with memory (rooms) and exploration
        self.depth_cfg = depth_cfg
        self.exploration = exploration           # shadow mode (mode A)
        self.last_exploration: ExplorationOutput | None = None
        self._last_select_t = -float("inf")
        self.on_frame = on_frame
        self.before_frame = before_frame          # e.g. real-time pacing for live demos
        self.log = RunLog()

    def _dispatch(self, t: float) -> list[Subtask]:
        sts = self.planner.pop_dispatched()
        for st in sts:
            self.perception.set_active_prompts(st.detector_prompts if st.type != "stop" else [])
            self.bag_sim.set_subtask(st, t)
            if self.exploration is not None:
                self.exploration.set_subtask(st)
            self.log.dispatched.append((t, st))
        return sts

    def _explore(self, res: FrameResult, info, robot: Pose, t: float) -> None:
        self.exploration.on_frame(res, info)
        if t - self._last_select_t >= 1.0 / self.exploration.cfg.select_rate_hz - 1e-6:
            self._last_select_t = t
            out = self.last_exploration = self.exploration.step(robot, t)
            self.log.exploration.append((t, out))
            self.log.exploration_events += out.events

    def run(self, tasks: list[TaskSpec]) -> RunLog:
        pending = sorted(tasks, key=lambda x: x.start_offset_s)
        t0 = t = None
        for k, frame in enumerate(self.source):
            if self.before_frame is not None:
                self.before_frame(frame)
            t = frame.stamp
            t0 = t if t0 is None else t0
            while pending and t - t0 >= pending[0].start_offset_s:
                self.planner.submit_task(pending.pop(0).instruction, t)
            new = self._dispatch(t)
            a = time.perf_counter()
            res = self.perception.process(frame)
            b = time.perf_counter()
            info = self.grid.update(points_of(frame, self.depth_cfg), frame.sensor_origin_map, t)
            c = time.perf_counter()
            self.memory.update(res, info)
            d = time.perf_counter()
            robot = Pose.from_matrix(frame.T_map_base, t)
            self.log.trajectory_xy.append(frame.T_map_base[:2, 3].copy())
            if self.exploration is not None:
                self._explore(res, info, robot, t)
            self.log.perception_ms.append(1e3 * (b - a))
            self.log.mapping_ms.append(1e3 * (c - b))
            self.log.memory_ms.append(1e3 * (d - c))
            self.log.exploration_ms.append(1e3 * (time.perf_counter() - d))
            events = self.bag_sim.step(robot, t) + self.planner.poll_internal(t)
            for e in events:
                self.planner.on_event(e)
            self.log.events += events
            new += self._dispatch(t)
            reports = self.planner.pop_reports()
            self.log.reports += reports
            self.log.frames += 1
            if self.on_frame is not None:
                self.on_frame(k, frame, res, events, new, reports)
        if t is not None:                                # bag end: close the open subtask
            self.planner.wait_idle()
            self._dispatch(t)
            for e in self.bag_sim.finish(t):
                self.planner.on_event(e)
                self.log.events.append(e)
            self.planner.wait_idle()
            self._dispatch(t)
            self.log.reports += self.planner.pop_reports()
        return self.log
