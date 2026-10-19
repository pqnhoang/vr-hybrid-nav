"""Live orchestration. All methods are called by one worker; VLM jobs remain asynchronous."""
from __future__ import annotations

from vlnav.common.models import Frame, PlannerEvent, Pose, Subtask
from vlnav.exploration.explorer import ExplorationOutput
from vlnav.runtime.system_runtime import points_of


def hold() -> ExplorationOutput:
    return ExplorationOutput(goal=None, goal_kind="hold", goal_id=None, path=None, events=[])


class LiveRuntime:
    def __init__(self, components, *, goto_object_approach: bool = False):
        self.c = components
        self.move = getattr(components, "move", None)     # MoveCore when move.enabled (DualVLN routes)
        self.goto_object_approach = goto_object_approach
        self._active_key = None
        self.last_frame_stamp = None
        self.last_reasoning = ""

    def submit(self, instruction: str, t: float) -> None:
        self.last_reasoning = ""
        if instruction.strip():
            self.c.planner.submit_task(instruction.strip(), t)
        else:
            self.c.planner.on_event(PlannerEvent(type="abort", stamp=t))
        self._sync()

    def _sync(self) -> bool:
        """Clear the old goal during decomposition/replanning, including task preemption."""
        self.c.planner.pop_dispatched()
        st = self.c.planner.state().current
        key = (st.task_id, st.id) if st is not None else None
        changed = key != self._active_key
        if changed:
            self.c.perception.set_active_prompts(st.detector_prompts if st else [])
            object_xy = None
            if self.goto_object_approach and st is not None and st.type == "goto" and st.node_id is not None:
                try:
                    object_xy = self.c.memory.get_object(st.node_id).centroid[:2]
                except KeyError:
                    pass
            self.c.exploration.set_subtask(st or Subtask(id=-1, task_id=-1, type="stop", target=""),
                                           object_xy=object_xy,
                                           require_object_approach=self.goto_object_approach)
            if self.move is not None:
                self.move.set_subtask(st)
            self._active_key = key
        return changed

    def ingest(self, frame: Frame):
        res = self.c.perception.process(frame)
        info = self.c.grid.update(points_of(frame, self.c.mapping.depth_points),
                                  frame.sensor_origin_map, frame.stamp)
        self.c.memory.update(res, info)
        self.c.exploration.on_frame(res, info)
        if self.move is not None:
            self.move.on_frame(frame)
        self.last_frame_stamp = frame.stamp
        return res

    def step(self, robot: Pose | None, t: float) -> ExplorationOutput:
        # Consume actual arrival/failure events before the timeout/candidate poll.
        # A newly dispatched subtask gets the next step to select its own goal.
        out = hold()
        if robot is not None and self._active_key is not None:
            st = self.c.planner.state().current
            mover = self.move if self.move is not None and st is not None and st.type == "move" else None
            out = mover.step(robot, t) if mover is not None else self.c.exploration.step(robot, t)
            for event in out.events:
                self.c.planner.on_event(event)
        for event in self.c.planner.poll_internal(t):
            self.c.planner.on_event(event)
        if self._sync() or self._active_key is None:
            out = hold()
        # Keep the last short planner explanation for the live camera overlay.
        for decision in self.c.planner.pop_decisions():
            if decision["kind"] in ("decompose", "replan"):
                self.last_reasoning = decision.get("reasoning") or decision.get("reason") or ""
        return out

    def close(self) -> None:
        if self.move is not None:
            self.move.close()
        self.c.planner.close()
        self.c.memory.close()
