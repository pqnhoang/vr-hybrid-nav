"""Bag event simulator (planner spec §6): stands in for exploration events in run mode A, where
the robot path is fixed by the bag.

- `oracle`: `goto` -> `reached` after `oracle_delay_s`; `explore` -> no events.
- `trajectory`: `goto` -> `reached` the first time the bag trajectory comes within
  `reach_radius_m` of the goal after dispatch.
`finish(t)` at bag end: an open `goto` -> `nav_failed`, an open `explore` -> `no_frontier`, an open
`move` -> `timeout` (move events otherwise come from MoveCore).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml
from vlnav.common.models import PlannerEvent, Pose, Subtask


class BagEventSimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["trajectory", "oracle"]
    reach_radius_m: float
    oracle_delay_s: float


def load_sim_config(path: str | Path = CONFIG_DIR / "sim.yaml",
                    overrides: dict | None = None) -> BagEventSimConfig:
    return BagEventSimConfig.model_validate(deep_merge(load_yaml(path), overrides)["bag_event_sim"])


class BagEventSim:
    def __init__(self, cfg: BagEventSimConfig):
        self.cfg = cfg
        self.current: Subtask | None = None
        self.dispatch_t = 0.0
        self.done = True

    def set_subtask(self, st: Subtask, t: float) -> None:
        self.current, self.dispatch_t, self.done = st, t, st.type == "stop"

    def _event(self, kind: str, t: float, **payload) -> PlannerEvent:
        return PlannerEvent(type=kind, stamp=t, task_id=self.current.task_id,
                            subtask_id=self.current.id, payload=payload)

    def step(self, robot_pose: Pose, t: float) -> list[PlannerEvent]:
        st = self.current
        if st is None or self.done or st.type != "goto" or st.goal_pose is None:
            return []
        if self.cfg.mode == "oracle":
            hit = t - self.dispatch_t >= self.cfg.oracle_delay_s
        else:
            d = float(np.hypot(robot_pose.position[0] - st.goal_pose.position[0],
                               robot_pose.position[1] - st.goal_pose.position[1]))
            hit = d <= self.cfg.reach_radius_m
        if not hit:
            return []
        self.done = True
        return [self._event("reached", t, source="bag_event_sim", mode=self.cfg.mode)]

    def finish(self, t: float) -> list[PlannerEvent]:
        if self.current is None or self.done:
            return []
        self.done = True
        kind = {"goto": "nav_failed", "move": "timeout"}.get(self.current.type, "no_frontier")
        return [self._event(kind, t, source="bag_event_sim", reason="bag ended")]
