"""Planner state (planner spec §5.1)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vlnav.common.models import Subtask

TargetStatus = Literal["pending", "exploring", "verified", "reported", "failed"]
FsmState = Literal["IDLE", "DECOMPOSING", "EXECUTING", "REPLANNING", "DONE", "FAILED"]
FINISHED: tuple[str, ...] = ("reported", "failed")


class TargetEntry(BaseModel):
    name: str
    detector_prompts: list[str]
    room_priors: list[str] = []
    status: TargetStatus = "pending"
    node_id: int | None = None
    attempts: int = 0
    notes: str = ""

    @property
    def active(self) -> bool:
        return self.status not in FINISHED


class PlannerState(BaseModel):
    task_id: int
    instruction: str
    fsm: FsmState
    targets: list[TargetEntry]
    current: Subtask | None
    history: list[dict]                  # finished subtasks with outcome + event
    started_t: float
    done_reason: str | None = None

    def target(self, name: str) -> TargetEntry | None:
        key = name.strip().lower()
        return next((t for t in self.targets if t.name.strip().lower() == key), None)
