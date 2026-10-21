"""Blackboard types (TASK_cap_mas.md §3): agents read a frozen WorldSnapshot and return a Proposal;
only the ProgramRuntime commits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vlnav.acquisition.memory_api import MemoryView
from vlnav.mapping.occupancy import OccupancyGrid2D


@dataclass
class TraceEntry:
    """One primitive of a program run (real or dry-run)."""
    call: str                          # e.g. 'locate("sofa")'
    line: int | None                   # program line, maps back to the `# Step k` comment
    result: str                        # short text, e.g. "found couch #12 (verified)", "nav_failed"
    ok: bool = True
    t: float | None = None
    detail: dict = field(default_factory=dict)

    def text(self) -> str:
        return f"{'L' + str(self.line) if self.line else '-'}: {self.call} -> {self.result}"


@dataclass(frozen=True)
class WorldSnapshot:
    """What the agents reason on. `grid` is a copy taken at snapshot time (safe to read in the
    deliberation thread while mapping keeps updating the live grid)."""
    version: int
    stamp: float
    instruction: str
    robot: tuple[float, float, float] | None      # x, y, yaw in the map frame
    memory: MemoryView
    grid: OccupancyGrid2D
    move_enabled: bool
    trace: tuple[TraceEntry, ...] = ()            # steps already done in this task
    failure: str | None = None                    # why the last program stopped (repair)


@dataclass
class RoundLog:
    round: int
    plan: dict | None = None                      # Planner JSON
    code: str | None = None
    error: str | None = None                      # parse / validation / dry-run exception
    trace: list[TraceEntry] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    observation: dict | None = None               # Observer JSON
    valid: bool = False
    calls: list[dict] = field(default_factory=list)   # agent calls: role, raw replies, latency

    def to_json(self) -> dict:
        return {"round": self.round, "plan": self.plan, "code": self.code, "error": self.error,
                "trace": [t.text() for t in self.trace], "hard_failures": self.hard_failures,
                "observation": self.observation, "valid": self.valid,
                "calls": [{k: v for k, v in c.items() if k != "prompt"} for c in self.calls]}


@dataclass(frozen=True)
class Proposal:
    program: str
    base_version: int
    rounds: tuple[RoundLog, ...]
    plan: dict | None
    actions: tuple[dict, ...]                     # dry-run actions in order: {type, target}
