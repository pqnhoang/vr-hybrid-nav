"""Controller interface (SYSTEM.md §4.4). Implementations are sim- or robot-specific: the Habitat
pure-pursuit follower comes with the Habitat harness (exploration milestone E5); on the robot the FAR
planner adapter lives in `vlnav_ros/` (S7)."""
from __future__ import annotations

from typing import Protocol

from vlnav.common.models import Pose


class Controller(Protocol):
    def step(self, path: list[tuple[float, float]], robot_pose: Pose) -> object: ...
