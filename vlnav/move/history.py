"""Image history for DualVLN S2 (TASK_move_subtask.md §6.2).

InternNav keeps one image per Habitat step (0.25 m forward or 15° turn) and, at each S2 query, sends
`num_history` of the previous ones picked by `np.unique(np.linspace(0, n - 1, num_history))`
(`internvla_n1_agent_realworld.py:184`). On the robot frames come at camera rate, so a frame is kept
only after `step_m` of travel or `step_rad` of turning since the last kept one **[DESIGN]**.
Kept images are already resized to the model input (PIL resize, as InternNav does).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vlnav.common.geometry import wrap_angle


@dataclass
class Kept:
    stamp: float
    rgb: np.ndarray                     # resized to the model input
    xy: tuple[float, float]
    yaw: float


def history_ids(n: int, num_history: int) -> list[int]:
    """InternNav's history selection over n previous frames (empty when n == 0)."""
    if n <= 0 or num_history <= 0:
        return []
    return np.unique(np.linspace(0, n - 1, num_history, dtype=np.int32)).tolist()


def resize_rgb(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB").resize(size_wh))


class FrameBuffer:
    def __init__(self, step_m: float, step_rad: float, size_wh: tuple[int, int], num_history: int):
        self.step_m = step_m
        self.step_rad = step_rad
        self.size_wh = size_wh
        self.num_history = num_history
        self.kept: list[Kept] = []

    def reset(self) -> None:
        self.kept = []

    def add(self, stamp: float, rgb: np.ndarray, xy, yaw: float) -> bool:
        """Keep the frame if the robot moved / turned enough since the last kept one."""
        if self.kept:
            last = self.kept[-1]
            moved = math.hypot(xy[0] - last.xy[0], xy[1] - last.xy[1])
            turned = abs(float(wrap_angle(yaw - last.yaw)))
            if moved < self.step_m and turned < self.step_rad:
                return False
        self.kept.append(Kept(stamp, resize_rgb(rgb, self.size_wh), (float(xy[0]), float(xy[1])), float(yaw)))
        return True

    def history(self, before_stamp: float) -> list[Kept]:
        """History for a query on the frame at `before_stamp`: kept frames strictly before it."""
        prev = [k for k in self.kept if k.stamp < before_stamp]
        return [prev[i] for i in history_ids(len(prev), self.num_history)]
