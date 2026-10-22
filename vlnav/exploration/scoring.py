"""NeSy score for frontier points (TASK_nesy_exploration.md §8, paper Eq. 8; AUDIT 2.3).

    additive_dist (default):  S = w_dist·S_dist + w_VL·(S_VL·S_unknown)
    multiplicative_unknown:   S = (w_dist·S_dist + w_VL·S_VL)·S_unknown

The selector subtracts a configurable goal-yaw-change penalty from these base scores.
With no relevant detections and no yaw penalty, the default form is nearest-frontier.
`room_bonus` (off by default, ablation): + w_room for frontiers in / next to a room labeled
`Subtask.room_hint`, via a callable injected by runtime/ (exploration never imports memory).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrontierScore:
    id: int
    xy: tuple[float, float]
    dist: float              # metric used for S_dist (Euclidean m, or path cost)
    s_dist: float
    s_vl: float              # 0 unless scored for the current subtask
    s_unknown: float
    room_bonus: float
    s_nesy: float
    yaw_penalty: float = 0.0


def s_nesy(s_dist: float, s_vl: float, s_unknown: float, w_dist: float, w_vl: float, form: str) -> float:
    if form == "additive_dist":
        return w_dist * s_dist + w_vl * (s_vl * s_unknown)
    return (w_dist * s_dist + w_vl * s_vl) * s_unknown
