"""DualVLN S2 text -> typed reply (TASK_move_subtask.md §2, §6.4), same rules as InternNav.

InternNav (`internvla_n1_policy.py`): any digit in the text -> pixel goal from the first two
integers, written by the model as (x, y) and stored by InternNav as [row, col]; otherwise the
arrows / STOP are read in order. Here the pixel is kept as (x, y) = (column, row) of the model input image.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

ACTIONS = {"STOP": "stop", "↑": "forward", "←": "left", "→": "right", "↓": "look_down"}
_ACTION_RE = re.compile("|".join(re.escape(a) for a in ACTIONS))


class S2Reply(BaseModel):
    kind: Literal["pixel", "actions", "invalid"]
    text: str
    pixel_xy: tuple[int, int] | None = None           # (x, y) in the model input image
    actions: list[Literal["stop", "forward", "left", "right", "look_down"]] = []

    @property
    def is_stop(self) -> bool:
        return self.kind == "actions" and bool(self.actions) and self.actions[0] == "stop"


def parse_s2_text(text: str) -> S2Reply:
    if re.search(r"\d", text):
        nums = [int(n) for n in re.findall(r"\d+", text)]
        if len(nums) < 2:
            return S2Reply(kind="invalid", text=text)
        return S2Reply(kind="pixel", text=text, pixel_xy=(nums[0], nums[1]))
    acts = [ACTIONS[m] for m in _ACTION_RE.findall(text)]
    if not acts:
        return S2Reply(kind="invalid", text=text)
    return S2Reply(kind="actions", text=text, actions=acts)
