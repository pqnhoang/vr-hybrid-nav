"""Display of the running task program (code-as-policy, TASK_cap_mas.md). ROS-free: one OpenCV image.

Header: task, FSM, planner (cap / NeSy fallback), deliberation rounds, the subtask running now.
Body: the program with line numbers; comments dimmed, the line whose action is running highlighted,
and under each executed line its results (green ok, red FAIL). The ROS node publishes it on
/planner/program (ProgramRuntime.program_view() -> render_program()).
"""
from __future__ import annotations

import cv2
import numpy as np

from vlnav.exploration.viz import FONT, wrap_px

SCALE, LINE_H, PAD = 0.5, 22, 12
C_BG, C_TEXT, C_DIM = (30, 30, 30), (235, 235, 235), (140, 140, 140)        # BGR
C_OK, C_FAIL, C_NOW = (90, 200, 90), (80, 80, 240), (110, 70, 0)              # C_NOW: running line background
C_FSM = {"EXECUTING": (255, 180, 60), "DONE": (90, 200, 90), "FAILED": (80, 80, 240)}


def _ascii(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")      # Hershey fonts draw ASCII only


def _width(text: str) -> int:
    return cv2.getTextSize(text, FONT, SCALE, 1)[0][0]


def _split(text: str, width_px: int) -> list[str]:
    """Wrap one code line by rendered width; the pieces after the first are indented by 4 spaces."""
    out, prefix = [], ""
    while True:
        n = len(text)
        while n > len(prefix) + 1 and _width(text[:n]) > width_px:
            n -= 1
        out.append(text[:n])
        if n >= len(text):
            return out
        indent = len(out[0]) - len(out[0].lstrip())
        prefix = " " * (indent + 4)
        text = prefix + text[n:].lstrip()


def render_program(view: dict, width: int = 960) -> np.ndarray:
    """`view`: ProgramRuntime.program_view(). Returns a BGR image `width` px wide."""
    text_w = width - 2 * PAD
    gutter = _width("000  ")
    rows: list[tuple[str, str, tuple, tuple | None]] = []          # (line number, text, color, background)
    for line in wrap_px(_ascii(f"Task #{view['task_id']}: {view['instruction']}"), text_w, SCALE):
        rows.append(("", line, C_TEXT, None))
    info = [view["fsm"], f"planner {view['planner']}"]
    if view.get("rounds"):
        info.append(f"{view['rounds']} round(s)")
    if view.get("subtask"):
        info.append(f"now: {view['subtask']}")
    if view.get("done_reason"):
        info.append(view["done_reason"])
    rows += [("", line, C_FSM.get(view["fsm"], C_TEXT), None)
             for line in wrap_px(_ascii("  |  ".join(info)), text_w, SCALE)]
    rows.append(("", "", C_TEXT, None))
    body = len(rows)                                                # program rows start after the gutter
    code = view.get("program")
    if not code:
        msg = ("the NeSy planner runs this task (no valid program)" if view["planner"] == "nesy" else
               "deliberating: Planner -> Coder -> dry run -> Observer ..."
               if view["fsm"] in ("DECOMPOSING", "REPLANNING") else "no program")
        rows.append(("", msg, C_DIM, None))
        body = len(rows)
    else:
        results: dict[int, list[tuple[bool, str]]] = {}
        for line, ok, res in view.get("steps", []):
            results.setdefault(line, []).append((ok, res))
        for n, src in enumerate(_ascii(code).rstrip("\n").splitlines(), 1):
            bg = C_NOW if n == view.get("current_line") else None
            color = C_DIM if src.lstrip().startswith("#") else C_TEXT
            for k, piece in enumerate(_split(src, text_w - gutter)):
                rows.append((str(n) if k == 0 else "", piece, color, bg))
            indent = " " * (len(src) - len(src.lstrip()) + 4)
            for ok, res in results.get(n, []):
                for piece in _split(_ascii(f"{indent}-> {'ok' if ok else 'FAIL'}: {res}"), text_w - gutter):
                    rows.append(("", piece, C_OK if ok else C_FAIL, None))
    img = np.full((2 * PAD + LINE_H * len(rows), width, 3), C_BG, np.uint8)
    for i, (num, text, color, bg) in enumerate(rows):
        y = PAD + i * LINE_H
        if bg is not None:
            cv2.rectangle(img, (0, y), (width - 1, y + LINE_H - 1), bg, -1)
        base = y + LINE_H - 6
        if num:
            cv2.putText(img, num, (PAD + gutter - _width(num + "  "), base), FONT, SCALE, C_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, text, (PAD + gutter if i >= body else PAD, base), FONT, SCALE, color, 1, cv2.LINE_AA)
    return img
