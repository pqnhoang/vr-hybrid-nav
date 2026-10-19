"""Draw detector boxes once for the ROS camera topic and optional JPEG snapshots.

During a `move` subtask the last DualVLN S2 cycle is drawn too: its pixel goal (in this camera's raw
pixels, `projection.uv`; it was asked on an earlier frame) and a line with the S2 texts and decision.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _box(image: np.ndarray, bbox, label: str, color=(0, 200, 255)) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, baseline = cv2.getTextSize(label, font, .7, 2)
    top = max(0, y1 - size[1] - baseline - 8)
    cv2.rectangle(image, (x1, top), (x1 + size[0] + 8, top + size[1] + baseline + 8), color, -1)
    cv2.putText(image, label, (x1 + 4, top + size[1] + 4), font, .7, (0, 0, 0), 2,
                cv2.LINE_AA)


def _fit(text: str, width: int, font: int, scale: float, thickness: int = 1) -> str:
    text = " ".join(str(text).split()).encode("ascii", "replace").decode("ascii")
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cv2.getTextSize(text[:mid] + "...", font, scale, thickness)[0][0] <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "..."


def _planner_lines(state, reasoning: str) -> list[str]:
    st = state.current
    mode = {"DECOMPOSING": "PLAN", "REPLANNING": "REPLAN"}.get(state.fsm, state.fsm)
    if state.fsm == "EXECUTING" and st is not None:
        mode = f"{st.type.upper()} {st.target}" + (f" #{st.node_id}" if st.node_id is not None else "")
    why = ("Waiting for planner decision" if state.fsm in ("DECOMPOSING", "REPLANNING") else
           state.done_reason if state.fsm in ("DONE", "FAILED") else
           f"route: {st.instruction}" if st is not None and st.type == "move" else reasoning)
    return [f"Task: {state.instruction or 'Awaiting instruction'}",
            f"Mode: {mode}", f"Reason: {why or 'Waiting for decision'}"]


MOVE_COLOR = (170, 0, 220)                       # BGR magenta, as in vlnav/move/viz.py
ARROWS = {"↑": "^", "←": "<", "→": ">", "↓": "v"}


def _move_line(move: dict) -> str:
    texts = " | ".join(("LD " if e["look_down"] else "") + "".join(ARROWS.get(c, c) for c in e["text"])
                       for e in move.get("exchanges", []))
    why = f" ({move['reason']})" if move.get("reason") else ""
    return f"S2 #{move.get('k')}: {texts or '-'} -> {move.get('decision')}{why}"


def _move_pixel(image: np.ndarray, move: dict) -> None:
    uv = (move.get("projection") or {}).get("uv")
    if uv is None:
        return
    h, w = image.shape[:2]
    x, y = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= x < w and 0 <= y < h):
        return
    r = max(8, w // 80)
    cv2.circle(image, (x, y), r + 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.circle(image, (x, y), r, MOVE_COLOR, 3, cv2.LINE_AA)
    cv2.drawMarker(image, (x, y), MOVE_COLOR, cv2.MARKER_CROSS, 2 * r, 3, cv2.LINE_AA)
    cv2.putText(image, "S2 goal", (x + r + 6, y - r), cv2.FONT_HERSHEY_SIMPLEX, max(.5, w / 2400),
                MOVE_COLOR, 2, cv2.LINE_AA)


def annotate_detections(result, state=None, reasoning: str = "", move: dict | None = None) -> np.ndarray:
    """Return one image for both the ROS camera topic and saved camera JPEGs.

    `move`: record of the last S2 cycle (`MoveCore.snapshot()["last"]`) while a move subtask runs."""
    image = cv2.cvtColor(result.frame.rgb, cv2.COLOR_RGB2BGR)
    relevant = set(result.relevant_idx)
    for i, det in enumerate(result.detections):
        color = (0, 200, 255) if i in relevant else (255, 220, 0)
        _box(image, det.bbox, f"{det.label} {det.score:.2f}", color)
    if move is not None:
        _move_pixel(image, move)
    h, w = image.shape[:2]
    if state is not None and w >= 240 and h >= 110:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(1.1, max(.5, w / 1700))
        thickness = 2 if scale >= .9 else 1
        (_, font_h), baseline = cv2.getTextSize("Ag", font, scale, thickness)
        line_h, pad = font_h + baseline + 5, 8
        lines = _planner_lines(state, reasoning) + ([_move_line(move)] if move is not None else [])
        panel_h = 2 * pad + len(lines) * line_h
        panel = image[:panel_h]
        dark = np.full_like(panel, (18, 18, 18))
        cv2.addWeighted(dark, .9, panel, .1, 0, panel)
        for i, line in enumerate(lines):
            color = (0, 200, 255) if i == 1 else (255, 120, 240) if i == 3 else (255, 255, 255)
            cv2.putText(image, _fit(line, w - 2 * pad, font, scale, thickness),
                        (pad, pad + font_h + i * line_h), font, scale, color, thickness, cv2.LINE_AA)
    return image


class CameraImageWriter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _save(self, image: np.ndarray, name: str, latest: str) -> Path:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError("OpenCV could not encode the camera JPEG")
        data = encoded.tobytes()
        path = self.directory / name
        for destination in (path, self.directory / latest):
            temporary = self.directory / f".{destination.name}.tmp"
            temporary.write_bytes(data)
            temporary.replace(destination)
        return path

    def write_frame(self, image: np.ndarray, stamp: float) -> Path:
        """Save the image published on /perception/annotated_image."""
        return self._save(image, f"camera_{stamp:.3f}.jpg", "latest_camera.jpg")

    def write_goal(self, node, task_id: int, subtask_id: int) -> Path:
        """Keep the full best-view frame used for a selected object goal."""
        image = cv2.imread(node.best_view_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(node.best_view_image_path)
        _box(image, node.best_view_bbox, f"GOAL {node.label} #{node.id}")
        return self._save(image, f"goal_{task_id}_{subtask_id}_node_{node.id}.jpg",
                          "latest_goal.jpg")
