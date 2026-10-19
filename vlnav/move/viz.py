"""Debug view of the move subtask (TASK_move_subtask.md §12 M3). ROS-free: images (OpenCV) + plain data.

Left: the model-view image of the last handled S2 cycle (what DualVLN saw) with the reply pixel
(magenta) and the S2 texts. Right: the occupancy grid with the robot path (blue), the projection ray
from the camera to the floor point (orange), the projected point (orange dot), the reachable goal
(magenta ring + heading) with its A* path, and the robot (blue disc + heading). Bottom: instruction,
cycle, decision / reason / flags, events.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from vlnav.exploration.viz import FONT, View, _background, _scale_bar, wrap_px
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.move.config import MoveVizCfg

C_PIX = C_GOAL = (220, 0, 170)
C_RAY = (255, 140, 0)
C_ROBOT = (30, 90, 255)
C_TRAJ = (120, 150, 230)
TEXT_H = 110
ASCII = {"↑": "^", "←": "<", "→": ">", "↓": "v"}


def ascii_text(text: str) -> str:
    """OpenCV fonts have no arrows."""
    return "".join(ASCII.get(ch, ch) for ch in text).encode("ascii", "replace").decode()


def _arrow(img, view, xy, yaw, length_m, color, thick=2):
    a = view.ipx(xy)
    b = view.ipx((xy[0] + length_m * math.cos(yaw), xy[1] + length_m * math.sin(yaw)))
    cv2.arrowedLine(img, a, b, color, thick, cv2.LINE_AA, tipLength=0.35)


def render_image(image: np.ndarray | None, last: dict | None, height: int) -> np.ndarray:
    if image is None:
        out = np.full((height, height * 4 // 3, 3), 235, np.uint8)
        cv2.putText(out, "no S2 cycle yet", (20, 40), FONT, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
        return out
    img = np.ascontiguousarray(image.copy())
    if last is not None and last.get("view_xy"):
        x, y = (int(round(v)) for v in last["view_xy"])
        cv2.circle(img, (x, y), 10, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img, (x, y), 10, C_PIX, 2, cv2.LINE_AA)
        cv2.drawMarker(img, (x, y), C_PIX, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    scale = height / img.shape[0]
    img = cv2.resize(img, (int(round(img.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    if last is not None:
        texts = " | ".join(("LD " if e["look_down"] else "") + ascii_text(e["text"]) for e in last["exchanges"])
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, f"cycle {last['k']}: {texts}"[:90], (8, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_map(grid: OccupancyGrid2D, snap: dict, robot_xyyaw, trajectory_xy, path, size: int,
               radius_m: float) -> np.ndarray:
    last, goal = snap.get("last") or {}, snap.get("goal")
    proj = last.get("projection") or {}
    view = View(xmin=robot_xyyaw[0] - radius_m, ymax=robot_xyyaw[1] + radius_m, m_per_px=2 * radius_m / size,
                size=size)
    img = _background(grid, np.empty((0, 2), int), view)
    if len(trajectory_xy) > 1:
        cv2.polylines(img, [np.round(view.px(np.asarray(trajectory_xy))).astype(np.int32)], False, C_TRAJ, 2,
                      cv2.LINE_AA)
    if path is not None and len(path) > 1:
        cv2.polylines(img, [np.round(view.px(np.asarray(path))).astype(np.int32)], False, C_GOAL, 2, cv2.LINE_AA)
    if proj.get("p_map") is not None:
        end = proj.get("floor_xy") or proj["p_map"][:2]
        cv2.line(img, view.ipx(proj["cam_xy"]), view.ipx(end), C_RAY, 1, cv2.LINE_AA)
        cv2.circle(img, view.ipx(proj["p_map"][:2]), 6, C_RAY, -1, cv2.LINE_AA)
    if goal is not None:
        gx, gy, gyaw = goal["pose"]
        cv2.circle(img, view.ipx((gx, gy)), 9, C_GOAL, 2, cv2.LINE_AA)
        _arrow(img, view, (gx, gy), gyaw, 0.6, C_GOAL)
    x, y, yaw = robot_xyyaw
    cv2.circle(img, view.ipx((x, y)), 7, C_ROBOT, -1, cv2.LINE_AA)
    _arrow(img, view, (x, y), yaw, 0.5, C_ROBOT)
    _scale_bar(img, view.m_per_px)
    return img


def render_text(snap: dict, t_rel: float, width: int, events: list[str]) -> np.ndarray:
    img = np.full((TEXT_H, width, 3), 250, np.uint8)
    st, last = snap.get("subtask") or {}, snap.get("last") or {}
    lines = wrap_px(f"t={t_rel:6.1f}s  MOVE: {st.get('instruction') or '-'}", width - 16, 0.5)[:2]
    flags = (last.get("projection") or {}).get("flags", []) + (last.get("goal_result") or {}).get("flags", [])
    lines.append(f"cycles {snap['n_cycles']}  in_flight {snap['in_flight']}  failures {snap['n_failures']}  "
                 f"history {snap['n_history_kept']}  done {snap['done']}")
    if last:
        lines.append(f"decision {last.get('decision')}  reason {last.get('reason')}  flags {flags}  "
                     f"goal {snap['goal']['kind'] if snap.get('goal') else None}")
    if events:
        lines.append("events: " + ", ".join(events[-3:]))
    for i, line in enumerate(lines[:5]):
        cv2.putText(img, ascii_text(line), (8, 20 + 21 * i), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def render(grid: OccupancyGrid2D, snap: dict, image: np.ndarray | None, robot_xyyaw, trajectory_xy, path,
           cfg: MoveVizCfg, t_rel: float, events: list[str]) -> np.ndarray:
    h = max(cfg.map_px, cfg.image_px)
    left = render_image(image, snap.get("last"), cfg.image_px)
    right = render_map(grid, snap, robot_xyyaw, trajectory_xy, path, cfg.map_px, cfg.map_radius_m)
    row = np.full((h, left.shape[1] + right.shape[1], 3), 255, np.uint8)
    row[:left.shape[0], :left.shape[1]] = left
    row[:right.shape[0], left.shape[1]:] = right
    return np.vstack([row, render_text(snap, t_rel, row.shape[1], events)])


class MoveRecorder:
    """shadow.mp4 (one frame per step) + frames/ (one PNG per handled S2 cycle)."""

    def __init__(self, out_dir: str | Path, cfg: MoveVizCfg):
        self.dir = Path(out_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.writer: cv2.VideoWriter | None = None
        self._last: str | None = None

    def add(self, img: np.ndarray, cycle: str | None) -> None:
        """`cycle`: name of the last handled S2 cycle; a PNG is kept when it changes."""
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.dir / "shadow_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.cfg.video_fps, (img.shape[1], img.shape[0]))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)
        if cycle is not None and cycle != self._last:
            cv2.imwrite(str(self.dir / "frames" / f"{cycle}.jpg"), bgr)
            self._last = cycle

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        raw, out = self.dir / "shadow_raw.mp4", self.dir / "shadow.mp4"
        if shutil.which("ffmpeg") and subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)]).returncode == 0:
            raw.unlink()
        else:
            raw.rename(out)
