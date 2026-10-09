"""Top-down debug view of exploration on the 2D occupancy grid (TASK_nesy_exploration.md §2, §12 E4
bag shadow mode). ROS-free: images (OpenCV) + plain data; Rerun / RViz render the same snapshot.

Map: occupancy (white free, dark occupied, gray unknown), frontier cells (light blue), FoV wedge of
the last frame (yellow), robot path (blue), frontier points colored by S_NeSy (turbo: blue low,
red high; green ring = S_VL >= 0.01; gray = not selectable: within δ_reached / blacklisted (×)), top
candidates numbered, instance points (diamonds: orange active, green visited, red unreachable;
small square = approach point), goal (magenta ring) with its A* path, robot (blue disc + heading).
Panel: the task (instruction, planner state, plan = targets with status, [>] = current target;
passed in by the runner, exploration never reads the planner), subtask, goal, counts, VL mixture of
the frame, candidate table with the score breakdown (S_NeSy = w_dist·S_dist + w_VL·S_VL·S_unknown),
instances, events / notes, timing.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vlnav.exploration.config import ExplorationVizCfg
from vlnav.mapping.occupancy import OccupancyGrid2D

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_FREE, C_OCC, C_UNK = (255, 255, 255), (55, 55, 55), (200, 200, 200)
C_FRONTIER = (120, 200, 255)
C_WEDGE = (255, 200, 0)
C_PATH = C_GOAL = (220, 0, 170)
C_ROBOT = (30, 90, 255)
C_TRAJ = (120, 150, 230)
C_VL = (0, 200, 0)
C_INST = {"active": (255, 140, 0), "visited": (0, 170, 0), "unreachable": (210, 0, 0)}
VL_EPS = 0.01          # "has VL evidence": the Gaussian tails make every point in the wedge > 0
TASK_LINES, TASK_TARGETS = 4, 6        # instruction lines / plan rows shown in the panel


@dataclass(frozen=True)
class View:
    """World xy -> pixel (u right, v down, north up) of the map area."""
    xmin: float
    ymax: float
    m_per_px: float
    size: int

    def px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.stack([(xy[..., 0] - self.xmin) / self.m_per_px, (self.ymax - xy[..., 1]) / self.m_per_px], -1)

    def ipx(self, xy) -> tuple[int, int]:
        u, v = self.px(xy)
        return int(round(u)), int(round(v))


def make_view(grid: OccupancyGrid2D, snap: dict, cfg: ExplorationVizCfg, extra_xy=()) -> View:
    """Square view around the known part of the grid (+ robot, goal, path), letterboxed."""
    arr, res = grid.as_array(), grid.resolution
    ox, oy = grid.origin_xy
    pts = [np.asarray(p, float).reshape(-1, 2) for p in extra_xy if p is not None and len(p)]
    known = arr != -1
    if known.any():
        iy, ix = np.nonzero(known.any(axis=1))[0], np.nonzero(known.any(axis=0))[0]
        pts.append(np.array([[ox + ix[0] * res, oy + iy[0] * res], [ox + (ix[-1] + 1) * res, oy + (iy[-1] + 1) * res]]))
    elif arr.size:
        pts.append(np.array([[ox, oy], [ox + arr.shape[1] * res, oy + arr.shape[0] * res]]))
    allp = np.concatenate(pts) if pts else np.zeros((1, 2))
    lo, hi = allp.min(0) - cfg.crop_margin_m, allp.max(0) + cfg.crop_margin_m
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    c = (lo + hi) / 2
    return View(xmin=c[0] - span / 2, ymax=c[1] + span / 2, m_per_px=span / cfg.map_px, size=cfg.map_px)


def _background(grid: OccupancyGrid2D, cells: np.ndarray, view: View) -> np.ndarray:
    arr = grid.as_array()
    lut = np.array([C_UNK, C_FREE, C_OCC], np.uint8)          # index = value + 1
    colored = lut[arr.astype(np.int16) + 1] if arr.size else np.zeros((0, 0, 3), np.uint8)
    if len(cells):
        colored[cells[:, 1], cells[:, 0]] = C_FRONTIER
    n, res = view.size, grid.resolution
    xs = view.xmin + (np.arange(n) + 0.5) * view.m_per_px
    ys = view.ymax - (np.arange(n) + 0.5) * view.m_per_px
    ix = np.floor((xs - grid.origin_xy[0]) / res).astype(int)
    iy = np.floor((ys - grid.origin_xy[1]) / res).astype(int)
    okx, oky = (ix >= 0) & (ix < arr.shape[1]), (iy >= 0) & (iy < arr.shape[0])
    img = np.empty((n, n, 3), np.uint8)
    img[:] = C_UNK
    if arr.size:
        img[np.ix_(oky, okx)] = colored[np.ix_(iy[oky], ix[okx])]
    return img


def _turbo(v: float) -> tuple[int, int, int]:
    bgr = cv2.applyColorMap(np.array([[int(255 * min(max(v, 0.0), 1.0))]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _diamond(img, c, r, color, filled=True):
    pts = np.array([[c[0], c[1] - r], [c[0] + r, c[1]], [c[0], c[1] + r], [c[0] - r, c[1]]], np.int32)
    if filled:
        cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def _scale_bar(img: np.ndarray, m_per_px: float) -> None:
    length = next(v for v in (1, 2, 5, 10, 20, 50, 100) if v / m_per_px >= 60)
    n, h = int(round(length / m_per_px)), img.shape[0]
    x0, y0 = 12, h - 14
    cv2.rectangle(img, (x0 - 4, y0 - 20), (x0 + n + 4, y0 + 6), (255, 255, 255), -1)
    cv2.line(img, (x0, y0), (x0 + n, y0), (0, 0, 0), 2)
    cv2.putText(img, f"{length} m", (x0 + n // 2 - 12, y0 - 6), FONT, 0.42, (0, 0, 0), 1, cv2.LINE_AA)


def frontier_order(snap: dict) -> list[dict]:
    """Selectable frontier points, best S_NeSy first."""
    return sorted((p for p in snap["points"] if p["score"] is not None),
                  key=lambda p: (-p["score"]["s_nesy"], p["id"]))


def render_map(grid: OccupancyGrid2D, snap: dict, cfg: ExplorationVizCfg, trajectory_xy=None) -> tuple[np.ndarray, View]:
    goal = snap.get("goal")
    extra = [trajectory_xy, snap.get("path"), [snap["robot"][:2]] if snap.get("robot") else None,
             [goal["xy"]] if goal else None]
    view = make_view(grid, snap, cfg, extra)
    img = _background(grid, snap["frontier_cells"], view)
    # wedge
    for w in snap.get("wedges", []):
        arc = [(w["x"] + w["radius"] * math.cos(w["psi"] + a), w["y"] + w["radius"] * math.sin(w["psi"] + a))
               for a in np.linspace(-w["hfov"] / 2, w["hfov"] / 2, 24)]
        poly = np.round(view.px([(w["x"], w["y"]), *arc])).astype(np.int32)
        over = img.copy()
        cv2.fillPoly(over, [poly], C_WEDGE)
        img = cv2.addWeighted(over, 0.18, img, 0.82, 0)
        cv2.polylines(img, [poly], True, C_WEDGE, 1, cv2.LINE_AA)
    if trajectory_xy is not None and len(trajectory_xy) > 1:
        cv2.polylines(img, [np.round(view.px(trajectory_xy)).astype(np.int32)], False, C_TRAJ, 1, cv2.LINE_AA)
    if snap.get("path"):
        cv2.polylines(img, [np.round(view.px(snap["path"])).astype(np.int32)], False, C_PATH, 2, cv2.LINE_AA)
    # frontier points
    order = frontier_order(snap)
    smax = max([p["score"]["s_nesy"] for p in order] + [1e-6])
    rank = {p["id"]: i + 1 for i, p in enumerate(order[:cfg.label_top_k])}
    for p in snap["points"]:
        c = view.ipx(p["xy"])
        if p["score"] is None:
            cv2.circle(img, c, 3, (150, 150, 150), 1, cv2.LINE_AA)
            if p["blacklisted"]:
                cv2.drawMarker(img, c, (90, 90, 90), cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA)
            continue
        cv2.circle(img, c, 5, _turbo(p["score"]["s_nesy"] / smax), -1, cv2.LINE_AA)
        cv2.circle(img, c, 5, (0, 0, 0), 1, cv2.LINE_AA)
        if p["s_vl"] >= VL_EPS:
            cv2.circle(img, c, 8, C_VL, 2, cv2.LINE_AA)
        if p["id"] in rank:
            cv2.putText(img, str(rank[p["id"]]), (c[0] + 7, c[1] - 6), FONT, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, str(rank[p["id"]]), (c[0] + 7, c[1] - 6), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # instances
    for q in snap["instances"]:
        c = view.ipx(q["xy"])
        col = C_INST[q["status"]]
        if q["approach_xy"] is not None:
            a = view.ipx(q["approach_xy"])
            cv2.line(img, a, c, col, 1, cv2.LINE_AA)
            cv2.rectangle(img, (a[0] - 3, a[1] - 3), (a[0] + 3, a[1] + 3), col, -1)
        _diamond(img, c, 7, col)
        cv2.putText(img, f"I{q['id']}", (c[0] + 9, c[1] + 4), FONT, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    # goal + robot
    if goal is not None:
        c = view.ipx(goal["xy"])
        cv2.circle(img, c, 12, C_GOAL, 2, cv2.LINE_AA)
        cv2.drawMarker(img, c, C_GOAL, cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
    st = snap.get("subtask")
    if st and st["type"] == "goto" and st.get("goal_xy"):
        cv2.drawMarker(img, view.ipx(st["goal_xy"]), C_GOAL, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
    if snap.get("robot"):
        x, y, yaw = snap["robot"]
        c = view.ipx((x, y))
        tip = view.ipx((x + 0.9 * math.cos(yaw), y + 0.9 * math.sin(yaw)))
        cv2.circle(img, c, 6, C_ROBOT, -1, cv2.LINE_AA)
        cv2.arrowedLine(img, c, tip, C_ROBOT, 2, cv2.LINE_AA, tipLength=0.35)
    _scale_bar(img, view.m_per_px)
    return img, view


def _goal_text(snap: dict) -> str:
    g, kind = snap.get("goal"), snap.get("goal_kind")
    if g is None:
        return f"goal: {kind or '-'}"
    if kind == "goto":
        return f"goal: goto node #{g['id']}"
    score = f"  score {g['score']:.2f}" if g.get("score") is not None else ""
    plen = sum(math.dist(a, b) for a, b in zip(snap["path"], snap["path"][1:])) if snap.get("path") else 0.0
    return f"goal: {kind} #{g['id']}{score}  path {plen:.1f} m"


TARGET_MARK = {"reported": ("[x]", (0, 150, 0)), "verified": ("[v]", (0, 150, 0)),
               "exploring": ("[ ]", (20, 20, 20)), "pending": ("[ ]", (130, 130, 130)),
               "failed": ("[!]", (200, 0, 0))}


def wrap_px(text: str, width_px: int, scale: float = 0.42) -> list[str]:
    """Greedy word wrap by rendered width (OpenCV Hershey font)."""
    out, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(cand, FONT, scale, 1)[0][0] > width_px:
            out.append(cur)
            cur = word
        else:
            cur = cand
    return out + ([cur] if cur else [])


def _task_block(line, task: dict, width_px: int) -> None:
    """Instruction + plan (targets with status) + current subtask, from the runtime's task dict."""
    state = task.get("state") or ""
    line("TASK" + (f"   planner: {state}" if state else ""), scale=0.5, bold=True, gap=19)
    if task.get("instruction"):
        for ln in wrap_px(f'"{task["instruction"]}"', width_px)[:TASK_LINES]:
            line(ln, color=(40, 40, 120))
    if task.get("done_reason"):
        line(f"done: {task['done_reason']}", color=(90, 90, 90))
    cur = task.get("current") or {}
    for tg in task.get("targets", [])[:TASK_TARGETS]:
        mark, color = TARGET_MARK.get(tg["status"], ("[ ]", (20, 20, 20)))
        now = cur.get("target") == tg["name"] and tg["status"] not in ("reported", "failed")
        if now:
            mark, color = "[>]", C_GOAL
        found = f" -> node #{tg['node_id']}" if tg.get("node_id") is not None else ""
        line(f" {mark} {tg['name']}: {tg['status']}{found}", color=color, bold=now)
    line("-" * 60, color=(180, 180, 180), gap=14)


def render_panel(snap: dict, cfg: ExplorationVizCfg, height: int, t0: float | None, k: int | None,
                 task: dict | None = None) -> np.ndarray:
    img = np.full((height, cfg.panel_px, 3), 250, np.uint8)
    y = [18]
    footer = height - 94                       # timing + legend are pinned at the bottom

    def line(text: str, color=(20, 20, 20), scale=0.42, bold=False, gap=17, pinned=False):
        if not pinned and y[0] > footer - 6:   # content would run into the footer: drop it
            return
        cv2.putText(img, text[:62], (8, y[0]), FONT, scale, color, 2 if bold else 1, cv2.LINE_AA)
        y[0] += gap

    t = snap.get("t") if snap.get("t") is not None else snap.get("frame_t")
    rel = (t - t0) if t is not None and t0 is not None else (t or 0.0)
    line(f"t = {rel:6.1f} s" + (f"   frame {k}" if k is not None else ""), scale=0.55, bold=True, gap=22)
    if task:
        _task_block(line, task, cfg.panel_px - 16)
    st = snap.get("subtask")
    if st is None:
        line("subtask: none (no goal)")
    else:
        line(f"subtask #{st['id']} {st['type']} '{st['target']}'", bold=True)
        if st["prompts"]:
            line(f"  prompts: {', '.join(st['prompts'])}")
        if st.get("room_hint"):
            line(f"  room hint: {st['room_hint']}")
    line(_goal_text(snap), color=C_GOAL, bold=True)
    pts = snap["points"]
    n_sel = sum(p["score"] is not None for p in pts)
    n_vl = sum(p["s_vl"] >= VL_EPS for p in pts)
    n_bl = sum(p["blacklisted"] for p in pts)
    line(f"frontier: {len(snap['frontier_cells'])} cells, {len(pts)} points ({n_sel} selectable)")
    line(f"  S_VL>={VL_EPS:g}: {n_vl}   scored: {sum(p['scored'] for p in pts)}   blacklisted: {n_bl}")
    inst = snap["instances"]
    by = {s: sum(q["status"] == s for q in inst) for s in C_INST}
    line(f"instances: {by['active']} active, {by['visited']} visited, {by['unreachable']} unreachable")
    mix = snap.get("mixture")
    if mix is not None:
        dets = ", ".join(f"{lb} {a:.2f}@{math.degrees(m):+.0f}" for lb, a, m in
                         zip(mix["labels"], mix["alpha"], mix["mu"])) or "none"
        line(f"VL dets (label conf@bearing): {dets}")
    y[0] += 4
    line("frontier candidates (S = w_d*S_d + w_vl*S_vl*S_unk)", bold=True)
    line(" #   id   S_NeSy  S_dist  S_VL  S_unk   d[m]", color=(90, 90, 90))
    goal = snap.get("goal")
    for i, p in enumerate(frontier_order(snap)[:cfg.top_k_text]):
        s = p["score"]
        mark = "*" if goal and goal["kind"] == "frontier" and goal["id"] == p["id"] else " "
        line(f"{mark}{i + 1:<2} {p['id']:>4}  {s['s_nesy']:6.3f}  {s['s_dist']:5.2f}  {s['s_vl']:5.2f} "
             f"{s['s_unknown']:5.2f} {s['dist']:6.1f}", color=C_GOAL if mark == "*" else (20, 20, 20))
    if inst:
        y[0] += 4
        line("instances (rank = S_VL at own bearing)", bold=True)
        for q in sorted(inst, key=lambda q: (q["status"] != "active", -q["rank"]))[:6]:
            line(f" I{q['id']:<3} {q['label'][:14]:<14} rank {q['rank']:.2f} conf {q['score']:.2f} {q['status']}",
                 color=C_INST[q["status"]])
    msgs = [f"EVENT {e['type']}" for e in snap.get("events", [])] + snap.get("notes", [])
    if msgs:
        y[0] += 4
        for m in msgs[:5]:
            line(m, color=(200, 0, 0), bold=m.startswith("EVENT"))
    tm = snap.get("timing_ms", {})
    y[0] = footer
    line(f"timing: on_frame {tm.get('on_frame', 0):.0f} ms, step {tm.get('step', 0):.0f} ms, "
         f"plans {snap.get('reach_checks', 0)}", color=(90, 90, 90), pinned=True)
    line(f"legend: color = S_NeSy (blue low, red high), green ring =", color=(90, 90, 90),
         scale=0.38, gap=14, pinned=True)
    line(f"  S_VL>={VL_EPS:g}, gray = not selectable, x = blacklisted,", color=(90, 90, 90), scale=0.38, gap=14,
         pinned=True)
    line("  diamond = instance, magenta = goal + path, yellow = FoV,", color=(90, 90, 90), scale=0.38, gap=14,
         pinned=True)
    line("  blue = robot", color=(90, 90, 90), scale=0.38, gap=14, pinned=True)
    return img


def render(grid: OccupancyGrid2D, snap: dict, cfg: ExplorationVizCfg, trajectory_xy=None,
           t0: float | None = None, k: int | None = None, task: dict | None = None) -> tuple[np.ndarray, View]:
    """Map + panel, RGB. Returns the map view (pixel mapping) for overlays. `task`: the task the
    subtasks come from (instruction, planner state, targets), from runtime.task_context."""
    m, view = render_map(grid, snap, cfg, trajectory_xy)
    return np.concatenate([m, render_panel(snap, cfg, m.shape[0], t0, k, task)], axis=1), view


def compact(snap: dict, t0: float | None = None) -> dict:
    """JSON-safe log line (no cell arrays, rounded)."""
    def r(v, n=3):
        return None if v is None else round(float(v), n)
    path = snap.get("path")
    return {
        "t": r(snap.get("t")), "t_rel": r(snap["t"] - t0) if snap.get("t") is not None and t0 is not None else None,
        "subtask": snap.get("subtask"), "goal_kind": snap.get("goal_kind"),
        "goal": None if snap.get("goal") is None else {**snap["goal"], "xy": [r(v) for v in snap["goal"]["xy"]],
                                                        "score": r(snap["goal"].get("score"))},
        "path_len_m": r(sum(math.dist(a, b) for a, b in zip(path, path[1:]))) if path else None,
        "robot": None if snap.get("robot") is None else [r(v) for v in snap["robot"]],
        "n_frontier_cells": int(len(snap["frontier_cells"])),
        "points": [{"id": p["id"], "xy": [r(v, 2) for v in p["xy"]], "s_vl": r(p["s_vl"]), "scored": p["scored"],
                    "blacklisted": p["blacklisted"],
                    "score": None if p["score"] is None else {k: r(v) for k, v in p["score"].items()}}
                   for p in snap["points"]],
        "instances": [{**q, "xy": [r(v, 2) for v in q["xy"]],
                       "approach_xy": None if q["approach_xy"] is None else [r(v, 2) for v in q["approach_xy"]]}
                      for q in snap["instances"]],
        "mixture": snap.get("mixture"), "events": snap.get("events"), "notes": snap.get("notes"),
        "n_switches": snap.get("n_switches"), "reach_checks": snap.get("reach_checks"),
        "timing_ms": {k: r(v, 1) for k, v in snap.get("timing_ms", {}).items()},
    }


class ShadowRecorder:
    """Bag shadow mode output (E4): video + key frames + per-step JSONL + statistics (§13)."""

    def __init__(self, out_dir: str | Path, cfg: ExplorationVizCfg):
        self.dir = Path(out_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.log = open(self.dir / "shadow_log.jsonl", "w")
        self.writer: cv2.VideoWriter | None = None
        self.t0: float | None = None
        self.last: np.ndarray | None = None
        self._prev_goal: tuple | None = None
        self._explore_s = 0.0
        self._prev_t: float | None = None
        self.stats = {"steps": 0, "explore_steps": 0, "switches": 0, "goal_kinds": {}, "events": [],
                      "frontier_points": [], "frontier_cells": [], "frac_vl_nonzero": [],
                      "on_frame_ms": [], "step_ms": [], "reach_checks": []}

    def add(self, k: int, grid: OccupancyGrid2D, snap: dict, trajectory_xy=None,
            task: dict | None = None) -> tuple[np.ndarray, View]:
        t = snap.get("t") if snap.get("t") is not None else snap.get("frame_t")
        self.t0 = t if self.t0 is None else self.t0
        img, view = render(grid, snap, self.cfg, trajectory_xy, self.t0, k, task)
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.dir / "shadow_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.cfg.video_fps, (img.shape[1], img.shape[0]))
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if k % self.cfg.keyframe_every == 0:
            cv2.imwrite(str(self.dir / "frames" / f"frame_{k:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.log.write(json.dumps({"k": k, **compact(snap, self.t0), "task": task}) + "\n")
        self._update_stats(snap, t)
        self.last = img
        return img, view

    def _update_stats(self, snap: dict, t: float | None) -> None:
        s = self.stats
        s["steps"] += 1
        kind = snap.get("goal_kind") or "none"
        s["goal_kinds"][kind] = s["goal_kinds"].get(kind, 0) + 1
        s["events"] += [{"t": round(e["stamp"] - self.t0, 2), "type": e["type"]} for e in snap.get("events", [])]
        tm = snap.get("timing_ms", {})
        s["on_frame_ms"].append(tm.get("on_frame", 0.0))
        s["step_ms"].append(tm.get("step", 0.0))
        s["reach_checks"].append(snap.get("reach_checks", 0))
        st = snap.get("subtask")
        if st is None or st["type"] != "explore":
            self._prev_t = None
            return
        s["explore_steps"] += 1
        if t is not None and self._prev_t is not None:
            self._explore_s += t - self._prev_t
        self._prev_t = t
        pts = snap["points"]
        s["frontier_points"].append(len(pts))
        s["frontier_cells"].append(int(len(snap["frontier_cells"])))
        scored = [p for p in pts if p["scored"]]
        if scored:
            s["frac_vl_nonzero"].append(sum(p["s_vl"] >= VL_EPS for p in scored) / len(scored))
        g = snap.get("goal")
        key = None if g is None else (g["kind"], g["id"])
        if key is not None and self._prev_goal is not None and key != self._prev_goal:
            s["switches"] += 1
        self._prev_goal = key if key is not None else self._prev_goal

    def close(self) -> dict:
        self.log.close()
        if self.writer is not None:
            self.writer.release()
            raw, out = self.dir / "shadow_raw.mp4", self.dir / "shadow.mp4"
            if shutil.which("ffmpeg") and subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)]).returncode == 0:
                raw.unlink()
            else:
                raw.rename(out)
        if self.last is not None:
            cv2.imwrite(str(self.dir / "final.png"), cv2.cvtColor(self.last, cv2.COLOR_RGB2BGR))
        s = self.stats

        def pct(v, q):
            return round(float(np.percentile(v, q)), 1) if v else None
        out = {
            "steps": s["steps"], "explore_steps": s["explore_steps"], "explore_s": round(self._explore_s, 1),
            "goal_kinds": s["goal_kinds"], "events": s["events"],
            "frontier_points": {"mean": round(float(np.mean(s["frontier_points"])), 1) if s["frontier_points"] else None,
                                "max": max(s["frontier_points"], default=0)},
            "frontier_cells": {"mean": round(float(np.mean(s["frontier_cells"])), 1) if s["frontier_cells"] else None,
                               "max": max(s["frontier_cells"], default=0)},
            "frac_scored_points_with_vl": round(float(np.mean(s["frac_vl_nonzero"])), 3) if s["frac_vl_nonzero"] else None,
            "switches": s["switches"],
            "switches_per_min": round(s["switches"] / (self._explore_s / 60), 2) if self._explore_s > 0 else None,
            "on_frame_ms": {"p50": pct(s["on_frame_ms"], 50), "p95": pct(s["on_frame_ms"], 95)},
            "step_ms": {"p50": pct(s["step_ms"], 50), "p95": pct(s["step_ms"], 95)},
            "reach_checks_mean": round(float(np.mean(s["reach_checks"])), 2) if s["reach_checks"] else None,
        }
        (self.dir / "stats.json").write_text(json.dumps(out, indent=2))
        return out
