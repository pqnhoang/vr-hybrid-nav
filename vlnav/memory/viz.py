"""ROS-free visualization of a memory snapshot (TASK_unified_memory.md §13). Returns RGB images."""
from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from vlnav.memory.snapshot import MemorySnapshot

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_color(label: str) -> tuple[int, int, int]:
    """Stable, saturated RGB color per label."""
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(255 * r), int(255 * g), int(255 * b)


@dataclass(frozen=True)
class TopdownBackground:
    image: np.ndarray                      # HxW uint8 gray, row 0 = max y
    bounds: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax
    m_per_px: float


def points_background(points_xy: np.ndarray, bounds, m_per_px: float) -> TopdownBackground:
    """Log point density of (e.g. LiDAR obstacle) points as a light-gray map."""
    xmin, ymin, xmax, ymax = bounds
    w, h = int(np.ceil((xmax - xmin) / m_per_px)), int(np.ceil((ymax - ymin) / m_per_px))
    u = ((points_xy[:, 0] - xmin) / m_per_px).astype(int)
    v = ((ymax - points_xy[:, 1]) / m_per_px).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    hist = np.zeros((h, w), np.float32)
    np.add.at(hist, (v[ok], u[ok]), 1.0)
    dens = np.log1p(hist)
    dens = dens / dens.max() if dens.max() > 0 else dens
    return TopdownBackground((255 - 150 * dens).astype(np.uint8), tuple(bounds), m_per_px)


GRID_GRAY = {-1: 205, 0: 255, 1: 60}              # unknown / free / occupied


def grid_background(grid) -> TopdownBackground:
    """Occupancy grid (ternary) as a gray top-down background, north up."""
    arr = grid.as_array()
    img = np.full(arr.shape, GRID_GRAY[-1], np.uint8)
    img[arr == 0] = GRID_GRAY[0]
    img[arr == 1] = GRID_GRAY[1]
    ox, oy = grid.origin_xy
    ny, nx = arr.shape
    return TopdownBackground(img[::-1].copy(), (ox, oy, ox + nx * grid.resolution, oy + ny * grid.resolution),
                             grid.resolution)


def snapshot_bounds(snap: MemorySnapshot, margin_m: float) -> tuple[float, float, float, float]:
    xy = [n.centroid[:2] for n in snap.all_objects()] + \
         [n.best_view_pose.position[:2] for n in snap.all_objects()]
    if not xy:
        return (-margin_m, -margin_m, margin_m, margin_m)
    a = np.asarray(xy)
    return (*(a.min(0) - margin_m), *(a.max(0) + margin_m))


def render_topdown(snap: MemorySnapshot, background: TopdownBackground | None = None,
                   bounds: tuple[float, float, float, float] | None = None,
                   m_per_px: float | None = None, trajectory_xy: np.ndarray | None = None,
                   show_text: bool = True, include_tentative: bool = False,
                   title: str | None = None, show_objects: bool = True) -> np.ndarray:
    """Background, rooms, trajectory, objects (`#id label score`), best-view poses (lines)."""
    if background is not None:
        bounds = bounds or background.bounds
        m_per_px = m_per_px or background.m_per_px
    bounds = bounds or snapshot_bounds(snap, 2.0)
    m_per_px = m_per_px or 0.1
    xmin, ymin, xmax, ymax = bounds
    w, h = int(np.ceil((xmax - xmin) / m_per_px)), int(np.ceil((ymax - ymin) / m_per_px))

    def px(xy):
        xy = np.asarray(xy, dtype=float)
        return np.stack([(xy[..., 0] - xmin) / m_per_px, (ymax - xy[..., 1]) / m_per_px], -1).astype(int)

    if background is not None:                      # nearest-neighbor resample of the background
        bg, bm = background.image, background.m_per_px
        us = np.floor((xmin - background.bounds[0] + (np.arange(w) + 0.5) * m_per_px) / bm).astype(int)
        vs = np.floor((background.bounds[3] - ymax + (np.arange(h) + 0.5) * m_per_px) / bm).astype(int)
        okx, oky = (us >= 0) & (us < bg.shape[1]), (vs >= 0) & (vs < bg.shape[0])
        crop = np.full((h, w), 255, np.uint8)
        crop[np.ix_(oky, okx)] = bg[np.ix_(vs[oky], us[okx])]
        img = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        img = np.full((h, w, 3), 255, np.uint8)

    overlay = img.copy()
    for room in snap.rooms():
        poly = snap.room_polygons.get(room.id, [])
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [px(poly)], label_color(f"room{room.id}"))
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    for room in snap.rooms():                       # labels on a white box, readable over clutter
        text = f"R{room.id} {room.label}"
        (tw, th), base = cv2.getTextSize(text, FONT, 0.55, 1)
        x, y = (int(v) for v in px(room.centroid_xy))
        x = min(max(x - tw // 2, 2), max(img.shape[1] - tw - 2, 2))
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), label_color(f"room{room.id}"), 2)
        cv2.putText(img, text, (x, y - 1), FONT, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    if trajectory_xy is not None and len(trajectory_xy) > 1:
        cv2.polylines(img, [px(trajectory_xy)], False, (120, 160, 255), 1, cv2.LINE_AA)

    nodes = snap.all_objects(include_tentative=include_tentative) if show_objects else []
    for n in nodes:
        c, b = px(n.centroid[:2]), px(n.best_view_pose.position[:2])
        cv2.line(img, tuple(int(v) for v in b), tuple(int(v) for v in c), (170, 170, 170), 1, cv2.LINE_AA)
    for n in nodes:
        c = tuple(int(v) for v in px(n.centroid[:2]))
        cv2.circle(img, c, 4, label_color(n.label), -1, cv2.LINE_AA)
        cv2.circle(img, c, 4, (30, 30, 30), 1, cv2.LINE_AA)
        if show_text:
            cv2.putText(img, f"#{n.id} {n.label} {n.best_score:.2f}", (c[0] + 5, c[1] - 4), FONT,
                        0.35, (20, 20, 20), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (10, 25), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def best_view_gallery(snap: MemorySnapshot, node_ids: list[int] | None = None,
                      labels: list[str] | None = None, cell_px: int = 192, cols: int = 8,
                      max_items: int = 64, margin: float = 0.15) -> np.ndarray:
    """Grid of best-view crops with `#id label score n_obs` captions."""
    nodes = [snap.get_object(i) for i in node_ids] if node_ids is not None else snap.all_objects()
    if labels is not None:
        nodes = [n for n in nodes if n.label in labels]
    nodes = nodes[:max_items]
    rows = max(1, int(np.ceil(len(nodes) / cols)))
    cap = 22
    out = np.full((rows * (cell_px + cap), cols * cell_px, 3), 255, np.uint8)
    for k, n in enumerate(nodes):
        r, c = divmod(k, cols)
        try:
            crop = snap.best_view_crop(n.id, margin, cell_px)
        except (FileNotFoundError, RuntimeError):
            continue
        y0, x0 = r * (cell_px + cap), c * cell_px
        out[y0 + cap:y0 + cap + crop.shape[0], x0:x0 + crop.shape[1]] = crop
        cv2.rectangle(out, (x0, y0), (x0 + cell_px - 1, y0 + cap - 1), label_color(n.label), -1)
        cv2.putText(out, f"#{n.id} {n.label} {n.best_score:.2f} x{n.n_obs}", (x0 + 3, y0 + 15),
                    FONT, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return out
