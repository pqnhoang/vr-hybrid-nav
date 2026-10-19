"""DualVLN pixel goal -> map point (TASK_move_subtask.md §7).

1. model pixel -> camera pixel (`to_camera_pixel`; the virtual view in view.py, §7.1);
2. camera ray ∩ floor plane at the grid's ground height (§7.2), works without depth;
3. depth check (§7.3): dense depth image or projected LiDAR points; a measured range shorter than the
   floor range means something stands in the way -> the point is pulled in before it;
4. range limits (§7.4): too close -> rejected, too far -> clamped, above the horizon -> depth point or
   a direction-only point.

Rays are parametrized by the optical-frame depth s: x_cam = s · K⁻¹[u, v, 1] (z = s), so
s compares directly with depth images; ranges along the ray are s · |K⁻¹[u, v, 1]|.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel

from vlnav.move.config import ProjectionCfg


class Projection(BaseModel):
    ok: bool
    reason: str | None = None                  # out_of_image | no_floor | too_close
    uv: tuple[float, float]                    # camera pixel
    cam_xy: tuple[float, float]                # camera center (ray start, for viz)
    p_map: tuple[float, float, float] | None = None
    floor_xy: tuple[float, float] | None = None    # raw ray ∩ floor before depth / range rules
    flags: list[str] = []                      # clipped_by_depth | no_depth | clamped | direction_only | depth_point
    r_floor: float | None = None               # range along the ray to the floor (m)
    r_meas: float | None = None                # measured range along the ray (m)


def to_camera_pixel(pixel_xy, ref_wh, image_wh) -> tuple[float, float]:
    """Model pixel (x, y) in a `ref_wh` image -> pixel in the `image_wh` camera image (pixel centers)."""
    sx, sy = image_wh[0] / ref_wh[0], image_wh[1] / ref_wh[1]
    return (pixel_xy[0] + 0.5) * sx - 0.5, (pixel_xy[1] + 0.5) * sy - 0.5


def camera_ray(K: np.ndarray, T_map_cam: np.ndarray, uv) -> tuple[np.ndarray, np.ndarray]:
    """(camera center, map-frame ray scaled so that optical depth s gives c + s · d)."""
    d_cam = np.linalg.solve(np.asarray(K, float), np.array([uv[0], uv[1], 1.0]))
    T = np.asarray(T_map_cam, float)
    return T[:3, 3].copy(), T[:3, :3] @ d_cam


def dense_depth_at(depth: np.ndarray, uv, window_px: int, min_m: float, max_m: float) -> float | None:
    """Median of valid depths in a (2·window//2+1)² square around uv (optical depth s)."""
    h, w = depth.shape[:2]
    r = max(window_px // 2, 0)
    u, v = int(round(uv[0])), int(round(uv[1]))
    patch = depth[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]
    vals = patch[np.isfinite(patch) & (patch > min_m) & (patch < max_m)]
    return float(np.median(vals)) if vals.size else None


def sparse_depth_at(points_map: np.ndarray, K: np.ndarray, T_map_cam: np.ndarray, uv, search_px: int, k: int,
                    min_m: float, max_m: float) -> float | None:
    """LiDAR points projected into the camera: median depth of the k points nearest to uv in pixels."""
    if points_map is None or not len(points_map):
        return None
    T = np.asarray(T_map_cam, float)
    pc = (np.asarray(points_map, float) - T[:3, 3]) @ T[:3, :3]
    pc = pc[(pc[:, 2] > min_m) & (pc[:, 2] < max_m)]
    if not len(pc):
        return None
    pu = pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]
    pv = pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]
    d2 = (pu - uv[0]) ** 2 + (pv - uv[1]) ** 2
    near = np.nonzero(d2 <= search_px * search_px)[0]
    if not len(near):
        return None
    near = near[np.argsort(d2[near])[:k]]
    return float(np.median(pc[near, 2]))


def measured_depth(frame, uv, cfg: ProjectionCfg) -> float | None:
    """Optical depth at uv from `frame.depth` (dense) or `frame.lidar_points_map` (sparse)."""
    if frame.depth is not None:
        return dense_depth_at(frame.depth, uv, cfg.depth_window_px, cfg.depth_min_m, cfg.depth_max_m)
    return sparse_depth_at(frame.lidar_points_map, frame.K, frame.T_map_cam, uv, cfg.depth_search_px,
                           cfg.depth_k, cfg.depth_min_m, cfg.depth_max_m)


def floor_fn(ground_heights: Callable[[np.ndarray], np.ndarray], robot_xyz, cfg: ProjectionCfg) -> Callable[[object], float]:
    """Floor height for `pixel_to_map` (§7.2).

    The grid's ground (min z per cell) is wrong next to the robot: a waist LiDAR does not see the floor
    there, so the cells hold body / near-robot points. The floor at the robot is `base z -
    base_height_m` when set, else the median grid ground on a ring around the robot
    (`ground_ring_m`; `ground_heights`: (N, 2) points -> (N,) heights, nan unknown). Farther than `ground_min_dist_m` the grid ground is used when it is within
    `max_floor_step_m` of that floor (ramps); otherwise the robot's floor. nan: no floor known."""
    r = np.asarray(robot_xyz, float)
    if cfg.base_height_m is not None:
        z0 = float(r[2] - cfg.base_height_m)
    else:
        lo, hi = cfg.ground_ring_m
        a = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
        rad = np.repeat(np.linspace(lo, hi, 5), len(a))
        ang = np.tile(a, 5)
        zs = ground_heights(r[:2] + rad[:, None] * np.stack([np.cos(ang), np.sin(ang)], -1))
        zs = zs[np.isfinite(zs)]
        z0 = float(np.median(zs)) if len(zs) else float("nan")

    def floor(xy) -> float:
        if not math.isfinite(z0):
            return z0
        if np.linalg.norm(np.asarray(xy, float)[:2] - r[:2]) < cfg.ground_min_dist_m:
            return z0
        z = float(ground_heights(np.asarray(xy, float)[None, :2])[0])
        return z if math.isfinite(z) and abs(z - z0) <= cfg.max_floor_step_m else z0
    return floor


def pixel_to_map(uv, K: np.ndarray, T_map_cam: np.ndarray, image_wh, robot_xy, floor_z,
                 s_meas: float | None, cfg: ProjectionCfg) -> Projection:
    """Camera pixel -> map point on the floor (§7.2–7.4).

    `floor_z(xy) -> float` gives the ground height at a map point (nan if unknown); it is called at
    the robot first, then once more at the hit point (sloped floors). `s_meas` is the measured
    optical depth at uv (None without depth).
    """
    c, d = camera_ray(K, T_map_cam, uv)
    base = Projection(ok=False, uv=(float(uv[0]), float(uv[1])), cam_xy=(float(c[0]), float(c[1])))
    if not (0 <= uv[0] < image_wh[0] and 0 <= uv[1] < image_wh[1]):
        return base.model_copy(update={"reason": "out_of_image"})
    norm = float(np.linalg.norm(d))
    z_f = floor_z(robot_xy)
    if not math.isfinite(z_f):
        return base.model_copy(update={"reason": "no_floor"})
    flags: list[str] = []
    r_meas = None if s_meas is None else s_meas * norm
    r_floor = None
    if d[2] / norm < -cfg.min_down_dz:                                  # ray hits the floor
        s = (z_f - c[2]) / d[2]
        p = c + s * d
        z2 = floor_z(p[:2])
        if math.isfinite(z2):
            z_f = z2
            s = (z_f - c[2]) / d[2]
            p = c + s * d
        r_floor = s * norm
        floor_xy = (float(p[0]), float(p[1]))
        if r_meas is None:
            flags.append("no_depth")
        elif r_meas < r_floor - max(cfg.depth_margin_m, cfg.depth_margin_rel * r_floor):
            r = max(r_meas - cfg.obstacle_backoff_m, 0.0)
            p = c + (r / norm) * d
            flags.append("clipped_by_depth")
    else:                                                               # above the horizon
        floor_xy = None
        if r_meas is not None:
            p = c + (r_meas / norm) * d
            flags.append("depth_point")
        else:
            dxy = d[:2] / max(float(np.linalg.norm(d[:2])), 1e-9)
            p = np.array([c[0] + cfg.fallback_range_m * dxy[0], c[1] + cfg.fallback_range_m * dxy[1], z_f])
            flags.append("direction_only")
    p = np.array([p[0], p[1], z_f])
    rxy = np.asarray(robot_xy, float)[:2]
    off = p[:2] - rxy
    dist = float(np.linalg.norm(off))
    if dist > cfg.max_range_m:
        p[:2] = rxy + off / dist * cfg.max_range_m
        flags.append("clamped")
    out = base.model_copy(update={"p_map": (float(p[0]), float(p[1]), float(p[2])), "floor_xy": floor_xy,
                                  "flags": flags, "r_floor": r_floor, "r_meas": r_meas})
    if dist < cfg.min_range_m:
        return out.model_copy(update={"reason": "too_close"})
    return out.model_copy(update={"ok": True})
