"""LiDAR points -> sparse depth image of a camera (SYSTEM.md §4.2), for cameras without depth.

Points (map frame, accumulated window) are projected with a z-buffer (nearest point wins per
pixel); the result plugs into the same mask lifting as dense depth (erosion + MAD + median).
"""
from __future__ import annotations

import numpy as np

NO_DEPTH = 0.0


def sparse_depth(points_map: np.ndarray, T_map_cam: np.ndarray, K: np.ndarray,
                 shape: tuple[int, int], min_depth_m: float, max_depth_m: float) -> np.ndarray:
    h, w = shape
    depth = np.full((h, w), np.inf, np.float32)
    if points_map is None or not len(points_map):
        return np.zeros((h, w), np.float32)
    pc = (points_map - T_map_cam[:3, 3]) @ T_map_cam[:3, :3]          # map -> camera optical
    pc = pc[(pc[:, 2] > min_depth_m) & (pc[:, 2] < max_depth_m)]
    u = np.round(pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]).astype(np.int64)
    v = np.round(pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    np.minimum.at(depth, (v[ok], u[ok]), pc[ok, 2].astype(np.float32))
    depth[~np.isfinite(depth)] = NO_DEPTH
    return depth
