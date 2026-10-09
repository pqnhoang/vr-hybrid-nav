"""Dense depth -> points (SYSTEM.md §4.2), for sim / depth cameras."""
from __future__ import annotations

import numpy as np


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel coordinates + depth -> Nx3 points in the camera optical frame (pinhole)."""
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def depth_to_points(depth: np.ndarray, K: np.ndarray, T_map_cam: np.ndarray, stride: int,
                    min_depth_m: float, max_depth_m: float) -> np.ndarray:
    """Every `stride`-th pixel with valid depth -> Nx3 points in the map frame."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u]
    ok = (z > min_depth_m) & (z < max_depth_m)
    pts = backproject(u[ok].astype(float), v[ok].astype(float), z[ok].astype(float), K)
    return pts @ T_map_cam[:3, :3].T + T_map_cam[:3, 3]
