"""LiDAR-on-image overlays for extrinsic checks (pure OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np


def project_points_overlay(rgb: np.ndarray, K: np.ndarray, points_map: np.ndarray,
                           T_map_cam: np.ndarray, max_range_m: float, radius_px: int = 2) -> np.ndarray:
    """RGB image with map points drawn at their projections, colored by depth (near = blue)."""
    img = rgb.copy()
    if points_map is None or not len(points_map):
        return img
    pc = (points_map - T_map_cam[:3, 3]) @ T_map_cam[:3, :3]
    pc = pc[(pc[:, 2] > 0.2) & (pc[:, 2] < max_range_m)]
    uv = pc[:, :2] / pc[:, 2:3] @ K[:2, :2].T + K[:2, 2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
    order = np.argsort(-pc[ok, 2])                       # far first, near drawn on top
    uv, z = uv[ok][order].astype(int), pc[ok, 2][order]
    colors = cv2.applyColorMap((255 * np.clip(z / max_range_m, 0, 1)).astype(np.uint8)[:, None],
                               cv2.COLORMAP_JET)[:, 0, ::-1]
    for (u, v), c in zip(uv, colors):
        cv2.circle(img, (int(u), int(v)), radius_px, tuple(int(x) for x in c), -1)
    return img
