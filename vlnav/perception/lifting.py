"""2D mask -> 3D observation (SYSTEM.md §4.2): erosion, depth samples, MAD rejection, median.

This build lifts with the dense aligned depth of the RGB-D camera (`Frame.depth`); the LiDAR
projection path (`perception/lidar_depth.py`) is not implemented yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vlnav.perception.config import LiftingCfg
from vlnav.perception.depth_utils import backproject

MAD_TO_SIGMA = 1.4826       # MAD of a normal distribution -> standard deviation


@dataclass(slots=True)
class Lifted:
    points_map: np.ndarray          # Mx3 voxel centers
    centroid_map: np.ndarray        # (3,)
    n_samples: int


def lift_mask(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T_map_cam: np.ndarray,
              cfg: LiftingCfg, rng: np.random.Generator) -> Lifted | None:
    if cfg.mask_erode_px > 0:
        k = 2 * cfg.mask_erode_px + 1
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if eroded.any():
            mask = eroded
    v, u = np.nonzero(mask)
    z = depth[v, u]
    ok = (z > cfg.min_depth_m) & (z < cfg.max_depth_m)
    u, v, z = u[ok], v[ok], z[ok]
    if len(z) > cfg.max_mask_pixels:
        keep = rng.choice(len(z), cfg.max_mask_pixels, replace=False)
        u, v, z = u[keep], v[keep], z[keep]
    if len(z) < cfg.min_points:
        return None
    med = np.median(z)
    mad = np.median(np.abs(z - med)) * MAD_TO_SIGMA
    if mad > 0:
        inl = np.abs(z - med) <= cfg.mad_k * mad
        u, v, z = u[inl], v[inl], z[inl]
    if len(z) < cfg.min_points:
        return None
    pts = backproject(u.astype(float), v.astype(float), z.astype(float), K)
    pts = pts @ T_map_cam[:3, :3].T + T_map_cam[:3, 3]
    keys = np.unique(np.floor(pts / cfg.voxel_m).astype(np.int64), axis=0)
    vox = (keys + 0.5) * cfg.voxel_m
    centroid = np.median(pts, axis=0)
    if np.linalg.norm(centroid - T_map_cam[:3, 3]) > cfg.max_obs_range_m:
        return None
    return Lifted(points_map=vox, centroid_map=centroid, n_samples=len(z))
