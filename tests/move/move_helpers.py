"""Synthetic camera / grid / frames for move-subtask tests (no GPU, no network)."""
from __future__ import annotations

import numpy as np

from vlnav.common.models import Frame
from vlnav.mapping.occupancy import FREE, OccupancyGrid2D
from vlnav.move.config import load_move_config
from vlnav.navigation import GridAStar, load_navigation_config

K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
W, H = 640, 480
R_BODY_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # optical -> body (x fwd)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def base_T(x=0.0, y=0.0, yaw=0.0, z=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw)
    T[:3, 3] = (x, y, z)
    return T


def cam_T(x=0.0, y=0.0, yaw=0.0, height=1.0, pitch_down_deg=15.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw) @ rot_y(np.radians(pitch_down_deg)) @ R_BODY_OPT
    T[:3, 3] = (x, y, height)
    return T


def project(T_map_cam: np.ndarray, p_map) -> tuple[float, float]:
    pc = (np.asarray(p_map, float) - T_map_cam[:3, 3]) @ T_map_cam[:3, :3]
    return pc[0] / pc[2] * K[0, 0] + K[0, 2], pc[1] / pc[2] * K[1, 1] + K[1, 2]


def frame(x=0.0, y=0.0, yaw=0.0, stamp=0.0, depth=None, lidar=None, rgb=None) -> Frame:
    rgb = np.zeros((H, W, 3), np.uint8) if rgb is None else rgb
    return Frame(stamp=stamp, camera_id="cam", rgb=rgb, K=K, hfov=2 * np.arctan(W / 2 / K[0, 0]),
                 T_map_cam=cam_T(x, y, yaw), T_map_base=base_T(x, y, yaw), depth=depth, lidar_points_map=lidar,
                 sensor_origin_map=(x, y, 1.0))


def free_grid(size_m=20.0, res=0.1, origin=(-10.0, -10.0)) -> OccupancyGrid2D:
    n = int(round(size_m / res))
    return OccupancyGrid2D.from_array(np.full((n, n), FREE, np.int8), res, origin)


def planner(**over) -> GridAStar:
    return GridAStar(load_navigation_config(overrides={"robot_radius_m": 0.2, **over}))


def cfg(**over):
    base = {"projection": {"base_height_m": 0.0}, "s2": {"blocking": True},
            "model_view": {"mode": "full"}, "loop": {"requery_period_s": None}}
    from vlnav.common.config import deep_merge
    return load_move_config(overrides=deep_merge(base, over))
