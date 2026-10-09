"""Synthetic frames, detections, grids for exploration tests (no GPU)."""
from __future__ import annotations

import math

import numpy as np

from vlnav.common.models import Detection2D, Frame, FrameResult, Observation, Pose, Subtask
from vlnav.exploration.config import load_exploration_config
from vlnav.mapping.config import load_mapping_config
from vlnav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid2D, UpdateInfo
from vlnav.navigation import GridAStar, load_navigation_config

W, H, FX = 160, 120, 100.0                          # hfov = 2 atan(80 / 100) = 77.3 deg
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])
HFOV = 2 * math.atan(W / 2 / FX)
R_BASE_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)     # optical: z fwd, x right, y down
RES = 0.1


def cfg(**over):
    return load_exploration_config(overrides=over)


def planner(**over):
    return GridAStar(load_navigation_config(overrides={"robot_radius_m": 0.0, **over}))


def T_base(x, y, yaw, z=0.0):
    T = np.eye(4)
    T[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    T[:3, 3] = (x, y, z)
    return T


def frame(x=0.0, y=0.0, yaw=0.0, t=0.0, cam_yaw=0.0, camera_id="cam") -> Frame:
    Tb = T_base(x, y, yaw)
    Tc = Tb @ T_base(0, 0, cam_yaw, 1.0)
    Tc[:3, :3] = Tc[:3, :3] @ R_BASE_CAM
    return Frame(stamp=t, camera_id=camera_id, rgb=np.zeros((H, W, 3), np.uint8), K=K, hfov=HFOV,
                 T_map_cam=Tc, T_map_base=Tb, sensor_origin_map=(x, y, 1.0))


def det(label, score, u, bbox=None) -> Detection2D:
    m = np.zeros((H, W), bool)
    m[50:70, int(u) - 5:int(u) + 5] = True
    return Detection2D(label=label, score=score, bbox=bbox or (int(u) - 5, 50, int(u) + 5, 70), mask=m,
                       mask_centroid_uv=(float(u), 60.0))


def result(f: Frame, dets=(), obs_xy=(), prompts=()) -> FrameResult:
    """`obs_xy[i]` = map xy of dets[i] (None: not lifted)."""
    dets = list(dets)
    obs = [Observation(stamp=f.stamp, camera_id=f.camera_id, label=d.label, score=d.score,
                       centroid_map=(xy[0], xy[1], 0.5), points_map=np.zeros((1, 3)), bbox=d.bbox,
                       T_map_cam=f.T_map_cam, T_map_base=f.T_map_base, frame_ref="x")
           for d, xy in zip(dets, obs_xy) if xy is not None]
    return FrameResult(frame=f, detections=dets, observations=obs,
                       relevant_idx=[i for i, d in enumerate(dets) if d.label in prompts])


def pose(x, y, yaw=0.0, t=0.0) -> Pose:
    return Pose.from_matrix(T_base(x, y, yaw), t)


def explore(sid=1, prompts=("sofa",), target="sofa", room_hint=None) -> Subtask:
    return Subtask(id=sid, task_id=0, type="explore", target=target, detector_prompts=list(prompts),
                   room_hint=room_hint)


def goto(sid, x, y, yaw=0.0) -> Subtask:
    return Subtask(id=sid, task_id=0, type="goto", target="sofa", node_id=7, goal_pose=pose(x, y, yaw))


def full_info(grid: OccupancyGrid2D, t=0.0) -> UpdateInfo:
    ny, nx = grid.shape
    return UpdateInfo(t=t, changed_bbox_cells=(0, 0, nx - 1, ny - 1), version=grid.version)


def no_change(grid: OccupancyGrid2D, t=0.0) -> UpdateInfo:
    return UpdateInfo(t=t, changed_bbox_cells=(0, 0, -1, -1), version=grid.version)


def corridor_grid(nx=80, ny=60, free_box=(5, 5, 40, 55), walls=()) -> OccupancyGrid2D:
    """Unknown grid (origin (0, 0), 0.1 m) with a free box [x0, y0, x1, y1) of cells + wall boxes."""
    arr = np.full((ny, nx), UNKNOWN, np.int8)
    x0, y0, x1, y1 = free_box
    arr[y0:y1, x0:x1] = FREE
    for wx0, wy0, wx1, wy1 in walls:
        arr[wy0:wy1, wx0:wx1] = OCCUPIED
    return OccupancyGrid2D.from_array(arr, RES, (0.0, 0.0))


# ------------------------------------------------------------------ LiDAR-like scans (mapping)
STEP = 0.05


def floor_pts(x0, y0, x1, y1, z=0.0):
    xs, ys = np.meshgrid(np.arange(x0 + STEP / 2, x1, STEP), np.arange(y0 + STEP / 2, y1, STEP))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)


def wall_pts(p0, p1, z_lo=0.2, z_hi=1.6):
    n = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / STEP) + 1
    xy = np.asarray(p0) + np.linspace(0, 1, n)[:, None] * (np.asarray(p1) - np.asarray(p0))
    return np.concatenate([np.column_stack([xy, np.full(n, z)]) for z in np.arange(z_lo, z_hi + 1e-9, 0.3)])


def scan_grid() -> OccupancyGrid2D:
    return OccupancyGrid2D.from_config(load_mapping_config(overrides={"resolution": RES, "initial_size_m": 16.0}))
