"""Gravity alignment of a LiDAR-odometry map frame (AUDIT pitfall 1).

FAST-LIO-style odometry uses the first LiDAR frame as map frame, which is not gravity aligned on a
tilted sensor. The accelerometer measures specific force (+g "up" when not accelerating), so the
average of R_map_imu(t) · a(t) over the trajectory points up in the map frame: motion
accelerations average out over a long run, no static period is required.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from vlnav.io.trajectory import Trajectory


def up_direction_in_map(traj: Trajectory, imu_stamps: np.ndarray, imu_acc: np.ndarray,
                        R_body_imu: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Mean specific-force direction in the map frame (unit vector)."""
    acc_map = []
    for t, a in zip(imu_stamps, imu_acc):
        T = traj.interpolate(float(t), max_gap_s)
        if T is not None:
            acc_map.append(T[:3, :3] @ R_body_imu @ a)
    if not acc_map:
        raise ValueError("no IMU sample inside the trajectory")
    up = np.mean(acc_map, axis=0)
    return up / np.linalg.norm(up)


def T_gravity_from_up(up_map: np.ndarray) -> np.ndarray:
    """4x4 rotation taking `up_map` to +z (minimal rotation, yaw unchanged)."""
    up = np.asarray(up_map, dtype=float) / np.linalg.norm(up_map)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z)
    s, c = np.linalg.norm(axis), float(up @ z)
    T = np.eye(4)
    if s > 1e-12:
        T[:3, :3] = Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()
    elif c < 0:
        T[:3, :3] = Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()
    return T
