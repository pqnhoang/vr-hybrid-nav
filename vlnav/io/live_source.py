"""Bounded camera/LiDAR synchronizer, independent of rclpy for replay and unit testing.

The adapter supplies lookup(frame_id, stamp) -> T_map_frame or None. Every LiDAR scan
uses its own timestamp; future scans are excluded from the camera's accumulation window.
"""
from __future__ import annotations

import threading
from collections import deque

import cv2
import numpy as np

from vlnav.common.models import Frame
from vlnav.io.ros_decode import camera_info_K_D, decode_image, decode_pointcloud_xyz, stamp_s


class LiveSource:
    def __init__(self, cfg, max_age_s: float = 5.0, max_scan_age_s: float = 0.5):
        if cfg.poses.kind != "tf" or cfg.depth_topic:
            raise ValueError("LiveSource requires TF poses and camera + PointCloud2 (no depth topic)")
        if not cfg.poses.map_frame or not cfg.poses.base_frame:
            raise ValueError("LiveSource requires map_frame and base_frame")
        self.cfg = cfg
        self.max_age_s = max_age_s
        self.max_scan_age_s = max_scan_age_s
        self._lock = threading.Lock()
        self._images = deque(maxlen=16)
        self._scans = deque(maxlen=128)
        self._info = None
        self._last_image = -float("inf")
        self._last_emitted = -float("inf")
        self.wait_reason = "waiting for camera, CameraInfo, LiDAR and TF"

    def camera_info(self, msg):
        with self._lock:
            self._info = msg

    def image(self, msg, received_at: float):
        t = stamp_s(msg.header)
        with self._lock:
            if t - self._last_image < 1.0 / self.cfg.frame_rate_hz - 1e-6:
                return
            self._last_image = t
            self._images.append((t, received_at, msg))

    def lidar(self, msg):
        with self._lock:
            self._scans.append(msg)

    def next_frame(self, lookup, now: float) -> Frame | None:
        with self._lock:
            ready = [x for x in self._images if x[0] > self._last_emitted
                     and self.cfg.pair_wait_s <= now - x[1] <= self.max_age_s]
            if not ready or self._info is None:
                self.wait_reason = "waiting for a recent image and CameraInfo"
                return None
            t, _, color = max(ready, key=lambda x: x[0])
            info = self._info
            scans = [m for m in self._scans if t - self.cfg.lidar_window_s <= stamp_s(m.header) <= t]
        if not scans or t - max(stamp_s(m.header) for m in scans) > self.max_scan_age_s:
            self.wait_reason = "waiting for LiDAR at the image timestamp"
            return None
        base = lookup(self.cfg.poses.base_frame, t)
        cam = lookup(color.header.frame_id, t)
        latest = max(scans, key=lambda m: stamp_s(m.header))
        lidar = lookup(latest.header.frame_id, t)
        if base is None or cam is None or lidar is None:
            self.wait_reason = "waiting for base/camera/LiDAR TF at the image timestamp"
            return None
        points = []
        for scan in scans:
            transform = lookup(scan.header.frame_id, stamp_s(scan.header))
            if transform is None:
                # The newest scan must be transformed; older missing scans may be omitted.
                if scan is latest:
                    self.wait_reason = "waiting for TF at the newest LiDAR scan timestamp"
                    return None
                continue
            xyz = decode_pointcloud_xyz(scan)
            if len(xyz):
                points.append(xyz @ transform[:3, :3].T + transform[:3, 3])
        if not points:
            self.wait_reason = "waiting for nonempty LiDAR scans"
            return None
        K, D, width, height = camera_info_K_D(info)
        rgb = decode_image(color)
        if rgb.shape[:2] != (height, width) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("CameraInfo dimensions/intrinsics do not match the image")
        if np.any(D):
            rgb = cv2.undistort(rgb, K, D)
        frame = Frame(stamp=t, camera_id=self.cfg.camera_id, rgb=rgb, K=K,
                      hfov=float(2 * np.arctan(width / (2 * K[0, 0]))),
                      T_map_cam=cam, T_map_base=base, lidar_points_map=np.concatenate(points),
                      sensor_origin_map=tuple(lidar[:3, 3]))
        self._last_emitted = t
        self.wait_reason = ""
        return frame
