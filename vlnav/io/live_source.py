"""Bounded camera/LiDAR synchronizer, independent of rclpy for replay and unit testing.

The adapter supplies lookup(frame_id, stamp) -> T_map_frame or None. Every LiDAR scan
uses its own timestamp; future scans are excluded from the camera's accumulation window.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from vlnav.common.models import Frame
from vlnav.io.ros_decode import camera_info_K_D, decode_image, decode_pointcloud_xyz, stamp_s


class LiveSource:
    def __init__(self, cfg, max_age_s: float = 5.0, max_scan_age_s: float = 0.5, *, debug: bool = False):
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
        self.debug = debug
        self.last_sync: dict = {}
        self._debug_counts = dict(images_received=0, images_throttled=0, images_overwritten=0,
                                  scans_received=0, scans_overwritten=0, frames_emitted=0)
        self._last_scan_received = None

    def camera_info(self, msg):
        with self._lock:
            self._info = msg

    def image(self, msg, received_at: float):
        t = stamp_s(msg.header)
        with self._lock:
            if self.debug:
                self._debug_counts["images_received"] += 1
            if t - self._last_image < 1.0 / self.cfg.frame_rate_hz - 1e-6:
                if self.debug:
                    self._debug_counts["images_throttled"] += 1
                return
            self._last_image = t
            if self.debug and len(self._images) == self._images.maxlen:
                self._debug_counts["images_overwritten"] += 1
            self._images.append((t, received_at, msg))

    def lidar(self, msg):
        with self._lock:
            if self.debug:
                self._debug_counts["scans_received"] += 1
                self._debug_counts["scans_overwritten"] += int(len(self._scans) == self._scans.maxlen)
                self._last_scan_received = time.monotonic()
            self._scans.append(msg)

    def debug_snapshot(self):
        with self._lock:
            return {**self.last_sync, "counters": dict(self._debug_counts),
                    "buffered_images": len(self._images), "buffered_scans": len(self._scans),
                    "latest_accepted_image_stamp": self._last_image,
                    "latest_buffered_scan_stamp": max((stamp_s(m.header) for m in self._scans), default=None),
                    "last_scan_received_monotonic_s": self._last_scan_received}

    def _wait(self, reason):
        self.wait_reason = reason
        if self.debug:
            self.last_sync["wait_reason"] = reason
        return None

    def next_frame(self, lookup, now: float) -> Frame | None:
        with self._lock:
            ready = [x for x in self._images if x[0] > self._last_emitted
                     and self.cfg.pair_wait_s <= now - x[1] <= self.max_age_s]
            if self.debug:
                self.last_sync = {"attempt_monotonic_s": now, "ready_images": len(ready),
                                  "camera_info_available": self._info is not None,
                                  "last_emitted_stamp": self._last_emitted}
            if not ready or self._info is None:
                return self._wait("waiting for a recent image and CameraInfo")
            t, received_at, color = max(ready, key=lambda x: x[0])
            info = self._info
            scans = [m for m in self._scans if t - self.cfg.lidar_window_s <= stamp_s(m.header) <= t]
            if self.debug:
                stamps = [stamp_s(m.header) for m in scans]
                self.last_sync.update(image_stamp=t, image_frame=color.header.frame_id,
                                      image_wait_s=now - received_at, selected_scan_stamps=stamps,
                                      newest_scan_gap_s=t - max(stamps) if stamps else None,
                                      future_scans_excluded=sum(stamp_s(m.header) > t for m in self._scans),
                                      scan_transforms=[])
        if not scans or t - max(stamp_s(m.header) for m in scans) > self.max_scan_age_s:
            return self._wait("waiting for LiDAR at the image timestamp")
        base = lookup(self.cfg.poses.base_frame, t)
        cam = lookup(color.header.frame_id, t)
        latest = max(scans, key=lambda m: stamp_s(m.header))
        lidar = lookup(latest.header.frame_id, t)
        if base is None or cam is None or lidar is None:
            return self._wait("waiting for base/camera/LiDAR TF at the image timestamp")
        points = []
        for scan in scans:
            transform = lookup(scan.header.frame_id, stamp_s(scan.header))
            if self.debug:
                self.last_sync["scan_transforms"].append(
                    {"stamp": stamp_s(scan.header), "frame": scan.header.frame_id,
                     "T_map_sensor": None if transform is None else transform.tolist()})
            if transform is None:
                # The newest scan must be transformed; older missing scans may be omitted.
                if scan is latest:
                    return self._wait("waiting for TF at the newest LiDAR scan timestamp")
                continue
            xyz = decode_pointcloud_xyz(scan)
            if len(xyz):
                points.append(xyz @ transform[:3, :3].T + transform[:3, 3])
        if not points:
            return self._wait("waiting for nonempty LiDAR scans")
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
        if self.debug:
            self.last_sync.update(wait_reason="", emitted_stamp=t, scans_used=len(points),
                                  points_used=len(frame.lidar_points_map),
                                  point_fields=[f.name for f in latest.fields],
                                  T_map_base=base.tolist(), T_map_camera=cam.tolist(),
                                  sensor_origin_map=frame.sensor_origin_map,
                                  camera_K=K.tolist(), image_shape=list(rgb.shape))
            with self._lock:
                self._debug_counts["frames_emitted"] += 1
        return frame
