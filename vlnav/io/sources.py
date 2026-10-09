"""Frame sources (SYSTEM.md §4.1). `BagSource`: camera (+ optional depth) + LiDAR from a ROS 2 bag.

Pose providers (`poses.kind` in the bag config):
- `trajectory`: external LiDAR-odometry file (T_odom_lidar) + calibration YAML (T_lidar_cam,
  T_gravity_odom). No base_link: the base is the LiDAR frame.
- `tf`: the bag's own TF tree (/tf_static + /tf), e.g. a robot publishing `base_link -> odom`;
  camera and LiDAR poses are chain lookups at the message stamp. Frames missing from the tree
  (e.g. an optical frame) come from `extra_static_tf`.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation

from vlnav.common.config import REPO_ROOT, deep_merge, load_yaml
from vlnav.common.models import Frame
from vlnav.io.ros_decode import (camera_info_K_D, decode_compressed_depth, decode_image,
                                 decode_pointcloud_xyz, stamp_s)
from vlnav.io.rosbag2_reader import open_bag
from vlnav.io.tf_buffer import TfBuffer, static_matrix
from vlnav.io.trajectory import Trajectory


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StaticTfCfg(_Cfg):
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]      # extrinsic xyz (ROS static_transform_publisher order)


class PosesCfg(_Cfg):
    kind: Literal["trajectory", "tf"]
    # kind: trajectory
    trajectory: str | None = None
    trajectory_format: str | None = None
    calibration: str | None = None           # YAML with T_lidar_cam (+ T_gravity_odom)
    # kind: tf
    map_frame: str | None = None
    base_frame: str | None = None
    tf_topic: str | None = None
    static_tf_topic: str | None = None
    extra_static_tf: list[StaticTfCfg] = []


class BagSourceConfig(_Cfg):
    bag_dir: str
    typestore: str
    poses: PosesCfg
    camera_id: str
    color_topic: str
    depth_topic: str | None                  # None: no depth camera (lift from LiDAR)
    camera_info_topic: str
    lidar_topic: str
    imu_topic: str | None
    frame_rate_hz: float
    max_pose_gap_s: float
    lidar_window_s: float
    depth_pair_tol_s: float
    pair_wait_s: float                       # wait for depth / late LiDAR scans before emitting
    start_offset_s: float | None             # relative to the start of the pose span
    end_offset_s: float | None


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_bag_config(path: str | Path, overrides: dict | None = None) -> BagSourceConfig:
    return BagSourceConfig.model_validate(deep_merge(load_yaml(path), overrides))


def matrix_from_yaml(d: dict) -> np.ndarray:
    """{xyz: [..], xyzw: [..]} or {matrix: 4x4} -> 4x4."""
    if "matrix" in d:
        return np.asarray(d["matrix"], dtype=float).reshape(4, 4)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(d["xyzw"]).as_matrix()
    T[:3, 3] = d["xyz"]
    return T


def matrix_to_yaml(T: np.ndarray) -> dict:
    return {"xyz": [float(v) for v in T[:3, 3]],
            "xyzw": [float(v) for v in Rotation.from_matrix(T[:3, :3]).as_quat()]}


def load_calibration(path: str | Path) -> dict[str, np.ndarray]:
    d = yaml.safe_load(Path(path).read_text())
    return {k: matrix_from_yaml(v) for k, v in d.items() if isinstance(v, dict)}


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


# ----------------------------------------------------------------------------- pose providers
class TrajectoryPoses:
    """External LiDAR trajectory; base = LiDAR frame, camera via T_lidar_cam."""

    def __init__(self, cfg: PosesCfg, max_gap_s: float):
        calib = load_calibration(resolve(cfg.calibration))
        self.T_lidar_cam = calib["T_lidar_cam"]
        self.T_gravity_odom = calib.get("T_gravity_odom", np.eye(4))
        self.traj = Trajectory.load(resolve(cfg.trajectory),
                                    cfg.trajectory_format).transformed(self.T_gravity_odom)
        self.max_gap_s = max_gap_s
        self.span = (float(self.traj.stamps[0]), float(self.traj.stamps[-1]))

    def base(self, t: float) -> np.ndarray | None:
        return self.traj.interpolate(t, self.max_gap_s)

    def lidar(self, t: float, frame_id: str) -> np.ndarray | None:
        return self.base(t)

    def camera(self, t: float, frame_id: str) -> np.ndarray | None:
        T = self.base(t)
        return None if T is None else T @ self.T_lidar_cam


class TfPoses:
    """Poses from the bag's TF tree."""

    def __init__(self, cfg: PosesCfg, bag, max_gap_s: float):
        extra = [(e.parent, e.child, static_matrix(e.xyz, e.rpy_deg)) for e in cfg.extra_static_tf]
        self.tf = TfBuffer.from_bag(bag, cfg.tf_topic, cfg.static_tf_topic, extra)
        self.map_frame, self.base_frame = cfg.map_frame, cfg.base_frame
        self.max_gap_s = max_gap_s
        chain = set(self.tf._chain(self.base_frame)) | set(self.tf._chain(self.map_frame))
        spans = [(tr.stamps[0], tr.stamps[-1]) for f, tr in self.tf.dynamic.items() if f in chain]
        if not spans:
            raise ValueError(f"no dynamic TF between {self.map_frame!r} and {self.base_frame!r}")
        self.span = (float(max(a for a, _ in spans)), float(min(b for _, b in spans)))

    def _lookup(self, frame: str, t: float) -> np.ndarray | None:
        return self.tf.lookup(self.map_frame, frame, t, self.max_gap_s)

    def base(self, t: float) -> np.ndarray | None:
        return self._lookup(self.base_frame, t)

    def lidar(self, t: float, frame_id: str) -> np.ndarray | None:
        return self._lookup(frame_id, t)

    def camera(self, t: float, frame_id: str) -> np.ndarray | None:
        return self._lookup(frame_id, t)


def make_pose_provider(cfg: PosesCfg, bag, max_gap_s: float):
    if cfg.kind == "trajectory":
        return TrajectoryPoses(cfg, max_gap_s)
    return TfPoses(cfg, bag, max_gap_s)


# --------------------------------------------------------------------------------- the source
class BagSource:
    """Iterates `Frame`s at `frame_rate_hz` (camera stamps) inside the pose span."""

    def __init__(self, cfg: BagSourceConfig):
        self.cfg = cfg
        self.bag = open_bag(resolve(cfg.bag_dir), cfg.typestore)
        self.poses = make_pose_provider(cfg.poses, self.bag, cfg.max_pose_gap_s)
        t0, t1 = self.poses.span
        self.t_start = t0 + cfg.start_offset_s if cfg.start_offset_s is not None else t0
        self.t_end = t0 + cfg.end_offset_s if cfg.end_offset_s is not None else t1
        self.stats = {"emitted": 0, "no_pose": 0, "no_depth": 0}
        self._maps = None
        self._lidar_frame: str | None = None

    def pose(self, t: float) -> np.ndarray | None:
        """T_map_base(t), None in pose gaps."""
        return self.poses.base(t)

    def __iter__(self) -> Iterator[Frame]:
        c = self.cfg
        K = D = None
        pending: deque[tuple[float, object]] = deque()          # (stamp, color msg)
        depths: dict[float, object] = {}
        scans: deque[tuple[float, np.ndarray]] = deque()         # (stamp, Nx3 map points)
        next_t = self.t_start
        topics = [c.color_topic, c.camera_info_topic, c.lidar_topic]
        if c.depth_topic:
            topics.append(c.depth_topic)
        for m in self.bag.messages(topics, start=self.t_start - c.lidar_window_s,
                                   end=self.t_end + c.pair_wait_s):
            if m.topic == c.camera_info_topic:
                if K is None:
                    K, D, _, _ = camera_info_K_D(m.msg)
                continue
            s = stamp_s(m.msg.header)
            if m.topic == c.lidar_topic:
                self._lidar_frame = m.msg.header.frame_id
                T = self.poses.lidar(s, self._lidar_frame)
                if T is not None:
                    scans.append((s, transform_points(T, decode_pointcloud_xyz(m.msg))))
            elif m.topic == c.depth_topic:
                depths[s] = m.msg
            elif m.topic == c.color_topic and self.t_start <= s <= self.t_end and s >= next_t:
                pending.append((s, m.msg))
                next_t = s + 1.0 / c.frame_rate_hz

            while pending and K is not None:
                s_c, color = pending[0]
                depth = None
                if c.depth_topic:
                    match = min(depths, key=lambda d: abs(d - s_c), default=None)
                    if match is not None and abs(match - s_c) <= c.depth_pair_tol_s:
                        depth = depths[match]
                    elif m.log_time < s_c + c.pair_wait_s:
                        break                                     # depth may still arrive
                elif m.log_time < s_c + c.pair_wait_s:
                    break                                         # let late LiDAR scans arrive
                pending.popleft()
                if c.depth_topic and depth is None:
                    self.stats["no_depth"] += 1
                    continue
                frame = self._frame(s_c, color, depth, K, D, scans)
                if frame is not None:
                    yield frame
            # keep what the oldest pending (or next) color frame can still need
            needed = (pending[0][0] if pending else m.log_time - c.pair_wait_s) \
                - c.lidar_window_s - c.pair_wait_s
            for d in [d for d in depths if d < needed]:
                del depths[d]
            while scans and scans[0][0] < needed:
                scans.popleft()

    def _undistort(self, img: np.ndarray, K: np.ndarray, D: np.ndarray, interp: int) -> np.ndarray:
        """Remove lens distortion (same K kept) so frames are pinhole downstream."""
        if not np.any(D):
            return img
        h, w = img.shape[:2]
        if self._maps is None or self._maps[0] != (h, w):
            m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
            self._maps = ((h, w), m1, m2)
        _, m1, m2 = self._maps
        return cv2.remap(img, m1, m2, interp)

    def _frame(self, s, color, depth, K, D, scans) -> Frame | None:
        T_map_base = self.poses.base(s)
        T_map_cam = self.poses.camera(s, color.header.frame_id)
        T_map_lidar = self.poses.lidar(s, self._lidar_frame) if self._lidar_frame else T_map_base
        if T_map_base is None or T_map_cam is None or T_map_lidar is None:
            self.stats["no_pose"] += 1
            return None
        rgb = self._undistort(decode_image(color), K, D, cv2.INTER_LINEAR)
        depth_m = None
        if depth is not None:
            depth_m = self._undistort(decode_compressed_depth(depth), K, D, cv2.INTER_NEAREST)
        window = [p for t, p in scans if s - self.cfg.lidar_window_s <= t <= s]
        self.stats["emitted"] += 1
        return Frame(
            stamp=s, camera_id=self.cfg.camera_id, rgb=rgb, K=K,
            hfov=float(2 * np.arctan(rgb.shape[1] / (2 * K[0, 0]))),
            T_map_cam=T_map_cam, T_map_base=T_map_base, depth=depth_m,
            lidar_points_map=np.concatenate(window) if window else np.empty((0, 3)),
            sensor_origin_map=tuple(float(v) for v in T_map_lidar[:3, 3]))
