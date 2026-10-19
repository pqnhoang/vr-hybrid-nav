"""Shared models (SYSTEM.md §3). Single definition used by every package."""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation


class _Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Pose(_Model):
    """SE(3) pose in the map frame."""

    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    stamp: float

    def yaw(self) -> float:
        """Heading of the body x axis projected on the ground plane (base poses, AUDIT pitfall 2)."""
        R = Rotation.from_quat(self.orientation_xyzw).as_matrix()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.orientation_xyzw).as_matrix()
        T[:3, 3] = self.position
        return T

    @classmethod
    def from_matrix(cls, T: np.ndarray, stamp: float) -> Pose:
        q = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()
        return cls(position=tuple(float(v) for v in np.asarray(T)[:3, 3]),
                   orientation_xyzw=tuple(float(v) for v in q), stamp=float(stamp))


class Frame(_Model):
    """One camera image + geometry at its stamp."""

    stamp: float
    camera_id: str
    rgb: np.ndarray                                   # HxWx3 uint8
    K: np.ndarray                                     # 3x3
    hfov: float                                       # rad, derived from K and width
    T_map_cam: np.ndarray                             # 4x4 (optical frame: z fwd, x right, y down)
    T_map_base: np.ndarray                            # 4x4 robot base pose
    depth: np.ndarray | None = None                   # HxW meters
    lidar_points_map: np.ndarray | None = None        # Nx3 accumulated window
    sensor_origin_map: tuple[float, float, float]     # for ray casting


class Detection2D(_Model):
    label: str
    score: float
    bbox: tuple[int, int, int, int]                   # x1, y1, x2, y2
    mask: np.ndarray                                  # HxW bool
    mask_centroid_uv: tuple[float, float]


class Observation(_Model):
    """3D-lifted detection."""

    stamp: float
    camera_id: str
    label: str
    score: float
    centroid_map: tuple[float, float, float]
    points_map: np.ndarray                            # Mx3 voxelized
    bbox: tuple[int, int, int, int]
    T_map_cam: np.ndarray
    T_map_base: np.ndarray
    frame_ref: str                                    # key into ImageStore / frame cache


class FrameResult(_Model):
    """Output of PerceptionPipeline for one frame."""

    frame: Frame
    detections: list[Detection2D]
    observations: list[Observation]                   # subset of detections that could be lifted
    relevant_idx: list[int]                           # detections whose label ∈ active prompts


class ObjectNode(_Model):
    """Owned by memory/, consumed by acquisition/ and planner/."""

    id: int
    label: str
    label_votes: dict[str, float]
    centroid: tuple[float, float, float]
    points: np.ndarray
    best_score: float
    best_view_pose: Pose                              # base pose at best detection (nav goal)
    best_view_cam_pose: Pose
    best_view_image_path: str
    best_view_bbox: tuple[int, int, int, int]
    room_id: int | None
    n_obs: int
    first_seen_t: float
    last_seen_t: float


class RoomNode(_Model):
    id: int
    label: str
    label_confidence: float
    mask_cells: np.ndarray                            # Kx2 grid indices
    centroid_xy: tuple[float, float]
    area_m2: float
    object_ids: list[int]
    adjacent_room_ids: list[int]


class Subtask(_Model):
    """Planner -> exploration."""

    id: int
    task_id: int
    type: Literal["explore", "goto", "stop", "move"]
    target: str                                       # move: short label of the route, e.g. "corridor end"
    detector_prompts: list[str] = []
    room_hint: str | None = None
    node_id: int | None = None
    goal_pose: Pose | None = None                     # required for goto
    instruction: str | None = None                    # required for move: route text sent to DualVLN


class PlannerEvent(_Model):
    """Exploration / sim / planner-internal -> planner FSM."""

    type: Literal["new_task", "reached", "no_frontier", "nav_failed",
                  "timeout", "candidate_found", "abort"]
    stamp: float
    task_id: int | None = None
    subtask_id: int | None = None
    payload: dict = {}


class TargetReport(_Model):
    task_id: int
    target: str
    node_id: int
    label: str
    position: tuple[float, float, float]
    image_path: str
    stamp: float
