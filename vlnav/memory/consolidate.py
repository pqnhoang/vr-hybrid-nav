"""Intra-frame merging of duplicate detections (TASK_unified_memory.md §6.2) [DESIGN].

YOLO-World often returns several boxes with different vocabulary labels for one object
("jacket" + "raincoat"). Observations of one frame whose masks overlap by
`same_object_mask_iou` are merged into one `ObsGroup` before association.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vlnav.common.models import Observation


@dataclass(slots=True)
class ObsGroup:
    """One consolidated observation (one physical object in one frame)."""

    stamp: float
    camera_id: str
    label: str                                    # label of the highest-score member
    label_votes: dict[str, float]                 # label -> max member score
    score: float
    points: np.ndarray                            # Mx3 union of member points
    centroid: np.ndarray                          # (3,)
    bbox: tuple[int, int, int, int]               # bbox of the highest-score member
    T_map_cam: np.ndarray
    T_map_base: np.ndarray
    image_hw: tuple[int, int]
    n_members: int


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray, box_a=None, box_b=None) -> float:
    """IoU of two masks; with boxes, computed only inside their union box (masks must lie inside
    their boxes, as produced by the detector) and 0 for disjoint boxes."""
    if box_a is not None and box_b is not None:
        if min(box_a[2], box_b[2]) <= max(box_a[0], box_b[0]) or \
                min(box_a[3], box_b[3]) <= max(box_a[1], box_b[1]):
            return 0.0
        x0, y0 = min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
        x1, y1 = max(box_a[2], box_b[2]), max(box_a[3], box_b[3])
        a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def consolidate(observations: list[Observation], masks: list[np.ndarray | None],
                image_hw: tuple[int, int], iou_thr: float) -> list[ObsGroup]:
    """Group observations of one frame by mask IoU (bbox IoU when a mask is missing)."""
    n = len(observations)
    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = masks[i], masks[j]
            if mi is not None and mj is not None:
                iou = mask_iou(mi, mj, observations[i].bbox, observations[j].bbox)
            else:
                iou = bbox_iou(observations[i].bbox, observations[j].bbox)
            if iou >= iou_thr:
                parent[_find(parent, j)] = _find(parent, i)

    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(_find(parent, i), []).append(i)

    groups = []
    for idx in sorted(members.values(), key=lambda m: m[0]):
        obs = [observations[i] for i in idx]
        top = max(obs, key=lambda o: o.score)       # first max wins -> deterministic
        votes: dict[str, float] = {}
        for o in obs:
            votes[o.label] = max(votes.get(o.label, 0.0), o.score)
        if len(obs) == 1:
            points = np.asarray(top.points_map, dtype=float)
            centroid = np.asarray(top.centroid_map, dtype=float)
        else:
            points = np.concatenate([np.asarray(o.points_map, dtype=float) for o in obs])
            centroid = np.median(points, axis=0)
        groups.append(ObsGroup(
            stamp=top.stamp, camera_id=top.camera_id, label=top.label, label_votes=votes,
            score=top.score, points=points, centroid=centroid, bbox=tuple(top.bbox),
            T_map_cam=top.T_map_cam, T_map_base=top.T_map_base, image_hw=image_hw,
            n_members=len(obs)))
    return groups
