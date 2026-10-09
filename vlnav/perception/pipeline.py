"""One perception pass per frame (SYSTEM.md §1, §4.2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vlnav.common.models import Frame, FrameResult, Observation
from vlnav.perception.config import DynamicCfg, LiftingCfg
from vlnav.perception.detector import OpenVocabDetector
from vlnav.perception.lidar_depth import sparse_depth
from vlnav.perception.lifting import lift_mask


def load_vocab_file(path: str | Path) -> list[str]:
    lines = (ln.split("#", 1)[0].strip() for ln in Path(path).read_text().splitlines())
    return [ln for ln in lines if ln]


class VocabularyManager:
    """generic_vocab ∪ active prompts (∪ dynamic labels); the detector vocabulary is set only on
    change."""

    def __init__(self, generic: list[str], dynamic: list[str] | None = None):
        self.generic = list(dict.fromkeys([*generic, *(dynamic or [])]))
        self.active: list[str] = []

    def set_active_prompts(self, prompts: list[str]) -> None:
        self.active = list(dict.fromkeys(prompts))

    def vocabulary(self) -> list[str]:
        return list(dict.fromkeys([*self.generic, *self.active]))

    def is_relevant(self, label: str) -> bool:
        return label in self.active


def box_overlap(a, b) -> float:
    """|a ∩ b| / |a| for xyxy boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


class PerceptionPipeline:
    def __init__(self, detector: OpenVocabDetector, vocab: VocabularyManager, lifting: LiftingCfg,
                 dynamic: DynamicCfg | None = None, seed: int = 0):
        self.detector = detector
        self.vocab = vocab
        self.lifting = lifting
        self.dynamic = dynamic
        self.n_dynamic_dropped = 0
        self.last_status: list[str] = []    # per detection of the last frame (debug / viz)
        self.last_depth: np.ndarray | None = None
        self.rng = np.random.default_rng(seed)
        self.detector.set_vocabulary(vocab.vocabulary())

    def set_active_prompts(self, prompts: list[str]) -> None:
        self.vocab.set_active_prompts(prompts)
        self.detector.set_vocabulary(self.vocab.vocabulary())

    def process(self, frame: Frame) -> FrameResult:
        if frame.depth is not None:
            depth = frame.depth
        else:                                       # no depth camera: project the LiDAR window
            depth = sparse_depth(frame.lidar_points_map, frame.T_map_cam, frame.K,
                                 frame.rgb.shape[:2], self.lifting.min_depth_m,
                                 self.lifting.max_depth_m)
        self.last_depth = depth
        key = f"{frame.camera_id}_{frame.stamp:.3f}"
        dets = self.detector.detect(frame.rgb, key=key)
        dyn = set(self.dynamic.labels) if self.dynamic else set()
        dyn_boxes = [d.bbox for d in dets if d.label in dyn]
        observations = []
        self.last_status = []
        for d in dets:
            if d.label in dyn:
                self.last_status.append("dynamic")
                continue
            if any(box_overlap(d.bbox, b) > self.dynamic.max_bbox_overlap for b in dyn_boxes):
                self.n_dynamic_dropped += 1
                self.last_status.append("on_dynamic")
                continue
            lifted = lift_mask(d.mask, depth, frame.K, frame.T_map_cam, self.lifting, self.rng)
            if lifted is None:
                self.last_status.append("no_3d")
                continue
            self.last_status.append("lifted")
            observations.append(Observation(
                stamp=frame.stamp, camera_id=frame.camera_id, label=d.label, score=d.score,
                centroid_map=tuple(float(v) for v in lifted.centroid_map),
                points_map=lifted.points_map, bbox=d.bbox, T_map_cam=frame.T_map_cam,
                T_map_base=frame.T_map_base, frame_ref=key))
        relevant = [i for i, d in enumerate(dets) if self.vocab.is_relevant(d.label)]
        return FrameResult(frame=frame, detections=dets, observations=observations,
                           relevant_idx=relevant)
