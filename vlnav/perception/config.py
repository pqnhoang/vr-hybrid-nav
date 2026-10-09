"""Typed view of `configs/perception.yaml` + `configs/checkpoints.yaml`."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DetectorCfg(_Cfg):
    device: str
    yolo_imgsz: int
    conf: float
    iou: float
    max_det: int
    fastsam_imgsz: int
    fastsam_conf: float
    fastsam_iou: float
    mask_min_iou: float          # below: fall back to the bbox as mask
    cache_dir: str | None


class VocabularyCfg(_Cfg):
    generic_vocab_file: str


class DynamicCfg(_Cfg):
    labels: list[str]            # detected but never lifted (e.g. person)
    max_bbox_overlap: float      # drop detections with |box ∩ dynamic box| / |box| above this


class LiftingCfg(_Cfg):
    mask_erode_px: int
    min_depth_m: float
    max_depth_m: float
    mad_k: float
    voxel_m: float
    min_points: int
    max_obs_range_m: float
    max_mask_pixels: int


class PerceptionConfig(_Cfg):
    detector: DetectorCfg
    vocabulary: VocabularyCfg
    dynamic: DynamicCfg
    lifting: LiftingCfg


class CheckpointPaths(_Cfg):
    yolo_world: str
    fastsam: str
    clip_text: str


def load_perception_config(path: str | Path = CONFIG_DIR / "perception.yaml",
                           overrides: dict | None = None) -> PerceptionConfig:
    return PerceptionConfig.model_validate(deep_merge(load_yaml(path), overrides))


def load_checkpoint_paths(path: str | Path = CONFIG_DIR / "checkpoints.yaml") -> CheckpointPaths:
    return CheckpointPaths.model_validate(load_yaml(path)["detector"])
