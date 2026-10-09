"""Typed view of `configs/memory.yaml`. Values live in the YAML only (no defaults in code)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml

DEFAULT_MEMORY_YAML = CONFIG_DIR / "memory.yaml"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GatingCfg(_Cfg):
    min_obs_score: float
    min_points: int
    max_obs_range_m: float
    min_bbox_area_px: float
    ignore_labels: list[str]


class ConsolidationCfg(_Cfg):
    same_object_mask_iou: float


class AssociationCfg(_Cfg):
    assoc_radius_m: float
    overlap_norm: Literal["observation", "min"]
    assoc_label_sim: float
    assoc_voxel_m: float
    assoc_overlap_min: float
    assoc_close_m: float
    w_dist: float


class NodesCfg(_Cfg):
    max_node_voxels: int
    min_obs: int
    confirm_score: float


class BestViewCfg(_Cfg):
    border_penalty: float
    small_penalty: float
    best_view_min_area_px: float
    best_view_min_gain: float
    best_view_require_free: bool


class MaintenanceCfg(_Cfg):
    dedup_every_s: float
    dedup_overlap_min: float
    prune_every_s: float
    prune_age_s: float
    negative_evidence: bool


class RoomsCfg(_Cfg):
    enabled: bool
    room_update_every_s: float
    room_update_min_new_m2: float
    room_grid_res_m: float
    closing_kernel_m: float
    min_free_comp_m2: float
    door_width_m: float
    min_seed_area_m2: float
    min_room_area_m2: float
    min_shared_border_m: float
    room_iou_min: float
    relabel_iou: float
    relabel_min_new_objects: int
    room_assign_fallback_m: float
    room_alias_overlap_min: float


class LabelingCfg(_Cfg):
    room_label_set: list[str]
    room_label_max_objects: int
    room_label_min_objects: int
    timeout_s: float
    cache_area_bucket_m2: float
    worker_thread: bool


class QueriesCfg(_Cfg):
    max_neighbors: int


class ImagesCfg(_Cfg):
    image_jpeg_quality: int
    crop_cache_size: int


class CheckpointsCfg(_Cfg):
    checkpoint_every_s: float
    max_checkpoints: int
    exact_snapshots: bool


class StatsCfg(_Cfg):
    window: int


class MemoryConfig(_Cfg):
    memory_rate_hz: float
    gating: GatingCfg
    consolidation: ConsolidationCfg
    association: AssociationCfg
    nodes: NodesCfg
    best_view: BestViewCfg
    maintenance: MaintenanceCfg
    rooms: RoomsCfg
    labeling: LabelingCfg
    queries: QueriesCfg
    images: ImagesCfg
    checkpoints: CheckpointsCfg
    stats: StatsCfg

    @model_validator(mode="after")
    def _not_yet_implemented(self) -> MemoryConfig:
        if self.checkpoints.exact_snapshots:
            raise ValueError("checkpoints.exact_snapshots=true is not implemented yet")
        if self.maintenance.negative_evidence:
            raise ValueError("maintenance.negative_evidence=true is not implemented yet")
        return self


def load_memory_config(path: str | Path = DEFAULT_MEMORY_YAML,
                       overrides: dict[str, Any] | None = None) -> MemoryConfig:
    return MemoryConfig.model_validate(deep_merge(load_yaml(path), overrides))
