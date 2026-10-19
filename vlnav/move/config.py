"""Typed view of `configs/move.yaml` (TASK_move_subtask.md §11)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class S2Cfg(_Cfg):
    base_url: str
    timeout_s: float
    cache_dir: str | None
    blocking: bool


class ModelInputCfg(_Cfg):
    resize_w: int
    resize_h: int
    num_history: int


class ModelViewCfg(_Cfg):
    mode: Literal["habitat", "full"]
    width: int
    height: int
    hfov_deg: float


class HistoryCfg(_Cfg):
    step_m: float
    step_rad: float


class ActionsCfg(_Cfg):
    forward_step_m: float
    turn_step_rad: float
    rotate_mode: Literal["in_place", "offset"]
    rotate_offset_m: float


class ProjectionCfg(_Cfg):
    min_down_dz: float
    base_height_m: float | None
    ground_ring_m: tuple[float, float]
    ground_min_dist_m: float
    max_floor_step_m: float
    depth_window_px: int
    depth_search_px: int
    depth_k: int
    depth_min_m: float
    depth_max_m: float
    depth_margin_m: float
    depth_margin_rel: float
    obstacle_backoff_m: float
    min_range_m: float
    max_range_m: float
    fallback_range_m: float


class GoalCfg(_Cfg):
    walkback_step_m: float
    max_unreachable: int
    switch_dist_m: float


class LoopCfg(_Cfg):
    requery_radius_m: float
    requery_period_s: float | None
    stall_s: float
    stall_dist_m: float
    rotate_tol_rad: float
    timeout_s: float
    max_queries: int


class MoveVizCfg(_Cfg):
    map_px: int
    map_radius_m: float
    image_px: int
    video_fps: float


class MoveConfig(_Cfg):
    enabled: bool
    s2: S2Cfg
    model_input: ModelInputCfg
    model_view: ModelViewCfg
    main_pixel_ref: Literal["resized", "view"]
    look_down_mode: Literal["same_frame", "ignore"]
    max_look_down: int
    history: HistoryCfg
    actions: ActionsCfg
    projection: ProjectionCfg
    goal: GoalCfg
    loop: LoopCfg
    viz: MoveVizCfg


def load_move_config(path: str | Path | None = None, overrides: dict | None = None) -> MoveConfig:
    data = load_yaml(path or CONFIG_DIR / "move.yaml")
    return MoveConfig.model_validate(deep_merge(data, overrides))
