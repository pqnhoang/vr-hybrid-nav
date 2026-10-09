"""Typed view of `configs/mapping.yaml`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GroundCfg(_Cfg):
    mode: Literal["cell_min", "constant"]
    constant_z: float
    smooth_radius_m: float


class LogOddsCfg(_Cfg):
    hit: float
    miss: float
    clamp_min: float
    clamp_max: float
    occupied_threshold: float
    free_threshold: float


class DepthPointsCfg(_Cfg):
    stride: int
    min_depth_m: float
    max_depth_m: float


class ExternalCfg(_Cfg):
    occupied_threshold: int
    free_threshold: int


class MappingConfig(_Cfg):
    resolution: float
    initial_size_m: float
    grow_margin_m: float
    max_range_m: float
    max_points: int
    ray_angle_bins: int
    ray_step_factor: float
    obstacle_z_min: float
    obstacle_z_max: float
    negative_obstacles: bool
    seed: int
    ground: GroundCfg
    log_odds: LogOddsCfg
    depth_points: DepthPointsCfg
    external: ExternalCfg

    @model_validator(mode="after")
    def _check(self) -> MappingConfig:
        if self.negative_obstacles:
            raise ValueError("negative_obstacles=true is not implemented yet")
        lo = self.log_odds
        if not lo.clamp_min <= lo.free_threshold < lo.occupied_threshold <= lo.clamp_max:
            raise ValueError("log-odds thresholds must satisfy clamp_min <= free < occupied <= clamp_max")
        return self


def load_mapping_config(path: str | Path = CONFIG_DIR / "mapping.yaml",
                        overrides: dict | None = None) -> MappingConfig:
    return MappingConfig.model_validate(deep_merge(load_yaml(path), overrides))
