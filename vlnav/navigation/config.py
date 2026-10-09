"""Typed view of `configs/navigation.yaml`."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class NavigationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    robot_radius_m: float
    unknown_cost: float
    goal_snap_radius_m: float
    start_snap_radius_m: float


def load_navigation_config(path: str | Path = CONFIG_DIR / "navigation.yaml",
                           overrides: dict | None = None) -> NavigationConfig:
    return NavigationConfig.model_validate(deep_merge(load_yaml(path), overrides))
