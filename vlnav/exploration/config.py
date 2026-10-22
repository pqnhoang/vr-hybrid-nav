"""Typed view of `configs/exploration.yaml` (TASK_nesy_exploration.md §11)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrontierCfg(_Cfg):
    sensor_range_m: float
    outside_fov: Literal["ignore", "zero_score"]
    initial_spin: bool
    frontier_connectivity: Literal[4, 8]
    min_cluster_cells: int
    small_cluster_cells: int
    sample_spacing_m: float
    inherit_radius_m: float


class VLCfg(_Cfg):
    sigma_rad: float
    vl_update_mode: Literal["first", "max", "conf_weighted"]
    rescore_on_new_subtask: bool


class InstanceCfg(_Cfg):
    tau_det: float
    instance_score: Literal["vl_eq3", "confidence"]
    inst_voxel_m: float
    inst_standoff_min_m: float
    inst_standoff_max_m: float
    visit_radius_m: float


class CuriosityCfg(_Cfg):
    unknown_bfs_radius_m: float
    unknown_k: float
    dist_metric: Literal["euclidean", "path"]


class RoomBonusCfg(_Cfg):
    enabled: bool
    w_room: float


class ScoringCfg(_Cfg):
    form: Literal["additive_dist", "multiplicative_unknown"]
    w_dist: float
    w_vl: float
    room_bonus: RoomBonusCfg
    w_yaw: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    yaw_change_window_s: float = Field(default=8.0, gt=0, allow_inf_nan=False)


class SelectionCfg(_Cfg):
    delta_reached_m: float
    switch_margin: float
    min_goal_hold_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    max_reach_checks: int
    max_plan_failures: int
    blacklist_s: float
    stuck_detection: bool
    stuck_dist_m: float
    stuck_window_s: float
    replan_period_s: float
    no_frontier_patience_s: float
    goto_reach_radius_m: float
    goto_yaw_tol_rad: float | None


class AblationCfg(_Cfg):
    no_ibtp: bool
    no_curiosity: bool
    nearest_frontier: bool


class HabitatCfg(_Cfg):
    scenes: str
    hfov_deg: float
    cam_height_m: float
    forward_m: float
    turn_deg: float
    max_steps: int
    stop_score: float
    stop_dist_m: float


class ExplorationVizCfg(_Cfg):
    map_px: int
    panel_px: int
    crop_margin_m: float
    top_k_text: int
    label_top_k: int
    video_fps: float
    keyframe_every: int


class ExplorationConfig(_Cfg):
    select_rate_hz: float
    angle_reference: Literal["robot", "camera"]
    max_camera_yaw_offset_deg: float
    frontier: FrontierCfg
    vl: VLCfg
    instance: InstanceCfg
    curiosity: CuriosityCfg
    scoring: ScoringCfg
    selection: SelectionCfg
    ablation: AblationCfg
    habitat: HabitatCfg
    viz: ExplorationVizCfg

    @model_validator(mode="after")
    def _check(self) -> ExplorationConfig:
        if self.frontier.initial_spin:
            raise ValueError("frontier.initial_spin needs a controller to turn in place "
                             "(Habitat harness, milestone E5); not available yet")
        i = self.instance
        if not 0 <= i.inst_standoff_min_m <= i.inst_standoff_max_m:
            raise ValueError("instance standoff: need 0 <= min <= max")
        return self

    # effective switches after ablations (spec §11)
    @property
    def ibtp_enabled(self) -> bool:
        return not (self.ablation.no_ibtp or self.ablation.nearest_frontier)

    @property
    def w_vl(self) -> float:
        return 0.0 if self.ablation.nearest_frontier else self.scoring.w_vl

    @property
    def w_dist(self) -> float:
        return 0.0 if self.ablation.no_curiosity else self.scoring.w_dist

    @property
    def curiosity_enabled(self) -> bool:
        return not self.ablation.no_curiosity


def load_exploration_config(path: str | Path = CONFIG_DIR / "exploration.yaml",
                            overrides: dict | None = None) -> ExplorationConfig:
    return ExplorationConfig.model_validate(deep_merge(load_yaml(path), overrides))


def add_ablation_args(ap) -> None:
    """CLI ablation switches of spec §11 (runners)."""
    g = ap.add_argument_group("exploration ablations (TASK_nesy_exploration.md §11)")
    g.add_argument("--no-ibtp", action="store_true", help="instance pool disabled")
    g.add_argument("--no-curiosity", action="store_true", help="frontiers: S_NeSy = w_vl * S_VL")
    g.add_argument("--nearest-frontier", action="store_true", help="w_vl = 0 and no IBTP")
    g.add_argument("--vl-update-mode", choices=["first", "max", "conf_weighted"])
    g.add_argument("--room-bonus", action="store_true", help="room-hint bonus (needs memory rooms)")


def ablation_overrides(args) -> dict:
    """Exploration config overrides from `add_ablation_args` flags (empty if none set)."""
    over: dict = {}
    abl = {k: True for k in ("no_ibtp", "no_curiosity", "nearest_frontier") if getattr(args, k, False)}
    if abl:
        over["ablation"] = abl
    if getattr(args, "vl_update_mode", None):
        over["vl"] = {"vl_update_mode": args.vl_update_mode}
    if getattr(args, "room_bonus", False):
        over["scoring"] = {"room_bonus": {"enabled": True}}
    return over
