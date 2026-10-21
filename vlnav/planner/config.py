"""Typed view of `configs/planner.yaml`."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class PlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    blocking: bool
    max_graph_tokens: int
    chars_per_token: int
    relevant_sim_threshold: float
    planning_images: int
    planning_image_px: int
    n_context_images: int
    check_period_s: float
    explore_candidate_score: float
    explore_timeout_s: float
    goto_timeout_s: float
    task_timeout_s: float
    max_attempts_per_target: int
    vlm_timeout_s: float
    decision_log: str | None
    debug_dumps: bool                # one folder per decompose / replan next to the decision log
    prompts: str                     # module in prompt/ with DECOMPOSE + REPLAN, e.g. planner_prompt_v1


def load_planner_config(path: str | Path = CONFIG_DIR / "planner.yaml",
                        overrides: dict | None = None) -> PlannerConfig:
    return PlannerConfig.model_validate(deep_merge(load_yaml(path), overrides))
