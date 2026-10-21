"""Typed view of `configs/agent.yaml` (TASK_cap_mas.md §4, §6)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RoleCfg(_Cfg):
    vlm: dict | None                 # None: shared client; dict: overrides on configs/vlm.yaml


class PromptsCfg(_Cfg):
    planner: str
    coder: str
    observer: str


class AgentConfig(_Cfg):
    enabled: bool
    blocking: bool | None            # None: planner.blocking
    max_round: int
    max_repairs: int
    max_primitives: int
    coder_retries: int
    observer_veto: bool
    fallback: Literal["nesy", "none"]
    vlm_timeout_s: float
    log_dir: str | None
    prompts: PromptsCfg
    planner: RoleCfg
    coder: RoleCfg
    observer: RoleCfg


def load_agent_config(path: str | Path = CONFIG_DIR / "agent.yaml", overrides: dict | None = None) -> AgentConfig:
    return AgentConfig.model_validate(deep_merge(load_yaml(path), overrides))
