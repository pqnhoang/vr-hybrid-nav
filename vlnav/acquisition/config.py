"""Typed view of `configs/acquisition.yaml`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from vlnav.common.config import CONFIG_DIR, deep_merge, load_yaml


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SymbolicCfg(_Cfg):
    k: int
    rank_strategy: Literal["confidence", "confidence_x_similarity"]
    min_det_score: float
    label_sim_threshold: float
    label_rel_margin: float | None
    text_encoder: str
    text_encoder_path: str
    text_template: str
    device: str
    synonyms: dict[str, list[str]]


class ContextCfg(_Cfg):
    crop_margin: float
    crop_max_px: int
    include_full_frame: bool
    frame_max_px: int
    bbox_color: tuple[int, int, int]
    bbox_thickness: int
    neighbor_radius_m: float
    neighbor_same_room: bool
    max_neighbors: int


class VerifierCfg(_Cfg):
    timeout_s: float
    min_vlm_confidence: float


class AcquirerCfg(_Cfg):
    max_rounds: int


class LoggingCfg(_Cfg):
    debug_jsonl: str | None


class AcquisitionConfig(_Cfg):
    symbolic: SymbolicCfg
    context: ContextCfg
    verifier: VerifierCfg
    acquirer: AcquirerCfg
    logging: LoggingCfg


def load_acquisition_config(path: str | Path = CONFIG_DIR / "acquisition.yaml",
                            overrides: dict | None = None) -> AcquisitionConfig:
    return AcquisitionConfig.model_validate(deep_merge(load_yaml(path), overrides))
