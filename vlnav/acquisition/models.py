"""Acquisition-only models (acquisition spec §5). Shared models come from `common.models`."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from vlnav.common.models import ObjectNode, Pose


class Candidate(BaseModel):
    node: ObjectNode
    match_score: float               # label/text similarity to the target
    rank_score: float                # score used for top-k ranking
    forced: bool = False             # force-included (proposed by the planner VLM)


class VerifyOutput(BaseModel):
    status: Literal["selected", "none", "error"]
    node_id: int | None = None
    letter: str | None = None
    reasoning: str | None = None
    confidence: float | None = None
    error: str | None = None
    latency_s: float = 0.0


class AcquisitionResult(BaseModel):
    status: Literal["selected", "rejected_all", "no_candidates", "error"]
    node_id: int | None
    goal_pose: Pose | None           # = node.best_view_pose when selected
    candidates: list[int]            # ids that were sent to the VLM (all rounds)
    vlm_reason: str | None
    vlm_confidence: float | None
    latency_s: dict[str, float]      # {"filter": ..., "vlm": ..., "total": ...}
