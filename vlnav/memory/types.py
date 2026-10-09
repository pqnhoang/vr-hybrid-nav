"""Internal memory models (TASK_unified_memory.md §4.2). Never exposed to consumers directly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from vlnav.common.models import ObjectNode, RoomNode

UNKNOWN_LABEL = "unknown"      # room label before / without LLM labeling

MemoryEventType = Literal["node_created", "node_merged", "best_view_updated", "node_pruned",
                          "nodes_deduped", "rooms_updated", "room_labeled"]


@dataclass(slots=True)
class NodeState:
    """Mutable per-node state. `node` is replaced (never mutated) on change -> safe to share."""

    node: ObjectNode
    voxels: dict[tuple[int, int, int], None]      # insertion-ordered set at assoc_voxel_m
    label_votes: dict[str, float]
    best_view_history: list[tuple[float, str]]    # (t, image path)
    best_q: float                                 # quality of the current best view (§6.5)
    confirmed: bool
    created_t: float
    updated_t: float


@dataclass(slots=True)
class RoomState:
    room: RoomNode                                # replaced on change, like NodeState.node
    polygon: list[tuple[float, float]]            # world xy, for JSON / viz
    object_set_at_label: frozenset[int]           # confirmed objects used for the last label
    needs_label: bool


class MemoryEvent(BaseModel):
    """For logging / debugging; not consumed by the planner."""

    t: float
    version: int
    type: MemoryEventType
    payload: dict


class MemoryStats(BaseModel):
    version: int
    n_objects: int
    n_confirmed: int
    n_tentative: int
    n_rooms: int
    n_frames_processed: int
    n_frames_skipped: int
    n_obs_gated: int
    n_checkpoints: int
    update_ms_p50: float | None
    update_ms_p95: float | None
    room_seg_ms_p95: float | None
