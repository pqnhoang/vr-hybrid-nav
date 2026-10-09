"""Per-candidate visual + textual context for the VLM (acquisition spec §8.1)."""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

from vlnav.acquisition.config import ContextCfg
from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.models import Candidate
from vlnav.common.images import draw_bbox, limit_size, load_rgb


@dataclass
class CandidateContext:
    letter: str
    node_id: int
    crop: np.ndarray
    frame: np.ndarray | None
    text: str


def letters(n: int) -> list[str]:
    if n > len(string.ascii_uppercase):
        raise ValueError(f"at most {len(string.ascii_uppercase)} candidates")
    return list(string.ascii_uppercase[:n])


class ContextBuilder:
    def __init__(self, cfg: ContextCfg):
        self.cfg = cfg

    def text(self, c: Candidate, memory: MemoryView, use_neighbors: bool = True) -> str:
        n = c.node
        room = memory.get_room(n.room_id) if n.room_id is not None else None
        parts = [f"detector label: {n.label}", f"detection confidence: {n.best_score:.2f}",
                 f"room: {room.label if room else 'unknown'}"]
        if use_neighbors:
            near = memory.neighbors(n.id, self.cfg.neighbor_radius_m, self.cfg.neighbor_same_room)
            near = near[:self.cfg.max_neighbors]
            if near:
                c0 = np.asarray(n.centroid)
                desc = ", ".join(f"{o.label} ({np.linalg.norm(np.asarray(o.centroid) - c0):.1f} m)"
                                 for o in near)
                parts.append(f"on/near: {desc}")
        return "; ".join(parts)

    def build(self, candidates: list[Candidate], memory: MemoryView, include_full_frame: bool | None = None,
              use_neighbors: bool = True) -> list[CandidateContext]:
        full = self.cfg.include_full_frame if include_full_frame is None else include_full_frame
        out = []
        for letter, c in zip(letters(len(candidates)), candidates):
            n = c.node
            crop = memory.best_view_crop(n.id, self.cfg.crop_margin, self.cfg.crop_max_px)
            frame = None
            if full:
                img = draw_bbox(load_rgb(n.best_view_image_path), n.best_view_bbox,
                                self.cfg.bbox_color, self.cfg.bbox_thickness)
                frame = limit_size(img, self.cfg.frame_max_px)
            out.append(CandidateContext(letter, n.id, np.asarray(crop), frame,
                                        self.text(c, memory, use_neighbors)))
        return out
