"""LLM room labeling from the objects of each room (TASK_unified_memory.md §8.3).

Paper: an LLM infers the room label from the object nodes connected to the room. The output is
constrained to `room_label_set + ["other"]`; responses are cached by (object multiset, area bucket).
First build: requests run synchronously inside `UnifiedMemory.tick`; the worker-thread queue of
the spec is a later step (the request/result API below is already shaped for it).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from vlnav.common.models import ObjectNode
from vlnav.memory.config import LabelingCfg
from vlnav.memory.types import UNKNOWN_LABEL

PROMPT_DIR = Path(__file__).parent / "prompts"
OTHER_LABEL = "other"


class LLMClient(Protocol):
    """Same call shape as `acquisition.vlm_client.VLMClient` (one VLM endpoint for the planner)."""

    def chat(self, messages: list[dict], json_schema: dict | None, timeout_s: float) -> str: ...


class MockLLMClient:
    """Scripted client for tests: a fixed reply, a list of replies, or a function of the messages."""

    def __init__(self, reply: str | list[str] | Callable[[list[dict]], str]):
        self._reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], json_schema: dict | None, timeout_s: float) -> str:
        self.calls.append(messages)
        if callable(self._reply):
            return self._reply(messages)
        if isinstance(self._reply, list):
            return self._reply[len(self.calls) - 1]
        return self._reply


@dataclass(frozen=True)
class RoomLabelRequest:
    room_id: int
    objects: tuple[tuple[str, int, float], ...]         # (label, count, mean best_score)
    area_m2: float
    adjacent_labels: tuple[str, ...]


@dataclass(frozen=True)
class RoomLabelResult:
    label: str
    confidence: float
    ok: bool
    from_cache: bool


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


class RoomLabeler:
    def __init__(self, cfg: LabelingCfg, llm: LLMClient, template: str = "room_label.j2"):
        self.cfg = cfg
        self.llm = llm
        self.allowed = [*cfg.room_label_set, OTHER_LABEL]
        env = Environment(loader=FileSystemLoader(PROMPT_DIR), undefined=StrictUndefined,
                          keep_trailing_newline=True)
        self._template = env.get_template(template)
        self._cache: dict[tuple, tuple[str, float]] = {}
        self.n_llm_calls = 0

    def schema(self) -> dict:
        return {"type": "object",
                "properties": {"label": {"type": "string", "enum": self.allowed},
                               "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["label", "confidence"]}

    def build_request(self, room_id: int, objects: list[ObjectNode], area_m2: float,
                      adjacent_labels: list[str]) -> RoomLabelRequest | None:
        """None when the room has fewer than room_label_min_objects objects (stays unknown)."""
        if len(objects) < self.cfg.room_label_min_objects:
            return None
        counts = Counter(o.label for o in objects)
        scores: dict[str, list[float]] = {}
        for o in objects:
            scores.setdefault(o.label, []).append(o.best_score)
        top = sorted(counts, key=lambda lbl: (-counts[lbl], lbl))[:self.cfg.room_label_max_objects]
        return RoomLabelRequest(
            room_id=room_id,
            objects=tuple((lbl, counts[lbl], sum(scores[lbl]) / len(scores[lbl])) for lbl in top),
            area_m2=area_m2,
            adjacent_labels=tuple(sorted(a for a in adjacent_labels if a != UNKNOWN_LABEL)))

    def _cache_key(self, req: RoomLabelRequest) -> tuple:
        multiset = tuple(sorted((lbl, n) for lbl, n, _ in req.objects))
        return multiset, int(req.area_m2 // self.cfg.cache_area_bucket_m2)

    def render(self, req: RoomLabelRequest) -> str:
        return self._template.render(objects=req.objects, area_m2=req.area_m2,
                                     adjacent_labels=req.adjacent_labels, allowed=self.allowed)

    def label(self, req: RoomLabelRequest) -> RoomLabelResult:
        key = self._cache_key(req)
        if key in self._cache:
            lbl, conf = self._cache[key]
            return RoomLabelResult(lbl, conf, ok=True, from_cache=True)
        messages = [{"role": "user", "content": self.render(req)}]
        self.n_llm_calls += 1
        try:
            raw = self.llm.chat(messages, self.schema(), self.cfg.timeout_s)
            data = json.loads(_strip_fences(raw))
            lbl = str(data["label"]).strip().lower()
            conf = min(1.0, max(0.0, float(data["confidence"])))
        except Exception:                      # timeout, transport or parse error -> stay unknown
            return RoomLabelResult(UNKNOWN_LABEL, 0.0, ok=False, from_cache=False)
        if lbl not in self.allowed:
            lbl = OTHER_LABEL
        self._cache[key] = (lbl, conf)
        return RoomLabelResult(lbl, conf, ok=True, from_cache=False)
