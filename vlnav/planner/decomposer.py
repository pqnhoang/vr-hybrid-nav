"""Decomposition and replanning VLM queries (planner spec §5.3, §5.4).

Both send the serialized scene graph and — because the paper uses the scene graph *and* image
memory for decomposition and replanning (AUDIT 2.4) — best-view crops of up to `planning_images`
objects most relevant to the instruction / pending targets, each tagged with its #id.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.vlm_client import VLMClient, extract_json, image_part, strip_images, text_part
from vlnav.planner.config import PlannerConfig
from vlnav.planner.serializer import ExplorationSummary, GraphSerializer
from vlnav.planner.state import PlannerState

PROMPT_DIR = Path(__file__).parent / "prompts"
REPAIR_MESSAGE = "Your previous answer was not valid JSON. Return valid JSON only, matching the schema."

_SUBTASK = {"type": "object",
            "properties": {"type": {"type": "string", "enum": ["explore", "goto"]},
                           "target": {"type": "string"},
                           "node_id": {"type": ["integer", "null"]},
                           "room_hint": {"type": ["string", "null"]}},
            "required": ["type", "target"]}
_TARGET = {"type": "object",
           "properties": {"name": {"type": "string"},
                          "detector_prompts": {"type": "array", "items": {"type": "string"}},
                          "room_priors": {"type": "array", "items": {"type": "string"}}},
           "required": ["name", "detector_prompts"]}
DECOMPOSE_SCHEMA = {"type": "object",
                    "properties": {"reasoning": {"type": "string"},
                                   "targets": {"type": "array", "items": _TARGET},
                                   "subtasks": {"type": "array", "items": _SUBTASK}},
                    "required": ["reasoning", "targets", "subtasks"]}
REPLAN_SCHEMA = {"type": "object",
                 "properties": {"reasoning": {"type": "string"},
                                "task_done": {"type": "boolean"},
                                "target_updates": {"type": "array", "items": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"},
                                                   "status": {"type": "string", "enum": ["failed"]},
                                                   "reason": {"type": "string"}},
                                    "required": ["name", "status"]}},
                                "subtasks": {"type": "array", "items": _SUBTASK}},
                 "required": ["reasoning", "task_done", "subtasks"]}


@dataclass
class QueryResult:
    data: dict | None                      # parsed JSON (None: malformed twice / error)
    raws: list[str] = field(default_factory=list)
    error: str | None = None
    prompt_hash: str = ""
    graph: str = ""
    image_ids: list[int] = field(default_factory=list)
    latency_s: float = 0.0
    prompt: list[dict] = field(default_factory=list)
    prompt_text: str = ""                  # full text sent (debug dumps)
    crops: dict = field(default_factory=dict)   # node id -> RGB crop sent (debug dumps)


class PlanQuery:
    def __init__(self, cfg: PlannerConfig, vlm: VLMClient, serializer: GraphSerializer):
        self.cfg = cfg
        self.vlm = vlm
        self.serializer = serializer
        # no trim_blocks: it swallowed the newline after inline {% endif %} and glued list lines
        self.env = Environment(loader=FileSystemLoader(PROMPT_DIR), undefined=StrictUndefined,
                               keep_trailing_newline=True)
        self.planning_images = cfg.planning_images        # ablation: 0 = text only

    def select_images(self, memory: MemoryView, texts: list[str]) -> list[int]:
        if self.planning_images <= 0:
            return []
        order = self.serializer.ordered(memory, texts)
        return [n.id for n, _ in order[:self.planning_images]]

    def _ask(self, prompt: str, memory: MemoryView, image_ids: list[int], schema: dict,
             graph: str) -> QueryResult:
        t0 = time.perf_counter()
        content = [text_part(prompt)]
        crops = {}
        for i in image_ids:
            n = memory.get_object(i)
            try:
                crop = memory.best_view_crop(i, 0.15, self.cfg.planning_image_px)
            except (FileNotFoundError, RuntimeError):
                continue
            crops[n.id] = crop
            content += [text_part(f"Image of #{n.id} ({n.label}):"), image_part(crop)]
        messages = [{"role": "user", "content": content}]
        blob = json.dumps(strip_images(messages), sort_keys=True)
        res = QueryResult(None, prompt_hash=hashlib.sha256(blob.encode()).hexdigest()[:16], graph=graph,
                          image_ids=list(crops), prompt=strip_images(messages),
                          prompt_text="\n".join(p["text"] for p in content if p["type"] == "text"),
                          crops=crops)
        convo = list(messages)
        for _ in range(2):                                # one repair retry (spec §5.5)
            try:
                raw = self.vlm.chat(convo, schema, self.cfg.vlm_timeout_s)
            except Exception as e:
                res.error = f"{type(e).__name__}: {e}"
                break
            res.raws.append(raw)
            try:
                res.data = extract_json(raw)
                break
            except ValueError:
                convo = [*convo, {"role": "assistant", "content": raw},
                         {"role": "user", "content": REPAIR_MESSAGE}]
        if res.data is None and res.error is None:
            res.error = "malformed JSON twice"
        res.latency_s = time.perf_counter() - t0
        return res


class Decomposer(PlanQuery):
    def decompose(self, instruction: str, memory: MemoryView,
                  exploration: ExplorationSummary | None = None) -> QueryResult:
        graph = self.serializer.serialize(memory, [instruction], exploration)
        images = self.select_images(memory, [instruction])
        prompt = self.env.get_template("decompose.j2").render(
            instruction=instruction, graph=graph, n_images=len(images))
        return self._ask(prompt, memory, images, DECOMPOSE_SCHEMA, graph)


class Replanner(PlanQuery):
    def replan(self, state: PlannerState, last_event: str, memory: MemoryView,
               exploration: ExplorationSummary | None = None) -> QueryResult:
        active = [t.name for t in state.targets if t.active]
        graph = self.serializer.serialize(memory, active, exploration)
        images = self.select_images(memory, active)
        prompt = self.env.get_template("replan.j2").render(
            instruction=state.instruction, targets=state.targets, history=state.history,
            last_event=last_event, graph=graph, n_images=len(images))
        return self._ask(prompt, memory, images, REPLAN_SCHEMA, graph)
