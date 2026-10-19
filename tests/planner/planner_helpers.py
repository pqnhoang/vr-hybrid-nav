"""Planner test helpers: fixture memory (shared with acquisition) + a prompt-routing mock VLM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "acquisition"))
from acq_helpers import cfg as acq_cfg, matcher, memory  # noqa: E402,F401

from vlnav.acquisition.acquirer import TargetAcquirer  # noqa: E402
from vlnav.acquisition.vlm_client import MockVLMClient  # noqa: E402
from vlnav.common.models import PlannerEvent  # noqa: E402
from vlnav.planner.config import load_planner_config  # noqa: E402
from vlnav.planner.core import PlannerCore  # noqa: E402

INSTR = "It is raining today. Find an umbrella and a rain jacket for Rob."


def text_of(messages) -> str:
    out = []
    for m in messages:
        c = m["content"]
        out += [c] if isinstance(c, str) else [p["text"] for p in c if p["type"] == "text"]
    return "\n".join(out)


class Router:
    """Scripted VLM: separate reply queues for decompose / replan / verify prompts."""

    def __init__(self, decompose=(), replan=(), verify=()):
        self.q = {"decompose": list(decompose), "replan": list(replan), "verify": list(verify)}
        self.log: list[tuple[str, str]] = []

    def __call__(self, messages):
        t = text_of(messages)
        kind = ("verify" if "perception module" in t else
                "decompose" if "Plan the task" in t else "replan")
        self.log.append((kind, t))
        if not self.q[kind]:
            raise AssertionError(f"unexpected {kind} call")
        r = self.q[kind].pop(0)
        return r if isinstance(r, Exception) else (r if isinstance(r, str) else json.dumps(r))


def plan(targets, subtasks, **extra):
    return {"reasoning": "r", "targets": targets, "subtasks": subtasks, **extra}


def target(name, prompts=None):
    return {"name": name, "detector_prompts": prompts or [name], "room_priors": []}


def explore(name, hint=None):
    return {"type": "explore", "target": name, "node_id": None, "room_hint": hint}


def goto(name, node):
    return {"type": "goto", "target": name, "node_id": node, "room_hint": None}


def move(label, instruction):
    return {"type": "move", "target": label, "instruction": instruction, "node_id": None, "room_hint": None}


def verify(choice, conf=0.9):
    return {"reasoning": "ok", "choice": choice, "confidence": conf}


def make_planner(tmp_path, router: Router, blocking=True, mem=None, move=False, **over):
    pcfg = load_planner_config(overrides={"blocking": blocking, "decision_log": str(tmp_path / "decisions.jsonl"),
                                          **over})
    vlm = MockVLMClient(router)
    mem = mem or memory(tmp_path)
    acq = TargetAcquirer(acq_cfg(), matcher(), vlm)
    return PlannerCore(pcfg, mem, acq, vlm, matcher(), move_enabled=move), mem


def event(planner, kind, t, **payload):
    s = planner.state()
    return PlannerEvent(type=kind, stamp=t, task_id=s.task_id,
                        subtask_id=s.current.id if s.current else None, payload=payload)
