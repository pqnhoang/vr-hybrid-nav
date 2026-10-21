"""CaP agent test helpers: fixture memory (acquisition), a role-routing mock VLM and a runtime."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "planner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "acquisition"))
from acq_helpers import cfg as acq_cfg, matcher, memory  # noqa: E402,F401
from planner_helpers import text_of  # noqa: E402

from vlnav.acquisition.acquirer import TargetAcquirer  # noqa: E402
from vlnav.acquisition.vlm_client import MockVLMClient  # noqa: E402
from vlnav.agent.coder import CoderAgent  # noqa: E402
from vlnav.agent.config import load_agent_config  # noqa: E402
from vlnav.agent.mas import Orchestrator  # noqa: E402
from vlnav.agent.observer import ObserverAgent  # noqa: E402
from vlnav.agent.planner import PlannerAgent  # noqa: E402
from vlnav.agent.runtime import ProgramRuntime  # noqa: E402
from vlnav.agent.state import WorldSnapshot  # noqa: E402
from vlnav.common.models import PlannerEvent  # noqa: E402
from vlnav.mapping.occupancy import OccupancyGrid2D  # noqa: E402
from vlnav.navigation import load_navigation_config  # noqa: E402
from vlnav.planner.config import load_planner_config  # noqa: E402
from vlnav.planner.core import PlannerCore  # noqa: E402
from vlnav.planner.serializer import GraphSerializer  # noqa: E402

KINDS = (("nesy", "Plan the task"), ("planner", "task planner of a mobile robot"), ("coder", "translate a robot plan"),
         ("observer", "review the program"), ("verify", "perception module"), ("check", "Question:"))


class AgentRouter:
    """Scripted VLM: one reply queue per role (planner / coder / observer / verify / check / nesy)."""

    def __init__(self, **queues):
        self.q = {k: list(queues.get(k, ())) for k, _ in KINDS}
        self.log: list[tuple[str, str]] = []

    def __call__(self, messages):
        t = text_of(messages)
        kind = next((k for k, key in KINDS if key in t), "nesy-replan")
        self.log.append((kind, t))
        if not self.q.get(kind):
            raise AssertionError(f"unexpected {kind} call:\n{t[-400:]}")
        r = self.q[kind].pop(0)
        return r if isinstance(r, Exception) else (r if isinstance(r, str) else json.dumps(r))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.log]


def plan(*steps, thought="t", done=False):
    return {"thought": thought, "plan": list(steps), "done": done}


def code(body: str) -> str:
    return "```python\ndef task(api):\n" + "\n".join("    " + line for line in body.strip().splitlines()) + "\n```"


def observe(valid=True, summary="ok"):
    return {"verdict": "VALID" if valid else "INVALID",
            "checklist": {"target_match": "yes", "spatial_relation": "yes", "route_progress": "yes",
                          "goal_reachable": "yes", "stuck_or_loop": "no"},
            "error_logs": "", "summary": summary}


def verify(choice, conf=0.9):
    return {"reasoning": "ok", "choice": choice, "confidence": conf}


def make_runtime(tmp_path, router: AgentRouter, move=False, blocking=True, mem=None, agent=None, fallback=False,
                 **planner_over):
    pcfg = load_planner_config(overrides={"blocking": blocking, "decision_log": str(tmp_path / "decisions.jsonl"),
                                          "explore_candidate_score": 0.35, **planner_over})
    acfg = load_agent_config(overrides={"enabled": True, **(agent or {})})
    ncfg = load_navigation_config()
    vlm = MockVLMClient(router)
    mem = mem or memory(tmp_path)
    m = matcher()
    acq = TargetAcquirer(acq_cfg(), m, vlm)
    fallback = PlannerCore(pcfg, mem, acq, vlm, m, move_enabled=move) if fallback else None
    rt = None

    def log(kind, **f):
        rt.log(kind, **f)
    mas = Orchestrator(acfg, PlannerAgent(vlm, acfg.prompts.planner, 10), CoderAgent(vlm, acfg.prompts.coder, 10),
                       ObserverAgent(vlm, acfg.prompts.observer, 10), acq.filter, GraphSerializer(pcfg, m), ncfg,
                       log, tmp_path / "rounds.jsonl")
    rt = ProgramRuntime(acfg, pcfg, ncfg, mem, empty_grid(), acq, vlm, mas, fallback,
                        robot_fn=lambda: (0.0, 0.0, 0.0), move_enabled=move)
    return rt, mem


def empty_grid() -> OccupancyGrid2D:
    return OccupancyGrid2D(0.1, (0.0, 0.0), (0, 0))


def snapshot(mem, instruction, move=True, robot=(0.0, 0.0, 0.0), grid=None, failure=None) -> WorldSnapshot:
    return WorldSnapshot(version=mem.version(), stamp=200.0, instruction=instruction, robot=robot, memory=mem,
                         grid=grid or empty_grid(), move_enabled=move, failure=failure)


def event(rt, kind, t, **payload):
    s = rt.state()
    return PlannerEvent(type=kind, stamp=t, task_id=s.task_id,
                        subtask_id=s.current.id if s.current else None, payload=payload)


def decisions(tmp_path) -> list[dict]:
    return [json.loads(x) for x in (tmp_path / "decisions.jsonl").read_text().splitlines()]
