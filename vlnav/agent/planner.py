"""Planner agent (TASK_cap_mas.md §4.1): instruction + scene graph + trace + Observer feedback ->
{thought, plan: ["Step 1: ...", ...], done}. Text only."""
from __future__ import annotations

import math

from vlnav.agent.api import Place
from vlnav.agent.chat import ChatAgent, Reply
from vlnav.agent.state import WorldSnapshot

PLAN_SCHEMA = {"type": "object",
               "properties": {"thought": {"type": "string"},
                              "plan": {"type": "array", "items": {"type": "string"}},
                              "done": {"type": "boolean"}},
               "required": ["thought", "plan", "done"]}


def robot_text(here: Place | None) -> str:
    if here is None:
        return "position unknown (no pose yet)"
    room = f", in the {here.room.label}" if here.room is not None else ""
    return f"at ({here.xy[0]:.1f}, {here.xy[1]:.1f}) m, heading {math.degrees(here.yaw):.0f} deg{room}"


def normalize_plan(data: dict) -> dict:
    """{thought: str, plan: [str], done: bool}; plan items that are objects become their text."""
    steps = data.get("plan") or []
    if isinstance(steps, str):
        steps = [s for s in steps.splitlines() if s.strip()]
    steps = [str(s if not isinstance(s, dict) else s.get("step") or s.get("text") or s).strip() for s in steps]
    return {"thought": str(data.get("thought", "")).strip(), "plan": [s for s in steps if s],
            "done": bool(data.get("done", False))}


class PlannerAgent(ChatAgent):
    role = "planner"

    def plan(self, snap: WorldSnapshot, graph: str, robot: str, previous: list[str] | None,
             feedback: str | None) -> Reply:
        system = self.render("SYSTEM", move=snap.move_enabled)
        user = self.render("REQUEST", instruction=snap.instruction, robot=robot, graph=graph,
                           done=[t.text() for t in snap.trace], failure=snap.failure,
                           previous=previous, feedback=feedback)
        r = self.ask(system, user, PLAN_SCHEMA)
        if r.data is not None:
            r.data = normalize_plan(r.data)
        return r
