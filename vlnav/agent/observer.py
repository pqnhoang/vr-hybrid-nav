"""Observer agent (TASK_cap_mas.md §4.3): judges a program from its dry-run trace (before it runs)
or from the real trace (after it failed). Text only for now: no camera images.

Hard checks from code (mock_world.dry_run) can only turn VALID into INVALID, never the reverse.
"""
from __future__ import annotations

from vlnav.agent.chat import ChatAgent, Reply

CHECKS = ("target_match", "spatial_relation", "route_progress", "goal_reachable", "stuck_or_loop")
OBSERVE_SCHEMA = {"type": "object",
                  "properties": {"verdict": {"type": "string", "enum": ["VALID", "INVALID"]},
                                 "checklist": {"type": "object",
                                               "properties": {c: {"type": "string"} for c in CHECKS}},
                                 "error_logs": {"type": "string"},
                                 "summary": {"type": "string"}},
                  "required": ["verdict", "checklist", "error_logs", "summary"]}


def verdict(obs: dict | None, hard_failures: list[str]) -> bool:
    """VALID only if the Observer says so (or could not answer) and no hard check failed."""
    if hard_failures:
        return False
    if obs is None:
        return True                         # Observer unavailable: the hard checks decide
    return str(obs.get("verdict", "")).strip().upper() == "VALID"


class ObserverAgent(ChatAgent):
    role = "observer"

    def review(self, instruction: str, plan: list[str], code: str, trace: str, problems: list[str],
               move: bool) -> Reply:
        user = self.render("DRY_RUN", instruction=instruction, plan=plan, code=code, trace=trace,
                           problems=problems)
        return self.ask(self.render("SYSTEM", move=move), user, OBSERVE_SCHEMA)

    def diagnose(self, instruction: str, plan: list[str], trace: str, failure: str, move: bool) -> Reply:
        user = self.render("EXECUTION", instruction=instruction, plan=plan, trace=trace, failure=failure)
        return self.ask(self.render("SYSTEM", move=move), user, OBSERVE_SCHEMA)
