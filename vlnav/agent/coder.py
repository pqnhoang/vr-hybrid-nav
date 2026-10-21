"""Coder agent (TASK_cap_mas.md §4.2): plan -> one fenced `def task(api):`. It only translates;
the NavAPI docstring (api.api_doc) is its whole tool documentation."""
from __future__ import annotations

from vlnav.agent.api import api_doc
from vlnav.agent.chat import ChatAgent, Reply
from vlnav.agent.validate import ProgramError, extract_code, validate


class CoderAgent(ChatAgent):
    role = "coder"

    def code(self, instruction: str, plan: list[str], move: bool, error: str | None = None,
             previous_code: str | None = None) -> tuple[Reply, str | None, str | None]:
        """(reply, program, error): program is validated; else error says why (back to the Coder)."""
        system = self.render("SYSTEM", api_doc=api_doc(move), move=move)
        user = self.render("REQUEST", instruction=instruction, plan=plan, error=error,
                           previous_code=previous_code)
        r = self.ask(system, user, None)
        if r.error is not None:
            return r, None, r.error
        try:
            code = extract_code(r.raw)
        except ProgramError as e:
            return r, None, str(e)
        try:
            validate(code)
        except ProgramError as e:
            return r, code, str(e)
        return r, code, None
