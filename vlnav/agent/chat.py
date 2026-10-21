"""Shared chat plumbing of the three agents: prompt module loading, rendering, JSON replies with
one repair retry (same rule as the NeSy planner, planner spec §5.5)."""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field

from jinja2 import Environment, StrictUndefined

from vlnav.acquisition.vlm_client import VLMClient, extract_json
from vlnav.planner.decomposer import REPAIR_MESSAGE, _text


@dataclass
class Reply:
    role: str
    data: dict | None = None             # parsed JSON (JSON replies only)
    raws: list[str] = field(default_factory=list)
    error: str | None = None
    latency_s: float = 0.0
    prompt_text: str = ""

    @property
    def raw(self) -> str:
        return self.raws[-1] if self.raws else ""

    def log(self) -> dict:
        """Decision-log `query` record (the planner metrics count lines with `query.raw`)."""
        return {"role": self.role, "prompt_text": self.prompt_text, "raw": self.raws, "parsed": self.data,
                "error": self.error, "vlm_latency_s": round(self.latency_s, 3)}


class ChatAgent:
    role = "agent"

    def __init__(self, vlm: VLMClient, prompt_module: str, timeout_s: float):
        self.vlm = vlm
        self.timeout_s = timeout_s
        # no trim_blocks: it swallowed the newline after inline {% endif %} (as in the NeSy planner)
        self._env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
        self.prompts = importlib.import_module(f"prompt.{prompt_module}")

    def render(self, name: str, **kw) -> str:
        return self._env.from_string(getattr(self.prompts, name)).render(**kw)

    def ask(self, system: str, user: str, schema: dict | None) -> Reply:
        """`schema` set: JSON reply, parsed into `data`; None: free text (raw only)."""
        t0 = time.perf_counter()
        convo = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = Reply(self.role, prompt_text=_text(convo))
        for _ in range(2 if schema is not None else 1):
            try:
                raw = self.vlm.chat(convo, schema, self.timeout_s)
            except Exception as e:
                r.error = f"{type(e).__name__}: {e}"
                break
            r.raws.append(raw)
            if schema is None:
                break
            try:
                r.data = extract_json(raw)
                break
            except ValueError:
                convo = [*convo, {"role": "assistant", "content": raw}, {"role": "user", "content": REPAIR_MESSAGE}]
        if schema is not None and r.data is None and r.error is None:
            r.error = "malformed JSON twice"
        r.latency_s = time.perf_counter() - t0
        return r
