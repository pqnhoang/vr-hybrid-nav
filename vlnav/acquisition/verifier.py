"""Stage 2, neural verification (acquisition spec §8.3-8.5).

Candidates are offered as letters A, B, C… (never raw node ids); the reply must choose one of the
offered letters or "none", so the VLM cannot select a node that was not a candidate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from vlnav.acquisition.config import ContextCfg, VerifierCfg
from vlnav.acquisition.context_builder import CandidateContext, ContextBuilder
from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.models import Candidate, VerifyOutput
from vlnav.acquisition.vlm_client import (VLMClient, extract_json, image_part, strip_images,
                                          text_part)

PROMPT_DIR = Path(__file__).parent / "prompts"
NONE = "none"
REPAIR_MESSAGE = "Your previous answer was not valid JSON. Return valid JSON only, matching the schema."


def verify_schema(letters: list[str]) -> dict:
    return {"type": "object",
            "properties": {"reasoning": {"type": "string"},
                           "choice": {"type": "string", "enum": [*letters, NONE]},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["reasoning", "choice", "confidence"]}


class NeuralVerifier:
    def __init__(self, cfg: VerifierCfg, context_cfg: ContextCfg, vlm: VLMClient,
                 debug_jsonl: str | Path | None = None):
        self.cfg = cfg
        self.builder = ContextBuilder(context_cfg)
        self.vlm = vlm
        env = Environment(loader=FileSystemLoader(PROMPT_DIR), undefined=StrictUndefined)
        self.tpl = env.get_template("verify_target.j2").module
        self.debug_jsonl = Path(debug_jsonl) if debug_jsonl else None
        if self.debug_jsonl:
            self.debug_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def messages(self, instruction: str, target: str, ctx: list[CandidateContext],
                 full_frame: bool) -> list[dict]:
        content = [text_part(str(self.tpl.header(instruction, target, len(ctx), full_frame)))]
        for c in ctx:
            content.append(text_part(str(self.tpl.candidate(c.letter, c.text))))
            content.append(image_part(c.crop))
            if c.frame is not None:
                content.append(image_part(c.frame))
        content.append(text_part(str(self.tpl.footer([c.letter for c in ctx]))))
        return [{"role": "user", "content": content}]

    def verify(self, instruction: str, target: str, candidates: list[Candidate],
               memory: MemoryView, include_full_frame: bool | None = None,
               use_neighbors: bool = True) -> VerifyOutput:
        t0 = time.perf_counter()
        ctx = self.builder.build(candidates, memory, include_full_frame, use_neighbors)
        full = any(c.frame is not None for c in ctx)
        letter_to_id = {c.letter: c.node_id for c in ctx}
        schema = verify_schema(list(letter_to_id))
        messages = self.messages(instruction, target, ctx, full)
        raws: list[str] = []
        out = self._ask(messages, schema, raws, letter_to_id)
        out.latency_s = time.perf_counter() - t0
        self._log(instruction, target, ctx, messages, raws, out)
        return out

    def _ask(self, messages, schema, raws, letter_to_id) -> VerifyOutput:
        convo = list(messages)
        for attempt in range(2):                       # one repair retry on malformed JSON
            try:
                raw = self.vlm.chat(convo, schema, self.cfg.timeout_s)
            except Exception as e:                     # timeout / transport error
                return VerifyOutput(status="error", error=f"{type(e).__name__}: {e}")
            raws.append(raw)
            try:
                data = extract_json(raw)
                choice = str(data["choice"]).strip()
                conf = float(data.get("confidence", 0.0))
                reasoning = str(data.get("reasoning", ""))
            except (ValueError, KeyError, TypeError):
                convo = [*convo, {"role": "assistant", "content": raw},
                         {"role": "user", "content": REPAIR_MESSAGE}]
                continue
            if choice.lower() == NONE:
                return VerifyOutput(status="none", reasoning=reasoning, confidence=conf)
            letter = choice.upper()
            if letter not in letter_to_id:
                return VerifyOutput(status="error", reasoning=reasoning, confidence=conf,
                                    error=f"invalid choice {choice!r}")
            if conf < self.cfg.min_vlm_confidence:
                return VerifyOutput(status="none", letter=letter, reasoning=reasoning,
                                    confidence=conf)
            return VerifyOutput(status="selected", node_id=letter_to_id[letter], letter=letter,
                                reasoning=reasoning, confidence=conf)
        return VerifyOutput(status="error", error="malformed JSON twice")

    def _log(self, instruction, target, ctx, messages, raws, out: VerifyOutput) -> None:
        if not self.debug_jsonl:
            return
        rec = {"t_wall": time.time(), "instruction": instruction, "target": target,
               "candidates": {c.letter: {"node_id": c.node_id, "context": c.text} for c in ctx},
               "prompt": strip_images(messages), "raw": raws, "result": out.model_dump()}
        with open(self.debug_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
