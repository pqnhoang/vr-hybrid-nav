"""Orchestrator (TASK_cap_mas.md §4.4): the deliberation loop of the multi-agent system.

    for round in range(max_round):
        Planner -> plan      Coder -> program (validated; errors go back to the Coder)
        dry run on the snapshot (MockNavAPI) -> trace + hard checks
        Observer -> verdict; VALID -> Proposal, else its summary goes back to the Planner
    no VALID program -> None (the runtime falls back to the NeSy planner or fails the task)

On a repair (snapshot.failure set) the Observer first explains the failure from the real trace and
its summary is the Planner's first feedback. Every round goes to rounds.jsonl; every agent call goes
to the decision log as a `query` record.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from vlnav.acquisition.symbolic_filter import SymbolicFilter
from vlnav.agent.chat import Reply
from vlnav.agent.coder import CoderAgent
from vlnav.agent.config import AgentConfig
from vlnav.agent.mock_world import MockNavAPI, dry_run
from vlnav.agent.observer import ObserverAgent, verdict
from vlnav.agent.planner import PlannerAgent, robot_text
from vlnav.agent.state import Proposal, RoundLog, WorldSnapshot
from vlnav.navigation import GridAStar, NavigationConfig
from vlnav.planner.serializer import GraphSerializer

@dataclass
class Deliberation:
    proposal: Proposal | None
    plan: dict | None = None                 # Planner JSON of the proposal (or of the last round)
    done: bool = False                       # repair: the Planner says nothing is left
    rounds: list[RoundLog] = field(default_factory=list)
    error: str | None = None
    latency_s: float = 0.0


class Orchestrator:
    def __init__(self, cfg: AgentConfig, planner: PlannerAgent, coder: CoderAgent, observer: ObserverAgent,
                 filt: SymbolicFilter, serializer: GraphSerializer, nav_cfg: NavigationConfig,
                 log: Callable[..., None] = lambda kind, **f: None, rounds_log: str | Path | None = None):
        self.cfg = cfg
        self.planner, self.coder, self.observer = planner, coder, observer
        self.filter, self.serializer, self.nav_cfg = filt, serializer, nav_cfg
        self.log = log
        self.rounds_log = Path(rounds_log) if rounds_log else None
        if self.rounds_log:
            self.rounds_log.parent.mkdir(parents=True, exist_ok=True)

    def _call(self, rnd: RoundLog | None, r: Reply, meta: dict) -> None:
        if rnd is not None:
            rnd.calls.append({"role": r.role, "raw": r.raws, "error": r.error, "latency_s": round(r.latency_s, 3)})
        self.log("agent_call", role=r.role, round=None if rnd is None else rnd.round,
                 deliberation=meta.get("kind"), query=r.log())

    def _write(self, rnd: RoundLog, snap: WorldSnapshot, meta: dict) -> None:
        if not self.rounds_log:
            return
        rec = {"task_id": meta.get("task_id"), "kind": meta.get("kind"), "t": snap.stamp,
               "instruction": snap.instruction, "failure": snap.failure, "memory_version": snap.version,
               **rnd.to_json()}
        with open(self.rounds_log, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def deliberate(self, snap: WorldSnapshot, previous_plan: dict | None = None,
                   meta: dict | None = None) -> Deliberation:
        meta = meta or {}
        t0 = time.perf_counter()
        out = Deliberation(None)
        graph = self.serializer.serialize(snap.memory, [snap.instruction])
        path = GridAStar(self.nav_cfg)                      # own instance: the field cache is not shared
        here = MockNavAPI(snap, self.filter, path, self.cfg.max_primitives).here() if snap.robot else None
        robot = robot_text(here)
        previous = previous_plan.get("plan") if previous_plan else None
        feedback = None
        if snap.failure:
            d = self.observer.diagnose(snap.instruction, previous or [], "\n".join(t.text() for t in snap.trace)
                                       or "(no action)", snap.failure, snap.move_enabled)
            self._call(None, d, meta)
            feedback = (d.data or {}).get("summary") or None
        for k in range(self.cfg.max_round):
            rnd = RoundLog(k)
            out.rounds.append(rnd)
            pr = self.planner.plan(snap, graph, robot, previous, feedback)
            self._call(rnd, pr, meta)
            if pr.data is None:
                rnd.error = f"planner: {pr.error}"
                self._write(rnd, snap, meta)
                continue
            plan = rnd.plan = out.plan = pr.data
            if not plan["plan"]:
                if plan["done"] and snap.failure:            # repair: everything is already done
                    out.done = True
                    self._write(rnd, snap, meta)
                    break
                rnd.error = "the plan is empty"
                feedback = "The plan was empty: plan the steps of the instruction."
                self._write(rnd, snap, meta)
                continue
            code, err = None, None
            for _ in range(self.cfg.coder_retries + 1):
                cr, code_k, err = self.coder.code(snap.instruction, plan["plan"], snap.move_enabled, err, code)
                self._call(rnd, cr, meta)
                code = code_k or code
                if err is None:
                    break
            rnd.code = code
            if err is not None:
                rnd.error = f"coder: {err}"
                previous = plan["plan"]
                feedback = f"The coder could not turn the plan into a valid program ({err}). Make the steps simpler."
                self._write(rnd, snap, meta)
                continue
            dry = dry_run(code, snap, self.filter, path, self.cfg.max_primitives)
            rnd.trace, rnd.hard_failures, rnd.error = dry.trace, dry.hard_failures, dry.error
            ob = self.observer.review(snap.instruction, plan["plan"], code, dry.text(), dry.hard_failures,
                                      snap.move_enabled)
            self._call(rnd, ob, meta)
            rnd.observation = ob.data
            rnd.valid = verdict(ob.data, dry.hard_failures)
            self._write(rnd, snap, meta)
            if rnd.valid:
                out.proposal = Proposal(program=code, base_version=snap.version, rounds=tuple(out.rounds),
                                        plan=plan, actions=tuple(dry.actions))
                break
            previous = plan["plan"]
            notes = [(ob.data or {}).get("summary") or ""] + [f"Code check: {h}" for h in dry.hard_failures]
            feedback = " ".join(n for n in notes if n) or "The reviewer rejected the plan."
        if out.proposal is None and not out.done:
            last = out.rounds[-1] if out.rounds else None
            out.error = f"no valid program in {len(out.rounds)} rounds" + (
                f" (last: {last.error or '; '.join(last.hard_failures) or 'observer: INVALID'})" if last else "")
        out.latency_s = time.perf_counter() - t0
        return out
