"""PlannerCore FSM (planner spec §5.7).

    new_task  -> DECOMPOSING -> validate -> acquire (goto) -> EXECUTING(first subtask)
    EXECUTING explore: candidate_found (internal check) | no_frontier | timeout -> end subtask
    EXECUTING goto:    reached -> TargetReport | nav_failed | timeout -> blacklist -> end subtask
    EXECUTING move:    reached (DualVLN STOP) | nav_failed | timeout -> end subtask, always replan
                       (a route is not a target: no TargetEntry, no TargetReport; TASK_move_subtask.md §10.2)
    end subtask -> REPLANNING -> validate -> DONE if task_done / all targets finished, else dispatch
    task_timeout_s -> FAILED; abort -> stop + FAILED

Only the first planned subtask is executed (receding horizon). VLM work (decompose / replan +
acquisition) runs in one worker thread; results for an outdated task are discarded. With
`blocking: true` the work runs inline, which keeps offline replays deterministic. Time only comes
from event stamps and `poll_internal(t)` (no wall clock).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from vlnav.acquisition.acquirer import TargetAcquirer
from vlnav.acquisition.label_matcher import LabelMatcher
from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.vlm_client import VLMClient
from vlnav.common.models import PlannerEvent, Pose, Subtask, TargetReport
from vlnav.planner.config import PlannerConfig
from vlnav.planner.decomposer import Decomposer, QueryResult, Replanner
from vlnav.planner.serializer import ExplorationSummary, GraphSerializer
from vlnav.planner.state import PlannerState, TargetEntry
from vlnav.planner.validation import (ValidatedPlan, fallback_plan, validate_decomposition,
                                      validate_replan)

ACTIVE_FSM = ("DECOMPOSING", "EXECUTING", "REPLANNING")
EXPLORE_END = ("candidate_found", "no_frontier", "timeout")
GOTO_FAIL = ("nav_failed", "timeout")
MOVE_END = ("reached", "nav_failed", "timeout")


@dataclass
class _Outcome:
    task_id: int
    kind: str
    targets: list[TargetEntry]
    subtask: dict | None = None            # {type, target, node_id, goal_pose, room_hint[, instruction]}
    done: bool = False
    failed: bool = False
    reason: str = ""
    log: dict = field(default_factory=dict)
    reject_floor: dict[str, float] = field(default_factory=dict)


def _query_log(q: QueryResult | None) -> dict:
    if q is None:
        return {}
    return {"prompt_hash": q.prompt_hash, "prompt_text": q.prompt_text, "images": q.image_ids,
            "raw": q.raws, "parsed": q.data, "error": q.error, "vlm_latency_s": round(q.latency_s, 3)}


class PlannerCore:
    def __init__(self, cfg: PlannerConfig, memory: MemoryView, acquirer: TargetAcquirer,
                 vlm: VLMClient, matcher: LabelMatcher | None,
                 exploration_fn: Callable[[], ExplorationSummary | None] | None = None,
                 decision_log: str | Path | None = None, move_enabled: bool = False):
        self.cfg = cfg
        self.move_enabled = move_enabled           # `move` subtasks in schema, prompts and validation
        self.memory = memory
        self.acquirer = acquirer
        self.serializer = GraphSerializer(cfg, matcher)
        self.decomposer = Decomposer(cfg, vlm, self.serializer)
        self.replanner = Replanner(cfg, vlm, self.serializer)
        self.decomposer.move = self.replanner.move = move_enabled
        self.exploration_fn = exploration_fn
        log = decision_log if decision_log is not None else cfg.decision_log
        self.decision_log = Path(log) if log else None
        if self.decision_log:
            self.decision_log.parent.mkdir(parents=True, exist_ok=True)
        self._state = PlannerState(task_id=-1, instruction="", fsm="IDLE", targets=[], current=None,
                                   history=[], started_t=0.0)
        self._dispatched: list[Subtask] = []
        self._reports: list[TargetReport] = []
        self._executor = None if cfg.blocking else ThreadPoolExecutor(1, thread_name_prefix="planner")
        self._future: Future | None = None
        self._acq_lock = threading.Lock()
        self._now = 0.0
        self._subtask_t0 = 0.0
        self._last_check = float("-inf")
        self._seen_candidates: dict[str, set[int]] = {}
        self._reject_floor: dict[str, float] = {}      # target -> best rank score rejected by the VLM
        self._next_task_id = 0
        self._next_subtask_id = 0
        self._dump_seq = 0
        self._decisions: list[dict] = []           # compact summaries for observers (viz)

    # ------------------------------------------------------------------------------ public API
    def submit_task(self, instruction: str, t: float, task_id: int | None = None) -> int:
        tid = self._next_task_id if task_id is None else task_id
        self.on_event(PlannerEvent(type="new_task", stamp=t, task_id=tid,
                                   payload={"instruction": instruction}))
        return tid

    def on_event(self, e: PlannerEvent) -> None:
        self._now = max(self._now, e.stamp)
        self._collect()
        if e.type == "new_task":
            return self._start(e)
        if e.type == "abort":
            if self._state.fsm in ACTIVE_FSM:
                self._finish("FAILED", "aborted")
            return
        s = self._state
        st = s.current
        if s.fsm != "EXECUTING" or st is None or e.task_id != s.task_id or e.subtask_id != st.id:
            self._log("ignored_event", event=e.model_dump(), reason="stale or no active subtask")
            return
        tgt = s.target(st.target) if st.type != "move" else None
        if st.type == "move" and e.type in MOVE_END:
            pass                                          # outcome goes to the history; replan decides
        elif st.type == "explore" and e.type in EXPLORE_END:
            if e.type == "candidate_found":
                nid = int(e.payload.get("node_id", -1))
                self._seen_candidates.setdefault(tgt.name, set()).add(nid)
                tgt.notes = (f"candidate #{nid} ({e.payload.get('label', '?')}, score "
                             f"{e.payload.get('score', '?')}) seen, not verified or reached yet")
            else:
                tgt.attempts += 1
        elif st.type == "goto" and e.type == "reached":
            self._report(st, tgt, e.stamp)
        elif st.type == "goto" and e.type in GOTO_FAIL:
            with self._acq_lock:
                self.acquirer.blacklist(tgt.name, st.node_id)
            tgt.attempts += 1
            tgt.status, tgt.node_id = "pending", None
            tgt.notes = f"goto #{st.node_id} ended with {e.type}; that object is excluded now"
        else:
            self._log("ignored_event", event=e.model_dump(), reason=f"not handled for {st.type}")
            return
        if tgt is not None and tgt.active and tgt.attempts >= self.cfg.max_attempts_per_target:
            tgt.status, tgt.notes = "failed", f"{tgt.attempts} failed attempts"
        s.history.append({"type": st.type, "target": st.target, "node_id": st.node_id,
                          "outcome": e.type, "t": e.stamp, "subtask_id": st.id,
                          **({"instruction": st.instruction} if st.type == "move" else {})})
        s.current = None
        self._log("subtask_end", event=e.model_dump())
        # with move subtasks a route may still follow the last target: the replanner decides
        if not self.move_enabled and not any(t.active for t in s.targets):
            return self._finish("DONE", "all targets reported or failed")
        s.fsm = "REPLANNING"
        self._launch("replan", f"{e.type} for {st.type} '{st.target}'"
                     + (f" (node #{st.node_id})" if st.node_id is not None else "")
                     + (f", candidate node #{e.payload['node_id']}" if "node_id" in e.payload else ""))

    def poll_internal(self, t: float) -> list[PlannerEvent]:
        """Timeouts and the explore candidate check; feed the returned events to `on_event`."""
        self._now = max(self._now, t)
        self._collect()
        s = self._state
        if s.fsm in ACTIVE_FSM and t - s.started_t >= self.cfg.task_timeout_s:
            self._finish("FAILED", "task timeout")
            return []
        st = s.current
        if s.fsm != "EXECUTING" or st is None:
            return []
        age = t - self._subtask_t0
        if st.type == "explore":
            if t - self._last_check >= self.cfg.check_period_s:
                self._last_check = t
                with self._acq_lock:
                    exclude = self.acquirer.blacklisted(st.target) | self._seen_candidates.get(st.target, set())
                    top = self.acquirer.filter.topk(st.target, self._mem(), 1, exclude=exclude)
                # after a rejection only stronger evidence re-triggers (no loop over 70 chairs)
                floor = max(self.cfg.explore_candidate_score, self._reject_floor.get(st.target, -1.0) + 1e-9)
                if top and top[0].rank_score >= floor:
                    return [self._internal("candidate_found", t, node_id=top[0].node.id,
                                           score=round(top[0].rank_score, 3), label=top[0].node.label)]
            if age >= self.cfg.explore_timeout_s:
                return [self._internal("timeout", t)]
        elif st.type == "goto" and age >= self.cfg.goto_timeout_s:
            return [self._internal("timeout", t)]
        return []

    def pop_dispatched(self) -> list[Subtask]:
        out, self._dispatched = self._dispatched, []
        return out

    def pop_decisions(self) -> list[dict]:
        """Summaries of applied decompose / replan decisions since the last call (viz, logs)."""
        out, self._decisions = self._decisions, []
        return out

    def pop_reports(self) -> list[TargetReport]:
        out, self._reports = self._reports, []
        return out

    def state(self) -> PlannerState:
        return self._state.model_copy(deep=True)

    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def wait_idle(self, timeout_s: float | None = None) -> None:
        """Block until the in-flight VLM job (if any) is done and applied (tests, bag end)."""
        if self._future is not None:
            self._future.exception(timeout=timeout_s)
        self._collect()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    # ----------------------------------------------------------------------------- internals
    def _mem(self) -> MemoryView:
        return self.memory.view() if hasattr(self.memory, "view") else self.memory

    def _internal(self, kind: str, t: float, **payload) -> PlannerEvent:
        st = self._state.current
        return PlannerEvent(type=kind, stamp=t, task_id=self._state.task_id, subtask_id=st.id,
                            payload={"source": "planner", **payload})

    def _start(self, e: PlannerEvent) -> None:
        if self._state.fsm in ACTIVE_FSM:
            self._finish("FAILED", "preempted by a new task")
        tid = e.task_id if e.task_id is not None else self._next_task_id
        self._next_task_id = max(self._next_task_id, tid + 1)
        self._state = PlannerState(task_id=tid, instruction=str(e.payload.get("instruction", "")),
                                   fsm="DECOMPOSING", targets=[], current=None, history=[],
                                   started_t=e.stamp)
        self._seen_candidates, self._reject_floor = {}, {}
        with self._acq_lock:
            self.acquirer.reset_blacklist()
        self._log("new_task", instruction=self._state.instruction)
        self._launch("decompose", "new task")

    def _launch(self, kind: str, last_event: str) -> None:
        snapshot = self._state.model_copy(deep=True)
        if self._executor is None:
            self._apply(self._safe_job(kind, snapshot, last_event))
        else:
            self._future = self._executor.submit(self._safe_job, kind, snapshot, last_event)

    def _collect(self) -> None:
        if self._future is not None and self._future.done():
            fut, self._future = self._future, None
            self._apply(fut.result())

    def _safe_job(self, kind: str, state: PlannerState, last_event: str) -> _Outcome:
        try:
            return self._job(kind, state, last_event)
        except Exception as e:                           # never kill the planner thread
            if kind == "decompose":
                return _Outcome(state.task_id, kind, state.targets, failed=True,
                                reason=f"decomposition crashed: {type(e).__name__}: {e}")
            plan = fallback_plan(state)
            out = self._choose(state, plan, kind, self._mem())
            out.log["exception"] = f"{type(e).__name__}: {e}"
            return out

    def _job(self, kind: str, state: PlannerState, last_event: str) -> _Outcome:
        mem = self._mem()
        expl = self.exploration_fn() if self.exploration_fn is not None else None
        inputs = {"kind": kind, "t": self._now, "task_id": state.task_id, "instruction": state.instruction,
                  "targets": [t.model_dump() for t in state.targets], "history": state.history,
                  "last_event": last_event, "memory_version": mem.version()}
        if kind == "decompose":
            q = self.decomposer.decompose(state.instruction, mem, expl)
            plan = (validate_decomposition(q.data, mem, self.move_enabled, state.instruction)
                    if q.data is not None else None)
            if plan is None or not (plan.targets or plan.subtasks):          # a route-only task has no targets
                out = _Outcome(state.task_id, kind, [], failed=True,
                               reason=f"decomposition failed: {q.error or 'no targets'}",
                               log={"query": _query_log(q), "input": inputs})
                out.log["debug_dir"] = self._dump(kind, inputs, q, None, out, mem)
                return out
        else:
            q = self.replanner.replan(state, last_event, mem, expl)
            plan = (validate_replan(q.data, state, mem, self.move_enabled, self.cfg.max_attempts_per_target)
                    if q.data is not None else fallback_plan(state))
        out = self._choose(state, plan, kind, mem)
        out.log["query"] = _query_log(q)
        out.log["input"] = inputs
        out.log["debug_dir"] = self._dump(kind, inputs, q, plan, out, mem)
        return out

    def _dump(self, kind, inputs, q: QueryResult, plan: ValidatedPlan | None, out: _Outcome,
              mem: MemoryView) -> str | None:
        """<log dir>/planner_debug/NNN_<kind>/ with everything needed to inspect / replay."""
        if not (self.cfg.debug_dumps and self.decision_log):
            return None
        root = self.decision_log.parent / "planner_debug"
        with self._acq_lock:
            self._dump_seq += 1
            seq = self._dump_seq
        d = root / f"{seq:03d}_{kind}"
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "prompt.txt").write_text(q.prompt_text)
        for k, raw in enumerate(q.raws):
            (d / f"reply_{k}.txt").write_text(raw)
        for nid, crop in q.crops.items():
            cv2.imwrite(str(d / "images" / f"node_{nid}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        result = {"parsed": q.data, "error": q.error, "vlm_latency_s": q.latency_s,
                  "validated": None if plan is None else {
                      "targets": [t.model_dump() for t in plan.targets],
                      "subtasks": [vars(p) for p in plan.subtasks], "fixes": plan.fixes,
                      "task_done": plan.task_done},
                  "outcome": {"subtask": out.subtask, "done": out.done, "failed": out.failed,
                              "reason": out.reason, "acquisition": out.log.get("acquisition")}}
        (d / "input.json").write_text(json.dumps(inputs, indent=2, default=_json_default))
        (d / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
        if hasattr(mem, "room_polygons"):                 # MemorySnapshot: exact memory the VLM saw
            from vlnav.memory.checkpoint import save_checkpoint
            save_checkpoint(mem, d / "memory")
            images = getattr(getattr(mem, "image_store", None), "root", None)
            (d / "memory" / "image_root.txt").write_text(str(images or ""))
        return str(d)

    def _choose(self, state: PlannerState, plan: ValidatedPlan, kind: str, mem: MemoryView) -> _Outcome:
        """First executable subtask; every goto goes through target acquisition (spec §5.6)."""
        log = {"reasoning": plan.reasoning, "fixes": plan.fixes,
               "plan": [vars(p) for p in plan.subtasks], "task_done": plan.task_done}
        targets = plan.targets
        moves = [p for p in plan.subtasks if p.type == "move"]
        if plan.task_done or not (any(t.active for t in targets) or moves):
            last = state.history[-1] if state.history else {}
            if not plan.task_done and last.get("type") == "move" and last.get("outcome") != "reached" \
                    and not any(t.status == "reported" for t in targets):
                return _Outcome(state.task_id, kind, targets, failed=True,
                                reason=f"route {last['target']!r} ended with {last['outcome']}", log=log)
            return _Outcome(state.task_id, kind, targets, done=True,
                            reason="planner: task done" if plan.task_done else "all targets finished",
                            log=log)
        by_name = {t.name.lower(): t for t in targets}
        for p in plan.subtasks:
            if p.type == "move":
                sub = {"type": "move", "target": p.target, "node_id": None, "goal_pose": None,
                       "room_hint": None, "instruction": p.instruction}
                return _Outcome(state.task_id, kind, targets, sub, log=log)
            tgt = by_name.get(p.target.lower())
            if tgt is None or not tgt.active:
                continue
            if p.type == "goto":
                with self._acq_lock:
                    acq = self.acquirer.acquire(state.instruction, tgt.name, mem,
                                                force_include=[p.node_id] if p.node_id is not None else None)
                log["acquisition"] = acq.model_dump(mode="json")
                if acq.status == "selected":
                    sub = {"type": "goto", "target": tgt.name, "node_id": acq.node_id,
                           "goal_pose": acq.goal_pose, "room_hint": p.room_hint}
                    return _Outcome(state.task_id, kind, targets, sub, log=log)
                sub = {"type": "explore", "target": tgt.name, "node_id": None, "goal_pose": None,
                       "room_hint": p.room_hint}
                log["fixes"] = log["fixes"] + [f"acquisition {acq.status} -> explore {tgt.name!r}"]
                out = _Outcome(state.task_id, kind, targets, sub, log=log)
                scores = []
                for i in acq.candidates:
                    try:
                        scores.append(mem.get_object(i).best_score)
                    except KeyError:
                        pass
                if acq.status == "rejected_all" and scores:
                    out.reject_floor = {tgt.name: max(scores)}
                return out
            sub = {"type": "explore", "target": tgt.name, "node_id": None, "goal_pose": None,
                   "room_hint": p.room_hint}
            return _Outcome(state.task_id, kind, targets, sub, log=log)
        pending = next(t for t in targets if t.active)
        sub = {"type": "explore", "target": pending.name, "node_id": None, "goal_pose": None,
               "room_hint": None}
        return _Outcome(state.task_id, kind, targets, sub, log=log)

    def _apply(self, out: _Outcome) -> None:
        s = self._state
        if out.task_id != s.task_id or s.fsm not in ("DECOMPOSING", "REPLANNING"):
            self._log("stale_result", result_kind=out.kind, result_task=out.task_id)
            return
        s.targets = out.targets
        for name, score in out.reject_floor.items():
            self._reject_floor[name] = max(self._reject_floor.get(name, -1.0), score)
        self._decisions.append({
            "kind": out.kind, "t": self._now, "reasoning": out.log.get("reasoning", ""),
            "plan": out.log.get("plan", []), "fixes": out.log.get("fixes", []),
            "acquisition": out.log.get("acquisition"), "subtask": out.subtask, "done": out.done,
            "failed": out.failed, "reason": out.reason, "debug_dir": out.log.get("debug_dir"),
            "vlm_latency_s": (out.log.get("query") or {}).get("vlm_latency_s")})
        if out.failed or out.done:
            self._log(out.kind, **out.log)
            return self._finish("FAILED" if out.failed else "DONE", out.reason)
        sub = out.subtask
        if sub["type"] == "move":                         # a route: no target entry to update
            st = Subtask(id=self._next_subtask_id, task_id=s.task_id, type="move", target=sub["target"],
                         instruction=sub["instruction"])
            self._next_subtask_id += 1
            s.current, s.fsm = st, "EXECUTING"
            self._subtask_t0, self._last_check = self._now, float("-inf")
            self._dispatched.append(st)
            self._log(out.kind, dispatched=st.model_dump(mode="json"), **out.log)
            return
        tgt = s.target(sub["target"])
        st = Subtask(id=self._next_subtask_id, task_id=s.task_id, type=sub["type"], target=tgt.name,
                     detector_prompts=list(tgt.detector_prompts), room_hint=sub["room_hint"],
                     node_id=sub["node_id"], goal_pose=sub["goal_pose"])
        self._next_subtask_id += 1
        if st.type == "goto":
            tgt.status, tgt.node_id = "verified", st.node_id
            tgt.notes = f"verified #{st.node_id}, robot on the way"
        else:
            tgt.status = "exploring"
        s.current, s.fsm = st, "EXECUTING"
        self._subtask_t0, self._last_check = self._now, float("-inf")
        self._dispatched.append(st)
        self._log(out.kind, dispatched=st.model_dump(mode="json"), **out.log)

    def _report(self, st: Subtask, tgt: TargetEntry, stamp: float) -> None:
        tgt.status, tgt.notes = "reported", f"reported: reached #{st.node_id}"
        try:
            n = self._mem().get_object(st.node_id)
            label, pos, img = n.label, n.centroid, n.best_view_image_path
        except KeyError:
            label, img = tgt.name, ""
            pos = st.goal_pose.position if st.goal_pose else (0.0, 0.0, 0.0)
        rep = TargetReport(task_id=self._state.task_id, target=tgt.name, node_id=st.node_id,
                           label=label, position=pos, image_path=img, stamp=stamp)
        self._reports.append(rep)
        self._log("report", report=rep.model_dump())

    def _finish(self, fsm: str, reason: str) -> None:
        s = self._state
        s.fsm, s.done_reason, s.current = fsm, reason, None
        stop = Subtask(id=self._next_subtask_id, task_id=s.task_id, type="stop", target="")
        self._next_subtask_id += 1
        self._dispatched.append(stop)
        self._log("finish", fsm=fsm, reason=reason,
                  targets=[t.model_dump() for t in s.targets])

    def _log(self, kind: str, **fields) -> None:
        if not self.decision_log:
            return
        rec = {"t": self._now, "wall": time.time(), "kind": kind, "task_id": self._state.task_id,
               "fsm": self._state.fsm, **fields}
        with open(self.decision_log, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, Pose):
        return o.model_dump()
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)
