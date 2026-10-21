"""ProgramRuntime: runs code-as-policy task programs on the real stack (TASK_cap_mas.md §2, §3, §6).

Drop-in for PlannerCore: the same public methods (`submit_task`, `on_event`, `poll_internal`,
`pop_dispatched`, `pop_decisions`, `pop_reports`, `state`, `busy`, `wait_idle`, `close`), so
SystemRuntime (bag mode A), LiveRuntime and the ROS node run it unchanged.

    new_task -> DECOMPOSING (deliberation: Planner -> Coder -> validate -> dry run -> Observer)
             -> EXECUTING: the program runs in its own thread; each action becomes one Subtask
                follow -> move (DualVLN), locate / explore -> explore, go_to -> goto
                and the end event of that subtask (reached / nav_failed / timeout / no_frontier /
                candidate_found) is the action's return value
    program returns          -> DONE
    program fails / raises   -> REPLANNING (repair deliberation with the trace so far), at most
                                max_repairs times, then FAILED
    no valid program         -> the NeSy PlannerCore takes the task (fallback: nesy) or FAILED

Only the runtime thread changes the planner state: the program thread sends everything (targets,
trace, reports) as messages that the runtime applies in order. With `blocking` (bag replay) the
deliberation runs inline and the runtime waits for the program after every answer, until it asks
for the next action or ends, so replays stay deterministic. The robot holds while the program
deliberates or verifies (no subtask is active then).
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from vlnav.acquisition.acquirer import TargetAcquirer
from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.vlm_client import VLMClient
from vlnav.agent.api import BudgetExceeded, Cancelled, Obj, TaskFailed
from vlnav.agent.config import AgentConfig
from vlnav.agent.mas import Deliberation, Orchestrator
from vlnav.agent.state import TraceEntry, WorldSnapshot
from vlnav.agent.validate import load_program
from vlnav.common.models import PlannerEvent, Subtask, TargetReport
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.navigation.config import NavigationConfig
from vlnav.planner.config import PlannerConfig
from vlnav.planner.core import ACTIVE_FSM, EXPLORE_END, GOTO_FAIL, MOVE_END, PlannerCore, _json_default
from vlnav.planner.state import PlannerState, TargetEntry

_CANCEL = object()


class ProgramRuntime:
    def __init__(self, cfg: AgentConfig, pcfg: PlannerConfig, nav_cfg: NavigationConfig, memory,
                 grid: OccupancyGrid2D, acquirer: TargetAcquirer, vlm: VLMClient, mas: Orchestrator,
                 fallback: PlannerCore | None = None,
                 robot_fn: Callable[[], tuple[float, float, float] | None] = lambda: None,
                 decision_log: str | Path | None = None, move_enabled: bool = False):
        self.cfg, self.pcfg, self.nav_cfg = cfg, pcfg, nav_cfg
        self.memory, self.grid, self.acquirer, self.vlm, self.mas = memory, grid, acquirer, vlm, mas
        self.fallback = fallback if cfg.fallback == "nesy" else None
        self.robot = robot_fn
        self.move_enabled = move_enabled
        self.blocking = pcfg.blocking if cfg.blocking is None else cfg.blocking
        log = decision_log if decision_log is not None else pcfg.decision_log
        self.decision_log = Path(log) if log else None
        if self.decision_log:
            self.decision_log.parent.mkdir(parents=True, exist_ok=True)
        self.acq_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._state = PlannerState(task_id=-1, instruction="", fsm="IDLE", targets=[], current=None,
                                   history=[], started_t=0.0)
        self._dispatched: list[Subtask] = []
        self._reports: list[TargetReport] = []
        self._decisions: list[dict] = []
        self._executor = None if self.blocking else ThreadPoolExecutor(1, thread_name_prefix="cap-mas")
        self._future: Future | None = None
        self._inbox: queue.Queue = queue.Queue()        # program thread -> runtime thread
        self._tls = threading.local()                   # program thread: its generation
        self._gen = 0                                   # current program; older messages are stale
        self._program: threading.Thread | None = None
        self._reply: queue.Queue | None = None
        self._pending: dict | None = None               # action request of the current subtask
        self._delegate = False                          # the fallback PlannerCore runs this task
        self._trace: list[TraceEntry] = []
        self._code: str | None = None                   # the running program (program_view)
        self._code_rounds = 0
        self._code_trace = 0                            # index of its first entry in _trace
        self._plan: dict | None = None
        self._repairs = 0
        self._now = 0.0
        self._subtask_t0 = 0.0
        self._last_check = float("-inf")
        self._seen_candidates: dict[str, set[int]] = {}
        self._reject_floor: dict[str, float] = {}
        self._next_task_id = 0
        self._next_subtask_id = 0

    # ------------------------------------------------------------------------------ public API
    @property
    def now(self) -> float:
        return self._now

    @property
    def instruction(self) -> str:
        return self._state.instruction

    def submit_task(self, instruction: str, t: float, task_id: int | None = None) -> int:
        tid = self._next_task_id if task_id is None else task_id
        self.on_event(PlannerEvent(type="new_task", stamp=t, task_id=tid,
                                   payload={"instruction": instruction}))
        return tid

    def program_view(self) -> dict | None:
        """What the program display shows (vlnav/agent/viz.py; ROS topic /planner/program): the
        program, the line whose action runs now and the results of the executed lines. Plain data,
        equal while nothing changed. None before the first task."""
        s = self._state
        if s.task_id < 0:
            return None
        st = s.current
        return {"task_id": s.task_id, "instruction": s.instruction, "fsm": s.fsm, "done_reason": s.done_reason,
                "planner": "nesy" if self._delegate else "cap", "program": self._code, "rounds": self._code_rounds,
                "steps": [(e.line, e.ok, e.result) for e in self._trace[self._code_trace:]],
                "current_line": (self._pending or {}).get("line"),
                "subtask": None if st is None else f"{st.type} {st.instruction or st.target!r}"}

    def on_event(self, e: PlannerEvent) -> None:
        self._now = max(self._now, e.stamp)
        if e.type == "new_task":
            self._start(e)
            return self._pump(wait=self.blocking)
        if self._delegate:
            return self.fallback.on_event(e)
        self._pump()
        if e.type == "abort":
            if self._state.fsm in ACTIVE_FSM:
                self._finish("FAILED", "aborted")
            return
        s, st = self._state, self._state.current
        if s.fsm != "EXECUTING" or st is None or e.task_id != s.task_id or e.subtask_id != st.id:
            self._log("ignored_event", event=e.model_dump(), reason="stale or no active subtask")
            return
        ok = (st.type == "move" and e.type in MOVE_END) or (st.type == "explore" and e.type in EXPLORE_END) \
            or (st.type == "goto" and (e.type == "reached" or e.type in GOTO_FAIL))
        if not ok:
            self._log("ignored_event", event=e.model_dump(), reason=f"not handled for {st.type}")
            return
        result = {"outcome": e.type}
        if e.type == "candidate_found":
            nid = int(e.payload.get("node_id", -1))
            self._seen_candidates.setdefault(st.target.lower(), set()).add(nid)
            result["node_id"] = nid
        tgt = s.target(st.target) if st.type != "move" else None
        if tgt is not None and e.type in ("no_frontier", "timeout", "nav_failed"):
            tgt.attempts += 1
        s.history.append({"type": st.type, "target": st.target, "node_id": st.node_id,
                          "outcome": e.type, "t": e.stamp, "subtask_id": st.id,
                          **({"instruction": st.instruction} if st.type == "move" else {})})
        s.current, self._pending = None, None
        self._log("subtask_end", event=e.model_dump())
        self._reply.put(result)
        self._pump(wait=self.blocking)

    def poll_internal(self, t: float) -> list[PlannerEvent]:
        """Timeouts and the explore candidate check (as PlannerCore); feed them to `on_event`."""
        self._now = max(self._now, t)
        if self._delegate:
            return self.fallback.poll_internal(t)
        self._pump()
        s = self._state
        if s.fsm in ACTIVE_FSM and t - s.started_t >= self.pcfg.task_timeout_s:
            self._finish("FAILED", "task timeout")
            return []
        st, req = s.current, self._pending
        if s.fsm != "EXECUTING" or st is None or req is None:
            return []
        age = t - self._subtask_t0
        if st.type == "explore" and t - self._last_check >= self.pcfg.check_period_s:
            self._last_check = t
            key = st.target.lower()
            with self.acq_lock:
                exclude = self.acquirer.blacklisted(st.target) | self._seen_candidates.get(key, set())
                top = self.acquirer.filter.topk(st.target, self._mem(), 1, exclude=exclude)
            floor = max(self.pcfg.explore_candidate_score, self._reject_floor.get(key, -1.0) + 1e-9)
            if top and top[0].rank_score >= floor:
                return [self._internal("candidate_found", t, node_id=top[0].node.id,
                                       score=round(top[0].rank_score, 3), label=top[0].node.label)]
        limit = req.get("timeout")
        if limit is None:
            limit = {"explore": self.pcfg.explore_timeout_s, "goto": self.pcfg.goto_timeout_s}.get(st.type)
        if limit is not None and age >= limit:
            return [self._internal("timeout", t)]
        return []

    def pop_dispatched(self) -> list[Subtask]:
        self._pump()
        out, self._dispatched = self._dispatched, []
        return out + (self.fallback.pop_dispatched() if self.fallback is not None else [])

    def pop_decisions(self) -> list[dict]:
        out, self._decisions = self._decisions, []
        return out + (self.fallback.pop_decisions() if self.fallback is not None else [])

    def pop_reports(self) -> list[TargetReport]:
        self._pump()
        out, self._reports = self._reports, []
        return out + (self.fallback.pop_reports() if self.fallback is not None else [])

    def state(self) -> PlannerState:
        if self._delegate:
            return self.fallback.state()
        return self._state.model_copy(deep=True)

    def busy(self) -> bool:
        if self._delegate:
            return self.fallback.busy()
        return (self._future is not None and not self._future.done()) or self._computing()

    def wait_idle(self, timeout_s: float | None = None) -> None:
        """Block until the deliberation and the program's work up to its next action are applied."""
        if self._delegate:
            return self.fallback.wait_idle(timeout_s)
        if self._future is not None:
            self._future.exception(timeout=timeout_s)
        self._pump(wait=True, timeout_s=timeout_s)

    def close(self) -> None:
        self._cancel_program()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        if self.fallback is not None:
            self.fallback.close()

    # ----------------------------------------------------------- accessors for the program thread
    def memory_view(self) -> MemoryView:
        return self._mem()

    def grid_copy(self) -> OccupancyGrid2D:
        g = self.grid
        return OccupancyGrid2D.from_array(g.as_array().copy(), g.resolution, g.origin_xy)

    def blacklisted(self, target: str) -> set[int]:
        with self.acq_lock:
            return self.acquirer.blacklisted(target)

    def act(self, gen: int, reply: queue.Queue, req: dict) -> dict:
        """Program thread: dispatch one action and block until its subtask ends."""
        if gen != self._gen:
            raise Cancelled()
        self._inbox.put(("act", gen, req))
        res = reply.get()
        if res is _CANCEL:
            raise Cancelled()
        return res

    def target_entry(self, name: str, prompts: list[str] | None) -> None:
        self._post("target", name, prompts)

    def set_target(self, name: str, status: str, node_id: int | None = None, notes: str = "") -> None:
        self._post("set_target", name, status, node_id, notes)

    def add_trace(self, entry: TraceEntry) -> None:
        self._post("trace", entry)

    def note_rejection(self, target: str, node_ids: list[int]) -> None:
        self._post("reject", target, list(node_ids))

    def report(self, obj: Obj) -> None:
        self._post("report", obj)

    def log(self, kind: str, **fields) -> None:
        self._log(kind, **fields)

    def _post(self, kind: str, *args) -> None:
        self._inbox.put((kind, getattr(self._tls, "gen", None), *args))

    # ----------------------------------------------------------------------------- internals
    def _mem(self) -> MemoryView:
        return self.memory.view() if hasattr(self.memory, "view") else self.memory

    def _internal(self, kind: str, t: float, **payload) -> PlannerEvent:
        st = self._state.current
        return PlannerEvent(type=kind, stamp=t, task_id=self._state.task_id, subtask_id=st.id,
                            payload={"source": "planner", **payload})

    def _computing(self) -> bool:
        """The program runs Python / VLM work between two actions."""
        return self._program is not None and self._pending is None

    def _pump(self, wait: bool = False, timeout_s: float | None = None) -> None:
        """Apply a finished deliberation and the program's messages. With `wait`, block while the
        program is computing, until it asks for an action or ends."""
        t_end = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            self._collect()
            try:
                msg = self._inbox.get_nowait()
            except queue.Empty:
                if not wait or (t_end is not None and time.monotonic() >= t_end):
                    return
                if self._future is not None and not self._future.done():
                    self._future.exception()
                    continue
                if not self._computing():
                    return
                try:
                    msg = self._inbox.get(timeout=0.5)
                except queue.Empty:
                    continue
            self._handle(msg)

    def _collect(self) -> None:
        if self._future is not None and self._future.done():
            fut, self._future = self._future, None
            self._apply(*fut.result())

    def _handle(self, msg: tuple) -> None:
        kind, gen, *args = msg
        if gen != self._gen or self._delegate:
            return                                         # from a cancelled program
        s = self._state
        if kind == "act":
            self._dispatch(args[0])
        elif kind == "target":
            name, prompts = args
            t = s.target(name)
            if t is None:
                s.targets.append(TargetEntry(name=name, detector_prompts=list(prompts or [name])))
            elif prompts:
                t.detector_prompts = list(prompts)
        elif kind == "set_target":
            name, status, node_id, notes = args
            t = s.target(name)
            if t is not None and t.status != "reported":
                t.status, t.node_id = status, node_id
                if notes:
                    t.notes = notes
        elif kind == "trace":
            self._trace.append(args[0])
            self._log("program_step", step=args[0].text(), ok=args[0].ok)
        elif kind == "reject":
            target, ids = args
            scores = []
            mem = self._mem()
            for i in ids:
                try:
                    scores.append(mem.get_object(i).best_score)
                except KeyError:
                    pass
            if scores:
                k = target.lower()
                self._reject_floor[k] = max(self._reject_floor.get(k, -1.0), max(scores))
        elif kind == "report":
            self._report(args[0])
        elif kind == "done":
            self._program_done(*args)

    def _dispatch(self, req: dict) -> None:
        s = self._state
        k = req["kind"]
        sid = self._next_subtask_id
        self._next_subtask_id += 1
        if k == "move":
            text = req["instruction"]
            st = Subtask(id=sid, task_id=s.task_id, type="move", target=_route_label(text), instruction=text)
        else:
            tgt = s.target(req["target"])
            st = Subtask(id=sid, task_id=s.task_id, type=k, target=req["target"],
                         detector_prompts=list(tgt.detector_prompts) if tgt else [],
                         room_hint=req.get("room_hint"), node_id=req.get("node_id"),
                         goal_pose=req.get("goal_pose"))
            if tgt is not None and k == "explore":
                tgt.status = "exploring"
        s.current, s.fsm, self._pending = st, "EXECUTING", req
        self._subtask_t0, self._last_check = self._now, float("-inf")
        self._dispatched.append(st)
        self._log("dispatch", dispatched=st.model_dump(mode="json"))

    def _report(self, obj: Obj) -> None:
        s = self._state
        name = obj.target or obj.name
        tgt = s.target(name)
        try:
            n = self._mem().get_object(obj._id)
            label, pos, img = n.label, n.centroid, n.best_view_image_path
        except KeyError:
            label, pos, img = obj.name, (*obj.xy, obj.height), ""
        if tgt is not None:
            tgt.status, tgt.node_id, tgt.notes = "reported", obj._id, f"reported: reached #{obj._id}"
        rep = TargetReport(task_id=s.task_id, target=name, node_id=obj._id, label=label,
                           position=pos, image_path=img, stamp=self._now)
        self._reports.append(rep)
        self._log("report", report=rep.model_dump())

    # ------------------------------------------------------------------------------ program
    def _start_program(self, code: str) -> None:
        fn = load_program(code)
        self._cancel_program()
        gen = self._gen
        self._reply = queue.Queue()
        self._program = threading.Thread(target=self._run_program, args=(gen, fn, self._reply),
                                         daemon=True, name=f"cap-program-{gen}")
        self._state.fsm, self._state.current = "EXECUTING", None
        self._program.start()

    def _run_program(self, gen: int, fn, reply: queue.Queue) -> None:
        from vlnav.agent.real_api import RealNavAPI
        self._tls.gen = gen
        status, reason = "ok", ""
        try:
            fn(RealNavAPI(self, gen, reply))
        except Cancelled:
            status = "cancelled"
        except TaskFailed as e:
            status, reason = "failed", f"the program called fail({str(e)!r})"
        except BudgetExceeded as e:
            status, reason = "failed", f"action budget: {e}"
        except Exception as e:                            # program bug or API error: repair
            line = next((fs.lineno for fs in reversed(traceback.extract_tb(e.__traceback__))
                         if fs.filename == "<program>"), None)
            status, reason = "failed", f"line {line}: {type(e).__name__}: {e}"
            self._log("program_exception", error=reason, traceback=traceback.format_exc())
        self._inbox.put(("done", gen, status, reason))

    def _program_done(self, status: str, reason: str) -> None:
        self._program, self._reply, self._pending = None, None, None
        if status == "cancelled":
            return
        self._log("program_end", status=status, reason=reason)
        if status == "ok":
            return self._finish("DONE", "program finished")
        if self._repairs >= self.cfg.max_repairs:
            return self._finish("FAILED", f"{reason} (no repairs left)")
        self._repairs += 1
        self._launch(failure=reason)

    def _cancel_program(self) -> None:
        self._gen += 1
        if self._reply is not None:
            self._reply.put(_CANCEL)
        self._program, self._reply, self._pending = None, None, None

    # -------------------------------------------------------------------------- deliberation
    def _start(self, e: PlannerEvent) -> None:
        if self._delegate:
            self.fallback.on_event(PlannerEvent(type="abort", stamp=e.stamp))
            self._dispatched += self.fallback.pop_dispatched()
            self._delegate = False
        elif self._state.fsm in ACTIVE_FSM:
            self._finish("FAILED", "preempted by a new task")
        tid = e.task_id if e.task_id is not None else self._next_task_id
        self._next_task_id = max(self._next_task_id, tid + 1)
        self._state = PlannerState(task_id=tid, instruction=str(e.payload.get("instruction", "")),
                                   fsm="DECOMPOSING", targets=[], current=None, history=[], started_t=e.stamp)
        self._trace, self._plan, self._repairs = [], None, 0
        self._code, self._code_rounds, self._code_trace = None, 0, 0
        self._seen_candidates, self._reject_floor = {}, {}
        with self.acq_lock:
            self.acquirer.reset_blacklist()
        self._log("new_task", instruction=self._state.instruction, planner="cap")
        self._launch()

    def snapshot(self, failure: str | None = None) -> WorldSnapshot:
        mem = self._mem()
        return WorldSnapshot(version=mem.version(), stamp=self._now, instruction=self._state.instruction,
                             robot=self.robot(), memory=mem, grid=self.grid_copy(),
                             move_enabled=self.move_enabled, trace=tuple(self._trace), failure=failure)

    def _launch(self, failure: str | None = None) -> None:
        s = self._state
        kind = "replan" if failure else "decompose"
        s.fsm, s.current = ("REPLANNING" if failure else "DECOMPOSING"), None
        snap = self.snapshot(failure)
        meta = {"task_id": s.task_id, "kind": kind}
        if self._executor is None:
            self._apply(*self._safe_deliberate(snap, meta))
        else:
            self._future = self._executor.submit(self._safe_deliberate, snap, meta)

    def _safe_deliberate(self, snap: WorldSnapshot, meta: dict) -> tuple[dict, Deliberation]:
        try:
            return meta, self.mas.deliberate(snap, self._plan, meta)
        except Exception as e:                              # never kill the worker
            self._log("deliberation_exception", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc())
            return meta, Deliberation(None, error=f"{type(e).__name__}: {e}")

    def _apply(self, meta: dict, d: Deliberation) -> None:
        s = self._state
        if meta["task_id"] != s.task_id or s.fsm not in ("DECOMPOSING", "REPLANNING") or self._delegate:
            self._log("stale_result", result_kind=meta["kind"], result_task=meta["task_id"])
            return
        kind = meta["kind"]
        p = d.proposal
        thought = (d.plan or {}).get("thought", "")
        self._decisions.append({
            "kind": kind, "t": self._now, "reasoning": thought, "plan": list(p.actions) if p else [],
            "steps": (d.plan or {}).get("plan", []), "fixes": [], "acquisition": None, "subtask": None,
            "done": d.done, "failed": p is None and not d.done, "reason": d.error or "",
            "debug_dir": None, "vlm_latency_s": round(d.latency_s, 3), "rounds": len(d.rounds)})
        if d.done:
            self._log(kind, plan=[], steps=[], reasoning=thought, done=True)
            return self._finish("DONE", "planner: nothing left to do")
        if p is not None:
            self._plan = d.plan
            for a in p.actions:                             # targets of the whole program (metrics, viz)
                if a["type"] != "move" and s.target(a["target"]) is None:
                    s.targets.append(TargetEntry(name=a["target"],
                                                 detector_prompts=list(a.get("prompts") or [a["target"]])))
            self._log(kind, plan=list(p.actions), steps=d.plan.get("plan", []), reasoning=thought,
                      program=p.program, base_version=p.base_version, memory_version=self._mem().version(),
                      rounds=len(d.rounds))
            self._code, self._code_rounds, self._code_trace = p.program, len(d.rounds), len(self._trace)
            return self._start_program(p.program)
        self._log("deliberation_failed", deliberation=kind, error=d.error, rounds=len(d.rounds))
        if kind == "decompose" and self.fallback is not None:
            self._log("fallback", planner="nesy")
            self._delegate = True
            self.fallback.submit_task(s.instruction, self._now, task_id=s.task_id)
            return
        self._finish("FAILED", f"no valid program{': ' + d.error if d.error else ''}")

    def _finish(self, fsm: str, reason: str) -> None:
        self._cancel_program()
        s = self._state
        s.fsm, s.done_reason, s.current = fsm, reason, None
        stop = Subtask(id=self._next_subtask_id, task_id=s.task_id, type="stop", target="")
        self._next_subtask_id += 1
        self._dispatched.append(stop)
        self._log("finish", fsm=fsm, reason=reason, targets=[t.model_dump() for t in s.targets])

    def _log(self, kind: str, **fields) -> None:
        if not self.decision_log:
            return
        rec = {"t": self._now, "wall": time.time(), "kind": kind, "task_id": self._state.task_id,
               "fsm": self._state.fsm, **fields}
        line = json.dumps(rec, default=_json_default) + "\n"
        with self._log_lock, open(self.decision_log, "a") as f:
            f.write(line)


def _route_label(text: str, n: int = 6) -> str:
    """Short label of a route for displays: its first words."""
    words = text.rstrip(".").split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")
