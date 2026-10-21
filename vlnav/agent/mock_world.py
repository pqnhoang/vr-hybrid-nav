"""MockNavAPI and dry_run (TASK_cap_mas.md §2, §5, §6): the program runs on a symbolic world built
from a WorldSnapshot (memory view + grid copy + robot pose), with no robot motion and no VLM.

Queries are the real ones (QueryAPI). Actions are simulated optimistically, so the dry run follows
the path the program takes when everything works:
- locate(name): a known match -> that object (as if the VLM confirms it, the robot then goes there);
  none -> a placeholder object (position unknown: the robot will explore).
- explore(name): always the placeholder.
- go_to: A* on the grid from the simulated pose; a known goal without a path fails (hard check).
- follow: success, unverified (DualVLN cannot be simulated). check: True (the yes branch is the
  success path). report / fail: traced.

Hard checks (code, not the Observer, §4.3): an exception; the action budget; no action at all;
follow() while move is disabled; a known goal without a path; report() of an object the program
did not reach; fail() reached on the success path; a searched object that a follow() sentence names
(a route landmark); a movement command in the instruction but no follow() (move enabled).
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field

from vlnav.acquisition.symbolic_filter import SymbolicFilter
from vlnav.agent.api import BudgetExceeded, Obj, Place, Room, TaskFailed
from vlnav.agent.real_api import QueryAPI, program_line
from vlnav.agent.state import TraceEntry, WorldSnapshot
from vlnav.agent.validate import ProgramError, load_program
from vlnav.navigation.path_planner import GridAStar

MOVEMENT = re.compile(r"\b(go|move|walk|drive|head|keep going)\s+(forward|straight|ahead|back|backward)\b|"
                      r"\bturn\s+(left|right|around)\b|\b(go|walk)\s+(through|past|along|down|up)\b|"
                      r"\bfollow\s+the\b|\b(on|to)\s+your\s+(left|right)\b", re.IGNORECASE)


def movement_phrases(instruction: str) -> list[str]:
    return [m.group(0) for m in MOVEMENT.finditer(instruction)]


def landmarks(actions: list[dict]) -> list[str]:
    """Searched objects that a follow() sentence of the same program names (full name or head noun)."""
    routes = " ".join(a["target"] for a in actions if a["type"] == "move").lower()
    out = []
    for a in actions:
        if a["type"] == "move" or a["target"] in out:
            continue
        name = a["target"].lower().strip()
        head = name.split()[-1] if name else ""
        if name and (re.search(rf"\b{re.escape(name)}\b", routes) or re.search(rf"\b{re.escape(head)}s?\b", routes)):
            out.append(a["target"])
    return out


@dataclass
class DryRun:
    trace: list[TraceEntry] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)       # {type: move|explore|goto, target}
    hard_failures: list[str] = field(default_factory=list)
    error: str | None = None                                # exception / validation error
    finished: bool = False

    def text(self) -> str:
        lines = [t.text() for t in self.trace] or ["(no action)"]
        if self.error:
            lines.append(f"ERROR: {self.error}")
        return "\n".join(lines)


class MockNavAPI(QueryAPI):
    def __init__(self, snap: WorldSnapshot, filt: SymbolicFilter, path: GridAStar, max_primitives: int):
        self.snap = snap
        self.pose = snap.robot
        super().__init__(lambda: snap.memory, lambda: snap.grid, lambda: self.pose, filt, path,
                         lambda: snap.stamp)
        self.run = DryRun()
        self.reached: set[tuple] = set()            # ("obj", id(handle)) and ("node", memory id)
        self.max_primitives = max_primitives
        self.n_actions = 0

    def _count(self) -> None:
        self.n_actions += 1
        if self.n_actions > self.max_primitives:
            raise BudgetExceeded(f"more than {self.max_primitives} actions")

    def _mark(self, target) -> None:
        self.reached.add(("obj", id(target)))
        if isinstance(target, Obj) and target._id >= 0:
            self.reached.add(("node", target._id))

    def _trace(self, call: str, result: str, ok: bool = True) -> None:
        self.run.trace.append(TraceEntry(call, program_line(), result, ok))

    def _hard(self, msg: str) -> None:
        if msg not in self.run.hard_failures:
            self.run.hard_failures.append(msg)

    def _placeholder(self, name: str) -> Obj:
        return Obj(name=name, target=name, room=None, xy=None, height=0.0, score=0.0, n_obs=0,
                   seen_s_ago=0.0, verified=True, _id=-1)

    # ------------------------------------------------------------------------------ actions
    def follow(self, instruction, timeout=None):
        self._count()
        call = f"follow({' '.join(str(instruction).split())!r})"
        if not self.snap.move_enabled:
            self._trace(call, "not available: move is disabled", ok=False)
            self._hard("follow() is used but move (DualVLN) is disabled: use locate/go_to only")
            return False
        self._trace(call, "route driven by DualVLN (not simulated, assumed done)")
        self.run.actions.append({"type": "move", "target": str(instruction)})
        return True

    def locate(self, name, prompts=None, room=None, timeout=None):
        return self._search(str(name), prompts, use_memory=True)

    def explore(self, name, prompts=None, room=None, timeout=None):
        return self._search(str(name), prompts, use_memory=False)

    def _search(self, name: str, prompts, use_memory: bool) -> Obj:
        self._count()
        call = f"{'locate' if use_memory else 'explore'}({name!r})"
        found = self.find(name) if use_memory else []
        if found:
            o = found[0]
            self._trace(call, f"known in the scene graph: {o.name} #{o._id} in "
                              f"{o.room.label if o.room else 'unknown room'} (the VLM checks it; the robot goes "
                              f"there without exploring)")
            self.run.actions.append({"type": "goto", "target": name, "prompts": prompts})
            return Obj(**{**o.__dict__, "verified": True})
        self._trace(call, "not in the scene graph: the robot will explore for it")
        self.run.actions.append({"type": "explore", "target": name, "prompts": prompts})
        return self._placeholder(name)

    def go_to(self, target, timeout=None):
        self._count()
        if isinstance(target, Obj):
            label = target.target or target.name
            if target._id < 0:
                self._trace(f"go_to({label})", "position known once found (not simulated)")
                self._mark(target)
                return True
            xy = self._goal_xy(self.snap.memory, target)
        elif isinstance(target, (Room, Place)):
            label = target.label if isinstance(target, Room) else "place"
            xy = target.xy
        else:
            raise TypeError(f"go_to() needs an Obj, Room or Place, got {type(target).__name__}")
        if self.pose is None or self.snap.grid.shape[0] == 0:
            self._trace(f"go_to({label})", "no map yet (not simulated)")
            self._mark(target)
            return True
        c = self._path.cost_to(self.snap.grid, self.pose[:2], xy)
        if c == float("inf"):
            self._trace(f"go_to({label})", "no path on the map", ok=False)
            self._hard(f"go_to({label}): no path to the goal on the current map")
            return False
        self._trace(f"go_to({label})", f"path {c:.1f} m")
        self.pose = (float(xy[0]), float(xy[1]), self.pose[2])
        self._mark(target)
        return True

    def check(self, question, obj=None):
        self._count()
        self._trace(f"check({str(question)!r})", "not simulated, assumed yes")
        return True

    def report(self, obj):
        self._count()
        if not isinstance(obj, Obj):
            self._trace("report(?)", "not an object", ok=False)
            self._hard("report() needs an Obj from locate/explore/find")
            return None
        label = obj.target or obj.name
        if ("obj", id(obj)) not in self.reached and ("node", obj._id) not in self.reached:
            self._hard(f"report({label}) before go_to({label}) succeeded")
        self._trace(f"report({label})", "reported")
        return None

    def fail(self, reason):
        self._trace(f"fail({str(reason)!r})", "task failed", ok=False)
        self._hard(f"fail({str(reason)!r}) is reached although every action succeeded in the dry run")
        raise TaskFailed(str(reason))


def dry_run(code: str, snap: WorldSnapshot, filt: SymbolicFilter, path: GridAStar, max_primitives: int) -> DryRun:
    api = MockNavAPI(snap, filt, path, max_primitives)
    run = api.run
    try:
        fn = load_program(code)
    except ProgramError as e:
        run.error = str(e)
        run.hard_failures.append(f"invalid program: {e}")
        return run
    try:
        fn(api)
        run.finished = True
    except TaskFailed:
        pass
    except BudgetExceeded as e:
        run.error = str(e)
        run.hard_failures.append(f"budget: {e}")
    except Exception as e:                                       # program bug: line + message
        line = next((fs.lineno for fs in reversed(traceback.extract_tb(e.__traceback__))
                     if fs.filename == "<program>"), None)
        run.error = f"line {line}: {type(e).__name__}: {e}"
        run.hard_failures.append(f"exception: {run.error}")
    if not run.trace:
        run.hard_failures.append("the program does no action")
    for name in landmarks(run.actions):
        run.hard_failures.append(f"{name!r} is a landmark inside a follow() sentence: keep it in the route, "
                                 "do not locate, reach or report it")
    if snap.move_enabled and not snap.failure:
        phrases = movement_phrases(snap.instruction)
        if phrases and not any(a["type"] == "move" for a in run.actions):
            run.hard_failures.append(f"the instruction has movement commands ({', '.join(phrases)}) "
                                     "but the program never calls api.follow() for them")
    return run
