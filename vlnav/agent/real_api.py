"""Queries shared by the real and the mock API, and RealNavAPI (TASK_cap_mas.md §5.3, §5.4, §6).

RealNavAPI runs in the program thread. Each action posts a request to the ProgramRuntime and
blocks until the runtime answers with the end event of the subtask it dispatched (explore / goto /
move). VLM verification (acquisition) runs here, in the program thread, never in the runtime thread.
"""
from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Callable

from vlnav.acquisition.label_matcher import LabelMatcher
from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.models import Candidate
from vlnav.acquisition.symbolic_filter import SymbolicFilter
from vlnav.acquisition.vlm_client import extract_json, image_part, text_part
from vlnav.agent.api import BudgetExceeded, NavAPI, Obj, Place, Room, TaskFailed
from vlnav.agent.state import TraceEntry
from vlnav.common.geometry import wrap_angle, yaw_quat
from vlnav.common.models import ObjectNode, Pose
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.navigation.path_planner import GridAStar

if TYPE_CHECKING:
    from vlnav.agent.runtime import ProgramRuntime

CHECK_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]}},
                "required": ["answer"]}


def program_line() -> int | None:
    """Line of the program (`<program>`) that made the current call."""
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_filename == "<program>":
            return f.f_lineno
        f = f.f_back
    return None


def side_of(robot: tuple[float, float, float], xy) -> str:
    b = float(wrap_angle(math.atan2(xy[1] - robot[1], xy[0] - robot[0]) - robot[2]))
    if abs(b) <= math.radians(45):
        return "ahead"
    if abs(b) >= math.radians(135):
        return "behind"
    return "left" if b > 0 else "right"


def pose_at(xy, yaw: float, t: float, z: float = 0.0) -> Pose:
    return Pose(position=(float(xy[0]), float(xy[1]), z), orientation_xyzw=yaw_quat(yaw), stamp=t)


class QueryAPI(NavAPI):
    """The instant queries on a memory view, a grid and a robot pose (real and mock)."""

    def __init__(self, memory: Callable[[], MemoryView], grid: Callable[[], OccupancyGrid2D],
                 robot: Callable[[], tuple[float, float, float] | None], filt: SymbolicFilter,
                 path: GridAStar, now: Callable[[], float], exclude: Callable[[str], set[int]] = lambda n: set()):
        self._memory, self._grid, self._robot, self._filter = memory, grid, robot, filt
        self._path, self._now, self._exclude = path, now, exclude

    @property
    def _matcher(self) -> LabelMatcher:
        return self._filter.matcher

    # ---------------------------------------------------------------------------- handles
    def _room(self, mem: MemoryView, room_id: int | None) -> Room | None:
        r = mem.get_room(room_id) if room_id is not None else None
        if r is None:
            return None
        labels = []
        for i in r.object_ids:
            try:
                labels.append(mem.get_object(i).label)
            except KeyError:
                pass
        return Room(r.label, (float(r.centroid_xy[0]), float(r.centroid_xy[1])), float(r.area_m2),
                    tuple(sorted(set(labels))), r.id)

    def _obj(self, mem: MemoryView, n: ObjectNode, target: str, verified: bool = False) -> Obj:
        return Obj(name=n.label, target=target, room=self._room(mem, n.room_id),
                   xy=(float(n.centroid[0]), float(n.centroid[1])), height=float(n.centroid[2]),
                   score=round(float(n.best_score), 3), n_obs=int(n.n_obs),
                   seen_s_ago=round(max(0.0, self._now() - n.last_seen_t), 1), verified=verified, _id=n.id)

    def _room_match(self, label: str, query: str) -> bool:
        a, b = label.lower().strip(), query.lower().strip()
        return a == b or b in a or a in b or self._matcher.sim(a, b) >= self._matcher.threshold

    def _xy(self, x) -> tuple[float, float] | None:
        if isinstance(x, (Obj, Room, Place)):
            return x.xy
        raise TypeError(f"expected an Obj, Room or Place, got {type(x).__name__}")

    def _goal_xy(self, mem: MemoryView, x) -> tuple[float, float] | None:
        """Where the robot would stand: the best-view pose for an object, else the point itself."""
        if isinstance(x, Obj) and x._id >= 0:
            try:
                return tuple(mem.get_object(x._id).best_view_pose.position[:2])
            except KeyError:
                return x.xy
        return self._xy(x)

    # ------------------------------------------------------------------------------ queries
    def find(self, name, room=None, near=None, radius=3.0):
        mem = self._memory()
        out = []
        for c in self._filter.candidates(str(name), mem, exclude=self._exclude(str(name))):
            o = self._obj(mem, c.node, str(name))
            if room is not None:
                rl = room.label if isinstance(room, Room) else str(room)
                if o.room is None or not self._room_match(o.room.label, rl):
                    continue
            if near is not None and (near.xy is None or math.dist(o.xy, near.xy) > radius):
                continue
            out.append(o)
        return out

    def rooms(self, label=None):
        mem = self._memory()
        rs = [self._room(mem, r.id) for r in sorted(mem.rooms(), key=lambda r: -r.area_m2)]
        return [r for r in rs if r is not None and (label is None or self._room_match(r.label, str(label)))]

    def here(self):
        r = self._robot()
        if r is None:
            return Place((0.0, 0.0), 0.0, None)
        mem = self._memory()
        best, room = float("inf"), None
        for rm in mem.rooms():                       # nearest room centre (room cells are not exposed here)
            d = math.dist(rm.centroid_xy, r[:2])
            if d < best:
                best, room = d, rm.id
        return Place((float(r[0]), float(r[1])), float(r[2]), self._room(mem, room))

    def distance(self, a, b=None):
        mem = self._memory()
        goal = self._goal_xy(mem, a)
        start = self._goal_xy(mem, b) if b is not None else (self._robot() or (None,))[:2]
        if goal is None or start is None or start[0] is None:
            return None
        c = self._path.cost_to(self._grid(), start, goal)
        return round(c, 2) if math.isfinite(c) else None

    def side(self, x):
        r = self._robot()
        xy = self._xy(x)
        if r is None or xy is None:
            return "ahead"
        return side_of(r, xy)


class RealNavAPI(QueryAPI):
    """Actions on the real stack through the ProgramRuntime (one instance per program)."""

    def __init__(self, rt: "ProgramRuntime", gen: int, reply):
        super().__init__(rt.memory_view, rt.grid_copy, rt.robot, rt.acquirer.filter,
                         GridAStar(rt.nav_cfg), lambda: rt.now, rt.blacklisted)
        self._rt, self._gen, self._reply = rt, gen, reply
        self._n_actions = 0

    # ---------------------------------------------------------------------------- plumbing
    def _act(self, kind: str, **req) -> dict:
        return self._rt.act(self._gen, self._reply, {"kind": kind, "line": program_line(), **req})

    def _count(self) -> None:
        self._n_actions += 1
        if self._n_actions > self._rt.cfg.max_primitives:
            raise BudgetExceeded(f"more than {self._rt.cfg.max_primitives} actions")

    def _trace(self, call: str, result: str, ok: bool = True, **detail) -> None:
        self._rt.add_trace(TraceEntry(call, program_line(), result, ok, self._rt.now, detail))

    def _verify(self, target: str, node_ids: list[int] | None) -> tuple[ObjectNode | None, str]:
        """VLM acquisition: the best of the symbolic top-k (+ `node_ids`) that the VLM confirms."""
        rt = self._rt
        mem = rt.memory_view()
        with rt.acq_lock:
            acq = rt.acquirer.acquire(rt.instruction, target, mem, force_include=node_ids)
        rt.log("acquisition", target=target, result=acq.model_dump(mode="json"))
        if acq.status != "selected":
            rt.note_rejection(target, acq.candidates)
            return None, acq.status
        return mem.get_object(acq.node_id), "selected"

    def _verify_one(self, target: str, node: ObjectNode) -> bool:
        rt = self._rt
        if rt.acquirer.verifier is None or not rt.acquirer.use_vlm:
            return True
        mem = rt.memory_view()
        sim = rt.acquirer.filter.matcher.score(target, [node.label])[node.label]
        with rt.acq_lock:
            out = rt.acquirer.verifier.verify(rt.instruction, target,
                                              [Candidate(node=node, match_score=sim, rank_score=node.best_score)], mem)
        rt.log("verify", target=target, node_id=node.id, result=out.model_dump(mode="json"))
        if out.status != "selected":
            with rt.acq_lock:
                rt.acquirer.blacklist(target, node.id)
            return False
        return True

    # ------------------------------------------------------------------------------ actions
    def follow(self, instruction, timeout=None):
        self._count()
        text = " ".join(str(instruction).split())
        call = f"follow({text!r})"
        if not self._rt.move_enabled:
            self._trace(call, "not available (move disabled)", ok=False)
            return False
        res = self._act("move", instruction=text, timeout=timeout)
        ok = res["outcome"] == "reached"
        self._trace(call, res["outcome"], ok)
        return ok

    def locate(self, name, prompts=None, room=None, timeout=None):
        return self._search(str(name), prompts, room, timeout, use_memory=True)

    def explore(self, name, prompts=None, room=None, timeout=None):
        return self._search(str(name), prompts, room, timeout, use_memory=False)

    def _search(self, name: str, prompts, room, timeout, use_memory: bool) -> Obj | None:
        self._count()
        rt = self._rt
        call = f"{'locate' if use_memory else 'explore'}({name!r})"
        rt.target_entry(name, prompts)
        room_label = room.label if isinstance(room, Room) else (str(room) if room else None)
        if use_memory and self._filter.topk(name, rt.memory_view(), 1, exclude=rt.blacklisted(name)):
            node, status = self._verify(name, None)
            if node is not None:
                obj = self._obj(rt.memory_view(), node, name, verified=True)
                rt.set_target(name, "verified", node.id, f"verified #{node.id} from the scene graph")
                self._trace(call, f"scene graph: {node.label} #{node.id} verified", node_id=node.id)
                return obj
            rt.add_trace(TraceEntry(call, program_line(), f"scene graph candidates {status} -> explore",
                                    True, rt.now))
        limit = timeout if timeout is not None else rt.pcfg.explore_timeout_s
        t0 = rt.now
        while True:
            remaining = limit - (rt.now - t0)
            if remaining <= 0:
                self._trace(call, "timeout", ok=False)
                return None
            rt.set_target(name, "exploring")
            res = self._act("explore", target=name, room_hint=room_label, timeout=remaining)
            if res["outcome"] != "candidate_found":
                self._trace(call, res["outcome"], ok=False)
                return None
            node, _ = self._verify(name, [res["node_id"]])
            if node is not None:
                rt.set_target(name, "verified", node.id, f"verified #{node.id} while exploring")
                self._trace(call, f"found {node.label} #{node.id} (verified)", node_id=node.id)
                return self._obj(rt.memory_view(), node, name, verified=True)
            rt.add_trace(TraceEntry(call, program_line(), f"candidate #{res['node_id']} rejected, exploring on",
                                    True, rt.now))

    def go_to(self, target, timeout=None):
        self._count()
        rt = self._rt
        limit = timeout if timeout is not None else rt.pcfg.goto_timeout_s
        if isinstance(target, Obj):
            call = f"go_to({target.target or target.name})"
            mem = rt.memory_view()
            try:
                node = mem.get_object(target._id)
            except KeyError:
                self._trace(call, "object not in memory", ok=False)
                return False
            name = target.target or target.name
            rt.target_entry(name, None)
            if not target.verified and not self._verify_one(name, node):
                self._trace(call, f"{node.label} #{node.id} rejected by the VLM", ok=False)
                return False
            rt.set_target(name, "verified", node.id, f"verified #{node.id}, robot on the way")
            res = self._act("goto", target=name, node_id=node.id, goal_pose=node.best_view_pose, timeout=limit)
            ok = res["outcome"] == "reached"
            if not ok:
                with rt.acq_lock:
                    rt.acquirer.blacklist(name, node.id)
                rt.set_target(name, "pending", None, f"goto #{node.id} ended with {res['outcome']}")
            self._trace(call, res["outcome"], ok, node_id=node.id)
            return ok
        if isinstance(target, (Room, Place)):
            label = target.label if isinstance(target, Room) else "place"
            r = self.here()
            yaw = target.yaw if isinstance(target, Place) else math.atan2(target.xy[1] - r.xy[1],
                                                                          target.xy[0] - r.xy[0])
            res = self._act("goto", target=label, node_id=None, goal_pose=pose_at(target.xy, yaw, rt.now),
                            timeout=limit)
            ok = res["outcome"] == "reached"
            self._trace(f"go_to({label})", res["outcome"], ok)
            return ok
        raise TypeError(f"go_to() needs an Obj, Room or Place, got {type(target).__name__}")

    def check(self, question, obj=None):
        self._count()
        rt = self._rt
        call = f"check({str(question)!r})"
        if obj is None or obj._id < 0:
            self._trace(call, "no image (needs an object)", ok=False)
            return None
        try:
            crop = rt.memory_view().best_view_crop(obj._id, 0.15, 512)
        except (KeyError, FileNotFoundError, RuntimeError):
            self._trace(call, "no image of the object", ok=False)
            return None
        msgs = [{"role": "user", "content": [
            text_part(f"Look at the {obj.name} in this image. Question: {question}\n"
                      'Answer with JSON {"answer": "yes"} or {"answer": "no"}, or {"answer": "unsure"} '
                      "if the image does not show it."), image_part(crop)]}]
        try:
            raw = rt.vlm.chat(msgs, CHECK_SCHEMA, rt.cfg.vlm_timeout_s)
            ans = str(extract_json(raw).get("answer", "")).lower()
        except Exception as e:                                   # timeout, bad reply: unknown
            self._trace(call, f"error {type(e).__name__}", ok=False)
            return None
        rt.log("check", question=str(question), node_id=obj._id, raw=raw)
        val = True if ans == "yes" else False if ans == "no" else None
        self._trace(call, ans or "unsure")
        return val

    def report(self, obj):
        self._count()
        if not isinstance(obj, Obj) or obj._id < 0:
            self._trace("report(?)", "not an object", ok=False)
            return None
        self._rt.report(obj)
        self._trace(f"report({obj.target or obj.name})", f"reported {obj.name} #{obj._id}")
        return None

    def fail(self, reason):
        self._trace(f"fail({str(reason)!r})", "task failed", ok=False)
        raise TaskFailed(str(reason))
