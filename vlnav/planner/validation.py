"""Plan repair rules applied to every decomposer / replanner output (planner spec §5.5).

- Unknown `node_id` or `goto` without a node -> `explore` for that target.
- Subtasks for targets not in the list -> dropped (decomposition: the target is added instead).
- Subtasks for reported / failed targets -> dropped.
- Empty subtask list while targets are pending -> `explore` the first pending target.
- `task_done` is ignored while a target is still active ➕: a target ends only when reported or
  explicitly failed in `target_updates` (the VLM confused "candidate seen" with "reported").
- No parsable reply (after the repair retry) -> deterministic fallback (explore pending targets
  in order); a decomposition without any target fails the task.
- `detector_prompts` of a subtask = the target's prompts (attached at dispatch).
- `move` (only with `allow_move`, TASK_move_subtask.md §10.2): a route for DualVLN, not a target.
  It needs a non-empty `instruction`; `node_id` is ignored; the label defaults to "route". A route
  that already failed `max_move_attempts` times is dropped. Without `allow_move` it is dropped.
- Landmarks (decomposition with a `move`): a target whose head noun occurs in the instruction only
  right after a landmark word ("walk past the sofa", "stop near the plant") describes the route,
  not an object to find -> dropped with its subtasks. Implicit targets (not in the text) are kept.
- Object search as a route: a `move` whose instruction names an active target, not as a landmark
  ("take me to an armchair"), is an object search -> `explore` that target (searching objects is
  the planner's job, with memory; TASK_move_subtask.md §16).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from vlnav.acquisition.memory_api import MemoryView
from vlnav.planner.state import PlannerState, TargetEntry


@dataclass
class PlannedSubtask:
    type: str                  # explore | goto | move
    target: str                # move: short label of the route
    node_id: int | None = None
    room_hint: str | None = None
    instruction: str | None = None     # move only


@dataclass
class ValidatedPlan:
    targets: list[TargetEntry]
    subtasks: list[PlannedSubtask]
    fixes: list[str] = field(default_factory=list)
    task_done: bool = False
    reasoning: str = ""


def _key(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _clean_prompts(name: str, prompts) -> list[str]:
    out = []
    for p in prompts if isinstance(prompts, list) else []:
        p = " ".join(str(p).strip().split())
        if p and p.lower() not in {q.lower() for q in out}:
            out.append(p)
    return out or [name]


def _node_ok(memory: MemoryView, node_id) -> int | None:
    try:
        return memory.get_object(int(node_id)).id
    except (KeyError, TypeError, ValueError):
        return None


def move_failures(history: list[dict]) -> dict[str, int]:
    """Failed `move` subtasks per route label (nav_failed / timeout)."""
    out: dict[str, int] = {}
    for h in history:
        if h.get("type") == "move" and h.get("outcome") != "reached":
            out[_key(h["target"])] = out.get(_key(h["target"]), 0) + 1
    return out


def _move(s: dict, allow: bool, failures: dict[str, int], max_attempts: int,
          fixes: list[str]) -> PlannedSubtask | None:
    instruction = " ".join(str(s.get("instruction") or "").split())
    label = " ".join(str(s.get("target") or "").split()) or "route"
    if not allow:
        fixes.append(f"dropped move {label!r}: move subtasks are disabled")
    elif not instruction:
        fixes.append(f"dropped move {label!r} without an instruction")
    elif failures.get(_key(label), 0) >= max_attempts:
        fixes.append(f"dropped move {label!r}: failed {failures[_key(label)]} times")
    else:
        if s.get("node_id") is not None:
            fixes.append(f"move {label!r}: node_id {s['node_id']!r} ignored")
        return PlannedSubtask("move", label, None, None, instruction)
    return None


def _subtasks(raw, targets: dict[str, TargetEntry], memory: MemoryView, add_unknown: bool,
              fixes: list[str], allow_move: bool = False, move_fails: dict[str, int] | None = None,
              max_move_attempts: int = 3) -> list[PlannedSubtask]:
    out = []
    for s in raw if isinstance(raw, list) else []:
        if isinstance(s, dict) and str(s.get("type", "")).lower() == "move":
            m = _move(s, allow_move, move_fails or {}, max_move_attempts, fixes)
            if m is not None:
                out.append(m)
            continue
        if not isinstance(s, dict) or not s.get("target"):
            fixes.append(f"dropped malformed subtask {s!r}")
            continue
        name = str(s["target"]).strip()
        key = _key(name)
        if key not in targets:
            if not add_unknown:
                fixes.append(f"dropped subtask for unknown target {name!r}")
                continue
            targets[key] = TargetEntry(name=name, detector_prompts=[name])
            fixes.append(f"added target {name!r} from a subtask")
        t = targets[key]
        if not t.active:
            fixes.append(f"dropped subtask for {t.status} target {t.name!r}")
            continue
        kind = str(s.get("type", "explore")).lower()
        hint = s.get("room_hint") or None
        if kind == "goto":
            node = _node_ok(memory, s.get("node_id"))
            if node is None:
                fixes.append(f"goto {t.name!r} with invalid node {s.get('node_id')!r} -> explore")
                out.append(PlannedSubtask("explore", t.name, None, hint))
                continue
            out.append(PlannedSubtask("goto", t.name, node, hint))
        else:
            if kind != "explore":
                fixes.append(f"unknown subtask type {kind!r} -> explore")
            out.append(PlannedSubtask("explore", t.name, None, hint))
    return out


LANDMARK_WORDS = {"past", "near", "along", "by", "beside", "behind", "toward", "towards", "at", "around",
                  "between", "next", "front", "follow", "following", "passing"}


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _mentions(text: str, name: str, window: int = 4) -> list[bool]:
    """One flag per occurrence of the head noun of `name` in `text`: True when it follows a landmark word."""
    words = [_stem(w) for w in re.findall(r"[a-z]+", text.lower())]
    names = re.findall(r"[a-z]+", name.lower())
    if not names:
        return []
    head = _stem(names[-1])
    return [bool(LANDMARK_WORDS & set(words[max(0, i - window):i])) for i, w in enumerate(words) if w == head]


def _landmarks(instruction: str, targets: dict[str, TargetEntry]) -> list[str]:
    """Keys of targets named in the instruction only as route landmarks (see module doc)."""
    return [key for key, t in targets.items() if (m := _mentions(instruction, t.name)) and all(m)]


def _object_moves(subtasks: list[PlannedSubtask], targets: dict[str, TargetEntry],
                  fixes: list[str]) -> list[PlannedSubtask]:
    """A move that names an active target outside a landmark phrase -> explore that target."""
    out = []
    for s in subtasks:
        if s.type == "move":
            hit = next((t for t in targets.values() if t.active
                        and any(not lm for lm in _mentions(s.instruction or "", t.name))), None)
            if hit is not None:
                planned = any(o.type != "move" and _key(o.target) == _key(hit.name) for o in subtasks)
                fixes.append(f"move {s.instruction!r} searches target {hit.name!r} -> "
                             + ("dropped (already planned)" if planned else "explore"))
                if planned:
                    continue
                s = PlannedSubtask("explore", hit.name)
        out.append(s)
    return out


def _ensure_progress(plan: ValidatedPlan) -> None:
    if plan.task_done or plan.subtasks:
        return
    pending = [t for t in plan.targets if t.active]
    if pending:
        plan.subtasks.append(PlannedSubtask("explore", pending[0].name))
        plan.fixes.append(f"no subtask while targets pending -> explore {pending[0].name!r}")


def validate_decomposition(data: dict, memory: MemoryView, allow_move: bool = False,
                           instruction: str = "") -> ValidatedPlan:
    fixes: list[str] = []
    targets: dict[str, TargetEntry] = {}
    for t in data.get("targets", []) if isinstance(data.get("targets"), list) else []:
        if not isinstance(t, dict) or not str(t.get("name", "")).strip():
            fixes.append(f"dropped malformed target {t!r}")
            continue
        name = " ".join(str(t["name"]).split())
        if _key(name) in targets:
            fixes.append(f"dropped duplicate target {name!r}")
            continue
        targets[_key(name)] = TargetEntry(
            name=name, detector_prompts=_clean_prompts(name, t.get("detector_prompts")),
            room_priors=[str(r) for r in t.get("room_priors", []) if r] if isinstance(t.get("room_priors"), list) else [])
    subtasks = _subtasks(data.get("subtasks"), targets, memory, True, fixes, allow_move)
    if any(s.type == "move" for s in subtasks):
        for key in _landmarks(instruction, targets):
            name = targets.pop(key).name
            subtasks = [s for s in subtasks if s.type == "move" or _key(s.target) != key]
            fixes.append(f"dropped target {name!r}: only a landmark of a route")
        subtasks = _object_moves(subtasks, targets, fixes)
    plan = ValidatedPlan(list(targets.values()), subtasks, fixes, False, str(data.get("reasoning", "")))
    _ensure_progress(plan)
    return plan


def validate_replan(data: dict, state: PlannerState, memory: MemoryView, allow_move: bool = False,
                    max_move_attempts: int = 3) -> ValidatedPlan:
    fixes: list[str] = []
    targets = {_key(t.name): t.model_copy(deep=True) for t in state.targets}
    for u in data.get("target_updates", []) if isinstance(data.get("target_updates"), list) else []:
        t = targets.get(_key(u.get("name", ""))) if isinstance(u, dict) else None
        if t is None:
            fixes.append(f"dropped update for unknown target {u!r}")
        elif str(u.get("status")) == "failed" and t.active:
            t.status, t.notes = "failed", str(u.get("reason", "")).strip()
    subtasks = _subtasks(data.get("subtasks"), targets, memory, False, fixes, allow_move,
                         move_failures(state.history), max_move_attempts)
    if any(s.type == "move" for s in subtasks):
        subtasks = _object_moves(subtasks, targets, fixes)
    plan = ValidatedPlan(list(targets.values()), subtasks, fixes, bool(data.get("task_done", False)),
                         str(data.get("reasoning", "")))
    if plan.task_done and any(t.active for t in plan.targets):
        # a target ends only when reported or explicitly failed (with a reason) -> keep going
        plan.task_done = False
        fixes.append("task_done ignored: targets still active "
                     + str([t.name for t in plan.targets if t.active]))
    _ensure_progress(plan)
    return plan


def fallback_plan(state: PlannerState) -> ValidatedPlan:
    targets = [t.model_copy(deep=True) for t in state.targets]
    subtasks = [PlannedSubtask("explore", t.name) for t in targets if t.active]
    return ValidatedPlan(targets, subtasks, ["fallback: explore pending targets in order"])
