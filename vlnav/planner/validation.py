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
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vlnav.acquisition.memory_api import MemoryView
from vlnav.planner.state import PlannerState, TargetEntry


@dataclass
class PlannedSubtask:
    type: str                  # explore | goto
    target: str
    node_id: int | None = None
    room_hint: str | None = None


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


def _subtasks(raw, targets: dict[str, TargetEntry], memory: MemoryView, add_unknown: bool,
              fixes: list[str]) -> list[PlannedSubtask]:
    out = []
    for s in raw if isinstance(raw, list) else []:
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


def _ensure_progress(plan: ValidatedPlan) -> None:
    if plan.task_done or plan.subtasks:
        return
    pending = [t for t in plan.targets if t.active]
    if pending:
        plan.subtasks.append(PlannedSubtask("explore", pending[0].name))
        plan.fixes.append(f"no subtask while targets pending -> explore {pending[0].name!r}")


def validate_decomposition(data: dict, memory: MemoryView) -> ValidatedPlan:
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
    subtasks = _subtasks(data.get("subtasks"), targets, memory, True, fixes)
    plan = ValidatedPlan(list(targets.values()), subtasks, fixes, False, str(data.get("reasoning", "")))
    _ensure_progress(plan)
    return plan


def validate_replan(data: dict, state: PlannerState, memory: MemoryView) -> ValidatedPlan:
    fixes: list[str] = []
    targets = {_key(t.name): t.model_copy(deep=True) for t in state.targets}
    for u in data.get("target_updates", []) if isinstance(data.get("target_updates"), list) else []:
        t = targets.get(_key(u.get("name", ""))) if isinstance(u, dict) else None
        if t is None:
            fixes.append(f"dropped update for unknown target {u!r}")
        elif str(u.get("status")) == "failed" and t.active:
            t.status, t.notes = "failed", str(u.get("reason", "")).strip()
    subtasks = _subtasks(data.get("subtasks"), targets, memory, False, fixes)
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
