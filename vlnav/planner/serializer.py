"""Scene graph -> compact planner text under a token budget (planner spec §5.2).

    ROOMS
    R0 bedroom (18.2 m2, adj: R1) objects: jacket[#17 .62], chair[#9 .71], +3 more
    UNASSIGNED: box[#30 .33]
    EXPLORATION: known free area 142 m2; frontiers remaining: yes

Objects relevant to the current targets come first (label similarity), then by score. Every
object carries its memory id; ids coming back from the VLM are validated (validation.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from vlnav.acquisition.label_matcher import LabelMatcher
from vlnav.acquisition.memory_api import MemoryView
from vlnav.common.models import ObjectNode
from vlnav.planner.config import PlannerConfig


@dataclass(frozen=True)
class ExplorationSummary:
    known_free_m2: float | None = None
    frontiers_remaining: bool | None = None


def item(n: ObjectNode) -> str:
    return f"{n.label}[#{n.id} .{min(99, round(n.best_score * 100)):02d}]"


class GraphSerializer:
    def __init__(self, cfg: PlannerConfig, matcher: LabelMatcher | None):
        self.cfg = cfg
        self.matcher = matcher

    def relevance(self, nodes: list[ObjectNode], texts: list[str]) -> dict[int, float]:
        if not texts or self.matcher is None or not nodes:
            return {n.id: 0.0 for n in nodes}
        labels = [n.label for n in nodes]
        best = {lbl: 0.0 for lbl in labels}
        for t in texts:
            for lbl, s in self.matcher.score(t, labels).items():
                best[lbl] = max(best[lbl], s)
        return {n.id: best[n.label] for n in nodes}

    def ordered(self, memory: MemoryView, texts: list[str]) -> list[tuple[ObjectNode, float]]:
        nodes = memory.all_objects()
        rel = self.relevance(nodes, texts)
        thr = self.cfg.relevant_sim_threshold
        # relevance is a gate (CLIP similarities are too compressed to rank on), then score
        return sorted(((n, rel[n.id]) for n in nodes),
                      key=lambda x: (-(x[1] >= thr), -x[0].best_score, x[0].id))

    def serialize(self, memory: MemoryView, targets: list[str],
                  exploration: ExplorationSummary | None = None) -> str:
        budget = self.cfg.max_graph_tokens * self.cfg.chars_per_token
        rooms = memory.rooms()
        order = self.ordered(memory, targets)
        header = [f"R{r.id} {r.label} ({r.area_m2:.1f} m2"
                  + (f", adj: {','.join(f'R{a}' for a in r.adjacent_room_ids)}" if r.adjacent_room_ids else "")
                  + ")" for r in rooms]
        tail = []
        if exploration is not None and exploration.known_free_m2 is not None:
            fr = "unknown" if exploration.frontiers_remaining is None else \
                ("yes" if exploration.frontiers_remaining else "no")
            tail.append(f"EXPLORATION: known free area {exploration.known_free_m2:.0f} m2; "
                        f"frontiers remaining: {fr}")
        used = sum(len(h) + 12 for h in header) + sum(len(t) + 1 for t in tail) + 40
        included: set[int] = set()
        for n, _ in order:                              # greedy fill by priority
            cost = len(item(n)) + 2
            if used + cost > budget:
                break
            included.add(n.id)
            used += cost
        room_ids = {r.id for r in rooms}
        groups: dict[int | None, list[ObjectNode]] = {}
        for n, _ in order:                              # keeps priority order inside a room
            groups.setdefault(n.room_id if n.room_id in room_ids else None, []).append(n)

        def line(nodes: list[ObjectNode]) -> str:
            shown = [item(n) for n in nodes if n.id in included]
            more = len(nodes) - len(shown)
            return ", ".join(shown + ([f"+{more} more"] if more else [])) or "none"

        lines = []
        if rooms:
            lines.append("ROOMS")
            for r, h in zip(rooms, header):
                lines.append(f"{h} objects: {line(groups.get(r.id, []))}")
            if groups.get(None):
                lines.append(f"UNASSIGNED: {line(groups[None])}")
        else:
            lines.append(f"OBJECTS: {line(groups.get(None, []))}")
        return "\n".join(lines + tail)
