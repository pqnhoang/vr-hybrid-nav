"""Stage 1, symbolic filtering (acquisition spec §7.2): label match -> score gate -> top-k."""
from __future__ import annotations

from vlnav.acquisition.config import SymbolicCfg
from vlnav.acquisition.label_matcher import LabelMatcher
from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.models import Candidate


class SymbolicFilter:
    def __init__(self, cfg: SymbolicCfg, matcher: LabelMatcher):
        self.cfg = cfg
        self.matcher = matcher

    def candidates(self, target: str, memory: MemoryView, exclude: set[int] = frozenset(),
                   until_t: float | None = None) -> list[Candidate]:
        """All matching, non-excluded nodes above min_det_score, best first (deterministic)."""
        nodes = memory.all_objects(until_t=until_t)
        matched = self.matcher.matches(target, [n.label for n in nodes])
        out = []
        for n in nodes:
            if n.id in exclude or n.label not in matched or n.best_score < self.cfg.min_det_score:
                continue
            sim = matched[n.label]
            rank = n.best_score if self.cfg.rank_strategy == "confidence" else n.best_score * sim
            out.append(Candidate(node=n, match_score=sim, rank_score=rank))
        out.sort(key=lambda c: (-c.rank_score, -c.node.n_obs, c.node.id))
        return out

    def topk(self, target: str, memory: MemoryView, k: int, exclude: set[int] = frozenset(),
             until_t: float | None = None) -> list[Candidate]:
        return self.candidates(target, memory, exclude, until_t)[:k]
