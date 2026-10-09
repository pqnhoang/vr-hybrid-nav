"""Target acquisition orchestration (acquisition spec §9): symbolic top-k -> VLM verification,
next k on rejection, per-target blacklist, best-view pose as goal."""
from __future__ import annotations

import time

from vlnav.acquisition.config import AcquisitionConfig
from vlnav.acquisition.label_matcher import LabelMatcher, norm_label
from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.models import AcquisitionResult, Candidate
from vlnav.acquisition.symbolic_filter import SymbolicFilter
from vlnav.acquisition.verifier import NeuralVerifier
from vlnav.acquisition.vlm_client import VLMClient


class TargetAcquirer:
    def __init__(self, cfg: AcquisitionConfig, matcher: LabelMatcher, vlm: VLMClient | None):
        self.cfg = cfg
        self.filter = SymbolicFilter(cfg.symbolic, matcher)
        self.verifier = NeuralVerifier(cfg.verifier, cfg.context, vlm, cfg.logging.debug_jsonl) \
            if vlm is not None else None
        self._blacklist: dict[str, set[int]] = {}
        # ablation switches (eval CLI)
        self.k = cfg.symbolic.k
        self.use_vlm = vlm is not None
        self.include_full_frame: bool | None = None
        self.use_neighbors = True

    # ---------------------------------------------------------------------------- blacklist
    def blacklist(self, target: str, node_id: int) -> None:
        self._blacklist.setdefault(norm_label(target), set()).add(node_id)

    def reset_blacklist(self, target: str | None = None) -> None:
        if target is None:
            self._blacklist.clear()
        else:
            self._blacklist.pop(norm_label(target), None)

    def blacklisted(self, target: str) -> set[int]:
        return set(self._blacklist.get(norm_label(target), set()))

    # ------------------------------------------------------------------------------ acquire
    def _forced(self, ids: list[int] | None, memory: MemoryView, exclude: set[int],
                until_t: float | None, target: str) -> list[Candidate]:
        out = []
        visible = {n.id for n in memory.all_objects(until_t=until_t)}
        for i in ids or []:
            try:
                n = memory.get_object(i)
            except KeyError:
                continue                                  # hallucinated id: ignored (AUDIT 26)
            if n.id in exclude or n.id not in visible:
                continue
            sim = self.filter.matcher.score(target, [n.label])[n.label]
            out.append(Candidate(node=n, match_score=sim, rank_score=n.best_score, forced=True))
        return out

    def acquire(self, instruction: str, target: str, memory: MemoryView,
                until_t: float | None = None,
                force_include: list[int] | None = None) -> AcquisitionResult:
        t0 = time.perf_counter()
        lat = {"filter": 0.0, "vlm": 0.0}
        bl = self._blacklist.setdefault(norm_label(target), set())
        sent: list[int] = []

        def result(status, node=None, reason=None, conf=None) -> AcquisitionResult:
            lat["total"] = time.perf_counter() - t0
            return AcquisitionResult(status=status, node_id=node.id if node else None,
                                     goal_pose=node.best_view_pose if node else None,
                                     candidates=sent, vlm_reason=reason, vlm_confidence=conf,
                                     latency_s=dict(lat))

        last_reason = last_conf = None
        for rnd in range(self.cfg.acquirer.max_rounds):
            a = time.perf_counter()
            cands = self.filter.topk(target, memory, self.k, exclude=bl, until_t=until_t)
            if rnd == 0:
                have = {c.node.id for c in cands}
                cands += [c for c in self._forced(force_include, memory, bl, until_t, target)
                          if c.node.id not in have]
            lat["filter"] += time.perf_counter() - a
            if not cands:
                return result("no_candidates" if rnd == 0 else "rejected_all")
            sent += [c.node.id for c in cands]
            if not self.use_vlm:                          # --no-vlm ablation: symbolic top-1
                return result("selected", cands[0].node)
            out = self.verifier.verify(instruction, target, cands, memory,
                                       self.include_full_frame, self.use_neighbors)
            lat["vlm"] += out.latency_s
            if out.status == "error":
                return result("error", reason=out.error)
            if out.status == "selected":
                return result("selected", memory.get_object(out.node_id), out.reasoning, out.confidence)
            bl |= {c.node.id for c in cands}
            last_reason, last_conf = out.reasoning, out.confidence
        return result("rejected_all", reason=last_reason, conf=last_conf)
