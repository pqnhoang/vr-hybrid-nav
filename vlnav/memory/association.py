"""Observation -> node association (TASK_unified_memory.md §6.3) [DESIGN].

Candidates within `assoc_radius_m`, semantic gate, geometric gate (voxel overlap or close
centroids), cost `1 - overlap + w_dist * dist / assoc_radius_m`, and one-to-one Hungarian matching
per frame so that two distinct objects seen in the same frame never merge into one node.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from vlnav.memory.config import AssociationCfg
from vlnav.memory.consolidate import ObsGroup
from vlnav.memory.spatial_index import SpatialIndex
from vlnav.memory.types import NodeState

LabelSim = Callable[[str, str], float]
VoxelKey = tuple[int, int, int]

_INVALID = 1e9   # sentinel cost for gated-out pairs (real costs are <= 1 + w_dist)


def voxel_keys(points: np.ndarray, voxel_m: float) -> set[VoxelKey]:
    if len(points) == 0:
        return set()
    keys = np.unique(np.floor(np.asarray(points) / voxel_m).astype(np.int64), axis=0)
    return set(map(tuple, keys.tolist()))


def voxel_centers(keys, voxel_m: float) -> np.ndarray:
    arr = np.asarray(list(keys), dtype=float).reshape(-1, 3)
    return (arr + 0.5) * voxel_m


def voxel_overlap(obs_vox, node_vox, norm: str) -> float:
    """|obs ∩ node| / |obs| ("observation", spec) or / min(|obs|, |node|) ("min")."""
    if not obs_vox or not node_vox:
        return 0.0
    inter = len(obs_vox & node_vox)
    if norm == "observation":
        return inter / len(obs_vox)
    return inter / min(len(obs_vox), len(node_vox))


def semantic_gate(votes_a: Mapping[str, float], votes_b: Mapping[str, float],
                  label_a: str, label_b: str, label_sim: LabelSim | None, min_sim: float) -> bool:
    """Any shared label, or embedding similarity of the top labels >= min_sim."""
    if votes_a.keys() & votes_b.keys():
        return True
    return label_sim is not None and label_sim(label_a, label_b) >= min_sim


@dataclass(slots=True)
class Match:
    obs_idx: int
    node_id: int
    overlap: float
    dist: float


def associate(groups: list[ObsGroup], group_voxels: list[set[VoxelKey]],
              nodes: Mapping[int, NodeState], index: SpatialIndex, cfg: AssociationCfg,
              label_sim: LabelSim | None) -> tuple[list[Match], list[int]]:
    """Return (matches, indices of unmatched groups)."""
    pairs: dict[tuple[int, int], tuple[float, float]] = {}
    for i, g in enumerate(groups):
        for node_id, dist in index.query_radius(g.centroid, cfg.assoc_radius_m):
            ns = nodes[node_id]
            if not semantic_gate(g.label_votes, ns.label_votes, g.label, ns.node.label,
                                 label_sim, cfg.assoc_label_sim):
                continue
            vox = group_voxels[i]
            overlap = voxel_overlap(vox, ns.voxels.keys(), cfg.overlap_norm)
            if overlap >= cfg.assoc_overlap_min or dist < cfg.assoc_close_m:
                pairs[(i, node_id)] = (overlap, dist)

    if not pairs:
        return [], list(range(len(groups)))

    node_ids = sorted({nid for _, nid in pairs})
    col = {nid: j for j, nid in enumerate(node_ids)}
    cost = np.full((len(groups), len(node_ids)), _INVALID)
    for (i, nid), (overlap, dist) in pairs.items():
        cost[i, col[nid]] = 1.0 - overlap + cfg.w_dist * dist / cfg.assoc_radius_m

    matches = []
    for r, c in zip(*linear_sum_assignment(cost)):
        if cost[r, c] < _INVALID:
            overlap, dist = pairs[(int(r), node_ids[c])]
            matches.append(Match(int(r), node_ids[c], overlap, dist))
    matched = {m.obs_idx for m in matches}
    return matches, [i for i in range(len(groups)) if i not in matched]
