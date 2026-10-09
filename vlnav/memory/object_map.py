"""Object node lifecycle (TASK_unified_memory.md §6.4–6.6).

Create / merge / best view / confirmed-vs-tentative / dedup with aliases / prune. Every change
replaces `NodeState.node` with a new `ObjectNode` (copy-on-write), so snapshots can share nodes.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from vlnav.common.models import Frame, ObjectNode, Pose
from vlnav.mapping.occupancy import OCCUPIED, OccupancyGrid2D
from vlnav.memory.association import LabelSim, associate, semantic_gate, voxel_centers, voxel_keys
from vlnav.memory.config import MemoryConfig
from vlnav.memory.consolidate import ObsGroup
from vlnav.memory.image_store import ImageStore
from vlnav.memory.spatial_index import SpatialIndex
from vlnav.memory.types import NodeState

Emit = Callable[[str, dict], None]


def view_quality(score: float, bbox: tuple[int, int, int, int], image_hw: tuple[int, int],
                 border_penalty: float, small_penalty: float, min_area_px: float) -> float:
    """q = score - border_penalty·[bbox touches border] - small_penalty·[area < min] (§6.5)."""
    x1, y1, x2, y2 = bbox
    h, w = image_hw
    q = score
    if x1 <= 0 or y1 <= 0 or x2 >= w - 1 or y2 >= h - 1:
        q -= border_penalty
    if (x2 - x1) * (y2 - y1) < min_area_px:
        q -= small_penalty
    return q


def _argmax_label(votes: dict[str, float], current: str | None) -> str:
    # ties keep the current label, then the alphabetically first one -> deterministic
    return max(sorted(votes), key=lambda lbl: (votes[lbl], lbl == current))


def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


class ObjectMap:
    def __init__(self, cfg: MemoryConfig, image_store: ImageStore, grid: OccupancyGrid2D,
                 label_sim: LabelSim | None, emit: Emit):
        self.cfg = cfg
        self.images = image_store
        self.grid = grid
        self.label_sim = label_sim
        self.emit = emit
        self.nodes: dict[int, NodeState] = {}
        self.aliases: dict[int, int] = {}
        self._next_id = 0
        self._index: SpatialIndex | None = None

    # ------------------------------------------------------------------ queries
    def resolve(self, node_id: int) -> int:
        while node_id in self.aliases:
            node_id = self.aliases[node_id]
        return node_id

    def index(self) -> SpatialIndex:
        """KD-tree over all nodes (incl. tentative), rebuilt lazily after changes."""
        if self._index is None:
            ids = sorted(self.nodes)
            pts = np.array([self.nodes[i].node.centroid for i in ids], dtype=float).reshape(-1, 3)
            self._index = SpatialIndex(ids, pts)
        return self._index

    # ------------------------------------------------------------------ write path
    def integrate(self, groups: list[ObsGroup], frame: Frame) -> set[int]:
        """Associate consolidated observations of one frame; return ids of changed nodes."""
        if not groups:
            return set()
        vox = [voxel_keys(g.points, self.cfg.association.assoc_voxel_m) for g in groups]
        matches, unmatched = associate(groups, vox, self.nodes, self.index(), self.cfg.association,
                                       self.label_sim)
        changed = set()
        for m in matches:
            self._merge_obs(self.nodes[m.node_id], groups[m.obs_idx], vox[m.obs_idx], frame)
            changed.add(m.node_id)
        for i in unmatched:
            changed.add(self._create(groups[i], vox[i], frame))
        self._index = None
        return changed

    def set_room(self, node_id: int, room_id: int | None) -> bool:
        ns = self.nodes[node_id]
        if ns.node.room_id == room_id:
            return False
        ns.node = ns.node.model_copy(update={"room_id": room_id})
        return True

    # ------------------------------------------------------------------ internals
    def _quality(self, g: ObsGroup) -> float:
        bv = self.cfg.best_view
        q = view_quality(g.score, g.bbox, g.image_hw, bv.border_penalty, bv.small_penalty,
                         bv.best_view_min_area_px)
        return q if self._view_allowed(g.T_map_base) else -np.inf

    def _view_allowed(self, T_map_base: np.ndarray) -> bool:
        """`best_view_require_free`: the base pose must not lie in an occupied cell."""
        if not self.cfg.best_view.best_view_require_free:
            return True
        cell = self.grid.world_to_cell(np.asarray(T_map_base)[:2, 3])
        return int(self.grid.value(cell)[0]) != OCCUPIED

    def _geometry(self, voxels: dict) -> tuple[np.ndarray, tuple[float, float, float]]:
        pts = _readonly(voxel_centers(voxels.keys(), self.cfg.association.assoc_voxel_m))
        c = np.median(pts, axis=0)
        return pts, (float(c[0]), float(c[1]), float(c[2]))

    def _cap(self, voxels: dict) -> None:
        excess = len(voxels) - self.cfg.nodes.max_node_voxels
        if excess > 0:
            for key in list(itertools.islice(voxels, excess)):   # drop oldest
                del voxels[key]

    def _is_confirmed(self, n_obs: int, best_score: float) -> bool:
        return n_obs >= self.cfg.nodes.min_obs or best_score >= self.cfg.nodes.confirm_score

    def _create(self, g: ObsGroup, vox: set, frame: Frame) -> int:
        nid = self._next_id
        self._next_id += 1
        voxels = dict.fromkeys(sorted(vox))
        self._cap(voxels)
        points, centroid = self._geometry(voxels)
        path = self.images.put_frame(frame)
        self.images.acquire(path)
        node = ObjectNode(
            id=nid, label=g.label, label_votes=dict(g.label_votes), centroid=centroid,
            points=points, best_score=g.score,
            best_view_pose=Pose.from_matrix(g.T_map_base, g.stamp),
            best_view_cam_pose=Pose.from_matrix(g.T_map_cam, g.stamp),
            best_view_image_path=path, best_view_bbox=g.bbox, room_id=None, n_obs=1,
            first_seen_t=g.stamp, last_seen_t=g.stamp)
        self.nodes[nid] = NodeState(
            node=node, voxels=voxels, label_votes=dict(g.label_votes),
            best_view_history=[(g.stamp, path)], best_q=self._quality(g),
            confirmed=self._is_confirmed(1, g.score), created_t=g.stamp, updated_t=g.stamp)
        self.emit("node_created", {"id": nid, "label": g.label, "score": g.score})
        return nid

    def _merge_obs(self, ns: NodeState, g: ObsGroup, vox: set, frame: Frame) -> None:
        for key in sorted(vox - ns.voxels.keys()):
            ns.voxels[key] = None
        self._cap(ns.voxels)
        for lbl, s in g.label_votes.items():
            ns.label_votes[lbl] = ns.label_votes.get(lbl, 0.0) + s
        points, centroid = self._geometry(ns.voxels)
        n_obs = ns.node.n_obs + 1
        best_score = max(ns.node.best_score, g.score)
        update = {
            "label": _argmax_label(ns.label_votes, ns.node.label),
            "label_votes": dict(ns.label_votes), "centroid": centroid, "points": points,
            "n_obs": n_obs, "best_score": best_score,
            "last_seen_t": max(ns.node.last_seen_t, g.stamp),
        }
        q = self._quality(g)
        if q > ns.best_q + self.cfg.best_view.best_view_min_gain:
            update.update(self._replace_view(ns, g, frame))
            ns.best_q = q
        ns.node = ns.node.model_copy(update=update)
        ns.confirmed = ns.confirmed or self._is_confirmed(n_obs, best_score)
        ns.updated_t = g.stamp
        self.emit("node_merged", {"id": ns.node.id, "label": g.label, "score": g.score})

    def _replace_view(self, ns: NodeState, g: ObsGroup, frame: Frame) -> dict:
        path = self.images.put_frame(frame)
        self.images.acquire(path)
        self.images.release(ns.node.best_view_image_path)
        ns.best_view_history.append((g.stamp, path))
        self.emit("best_view_updated", {"id": ns.node.id, "image": path, "score": g.score})
        return {"best_view_image_path": path, "best_view_bbox": g.bbox, 
                "best_view_pose": Pose.from_matrix(g.T_map_base, g.stamp),
                "best_view_cam_pose": Pose.from_matrix(g.T_map_cam, g.stamp)}

    # ------------------------------------------------------------------ maintenance
    def revalidate_best_views(self) -> None:
        """Views whose base pose became occupied lose their quality -> next observation replaces."""
        if not self.cfg.best_view.best_view_require_free:
            return
        for ns in self.nodes.values():
            if np.isfinite(ns.best_q) and not self._view_allowed(ns.node.best_view_pose.matrix()):
                ns.best_q = -np.inf

    def dedup(self) -> set[int]:
        """Merge duplicate confirmed nodes into the older id; return ids of kept nodes."""
        a_cfg, m_cfg = self.cfg.association, self.cfg.maintenance
        ids = sorted(i for i, ns in self.nodes.items() if ns.confirmed)
        if len(ids) < 2:
            return set()
        pts = np.array([self.nodes[i].node.centroid for i in ids], dtype=float)
        kept = set()
        for a, b, dist in SpatialIndex(ids, pts).pairs_within(a_cfg.assoc_radius_m):
            if a not in self.nodes or b not in self.nodes:
                continue                                   # already merged this round
            na, nb = self.nodes[a], self.nodes[b]
            close = dist < a_cfg.assoc_close_m and semantic_gate(
                na.label_votes, nb.label_votes, na.node.label, nb.node.label, self.label_sim,
                a_cfg.assoc_label_sim)
            smaller = min(len(na.voxels), len(nb.voxels))
            overlap = len(na.voxels.keys() & nb.voxels.keys()) / smaller if smaller else 0.0
            if close or overlap >= m_cfg.dedup_overlap_min:
                self._merge_nodes(keep=a, drop=b)          # ids are monotonic -> a is older
                kept.add(a)
        if kept:
            self._index = None
        return {self.resolve(k) for k in kept}        # a kept node may be merged again later

    def _merge_nodes(self, keep: int, drop: int) -> None:
        k, d = self.nodes[keep], self.nodes.pop(drop)
        for key in d.voxels:
            k.voxels.setdefault(key, None)
        self._cap(k.voxels)
        for lbl, s in d.label_votes.items():
            k.label_votes[lbl] = k.label_votes.get(lbl, 0.0) + s
        points, centroid = self._geometry(k.voxels)
        update = {
            "label": _argmax_label(k.label_votes, k.node.label),
            "label_votes": dict(k.label_votes), "centroid": centroid, "points": points,
            "n_obs": k.node.n_obs + d.node.n_obs,
            "best_score": max(k.node.best_score, d.node.best_score),
            "first_seen_t": min(k.node.first_seen_t, d.node.first_seen_t),
            "last_seen_t": max(k.node.last_seen_t, d.node.last_seen_t),
        }
        if d.best_q > k.best_q:            # take over drop's view and its image reference
            self.images.release(k.node.best_view_image_path)
            for f in ("best_view_image_path", "best_view_bbox", "best_view_pose",
                      "best_view_cam_pose"):
                update[f] = getattr(d.node, f)
            k.best_q = d.best_q
        else:
            self.images.release(d.node.best_view_image_path)
        k.best_view_history = sorted(k.best_view_history + d.best_view_history)
        k.node = k.node.model_copy(update=update)
        k.created_t = min(k.created_t, d.created_t)
        k.updated_t = max(k.updated_t, d.updated_t)
        self.aliases[drop] = keep
        self.emit("nodes_deduped", {"kept": keep, "merged": drop})

    def prune(self, t: float) -> list[int]:
        """Remove tentative nodes older than prune_age_s. Confirmed nodes are never pruned."""
        age = self.cfg.maintenance.prune_age_s
        pruned = [i for i, ns in self.nodes.items() if not ns.confirmed and t - ns.created_t > age]
        for i in pruned:
            ns = self.nodes.pop(i)
            self.images.release(ns.node.best_view_image_path)
            self.emit("node_pruned", {"id": i, "label": ns.node.label})
        if pruned:
            self._index = None
        return pruned
