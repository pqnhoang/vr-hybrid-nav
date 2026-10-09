"""UnifiedMemory: 3D scene graph (objects + rooms) + object-centric image memory.

TASK_unified_memory.md §5. Single writer (the core-process loop calls `update` per frame); readers
get immutable `MemorySnapshot`s from `view()` / `snapshot(t)`. `update` also runs `tick`, so the
SYSTEM §6 loop needs only `memory.update(res, info)`; calling `tick(t)` separately is harmless.
"""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from vlnav.acquisition.label_matcher import TextEmbedder
from vlnav.common.models import FrameResult, ObjectNode, Observation, RoomNode
from vlnav.mapping.occupancy import FREE, OccupancyGrid2D, UpdateInfo
from vlnav.memory.checkpoint import CheckpointHistory, save_checkpoint
from vlnav.memory.config import MemoryConfig
from vlnav.memory.consolidate import consolidate
from vlnav.memory.embedding import LabelSimilarity
from vlnav.memory.image_store import ImageStore
from vlnav.memory.object_map import ObjectMap
from vlnav.memory.room_labeler import RoomLabeler
from vlnav.memory.room_tracker import RoomTracker
from vlnav.memory.rooms import segment_rooms
from vlnav.memory.snapshot import MemorySnapshot
from vlnav.memory.types import UNKNOWN_LABEL, MemoryEvent, MemoryStats


def _pct(values, q: float) -> float | None:
    return float(np.percentile(list(values), q)) if values else None


class UnifiedMemory:
    def __init__(self, cfg: MemoryConfig, grid: OccupancyGrid2D, image_store: ImageStore,
                 embedder: TextEmbedder | None = None, labeler: RoomLabeler | None = None,
                 checkpoint_dir: str | Path | None = None):
        self.cfg = cfg
        self.grid = grid
        self.images = image_store
        self.label_sim = LabelSimilarity(embedder) if embedder is not None else None
        self.labeler = labeler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

        self._version = 0
        self._t = -np.inf
        self._events: list[MemoryEvent] = []
        self.objects = ObjectMap(cfg, image_store, grid, self.label_sim, self._emit)
        self.tracker = RoomTracker(cfg.rooms)
        self.history = CheckpointHistory(cfg.checkpoints.max_checkpoints, image_store)
        self._view: MemorySnapshot | None = None

        self._last_cam_t: dict[str, float] = {}
        self._started = False
        self._last_dedup = self._last_prune = self._last_ckpt = self._last_rooms = -np.inf
        self._rooms_grid_version = -1
        self._rooms_free_m2 = 0.0
        self._best_view_grid_version = -1
        self._rooms_dirty = False
        self._label_pool = ThreadPoolExecutor(1, thread_name_prefix="room-labeler") \
            if labeler is not None and cfg.labeling.worker_thread else None
        self._label_jobs: dict[int, tuple[Future, frozenset[int]]] = {}

        w = cfg.stats.window
        self._update_ms: deque[float] = deque(maxlen=w)
        self._room_ms: deque[float] = deque(maxlen=w)
        self._n_processed = self._n_skipped = self._n_gated = 0

    # ------------------------------------------------------------------ write path
    def update(self, res: FrameResult, info: UpdateInfo | None = None) -> None:
        frame = res.frame
        t = frame.stamp
        self._t = max(self._t, t)
        last = self._last_cam_t.get(frame.camera_id)
        if last is not None and t - last < 1.0 / self.cfg.memory_rate_hz:
            self._n_skipped += 1
        else:
            t0 = time.perf_counter()
            self._revalidate_best_views()
            self._last_cam_t[frame.camera_id] = t
            self._n_processed += 1
            kept = [o for o in res.observations if self._passes_gate(o)]
            self._n_gated += len(res.observations) - len(kept)
            masks_by_key = {(d.label, tuple(d.bbox)): d.mask for d in res.detections}
            masks = [masks_by_key.get((o.label, tuple(o.bbox))) for o in kept]
            groups = consolidate(kept, masks, frame.rgb.shape[:2],
                                 self.cfg.consolidation.same_object_mask_iou)
            changed = self.objects.integrate(groups, frame)
            for nid in sorted(changed):
                self._assign_room(nid)
            self._rooms_dirty |= bool(changed)
            self._update_ms.append(1e3 * (time.perf_counter() - t0))
        self.tick(t)

    def _passes_gate(self, o: Observation) -> bool:
        g = self.cfg.gating
        x1, y1, x2, y2 = o.bbox
        rng = float(np.linalg.norm(np.asarray(o.centroid_map) - np.asarray(o.T_map_cam)[:3, 3]))
        return (o.score >= g.min_obs_score and len(o.points_map) >= g.min_points
                and rng <= g.max_obs_range_m and (x2 - x1) * (y2 - y1) >= g.min_bbox_area_px
                and o.label.casefold() not in {lbl.casefold() for lbl in g.ignore_labels})

    def tick(self, t: float) -> None:
        """Periodic jobs: dedup, prune, rooms, room labels, best-view validity, checkpoints."""
        self._t = max(self._t, t)
        if not self._started:          # first dedup / prune one period after start; the
            self._started = True       # first checkpoint is taken right away (state at start)
            self._last_dedup = self._last_prune = t
        m = self.cfg.maintenance
        if t - self._last_dedup >= m.dedup_every_s:
            self._last_dedup = t
            for nid in sorted(self.objects.dedup()):
                self._assign_room(nid)
                self._rooms_dirty = True
        if t - self._last_prune >= m.prune_every_s:
            self._last_prune = t
            self._rooms_dirty |= bool(self.objects.prune(t))
        if self.cfg.rooms.enabled and self._rooms_due(t):
            self._update_rooms(t)
        if self._rooms_dirty:
            self._sync_room_objects()
        if self.cfg.rooms.enabled and self.labeler is not None:
            self._label_rooms()
        self._revalidate_best_views()
        if t - self._last_ckpt >= self.cfg.checkpoints.checkpoint_every_s:
            self._last_ckpt = t
            self._checkpoint()

    def _revalidate_best_views(self) -> None:
        if self.grid.version != self._best_view_grid_version:
            self._best_view_grid_version = self.grid.version
            self.objects.revalidate_best_views()

    # ------------------------------------------------------------------ rooms
    def _rooms_due(self, t: float) -> bool:
        if self.grid.version == self._rooms_grid_version or 0 in self.grid.shape:
            return False
        if t - self._last_rooms >= self.cfg.rooms.room_update_every_s:
            return True
        free_m2 = float((self.grid.as_array() == FREE).sum()) * self.grid.resolution ** 2
        return free_m2 - self._rooms_free_m2 >= self.cfg.rooms.room_update_min_new_m2

    def _update_rooms(self, t: float) -> None:
        t0 = time.perf_counter()
        arr = self.grid.as_array()
        seg = segment_rooms(arr, self.grid.resolution, self.grid.origin_xy, self.cfg.rooms)
        summary = self.tracker.update(seg)
        self._room_ms.append(1e3 * (time.perf_counter() - t0))
        self._last_rooms = t
        self._rooms_grid_version = self.grid.version
        self._rooms_free_m2 = float((arr == FREE).sum()) * self.grid.resolution ** 2
        for nid in sorted(self.objects.nodes):
            self._assign_room(nid)
        self._rooms_dirty = True
        self._emit("rooms_updated", summary)

    def _assign_room(self, node_id: int) -> None:
        """Paper rule: room whose mask contains the centroid; [DESIGN] nearest-room fallback."""
        room_id = None
        if self.cfg.rooms.enabled:
            c = self.objects.nodes[node_id].node.centroid
            room_id = self.tracker.room_at(c[:2], self.cfg.rooms.room_assign_fallback_m)
        if self.objects.set_room(node_id, room_id):
            self._bump()

    def _sync_room_objects(self) -> None:
        """Keep RoomNode.object_ids == confirmed nodes whose room_id points to the room."""
        members: dict[int, list[int]] = {rid: [] for rid in self.tracker.rooms}
        for nid, ns in sorted(self.objects.nodes.items()):
            if ns.confirmed and ns.node.room_id in members:
                members[ns.node.room_id].append(nid)
        for rid, ids in members.items():
            if self.tracker.set_object_ids(rid, ids):
                self._bump()
        self._rooms_dirty = False

    def _label_rooms(self) -> None:
        """§8.3: requests go to one worker thread, at most one per room in flight. A result is
        applied only if the room's objects did not change meanwhile — [DESIGN] changes smaller
        than `relabel_min_new_objects` are tolerated, otherwise busy rooms would never get a label
        — else the room stays `needs_label` and is re-queued."""
        tol = self.cfg.rooms.relabel_min_new_objects
        for rid, (fut, objs) in list(self._label_jobs.items()):
            if not fut.done():
                continue
            del self._label_jobs[rid]
            rs = self.tracker.rooms.get(rid)
            if rs is not None and len(frozenset(rs.room.object_ids) ^ objs) < tol:
                self._apply_label(rid, fut.result(), objs)
        for rid in sorted(self.tracker.rooms):
            rs = self.tracker.rooms[rid]
            if len(rs.room.object_ids) < self.cfg.labeling.room_label_min_objects:
                if rs.room.label != UNKNOWN_LABEL:   # lost its objects (split, dedup, prune):
                    self.tracker.clear_label(rid)    # §8.3 too few objects -> "unknown"
                    self._emit("room_labeled", {"room_id": rid, "label": UNKNOWN_LABEL, "confidence": 0.0,
                                                "ok": True, "cached": False})
                continue
            if not rs.needs_label or rid in self._label_jobs:
                continue
            objs = [self.objects.nodes[i].node for i in rs.room.object_ids]
            adjacent = [self.tracker.rooms[a].room.label for a in rs.room.adjacent_room_ids
                        if a in self.tracker.rooms]
            req = self.labeler.build_request(rid, objs, rs.room.area_m2, adjacent)
            used = frozenset(rs.room.object_ids)
            if self._label_pool is None:
                self._apply_label(rid, self.labeler.label(req), used)
            else:
                self._label_jobs[rid] = (self._label_pool.submit(self.labeler.label, req), used)

    def _apply_label(self, rid: int, res, objs: frozenset[int]) -> None:
        self.tracker.set_label(rid, res.label, res.confidence, objs)
        self._emit("room_labeled", {"room_id": rid, "label": res.label, "confidence": res.confidence,
                                    "ok": res.ok, "cached": res.from_cache})

    def wait_labels(self, timeout_s: float | None = None, max_rounds: int = 3) -> None:
        """End of a run: wait for in-flight room labels and apply them; stale results re-queue
        (objects are final by now, so the next round sticks). Gives up after `timeout_s` per job."""
        if self.labeler is None or not self.cfg.rooms.enabled:
            return
        for _ in range(max_rounds):
            self._label_rooms()
            if not self._label_jobs:
                return
            try:
                wait([f for f, _ in self._label_jobs.values()], timeout=timeout_s)
            except TimeoutError:
                return

    def close(self) -> None:
        if self._label_pool is not None:
            self._label_pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint(self) -> None:
        snap = self.view()
        self.history.add(snap)
        if self.checkpoint_dir is not None:
            save_checkpoint(snap, self.checkpoint_dir / f"ckpt_{snap.stamp:012.3f}", self.grid)
        self.images.gc()

    # ------------------------------------------------------------------ versioning / events
    def _bump(self) -> None:
        self._version += 1

    def _emit(self, type_: str, payload: dict) -> None:
        self._bump()
        self._events.append(MemoryEvent(t=float(self._t), version=self._version, type=type_,
                                        payload=payload))

    # ------------------------------------------------------------------ read path
    def version(self) -> int:
        return self._version

    def view(self) -> MemorySnapshot:
        """Current state; immutable and cached per (version, time)."""
        v = self._view
        if v is None or v.version() != self._version or v.stamp != self._t:
            nodes = self.objects.nodes
            rooms = self.tracker.rooms
            self._view = MemorySnapshot(
                version=self._version, stamp=float(self._t),
                objects={i: ns.node for i, ns in nodes.items()},
                confirmed=frozenset(i for i, ns in nodes.items() if ns.confirmed),
                rooms={rid: rs.room for rid, rs in rooms.items()},
                room_polygons={rid: rs.polygon for rid, rs in rooms.items()},
                aliases=self.objects.aliases, room_aliases=self.tracker.aliases,
                max_neighbors=self.cfg.queries.max_neighbors, image_store=self.images,
                label_sim=self.label_sim, history=self.snapshot)
        return self._view

    def snapshot(self, t: float) -> MemorySnapshot:
        """Exact state as of the latest checkpoint <= t (never the live state filtered by t)."""
        if t >= self._t:
            return self.view()
        snap = self.history.at(t)
        if snap is not None:
            return snap
        return MemorySnapshot(version=0, stamp=t, objects={}, confirmed=frozenset(), rooms={},
                              room_polygons={}, aliases={}, room_aliases={},
                              max_neighbors=self.cfg.queries.max_neighbors,
                              image_store=self.images, label_sim=self.label_sim,
                              history=self.snapshot)

    def events(self, since_version: int) -> list[MemoryEvent]:
        return [e for e in self._events if e.version > since_version]

    def stats(self) -> MemoryStats:
        n_conf = sum(ns.confirmed for ns in self.objects.nodes.values())
        return MemoryStats(
            version=self._version, n_objects=len(self.objects.nodes), n_confirmed=n_conf,
            n_tentative=len(self.objects.nodes) - n_conf, n_rooms=len(self.tracker.rooms),
            n_frames_processed=self._n_processed, n_frames_skipped=self._n_skipped,
            n_obs_gated=self._n_gated, n_checkpoints=len(self.history),
            update_ms_p50=_pct(self._update_ms, 50), update_ms_p95=_pct(self._update_ms, 95),
            room_seg_ms_p95=_pct(self._room_ms, 95))

    # ------------------------------------------------------------------ MemoryView delegation
    def all_objects(self, until_t: float | None = None,
                    include_tentative: bool = False) -> list[ObjectNode]:
        return self.view().all_objects(until_t, include_tentative)

    def get_object(self, node_id: int) -> ObjectNode:
        return self.view().get_object(node_id)

    def get_room(self, room_id: int) -> RoomNode | None:
        return self.view().get_room(room_id)

    def neighbors(self, node_id: int, radius_m: float, same_room: bool) -> list[ObjectNode]:
        return self.view().neighbors(node_id, radius_m, same_room)

    def rooms(self) -> list[RoomNode]:
        return self.view().rooms()

    def objects_in_room(self, room_id: int) -> list[ObjectNode]:
        return self.view().objects_in_room(room_id)

    def find_by_text(self, text: str, min_sim: float,
                     top_n: int) -> list[tuple[ObjectNode, float]]:
        return self.view().find_by_text(text, min_sim, top_n)

    def best_view_crop(self, node_id: int, margin: float, max_px: int) -> np.ndarray:
        return self.view().best_view_crop(node_id, margin, max_px)

