"""Immutable memory snapshot = the `MemoryView` implementation (TASK_unified_memory.md §5, §9.2).

Snapshots share `ObjectNode` / `RoomNode` objects with the live memory (copy-on-write: the writer
replaces nodes instead of mutating them), so taking one costs a shallow dict copy.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np

from vlnav.common.models import ObjectNode, RoomNode
from vlnav.memory.embedding import LabelSimilarity
from vlnav.memory.image_store import ImageStore
from vlnav.memory.spatial_index import SpatialIndex

History = Callable[[float], "MemorySnapshot"]


class MemorySnapshot:
    def __init__(self, *, version: int, stamp: float, objects: Mapping[int, ObjectNode],
                 confirmed: frozenset[int], rooms: Mapping[int, RoomNode],
                 room_polygons: Mapping[int, list[tuple[float, float]]],
                 aliases: Mapping[int, int], room_aliases: Mapping[int, int], max_neighbors: int,
                 image_store: ImageStore | None = None, label_sim: LabelSimilarity | None = None,
                 history: History | None = None):
        self._version = version
        self.stamp = stamp
        self._objects = MappingProxyType(dict(objects))
        self._confirmed = frozenset(confirmed)
        self._rooms = MappingProxyType(dict(rooms))
        self._room_polygons = MappingProxyType(dict(room_polygons))
        self._aliases = MappingProxyType(dict(aliases))
        self._room_aliases = MappingProxyType(dict(room_aliases))
        self.max_neighbors = max_neighbors
        self.image_store = image_store
        self.label_sim = label_sim
        self._history = history
        self._index: SpatialIndex | None = None

    def replace(self, **kw) -> MemorySnapshot:
        """New snapshot with some fields replaced (used for stripped checkpoint copies)."""
        args = dict(version=self._version, stamp=self.stamp, objects=self._objects,
                    confirmed=self._confirmed, rooms=self._rooms, room_polygons=self._room_polygons,
                    aliases=self._aliases, room_aliases=self._room_aliases,
                    max_neighbors=self.max_neighbors, image_store=self.image_store,
                    label_sim=self.label_sim, history=self._history)
        args.update(kw)
        return MemorySnapshot(**args)

    # ------------------------------------------------------------------ raw access
    @property
    def objects(self) -> Mapping[int, ObjectNode]:
        return self._objects

    @property
    def confirmed(self) -> frozenset[int]:
        return self._confirmed

    @property
    def aliases(self) -> Mapping[int, int]:
        return self._aliases

    @property
    def room_aliases(self) -> Mapping[int, int]:
        return self._room_aliases

    @property
    def room_polygons(self) -> Mapping[int, list[tuple[float, float]]]:
        return self._room_polygons

    def is_confirmed(self, node_id: int) -> bool:
        return node_id in self._confirmed

    # ------------------------------------------------------------------ MemoryView
    def version(self) -> int:
        return self._version

    def snapshot(self, t: float) -> MemorySnapshot:
        if t >= self.stamp:
            return self
        if self._history is None:
            raise ValueError(f"snapshot({t}) before stamp {self.stamp}: no history available")
        return self._history(t)

    def all_objects(self, until_t: float | None = None,
                    include_tentative: bool = False) -> list[ObjectNode]:
        # never filter-only by first_seen_t: that leaks later centroids / views (§10)
        if until_t is not None and until_t < self.stamp:
            return self.snapshot(until_t).all_objects(include_tentative=include_tentative)
        return [n for i, n in sorted(self._objects.items())
                if include_tentative or i in self._confirmed]

    def get_object(self, node_id: int) -> ObjectNode:
        while node_id in self._aliases:
            node_id = self._aliases[node_id]
        return self._objects[node_id]

    def get_room(self, room_id: int) -> RoomNode | None:
        while room_id in self._room_aliases:
            room_id = self._room_aliases[room_id]
        return self._rooms.get(room_id)

    def rooms(self) -> list[RoomNode]:
        return [r for _, r in sorted(self._rooms.items())]

    def objects_in_room(self, room_id: int) -> list[ObjectNode]:
        room = self.get_room(room_id)
        if room is None:
            return []
        return [n for i, n in sorted(self._objects.items())
                if i in self._confirmed and n.room_id == room.id]

    def _confirmed_index(self) -> SpatialIndex:
        if self._index is None:
            ids = sorted(self._confirmed & self._objects.keys())
            pts = np.array([self._objects[i].centroid for i in ids], dtype=float).reshape(-1, 3)
            self._index = SpatialIndex(ids, pts)
        return self._index

    def neighbors(self, node_id: int, radius_m: float, same_room: bool) -> list[ObjectNode]:
        """Confirmed nodes within radius_m of the node, by distance, capped at max_neighbors."""
        node = self.get_object(node_id)
        out = []
        for nid, _ in self._confirmed_index().query_radius(node.centroid, radius_m):
            other = self._objects[nid]
            if nid == node.id or (same_room and other.room_id != node.room_id):
                continue
            out.append(other)
            if len(out) >= self.max_neighbors:
                break
        return out

    def find_by_text(self, text: str, min_sim: float,
                     top_n: int) -> list[tuple[ObjectNode, float]]:
        """Confirmed nodes by label similarity to `text` (exact match without an embedder)."""
        nodes = [self._objects[i] for i in sorted(self._confirmed & self._objects.keys())]
        labels = sorted({n.label for n in nodes})
        if self.label_sim is not None:
            sim = dict(zip(labels, self.label_sim.sims(text, labels)))
        else:
            sim = {lbl: float(lbl.casefold() == text.casefold()) for lbl in labels}
        hits = [(n, float(sim[n.label])) for n in nodes if sim[n.label] >= min_sim]
        hits.sort(key=lambda h: (-h[1], -h[0].best_score, h[0].id))
        return hits[:top_n]

    def best_view_crop(self, node_id: int, margin: float, max_px: int) -> np.ndarray:
        if self.image_store is None:
            raise RuntimeError("snapshot has no image store")
        node = self.get_object(node_id)
        return self.image_store.crop(node.best_view_image_path, node.best_view_bbox, margin, max_px)
