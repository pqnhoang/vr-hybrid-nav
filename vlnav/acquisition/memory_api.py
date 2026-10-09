"""MemoryView protocol (acquisition spec §6, extended by TASK_unified_memory.md §5) and the
fixture-backed `InMemoryMemory` used by acquisition tests and offline evaluation.

Implemented by `memory.MemorySnapshot` (hence `UnifiedMemory.view()`), `memory.RemoteMemoryMirror`
and `InMemoryMemory`. Consumers code only against this protocol.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from vlnav.common.images import crop_bbox, load_rgb
from vlnav.common.models import ObjectNode, Pose, RoomNode


@runtime_checkable
class MemoryView(Protocol):
    def all_objects(self, until_t: float | None = None,
                    include_tentative: bool = False) -> list[ObjectNode]: ...

    def get_object(self, node_id: int) -> ObjectNode: ...          # follows dedup aliases

    def get_room(self, room_id: int) -> RoomNode | None: ...

    def neighbors(self, node_id: int, radius_m: float, same_room: bool) -> list[ObjectNode]: ...

    # extensions used by the planner
    def rooms(self) -> list[RoomNode]: ...

    def objects_in_room(self, room_id: int) -> list[ObjectNode]: ...

    def find_by_text(self, text: str, min_sim: float,
                     top_n: int) -> list[tuple[ObjectNode, float]]: ...

    def best_view_crop(self, node_id: int, margin: float, max_px: int) -> np.ndarray: ...

    def snapshot(self, t: float) -> MemoryView: ...

    def version(self) -> int: ...


_EMPTY_POINTS = np.empty((0, 3))
_EMPTY_CELLS = np.empty((0, 2), dtype=np.int32)


def _pose(d: dict) -> Pose:
    return Pose(position=tuple(d["position"]), orientation_xyzw=tuple(d["orientation_xyzw"]),
                stamp=float(d.get("stamp", 0.0)))


def object_from_fixture(o: dict, image_root: Path) -> ObjectNode:
    """Fixture object (acquisition spec §12): only the acquisition fields are required."""
    image = o.get("best_view_image_path", o.get("best_view_image", ""))
    path = Path(image)
    return ObjectNode(
        id=int(o["id"]), label=o["label"],
        label_votes=o.get("label_votes", {o["label"]: float(o["best_score"])}),
        centroid=tuple(float(v) for v in o["centroid"]), points=_EMPTY_POINTS,
        best_score=float(o["best_score"]), best_view_pose=_pose(o["best_view_pose"]),
        best_view_cam_pose=_pose(o.get("best_view_cam_pose", o["best_view_pose"])),
        best_view_image_path=str(path if path.is_absolute() else image_root / path),
        best_view_bbox=tuple(int(v) for v in o["best_view_bbox"]), room_id=o.get("room_id"),
        n_obs=int(o.get("n_obs", 1)), first_seen_t=float(o.get("first_seen_t", 0.0)),
        last_seen_t=float(o.get("last_seen_t", o.get("first_seen_t", 0.0))))


def room_from_fixture(r: dict) -> RoomNode:
    return RoomNode(id=int(r["id"]), label=r.get("label", "unknown"),
                    label_confidence=float(r.get("label_confidence", 1.0)), mask_cells=_EMPTY_CELLS,
                    centroid_xy=tuple(r.get("centroid_xy", (0.0, 0.0))),
                    area_m2=float(r.get("area_m2", 0.0)), object_ids=list(r.get("object_ids", [])),
                    adjacent_room_ids=list(r.get("adjacent_room_ids", [])))


class InMemoryMemory:
    """Static fixture memory. `until_t` / `snapshot(t)` filter by `first_seen_t` — acceptable here
    because fixtures do not change over time (memory spec §10)."""

    def __init__(self, objects: Iterable[ObjectNode], rooms: Iterable[RoomNode] = (),
                 max_neighbors: int = 6, aliases: dict[int, int] | None = None,
                 tentative_ids: Iterable[int] = (), text_sim=None, version: int = 1,
                 stamp: float = float("inf")):
        self._objects = {o.id: o for o in objects}
        self._rooms = {r.id: r for r in rooms}
        if self._rooms and not any(r.object_ids for r in self._rooms.values()):
            self._rooms = {rid: r.model_copy(update={"object_ids": sorted(   # derive the edges
                o.id for o in self._objects.values() if o.room_id == rid)})
                for rid, r in self._rooms.items()}
        self._aliases = dict(aliases or {})
        self._tentative = set(tentative_ids)
        self.max_neighbors = max_neighbors
        self.text_sim = text_sim                    # callable(text, label) -> float, optional
        self._version = version
        self.stamp = stamp

    @classmethod
    def from_json(cls, path: str | Path, max_neighbors: int = 6,
                  image_root: str | Path | None = None, text_sim=None) -> InMemoryMemory:
        path = Path(path)
        d = json.loads(path.read_text())
        root = Path(image_root) if image_root is not None else path.parent
        objects = [object_from_fixture(o, root) for o in d["objects"]]
        tentative = [o["id"] for o in d["objects"] if not o.get("confirmed", True)]
        return cls(objects, [room_from_fixture(r) for r in d.get("rooms", [])], max_neighbors,
                   {int(k): v for k, v in d.get("aliases", {}).items()}, tentative, text_sim,
                   int(d.get("version", 1)))

    # ----------------------------------------------------------------------------- MemoryView
    def version(self) -> int:
        return self._version

    def snapshot(self, t: float) -> InMemoryMemory:
        objs = [o for o in self._objects.values() if o.first_seen_t <= t]
        return InMemoryMemory(objs, self._rooms.values(), self.max_neighbors, self._aliases,
                              self._tentative, self.text_sim, self._version, stamp=t)

    def all_objects(self, until_t: float | None = None,
                    include_tentative: bool = False) -> list[ObjectNode]:
        return [o for i, o in sorted(self._objects.items())
                if (until_t is None or o.first_seen_t <= until_t)
                and (include_tentative or i not in self._tentative)]

    def get_object(self, node_id: int) -> ObjectNode:
        while node_id in self._aliases:
            node_id = self._aliases[node_id]
        return self._objects[node_id]

    def get_room(self, room_id: int) -> RoomNode | None:
        return self._rooms.get(room_id)

    def rooms(self) -> list[RoomNode]:
        return [r for _, r in sorted(self._rooms.items())]

    def objects_in_room(self, room_id: int) -> list[ObjectNode]:
        if room_id not in self._rooms:
            return []
        return [o for o in self.all_objects() if o.room_id == room_id]

    def neighbors(self, node_id: int, radius_m: float, same_room: bool) -> list[ObjectNode]:
        node = self.get_object(node_id)
        c = np.asarray(node.centroid)
        near = []
        for o in self.all_objects():
            d = float(np.linalg.norm(np.asarray(o.centroid) - c))
            if o.id != node.id and d <= radius_m and (not same_room or o.room_id == node.room_id):
                near.append((d, o.id, o))
        return [o for _, _, o in sorted(near, key=lambda x: (x[0], x[1]))][:self.max_neighbors]

    def find_by_text(self, text: str, min_sim: float,
                     top_n: int) -> list[tuple[ObjectNode, float]]:
        hits = []
        for o in self.all_objects():
            s = (float(self.text_sim(text, o.label)) if self.text_sim is not None
                 else float(o.label.casefold() == text.casefold()))
            if s >= min_sim:
                hits.append((o, s))
        hits.sort(key=lambda h: (-h[1], -h[0].best_score, h[0].id))
        return hits[:top_n]

    def best_view_crop(self, node_id: int, margin: float, max_px: int) -> np.ndarray:
        o = self.get_object(node_id)
        return crop_bbox(load_rgb(o.best_view_image_path), o.best_view_bbox, margin, max_px)
