"""JSON schema of the scene graph (TASK_unified_memory.md §11.1).

`include_geometry=False` is the `/memory/scene_graph` format (no points, no masks);
`include_geometry=True` adds them (checkpoints store them in .npz files instead, see checkpoint.py).
Extension to the spec schema: per-object `"confirmed"` and top-level `"room_aliases"`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vlnav.common.models import ObjectNode, Pose, RoomNode
from vlnav.memory.embedding import LabelSimilarity
from vlnav.memory.image_store import ImageStore
from vlnav.memory.snapshot import History, MemorySnapshot


def _pose(p: Pose) -> dict:
    return {"position": list(p.position), "orientation_xyzw": list(p.orientation_xyzw),
            "stamp": p.stamp}


def snapshot_to_dict(snap: MemorySnapshot, include_geometry: bool = False,
                     frame_id: str = "map") -> dict:
    objects = []
    for i, n in sorted(snap.objects.items()):
        o = {"id": n.id, "label": n.label, "label_votes": dict(n.label_votes),
             "centroid": list(n.centroid), "best_score": n.best_score,
             "best_view_pose": _pose(n.best_view_pose),
             "best_view_cam_pose": _pose(n.best_view_cam_pose),
             "best_view_image": Path(n.best_view_image_path).name,
             "best_view_bbox": list(n.best_view_bbox), "room_id": n.room_id, "n_obs": n.n_obs,
             "first_seen_t": n.first_seen_t, "last_seen_t": n.last_seen_t,
             "confirmed": snap.is_confirmed(i)}
        if include_geometry:
            o["points"] = np.asarray(n.points).tolist()
        objects.append(o)
    rooms = []
    for rid, r in sorted((r.id, r) for r in snap.rooms()):
        d = {"id": r.id, "label": r.label, "label_confidence": r.label_confidence,
             "centroid_xy": list(r.centroid_xy), "area_m2": r.area_m2,
             "adjacent_room_ids": list(r.adjacent_room_ids), "object_ids": list(r.object_ids),
             "polygon": [list(p) for p in snap.room_polygons.get(rid, [])]}
        if include_geometry:
            d["mask_cells"] = np.asarray(r.mask_cells).tolist()
        rooms.append(d)
    return {"version": snap.version(), "stamp": snap.stamp, "frame": frame_id, "rooms": rooms,
            "objects": objects, "aliases": {str(k): v for k, v in sorted(snap.aliases.items())},
            "room_aliases": {str(k): v for k, v in sorted(snap.room_aliases.items())}}


def snapshot_to_json(snap: MemorySnapshot, include_geometry: bool = False,
                     frame_id: str = "map") -> str:
    return json.dumps(snapshot_to_dict(snap, include_geometry, frame_id))


def _ro(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def snapshot_from_dict(d: dict, *, image_root: str | Path, max_neighbors: int,
                       image_store: ImageStore | None = None,
                       label_sim: LabelSimilarity | None = None,
                       history: History | None = None) -> MemorySnapshot:
    objects, confirmed = {}, set()
    for o in d["objects"]:
        pts = np.asarray(o.get("points", []), dtype=float).reshape(-1, 3)
        objects[o["id"]] = ObjectNode(
            id=o["id"], label=o["label"], label_votes=o["label_votes"],
            centroid=tuple(o["centroid"]), points=_ro(pts), best_score=o["best_score"],
            best_view_pose=Pose(**o["best_view_pose"]),
            best_view_cam_pose=Pose(**o["best_view_cam_pose"]),
            best_view_image_path=str(Path(image_root) / o["best_view_image"]),
            best_view_bbox=tuple(o["best_view_bbox"]), room_id=o["room_id"], n_obs=o["n_obs"],
            first_seen_t=o["first_seen_t"], last_seen_t=o["last_seen_t"])
        if o.get("confirmed", True):
            confirmed.add(o["id"])
    rooms, polygons = {}, {}
    for r in d["rooms"]:
        cells = np.asarray(r.get("mask_cells", []), dtype=np.int32).reshape(-1, 2)
        rooms[r["id"]] = RoomNode(
            id=r["id"], label=r["label"], label_confidence=r["label_confidence"],
            mask_cells=_ro(cells), centroid_xy=tuple(r["centroid_xy"]), area_m2=r["area_m2"],
            object_ids=list(r["object_ids"]), adjacent_room_ids=list(r["adjacent_room_ids"]))
        polygons[r["id"]] = [tuple(p) for p in r.get("polygon", [])]
    return MemorySnapshot(
        version=d["version"], stamp=d["stamp"], objects=objects, confirmed=frozenset(confirmed),
        rooms=rooms, room_polygons=polygons,
        aliases={int(k): v for k, v in d.get("aliases", {}).items()},
        room_aliases={int(k): v for k, v in d.get("room_aliases", {}).items()},
        max_neighbors=max_neighbors, image_store=image_store, label_sim=label_sim, history=history)


def snapshot_from_json(text: str, **kw) -> MemorySnapshot:
    return snapshot_from_dict(json.loads(text), **kw)
