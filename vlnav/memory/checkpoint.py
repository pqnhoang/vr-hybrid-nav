"""Checkpoint history, state-at-time-t and save/load (TASK_unified_memory.md §10).

`snapshot(t)` returns the latest checkpoint taken at or before t (granularity =
checkpoint_every_s). Historical snapshots keep everything the MemoryView API exposes except
geometry (`points`, `mask_cells` are dropped to bound memory); unchanged nodes are shared between
checkpoints. Each retained checkpoint holds a reference on the best-view images it uses, so the
image store never deletes a frame an older snapshot still points to.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

from vlnav.common.models import ObjectNode, RoomNode
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.memory.embedding import LabelSimilarity
from vlnav.memory.image_store import ImageStore
from vlnav.memory.serialize import snapshot_from_dict, snapshot_to_json
from vlnav.memory.snapshot import History, MemorySnapshot


def _empty(shape_cols: int, dtype) -> np.ndarray:
    a = np.empty((0, shape_cols), dtype=dtype)
    a.flags.writeable = False
    return a


_NO_POINTS = _empty(3, float)
_NO_CELLS = _empty(2, np.int32)


class CheckpointHistory:
    def __init__(self, max_checkpoints: int, image_store: ImageStore):
        self.max_checkpoints = max_checkpoints          # 0 = keep all
        self.images = image_store
        self._stamps: list[float] = []
        self._snaps: list[MemorySnapshot] = []
        self._obj_cache: dict[int, tuple[ObjectNode, ObjectNode]] = {}
        self._room_cache: dict[int, tuple[RoomNode, RoomNode]] = {}

    def __len__(self) -> int:
        return len(self._snaps)

    def _strip_obj(self, n: ObjectNode) -> ObjectNode:
        cached = self._obj_cache.get(n.id)
        if cached is None or cached[0] is not n:
            cached = (n, n.model_copy(update={"points": _NO_POINTS}))
            self._obj_cache[n.id] = cached
        return cached[1]

    def _strip_room(self, r: RoomNode) -> RoomNode:
        cached = self._room_cache.get(r.id)
        if cached is None or cached[0] is not r:
            cached = (r, r.model_copy(update={"mask_cells": _NO_CELLS}))
            self._room_cache[r.id] = cached
        return cached[1]

    def add(self, snap: MemorySnapshot) -> MemorySnapshot:
        if self._stamps and snap.stamp < self._stamps[-1]:
            raise ValueError(f"checkpoint at {snap.stamp} older than {self._stamps[-1]}")
        stripped = snap.replace(
            objects={i: self._strip_obj(n) for i, n in snap.objects.items()},
            rooms={r.id: self._strip_room(r) for r in snap.rooms()})
        for n in stripped.objects.values():
            self.images.acquire(n.best_view_image_path)
        self._stamps.append(snap.stamp)
        self._snaps.append(stripped)
        while self.max_checkpoints and len(self._snaps) > self.max_checkpoints:
            self._stamps.pop(0)
            for n in self._snaps.pop(0).objects.values():
                self.images.release(n.best_view_image_path)
        return stripped

    def at(self, t: float) -> MemorySnapshot | None:
        """Latest checkpoint with stamp <= t, or None."""
        k = bisect.bisect_right(self._stamps, t) - 1
        return self._snaps[k] if k >= 0 else None


def save_checkpoint(snap: MemorySnapshot, out_dir: str | Path,
                    grid: OccupancyGrid2D | None = None) -> Path:
    """graph.json + objects.npz (points) + rooms.npz (mask cells) [+ grid.npz]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(snapshot_to_json(snap, include_geometry=False))
    np.savez_compressed(out / "objects.npz",
                        **{str(i): np.asarray(n.points) for i, n in snap.objects.items()})
    np.savez_compressed(out / "rooms.npz",
                        **{str(r.id): np.asarray(r.mask_cells) for r in snap.rooms()})
    if grid is not None:
        np.savez_compressed(out / "grid.npz", grid=grid.as_array(), resolution=grid.resolution,
                            origin_xy=np.asarray(grid.origin_xy))
    return out


def load_checkpoint(ckpt_dir: str | Path, *, image_root: str | Path, max_neighbors: int,
                    image_store: ImageStore | None = None,
                    label_sim: LabelSimilarity | None = None,
                    history: History | None = None) -> MemorySnapshot:
    ckpt = Path(ckpt_dir)
    d = json.loads((ckpt / "graph.json").read_text())
    with np.load(ckpt / "objects.npz") as pts:
        for o in d["objects"]:
            o["points"] = pts[str(o["id"])]
    with np.load(ckpt / "rooms.npz") as cells:
        for r in d["rooms"]:
            r["mask_cells"] = cells[str(r["id"])]
    return snapshot_from_dict(d, image_root=image_root, max_neighbors=max_neighbors,
                              image_store=image_store, label_sim=label_sim, history=history)


def load_grid(ckpt_dir: str | Path) -> OccupancyGrid2D:
    with np.load(Path(ckpt_dir) / "grid.npz") as g:
        return OccupancyGrid2D.from_array(g["grid"], float(g["resolution"]),
                                          tuple(g["origin_xy"].tolist()))
