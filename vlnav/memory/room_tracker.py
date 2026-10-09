"""Stable room ids across re-segmentations (TASK_unified_memory.md §8.2).

Hungarian matching on region IoU; matched regions keep id and label; a split keeps the id on the
best-matching part; a merge keeps one id and aliases the absorbed ones.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from vlnav.common.models import RoomNode
from vlnav.memory.config import RoomsCfg
from vlnav.memory.rooms import RoomSegmentation
from vlnav.memory.types import UNKNOWN_LABEL, RoomState

NO_ROOM = -1


def _paste(prev: np.ndarray, prev_origin, new_shape, new_origin, res: float) -> np.ndarray:
    """Re-express `prev` (id image) in the frame of a grid with `new_origin` / `new_shape`."""
    out = np.full(new_shape, NO_ROOM, dtype=np.int32)
    dx = int(round((prev_origin[0] - new_origin[0]) / res))
    dy = int(round((prev_origin[1] - new_origin[1]) / res))
    ny, nx = prev.shape
    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(new_shape[0], dy + ny), min(new_shape[1], dx + nx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = prev[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


class RoomTracker:
    def __init__(self, cfg: RoomsCfg):
        self.cfg = cfg
        self.rooms: dict[int, RoomState] = {}
        self.aliases: dict[int, int] = {}
        self.id_image: np.ndarray | None = None          # room id per grid cell, NO_ROOM = none
        self.origin_xy: tuple[float, float] = (0.0, 0.0)
        self.resolution: float = 1.0
        self._next_id = 0

    def resolve(self, room_id: int) -> int:
        while room_id in self.aliases:
            room_id = self.aliases[room_id]
        return room_id

    # ------------------------------------------------------------------ segmentation update
    def update(self, seg: RoomSegmentation) -> dict:
        regions = seg.region_ids
        assign: dict[int, int] = {}                      # region -> room id
        ious: dict[int, float] = {}
        removed, aliased = [], {}

        if self.id_image is not None and self.rooms and regions:
            prev = _paste(self.id_image, self.origin_xy, seg.labels.shape, seg.origin_xy,
                          seg.resolution)
            old_ids = sorted(self.rooms)
            inter = np.zeros((len(regions), len(old_ids)))
            m = (seg.labels > 0) & (prev >= 0)
            if m.any():
                pairs, counts = np.unique(np.stack([seg.labels[m], prev[m]], 1), axis=0,
                                          return_counts=True)
                col = {o: j for j, o in enumerate(old_ids)}
                for (r, o), c in zip(pairs, counts):
                    if int(o) in col:
                        inter[int(r) - 1, col[int(o)]] = c
            area_new = np.array([(seg.labels == r).sum() for r in regions], dtype=float)
            area_old = np.array([(prev == o).sum() for o in old_ids], dtype=float)
            union = area_new[:, None] + area_old[None, :] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            for i, j in zip(*linear_sum_assignment(-iou)):
                if iou[i, j] >= self.cfg.room_iou_min:
                    assign[regions[i]] = old_ids[j]
                    ious[regions[i]] = float(iou[i, j])
            matched_old = set(assign.values())
            for j, o in enumerate(old_ids):
                if o in matched_old:
                    continue
                i = int(np.argmax(inter[:, j])) if len(regions) else 0
                if area_old[j] > 0 and inter[i, j] / area_old[j] >= self.cfg.room_alias_overlap_min \
                        and regions[i] in assign:
                    aliased[o] = assign[regions[i]]
                else:
                    removed.append(o)

        new_ids = []
        for r in regions:
            if r not in assign:
                assign[r] = self._next_id
                self._next_id += 1
                new_ids.append(assign[r])

        rooms: dict[int, RoomState] = {}
        for r in regions:
            rid = assign[r]
            old = self.rooms.get(rid)
            cells = np.argwhere(seg.labels == r)[:, ::-1].astype(np.int32)     # (ix, iy)
            cells.flags.writeable = False
            node = RoomNode(
                id=rid, label=old.room.label if old else UNKNOWN_LABEL,
                label_confidence=old.room.label_confidence if old else 0.0, mask_cells=cells,
                centroid_xy=seg.centroids_xy[r], area_m2=seg.areas_m2[r],
                object_ids=list(old.room.object_ids) if old else [],
                adjacent_room_ids=sorted(assign[a] for a in seg.adjacency.get(r, [])))
            needs = old is None or old.needs_label or ious.get(r, 1.0) < self.cfg.relabel_iou
            rooms[rid] = RoomState(room=node, polygon=seg.polygons[r],
                                   object_set_at_label=old.object_set_at_label if old else frozenset(),
                                   needs_label=needs)
        self.aliases.update(aliased)
        self.rooms = rooms
        lut = np.full(len(regions) + 1, NO_ROOM, dtype=np.int32)
        for r in regions:
            lut[r] = assign[r]
        self.id_image = lut[seg.labels]
        self.origin_xy, self.resolution = seg.origin_xy, seg.resolution
        return {"n_rooms": len(rooms), "new": new_ids, "removed": removed,
                "aliased": {str(k): v for k, v in aliased.items()}}

    # ------------------------------------------------------------------ queries / edits
    def room_at(self, xy, fallback_m: float) -> int | None:
        """Room whose mask contains xy; else nearest room cell within fallback_m; else None."""
        if self.id_image is None:
            return None
        res = self.resolution
        cx = int(np.floor((xy[0] - self.origin_xy[0]) / res))
        cy = int(np.floor((xy[1] - self.origin_xy[1]) / res))
        ny, nx = self.id_image.shape
        if 0 <= cx < nx and 0 <= cy < ny and self.id_image[cy, cx] != NO_ROOM:
            return int(self.id_image[cy, cx])
        r = int(np.ceil(fallback_m / res))
        y0, y1, x0, x1 = max(0, cy - r), min(ny, cy + r + 1), max(0, cx - r), min(nx, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            return None
        win = self.id_image[y0:y1, x0:x1]
        ys, xs = np.nonzero(win != NO_ROOM)
        if not len(xs):
            return None
        wx = self.origin_xy[0] + (xs + x0 + 0.5) * res
        wy = self.origin_xy[1] + (ys + y0 + 0.5) * res
        d = np.hypot(wx - xy[0], wy - xy[1])
        k = int(np.argmin(d))
        return int(win[ys[k], xs[k]]) if d[k] <= fallback_m else None

    def set_object_ids(self, room_id: int, object_ids: list[int]) -> bool:
        rs = self.rooms[room_id]
        if rs.room.object_ids == object_ids:
            return False
        rs.room = rs.room.model_copy(update={"object_ids": object_ids})
        if len(set(object_ids) ^ rs.object_set_at_label) >= self.cfg.relabel_min_new_objects:
            rs.needs_label = True
        return True

    def set_label(self, room_id: int, label: str, confidence: float,
                  object_set: frozenset[int]) -> None:
        rs = self.rooms[room_id]
        rs.room = rs.room.model_copy(update={"label": label, "label_confidence": confidence})
        rs.object_set_at_label = object_set
        rs.needs_label = False

    def clear_label(self, room_id: int) -> None:
        """Back to unknown and still `needs_label` (relabeled once it has enough objects)."""
        rs = self.rooms[room_id]
        rs.room = rs.room.model_copy(update={"label": UNKNOWN_LABEL, "label_confidence": 0.0})
        rs.object_set_at_label = frozenset()
        rs.needs_label = True
