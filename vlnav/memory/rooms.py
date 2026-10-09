"""Morphological room segmentation on the shared occupancy grid (TASK_unified_memory.md §8.1).

closing (unknown speckle only) -> drop small free components -> erosion by door_width/2 ->
seeds -> watershed on -distance -> merge small regions -> adjacency. [DESIGN details]
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from vlnav.mapping.occupancy import FREE, OCCUPIED
from vlnav.memory.config import RoomsCfg


@dataclass(frozen=True)
class RoomSegmentation:
    """Regions in the grid frame at segmentation time (labels 1..N, 0 = no room)."""

    labels: np.ndarray                                  # int32 (ny, nx), grid resolution
    resolution: float
    origin_xy: tuple[float, float]
    areas_m2: dict[int, float]
    centroids_xy: dict[int, tuple[float, float]]
    polygons: dict[int, list[tuple[float, float]]]
    adjacency: dict[int, list[int]]

    @property
    def region_ids(self) -> list[int]:
        return sorted(self.areas_m2)


def _disk(radius_cells: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_cells + 1,) * 2)


def _radius_cells(radius_m: float, res: float) -> int:
    # ceil: a door of exactly door_width_m must be closed by the erosion
    return max(0, int(np.ceil(radius_m / res - 1e-9)))


def _downsample(grid: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    """(free, occupied) masks on blocks of f x f cells; a block with any occupied cell is occupied."""
    if f == 1:
        return grid == FREE, grid == OCCUPIED
    ny, nx = grid.shape
    pad = ((0, (-ny) % f), (0, (-nx) % f))
    free = np.pad(grid == FREE, pad).reshape((ny + pad[0][1]) // f, f, (nx + pad[1][1]) // f, f)
    occ = np.pad(grid == OCCUPIED, pad).reshape(free.shape)
    occ_c = occ.any(axis=(1, 3))
    return (free.mean(axis=(1, 3)) >= 0.5) & ~occ_c, occ_c


def shared_borders(labels: np.ndarray) -> dict[tuple[int, int], int]:
    """Number of 4-connected cell pairs between each pair of different regions (a < b)."""
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        pairs.append(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], axis=1))
    allp = np.concatenate(pairs)
    if not len(allp):
        return {}
    uniq, counts = np.unique(allp, axis=0, return_counts=True)
    return {(int(a), int(b)): int(c) for (a, b), c in zip(uniq, counts)}


def _merge_small(labels: np.ndarray, min_cells: float) -> None:
    """Merge regions below min_cells into the neighbor with the longest shared border (in place)."""
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        small = sorted((int(c), int(i)) for i, c in zip(ids, counts) if c < min_cells)
        if not small:
            return
        _, rid = small[0]
        nbrs = {(b if a == rid else a): n for (a, b), n in shared_borders(labels).items()
                if rid in (a, b)}
        target = max(sorted(nbrs), key=lambda k: nbrs[k]) if nbrs else 0
        labels[labels == rid] = target


def _relabel(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels[labels > 0])
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[labels]


def _polygon(mask: np.ndarray, res: float, origin_xy) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    return [(float(origin_xy[0] + (x + 0.5) * res), float(origin_xy[1] + (y + 0.5) * res))
            for x, y in c]


def segment_rooms(grid: np.ndarray, resolution: float, origin_xy: tuple[float, float],
                  cfg: RoomsCfg) -> RoomSegmentation:
    f = max(1, int(round(cfg.room_grid_res_m / resolution)))
    res = resolution * f
    cell_m2 = res * res
    free, occ = _downsample(np.asarray(grid), f)

    # 1-2. fill unknown speckle (never through occupied cells), drop tiny free components
    r = _radius_cells(cfg.closing_kernel_m / 2, res)
    if r > 0:
        free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, _disk(r)).astype(bool) & ~occ
    n, comp, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] * cell_m2 < cfg.min_free_comp_m2)
    free &= ~np.isin(comp, small[small > 0])

    # 3-4. erosion disconnects rooms at doors; large components become seeds
    r = _radius_cells(cfg.door_width_m / 2, res)
    eroded = cv2.erode(free.astype(np.uint8), _disk(r)) if r > 0 else free.astype(np.uint8)
    n, seeds, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] * cell_m2 >= cfg.min_seed_area_m2)
    seeds = np.where(np.isin(seeds, keep[keep > 0]), seeds, 0)

    # 5. grow seeds back over the free space along the distance-transform ridges
    if seeds.max() > 0:
        dist = ndimage.distance_transform_edt(free)
        labels = watershed(-dist, seeds, mask=free).astype(np.int32)
    else:
        labels = np.zeros(free.shape, dtype=np.int32)

    # 6. merge small regions
    _merge_small(labels, cfg.min_room_area_m2 / cell_m2)
    labels = _relabel(labels)

    # back to grid resolution, restricted to free cells of the grid
    if f > 1:
        ny, nx = grid.shape
        labels = np.repeat(np.repeat(labels, f, axis=0), f, axis=1)[:ny, :nx]
    labels = np.where(np.asarray(grid) == FREE, labels, 0).astype(np.int32)
    labels = _relabel(labels)

    # 7. adjacency by shared border length, 8. per-room geometry
    borders = shared_borders(labels)
    adjacency: dict[int, list[int]] = {}
    for (a, b), cnt in borders.items():
        if cnt * resolution >= cfg.min_shared_border_m:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    areas, centroids, polygons = {}, {}, {}
    for rid in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == rid)
        areas[rid] = float(len(xs) * resolution * resolution)
        centroids[rid] = (float(origin_xy[0] + (xs.mean() + 0.5) * resolution),
                          float(origin_xy[1] + (ys.mean() + 0.5) * resolution))
        polygons[rid] = _polygon(labels == rid, resolution, origin_xy)
    return RoomSegmentation(
        labels=labels, resolution=resolution, origin_xy=tuple(origin_xy), areas_m2=areas,
        centroids_xy=centroids, polygons=polygons,
        adjacency={k: sorted(v) for k, v in adjacency.items()})
