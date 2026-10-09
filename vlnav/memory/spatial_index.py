"""KD-tree over node centroids (TASK_unified_memory.md §9.2). Immutable; rebuild on change."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree


class SpatialIndex:
    def __init__(self, ids: Sequence[int], points: np.ndarray):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self._tree = cKDTree(self.points) if len(self.ids) else None

    def __len__(self) -> int:
        return len(self.ids)

    def query_radius(self, p, radius: float) -> list[tuple[int, float]]:
        """(id, distance) of all points within `radius` of `p`, sorted by distance then id."""
        if self._tree is None:
            return []
        p = np.asarray(p, dtype=float)
        idx = self._tree.query_ball_point(p, radius)
        out = [(int(self.ids[i]), float(np.linalg.norm(self.points[i] - p))) for i in idx]
        return sorted(out, key=lambda x: (x[1], x[0]))

    def pairs_within(self, radius: float) -> list[tuple[int, int, float]]:
        """(id_a, id_b, distance) with id_a < id_b, sorted by distance then ids."""
        if self._tree is None:
            return []
        out = []
        for i, j in self._tree.query_pairs(radius):
            a, b = sorted((int(self.ids[i]), int(self.ids[j])))
            out.append((a, b, float(np.linalg.norm(self.points[i] - self.points[j]))))
        return sorted(out, key=lambda x: (x[2], x[0], x[1]))
