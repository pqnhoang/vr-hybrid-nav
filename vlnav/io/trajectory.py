"""Stamped SE(3) trajectories with interpolation (SYSTEM.md §4.1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MATRIX_BLOCK_FLOATS = 17       # stamp + row-major 4x4
TUM_COLUMNS = 8                # t x y z qx qy qz qw


class Trajectory:
    """Poses T_ref_body(t) at increasing stamps; slerp + lerp in between."""

    def __init__(self, stamps: np.ndarray, poses: np.ndarray):
        stamps = np.asarray(stamps, dtype=float)
        poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
        order = np.argsort(stamps, kind="stable")
        stamps, poses = stamps[order], poses[order]
        keep = np.concatenate([[True], np.diff(stamps) > 0])        # drop duplicate stamps
        self.stamps, self.poses = stamps[keep], poses[keep]
        self._rot = Rotation.from_matrix(self.poses[:, :3, :3])

    def __len__(self) -> int:
        return len(self.stamps)

    @classmethod
    def load_matrix_blocks(cls, path: str | Path) -> Trajectory:
        """Text file of blocks `stamp` + 4 rows of a 4x4 matrix (whitespace separated)."""
        vals = np.array(Path(path).read_text().split(), dtype=float)
        if len(vals) % MATRIX_BLOCK_FLOATS:
            raise ValueError(f"{path}: {len(vals)} numbers is not a multiple of 17")
        blocks = vals.reshape(-1, MATRIX_BLOCK_FLOATS)
        return cls(blocks[:, 0], blocks[:, 1:].reshape(-1, 4, 4))

    @classmethod
    def load_tum(cls, path: str | Path) -> Trajectory:
        rows = np.loadtxt(path, comments="#").reshape(-1, TUM_COLUMNS)
        T = np.tile(np.eye(4), (len(rows), 1, 1))
        T[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
        T[:, :3, 3] = rows[:, 1:4]
        return cls(rows[:, 0], T)

    @classmethod
    def load(cls, path: str | Path, fmt: str) -> Trajectory:
        loaders = {"matrix_blocks": cls.load_matrix_blocks, "tum": cls.load_tum}
        if fmt not in loaders:
            raise ValueError(f"unknown trajectory format {fmt!r}, expected one of {list(loaders)}")
        return loaders[fmt](path)

    def transformed(self, T_new_ref: np.ndarray) -> Trajectory:
        """Express the trajectory in another reference frame: T_new_body = T_new_ref @ T_ref_body."""
        return Trajectory(self.stamps, np.asarray(T_new_ref) @ self.poses)

    def gap_at(self, t: float) -> float:
        """Length of the stamp interval containing t (inf outside the trajectory)."""
        if t < self.stamps[0] or t > self.stamps[-1]:
            return np.inf
        i = int(np.searchsorted(self.stamps, t, side="right"))     # stamps[i-1] <= t < stamps[i]
        if i == len(self.stamps):
            return 0.0                                              # t is the last stamp
        return float(self.stamps[i] - self.stamps[i - 1])

    def interpolate(self, t: float, max_gap_s: float = np.inf) -> np.ndarray | None:
        """T_ref_body(t), or None outside the trajectory or inside a gap longer than max_gap_s."""
        if t < self.stamps[0] or t > self.stamps[-1] or self.gap_at(t) > max_gap_s:
            return None
        i = int(np.searchsorted(self.stamps, t, side="right"))
        if i == len(self.stamps):
            return self.poses[-1].copy()
        t0, t1 = self.stamps[i - 1], self.stamps[i]
        a = (t - t0) / (t1 - t0)
        T = np.eye(4)
        T[:3, :3] = Slerp([t0, t1], self._rot[[i - 1, i]])([t]).as_matrix()[0]
        T[:3, 3] = (1 - a) * self.poses[i - 1, :3, 3] + a * self.poses[i, :3, 3]
        return T
