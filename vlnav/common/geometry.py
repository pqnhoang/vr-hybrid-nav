"""Small geometry helpers shared by packages (SYSTEM.md §2 `common/geometry.py`)."""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap to (-pi, pi] (AUDIT pitfall 5: before every FoV test and Eq. 3)."""
    return np.arctan2(np.sin(a), np.cos(a))


def ground_heading(R: np.ndarray, axis: int) -> float:
    """Heading of a body axis projected on the ground plane (AUDIT pitfall 2): column `axis` of R.
    Base frames use x (axis 0); optical camera frames use z (axis 2)."""
    v = np.asarray(R)[:3, axis]
    return float(np.arctan2(v[1], v[0]))


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """xyzw quaternion of a rotation about z."""
    return 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
