"""TF tree from a bag (/tf_static + /tf) with time-interpolated chain lookups. ROS-free.

Each child frame has one parent (TF tree). Static edges are constant, dynamic edges are
interpolated (slerp + lerp) between their stamps; lookups fail (None) outside a dynamic edge's time
range or inside a gap longer than `max_gap_s`. Missing edges (e.g. a camera optical frame that the
robot description does not publish) can be added with `add_static`.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from vlnav.io.ros_decode import stamp_s, transform_to_matrix
from vlnav.io.trajectory import Trajectory

ROS_BODY_TO_OPTICAL_RPY_DEG = (-90.0, 0.0, -90.0)   # x fwd/y left/z up -> z fwd/x right/y down


def static_matrix(xyz, rpy_deg) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


class TfBuffer:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.static: dict[str, np.ndarray] = {}                 # child -> T_parent_child
        self._dyn_raw: dict[str, list[tuple[float, np.ndarray]]] = {}
        self.dynamic: dict[str, Trajectory] = {}

    # ------------------------------------------------------------------------------- building
    def add_static(self, parent: str, child: str, T: np.ndarray) -> None:
        self._set_parent(parent, child)
        self.static[child] = np.asarray(T, dtype=float)

    def add_dynamic(self, parent: str, child: str, stamp: float, T: np.ndarray) -> None:
        self._set_parent(parent, child)
        self._dyn_raw.setdefault(child, []).append((stamp, np.asarray(T, dtype=float)))

    def _set_parent(self, parent: str, child: str) -> None:
        old = self.parent.setdefault(child, parent)
        if old != parent:
            raise ValueError(f"frame {child!r} has two parents: {old!r} and {parent!r}")

    def add_tf_messages(self, msgs: Iterable, static: bool) -> None:
        for m in msgs:
            for tr in m.transforms:
                T = transform_to_matrix(tr.transform)
                if static:
                    self.add_static(tr.header.frame_id, tr.child_frame_id, T)
                else:
                    self.add_dynamic(tr.header.frame_id, tr.child_frame_id, stamp_s(tr.header), T)

    def finalize(self) -> TfBuffer:
        for child, samples in self._dyn_raw.items():
            if child in self.static:
                raise ValueError(f"frame {child!r} is both static and dynamic")
            self.dynamic[child] = Trajectory([s for s, _ in samples], [T for _, T in samples])
        self._dyn_raw = {}
        return self

    @classmethod
    def from_bag(cls, bag, tf_topic: str = "/tf", static_topic: str = "/tf_static",
                 extra_static: Iterable[tuple[str, str, np.ndarray]] = ()) -> TfBuffer:
        buf = cls()
        buf.add_tf_messages((m.msg for m in bag.messages([static_topic])), static=True)
        buf.add_tf_messages((m.msg for m in bag.messages([tf_topic])), static=False)
        for parent, child, T in extra_static:
            buf.add_static(parent, child, T)
        return buf.finalize()

    # -------------------------------------------------------------------------------- lookup
    def frames(self) -> set[str]:
        return set(self.parent) | set(self.parent.values())

    def _chain(self, frame: str) -> list[str]:
        chain = [frame]
        while chain[-1] in self.parent:
            chain.append(self.parent[chain[-1]])
            if len(chain) > len(self.parent) + 1:
                raise ValueError(f"TF cycle at {frame!r}")
        return chain

    def _edge(self, child: str, t: float, max_gap_s: float) -> np.ndarray | None:
        if child in self.static:
            return self.static[child]
        return self.dynamic[child].interpolate(t, max_gap_s)

    def _T_ancestor_frame(self, chain: list[str], upto: str, t: float,
                          max_gap_s: float) -> np.ndarray | None:
        T = np.eye(4)
        for child in chain[:chain.index(upto)]:
            E = self._edge(child, t, max_gap_s)
            if E is None:
                return None
            T = E @ T
        return T

    def lookup(self, target: str, source: str, t: float,
               max_gap_s: float = np.inf) -> np.ndarray | None:
        """T_target_source at time t (maps points in `source` to `target`), None if unavailable."""
        ca, cb = self._chain(source), self._chain(target)
        common = next((f for f in ca if f in set(cb)), None)
        if common is None:
            raise KeyError(f"no TF path between {target!r} and {source!r}")
        Ts = self._T_ancestor_frame(ca, common, t, max_gap_s)
        Tt = self._T_ancestor_frame(cb, common, t, max_gap_s)
        if Ts is None or Tt is None:
            return None
        return np.linalg.inv(Tt) @ Ts
