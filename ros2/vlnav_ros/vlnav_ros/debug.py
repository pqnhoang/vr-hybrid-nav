"""Opt-in, bounded diagnostic records; never store images or point-cloud payloads."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return value


class DebugRecorder:
    """One flushed JSONL file per process/run, shared by worker and ROS callbacks."""

    def __init__(self, run_dir, on_error):
        self._lock = threading.Lock()
        self._file = None
        self._sequence = 0
        self._on_error = on_error
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = Path(run_dir) / "debug" / f"navigation_{name}_{os.getpid()}.jsonl"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("x", encoding="utf-8", buffering=1)
        except OSError as exc:
            self._on_error(f"Debug recording unavailable: {exc}")

    def emit(self, event, **data):
        with self._lock:
            if self._file is None:
                return
            try:
                row = _json_value({"seq": self._sequence, "event": event,
                                   "wall_time": time.time(), "monotonic_s": time.monotonic(), **data})
                self._file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                self._file.flush()
                self._sequence += 1
            except (OSError, TypeError, ValueError) as exc:
                # Diagnostics must not stop navigation if storage fails.
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None
                self._on_error(f"Debug recording disabled: {exc}")

    def close(self):
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError as exc:
                    self._on_error(f"Debug recording close failed: {exc}")
                self._file = None


def pose_data(pose, frame=None):
    if pose is None:
        return None
    if hasattr(pose, "header"):
        p, q = pose.pose.position, pose.pose.orientation
        xyz, quat = [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
        stamp = pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9
        frame = pose.header.frame_id
    else:
        xyz, quat, stamp = list(pose.position), list(pose.orientation_xyzw), pose.stamp
    x, y, z, w = quat
    return {"frame": frame, "stamp": stamp, "position": xyz, "orientation_xyzw": quat,
            "yaw_rad": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))}


def goal_delta(old, new):
    a, b = pose_data(old), pose_data(new)
    dot = abs(sum(x * y for x, y in zip(a["orientation_xyzw"], b["orientation_xyzw"])))
    return {"distance_xyz_m": math.dist(a["position"], b["position"]),
            "distance_xy_m": math.dist(a["position"][:2], b["position"][:2]),
            "delta_z_m": b["position"][2] - a["position"][2],
            "orientation_delta_rad": 2 * math.acos(min(1.0, dot))}


def goal_geometry(out, robot):
    """Express the actual selected position in the measured robot base frame."""
    if out.goal is None or robot is None:
        return None
    delta = np.asarray(out.goal.position) - np.asarray(robot.position)
    local = delta @ robot.matrix()[:3, :3]
    yaw_error = out.goal.yaw() - robot.yaw()
    return {"position_base": local.tolist(), "behind_robot": bool(local[0] < 0),
            "distance_xy_m": float(np.linalg.norm(delta[:2])),
            "bearing_base_rad": math.atan2(local[1], local[0]),
            "goal_yaw_error_rad": math.atan2(math.sin(yaw_error), math.cos(yaw_error))}


def selection_data(core):
    """Copy selection evidence on its owner thread; bound the candidate lists."""
    snap = core.snapshot()
    previous = core.selector.last_debug.get("previous")
    previous_point = None
    if previous is not None:
        pool = snap["points"] if previous["kind"] == "frontier" else snap["instances"]
        previous_point = next((p for p in pool if p["id"] == previous["id"]), None)
    ranked = sorted((p for p in snap["points"] if p["score"] is not None),
                    key=lambda p: (-p["score"]["s_nesy"], p["id"]))
    return {"subtask": snap["subtask"], "selector": dict(core.selector.last_debug),
            "previous_point": previous_point, "top_frontiers": ranked[:5],
            "top_instances": sorted(snap["instances"], key=lambda p: (-p["rank"], p["id"]))[:5],
            "frontier_count": len(snap["points"]), "instance_count": len(snap["instances"]),
            "goto_object_xy": core.goto_object_xy, "notes": snap["notes"],
            "events": snap["events"], "timing_ms": snap["timing_ms"], "grid": snap["grid"]}


def path_data(path):
    if not path:
        return {"points": 0, "length_m": 0.0, "sample_xy": []}
    stride = max(1, math.ceil(len(path) / 16))
    sample = list(path[::stride])
    if sample[-1] != path[-1]:
        sample.append(path[-1])
    return {"points": len(path), "length_m": sum(math.dist(a, b) for a, b in zip(path, path[1:])),
            "sample_xy": sample}
