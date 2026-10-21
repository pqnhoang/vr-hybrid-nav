import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vlnav.common.models import Pose
from vlnav.exploration.explorer import ExplorationOutput
from vlnav_ros.debug import DebugRecorder, goal_geometry


def test_recorder_flushes_concurrent_events_and_preserves_runs(tmp_path):
    errors = []
    recorder = DebugRecorder(tmp_path, errors.append)
    recorder.emit("startup", uninitialized_age=float("inf"), vector=np.array([1., 2.]))
    # The file is readable before shutdown, including non-finite startup values.
    assert json.loads(recorder.path.read_text())["uninitialized_age"] is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda k: recorder.emit("tick", k=k), range(30)))
    recorder.close()
    recorder.close()
    recorder.emit("after_close")
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert len(rows) == 31 and [r["seq"] for r in rows] == list(range(31))
    assert sorted(r["k"] for r in rows[1:]) == list(range(30))
    second = DebugRecorder(tmp_path, errors.append)
    assert second.path != recorder.path
    second.close()
    assert not errors


def test_goal_geometry_uses_robot_heading_not_world_x():
    robot = Pose(position=(1., 2., 0.), orientation_xyzw=(0., 0., 1., 0.), stamp=10.)
    goal = Pose(position=(2., 2., 0.), orientation_xyzw=(0., 0., 0., 1.), stamp=10.)
    out = ExplorationOutput(goal=goal, goal_kind="frontier", goal_id=3,
                            path=[(1., 2.), (2., 2.)], events=[])
    data = goal_geometry(out, robot)
    assert data["behind_robot"]
    np.testing.assert_allclose(data["position_base"], [-1., 0., 0.], atol=1e-9)
    assert math.isclose(abs(data["bearing_base_rad"]), math.pi)


def test_recorder_storage_failure_is_reported_without_raising(tmp_path):
    errors = []
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    recorder = DebugRecorder(blocker, errors.append)
    recorder.emit("ignored")
    recorder.close()
    assert len(errors) == 1 and "unavailable" in errors[0]
