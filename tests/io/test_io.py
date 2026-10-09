"""S1: trajectory interpolation, gravity alignment, ROS message decoding."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlnav.io.gravity_align import T_gravity_from_up, up_direction_in_map
from vlnav.io.ros_decode import decode_compressed_depth, decode_pointcloud_xyz
from vlnav.io.trajectory import Trajectory


def pose(xyz, yaw_deg):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


TRAJ = Trajectory([0.0, 1.0, 5.0], [pose((0, 0, 0), 0), pose((2, 0, 0), 90), pose((2, 4, 0), 90)])


def test_interpolation_slerp_lerp():
    T = TRAJ.interpolate(0.5)
    assert np.allclose(T[:3, 3], (1, 0, 0))
    assert Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)[2] == pytest.approx(45)
    assert np.allclose(TRAJ.interpolate(5.0), TRAJ.poses[-1])


def test_gaps_and_bounds():
    assert TRAJ.interpolate(-0.1) is None and TRAJ.interpolate(5.1) is None
    assert TRAJ.interpolate(3.0, max_gap_s=1.0) is None                # inside the 4 s gap
    assert TRAJ.interpolate(0.5, max_gap_s=1.0) is not None
    assert TRAJ.gap_at(3.0) == 4.0 and TRAJ.gap_at(-1) == np.inf


def test_load_matrix_blocks(tmp_path):
    p = tmp_path / "odo.txt"
    p.write_text("".join(f"{t}\n" + "\n".join(" ".join(map(str, r)) for r in pose((t, 0, 0), 0)) + "\n\n"
                         for t in (10.0, 10.1)))
    tr = Trajectory.load(p, "matrix_blocks")
    assert len(tr) == 2 and np.allclose(tr.interpolate(10.05)[:3, 3], (10.05, 0, 0))


def test_gravity_alignment_levels_a_tilted_frame():
    tilt = Rotation.from_euler("y", 3, degrees=True).as_matrix()     # sensor pitched 3 deg
    up_body = tilt.T @ np.array([0, 0, 1.0])                          # accelerometer reading
    stamps = np.linspace(0, 5, 50)
    up = up_direction_in_map(TRAJ, stamps, np.tile(up_body, (50, 1)), np.eye(3), np.inf)
    # TRAJ rotates about z only, so the map-frame up is the body-frame up rotated by yaw
    Tg = T_gravity_from_up(up)
    assert np.allclose(Tg[:3, :3] @ up, (0, 0, 1), atol=1e-9)
    assert np.allclose(T_gravity_from_up((0, 0, 1)), np.eye(4))


def test_decode_compressed_depth():
    depth_mm = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 250)
    ok, png = cv2.imencode(".png", depth_mm)
    msg = SimpleNamespace(format="16UC1; compressedDepth", data=bytes(12) + png.tobytes())
    assert np.allclose(decode_compressed_depth(msg), depth_mm / 1000.0)
    with pytest.raises(ValueError):
        decode_compressed_depth(SimpleNamespace(format="32FC1; compressedDepth", data=msg.data))


def test_decode_pointcloud_xyz_drops_invalid():
    pts = np.array([[1, 2, 3, 9], [np.nan, 0, 0, 9], [0, 0, 0, 9], [4, 5, 6, 9]], dtype=np.float32)
    fields = [SimpleNamespace(name=n, offset=4 * i, datatype=7) for i, n in enumerate("xyzi")]
    msg = SimpleNamespace(fields=fields, width=4, height=1, point_step=16, is_bigendian=False,
                          data=pts.tobytes())
    assert np.allclose(decode_pointcloud_xyz(msg), [[1, 2, 3], [4, 5, 6]])
