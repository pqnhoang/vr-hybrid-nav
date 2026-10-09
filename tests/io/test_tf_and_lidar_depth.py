"""TF buffer lookups, raw image decoding, LiDAR sparse depth."""
from types import SimpleNamespace

import numpy as np
import pytest

from vlnav.io.ros_decode import decode_raw_image
from vlnav.io.tf_buffer import ROS_BODY_TO_OPTICAL_RPY_DEG, TfBuffer, static_matrix
from vlnav.perception.lidar_depth import sparse_depth


def T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    return static_matrix(xyz, rpy)


def robot_tf():
    """base_link -> odom (inverted odometry edge, like the vr_h3_2 bag), camera + optical."""
    buf = TfBuffer()
    for t, x in ((0.0, 0.0), (1.0, 2.0)):                  # robot drives +x: T_base_odom = -x
        buf.add_dynamic("base_link", "odom", t, T((-x, 0, 0)))
    buf.add_static("odom", "map", T())
    buf.add_static("base_link", "cam", T((0.1, 0, 0.5)))
    buf.add_static("cam", "cam_optical", T(rpy=ROS_BODY_TO_OPTICAL_RPY_DEG))
    return buf.finalize()


def test_lookup_through_inverted_odometry_edge_with_interpolation():
    tf = robot_tf()
    T_map_base = tf.lookup("map", "base_link", 0.5)
    assert np.allclose(T_map_base[:3, 3], (1.0, 0, 0))
    assert np.allclose(tf.lookup("map", "cam", 0.5)[:3, 3], (1.1, 0, 0.5))
    assert tf.lookup("map", "base_link", 2.0) is None           # outside the odometry span
    with pytest.raises(KeyError):
        tf.lookup("map", "nowhere", 0.5)


def test_optical_convention_looks_forward():
    R = robot_tf().lookup("base_link", "cam_optical", 0.0)[:3, :3]
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])               # optical z = body x (forward)
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0])              # optical x = body -y (right)


def test_two_parents_rejected():
    buf = TfBuffer()
    buf.add_static("a", "c", T())
    with pytest.raises(ValueError):
        buf.add_static("b", "c", T())


def test_decode_raw_bgra_with_row_padding():
    bgra = np.zeros((2, 3, 4), np.uint8)
    bgra[..., 0], bgra[..., 2] = 10, 200                         # B=10, R=200
    data = np.concatenate([bgra.reshape(2, 12), np.zeros((2, 4), np.uint8)], axis=1)   # step 16
    msg = SimpleNamespace(encoding="bgra8", height=2, width=3, step=16, data=data.tobytes())
    rgb = decode_raw_image(msg)
    assert rgb.shape == (2, 3, 3) and rgb[0, 0].tolist() == [200, 0, 10]


def test_sparse_depth_zbuffer_keeps_nearest():
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    T_map_cam = np.eye(4)                                       # camera = map, optical axes
    pts = np.array([[0, 0, 2.0], [0, 0, 5.0], [0.5, 0, 1.0], [0, 0, -1.0]])
    d = sparse_depth(pts, T_map_cam, K, (100, 100), 0.3, 10.0)
    assert d[50, 50] == pytest.approx(2.0)          # 2 m and 5 m hit the same pixel: nearest wins
    assert np.count_nonzero(d) == 1                 # (0.5, 0, 1) projects to u = 100 (off-image),
                                                    # (0, 0, -1) is behind the camera
