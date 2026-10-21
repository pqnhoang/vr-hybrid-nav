from types import SimpleNamespace as NS

import numpy as np

from vlnav.io.live_source import LiveSource
from vlnav.io.sources import load_bag_config, resolve


def hdr(t, frame="sensor"):
    ns = round(t * 1e9)
    sec, nanosec = divmod(ns, 10**9)
    return NS(stamp=NS(sec=sec, nanosec=nanosec), frame_id=frame)


def image(t):
    return NS(header=hdr(t, "camera"), width=2, height=2, step=6,
              encoding="rgb8", data=bytes(12))


def cloud(t):
    return NS(header=hdr(t, "lidar"), width=1, height=1, point_step=12, is_bigendian=False,
              fields=[NS(name=n, offset=i * 4, datatype=7) for i, n in enumerate("xyz")],
              data=np.array([[1, 0, 1]], np.float32).tobytes())


def source(**kwargs):
    s = LiveSource(load_bag_config(resolve("configs/bags/vr_h3_2_calib_1.yaml")), **kwargs)
    s.camera_info(NS(k=[2., 0., 1., 0., 2., 1., 0., 0., 1.], d=[], width=2, height=2))
    return s


def lookup(frame, t):
    T = np.eye(4)
    T[0, 3] = t
    return T


def test_scan_time_transforms_past_window_and_pair_wait():
    s = source()
    s.image(image(10.), 100.)
    for t in (8.9, 9.2, 9.8, 10.1):
        s.lidar(cloud(t))
    assert s.next_frame(lookup, 100.2) is None
    f = s.next_frame(lookup, 100.4)
    assert f.stamp == 10.
    np.testing.assert_allclose(f.lidar_points_map, [[10.2, 0, 1], [10.8, 0, 1]])
    assert f.T_map_cam[0, 3] == f.T_map_base[0, 3] == 10.
    assert f.sensor_origin_map == (10., 0., 0.)
    assert s.next_frame(lookup, 100.5) is None  # don't process a camera image twice


def test_waits_for_exact_tf_and_fresh_lidar_then_recovers():
    s = source()
    s.image(image(10.), 100.)
    s.lidar(cloud(9.1))
    assert s.next_frame(lookup, 100.4) is None
    s.lidar(cloud(9.9))
    assert s.next_frame(lambda frame, t: None if frame == "camera" else lookup(frame, t), 100.5) is None
    assert s.next_frame(lookup, 100.6) is not None


def test_latest_ready_image_and_bounded_buffers():
    s = source()
    for i in range(1000):
        t = float(i)
        s.image(image(t), t)
        s.lidar(cloud(t))
    assert len(s._images) <= 16 and len(s._scans) <= 128
    assert s.next_frame(lookup, 999.5).stamp == 999.
    s.image(image(1000.), 1000.)
    s.lidar(cloud(1000.))
    assert s.next_frame(lookup, 1006.) is None  # don't revive stale input


def test_debug_records_pairing_and_missing_tf_without_changing_frames():
    s = source(debug=True)
    s.image(image(10.), 100.)
    for t in (9.2, 9.8, 10.1):
        s.lidar(cloud(t))
    assert s.next_frame(lambda frame, t: None if frame == "camera" else lookup(frame, t), 100.4) is None
    assert "TF" in s.debug_snapshot()["wait_reason"]
    f = s.next_frame(lookup, 100.5)
    info = s.debug_snapshot()
    assert info["selected_scan_stamps"] == [9.2, 9.8]
    assert info["future_scans_excluded"] == 1
    assert abs(info["newest_scan_gap_s"] - .2) < 1e-6
    assert [r["T_map_sensor"][0][3] for r in info["scan_transforms"]] == [9.2, 9.8]
    assert info["counters"]["frames_emitted"] == 1
    assert info["points_used"] == len(f.lidar_points_map) == 2
    assert source().last_sync == {}
