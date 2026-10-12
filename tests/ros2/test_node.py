"""Real DDS/TF transport; actual core with scripted detector/VLM (no GPU needed)."""
import json
import time

import numpy as np
import pytest

pytest.importorskip("rclpy")
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from planner_helpers import Router, explore, matcher, plan, target
from vlnav.acquisition.vlm_client import MockVLMClient
from vlnav.perception.detector import MockDetector
from vlnav.runtime.builder import build_components
from vlnav_ros.messages import header
from vlnav_ros.node import VlnavNode


@pytest.mark.parametrize("use_sim_time", [False, True])
def test_sensors_to_goal_map_and_cancel_over_ros(tmp_path, monkeypatch, use_sim_time):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    # Isolate from other local ROS applications.
    rclpy.init(domain_id=173)
    executor = SingleThreadedExecutor()
    node = peer = None
    c = build_components({
        "memory": {"rooms": {"enabled": False}, "checkpoints": {"checkpoint_every_s": 1000}},
        "planner": {"blocking": False},
        "mapping": {"resolution": .2, "initial_size_m": 8., "grow_margin_m": 2.,
                    "ground": {"mode": "constant"}, "log_odds": {"miss": -1.}, "ray_angle_bins": 360},
        "exploration": {"select_rate_hz": 5., "frontier": {"min_cluster_cells": 2}},
    }, tmp_path, vlm=MockVLMClient(Router(decompose=[plan([target("sofa")], [explore("sofa")])])),
        matcher=matcher(), detector=MockDetector())
    try:
        node = VlnavNode(components=c, parameter_overrides=[
            Parameter("instruction", value="Find a sofa"),
            Parameter("sensor_config", value=""),
            Parameter("run_config", value=""),
            Parameter("run_dir", value=""),
            Parameter("use_sim_time", value=use_sim_time),
            Parameter("save_map_images", value=use_sim_time),
            Parameter("map_image_dir", value=str(tmp_path / "map_images")),
            Parameter("map_image_every_s", value=1.0),
        ])
        peer = rclpy.create_node("vlnav_test_sensors")
        executor.add_node(node)
        executor.add_node(peer)
        cfg = node.sensor_cfg
        pubs = [peer.create_publisher(typ, topic, qos_profile_sensor_data) for typ, topic in
                [(Image, cfg.color_topic), (CameraInfo, cfg.camera_info_topic), (PointCloud2, cfg.lidar_topic)]]
        task_pub = peer.create_publisher(String, "/task_instruction", 10)
        clock_pub = peer.create_publisher(Clock, "/clock", 10)
        received = {k: [] for k in ("map", "camera", "goal", "active", "markers", "frontiers",
                                     "robot", "path", "status")}
        subs = [peer.create_subscription(typ, topic, lambda msg, k=k: received[k].append(msg), 10)
                for k, typ, topic in [
                    ("map", OccupancyGrid, "/mapping/occupancy"),
                    ("camera", Image, "/perception/annotated_image"),
                    ("goal", PoseStamped, "/exploration/goal"),
                    ("active", Bool, "/exploration/goal_active"),
                    ("markers", MarkerArray, "/exploration/markers"),
                    ("frontiers", MarkerArray, "/exploration/frontiers"),
                    ("robot", MarkerArray, "/exploration/robot_markers"),
                    ("path", Path, "/exploration/path"),
                    ("status", String, "/planner/status")]]
        tf = TransformBroadcaster(peer)
        static = StaticTransformBroadcaster(peer)
        tr = TransformStamped(header=header("base_link", 0.),
                              child_frame_id=cfg.poses.extra_static_tf[0].parent)
        tr.transform.rotation.w = 1.
        static.sendTransform(tr)
        angle = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        points = np.column_stack((3 * np.cos(angle), 3 * np.sin(angle), np.zeros_like(angle))).astype(np.float32)
        fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate("xyz")]
        stream_tf = True
        started = time.monotonic()
        sensor_sim_time = use_sim_time

        def send():
            now = peer.get_clock().now().nanoseconds * 1e-9
            if sensor_sim_time:
                now = 1000. + time.monotonic() - started
                clock_pub.publish(Clock(clock=header("", now).stamp))
            tr = TransformStamped(header=header("odom_lidar", now), child_frame_id="base_link")
            tr.transform.rotation.w = 1.
            if stream_tf:
                tf.sendTransform(tr)
            # Slightly delayed camera/scan stamps let TF bracket the acquisition time.
            h = header(cfg.poses.extra_static_tf[0].child, now - .1)
            pubs[0].publish(Image(header=h, width=2, height=2, step=6, encoding="rgb8", data=bytes(12)))
            pubs[1].publish(CameraInfo(header=h, width=2, height=2,
                                      k=[2., 0., 1., 0., 2., 1., 0., 0., 1.]))
            pubs[2].publish(PointCloud2(header=header("base_link", now - .1), width=len(points), height=1,
                                       fields=fields, point_step=12, row_step=12 * len(points), data=points.tobytes()))

        def until(predicate, seconds=10):
            end = time.monotonic() + seconds
            last_send = 0.
            while time.monotonic() < end:
                if time.monotonic() - last_send > .05:
                    send()
                    last_send = time.monotonic()
                executor.spin_once(timeout_sec=.02)
                if predicate():
                    return
            pytest.fail(f"ROS condition timed out. Status: {received['status'][-1:]}")

        if use_sim_time:
            # A missing /clock must still produce diagnostics and inactive goals.
            end = time.monotonic() + 3
            while time.monotonic() < end and not received["status"]:
                executor.spin_once(timeout_sec=.02)
            assert received["status"]
            status = json.loads(received["status"][-1].data)
            assert "waiting for /clock" in status["input"]
            assert status["state"] == "IDLE"
            assert not status["goal_active"]

        else:
            # Reproduce a bag replayed into a wall-clock node. No TF is required
            # to diagnose mismatched clocks; even new tasks must remain deferred.
            sensor_sim_time = True
            # Node remains on wall time; publishing /clock cannot change that.
            until(lambda: received["status"] and
                  "too old for ROS time" in json.loads(received["status"][-1].data)["input"])
            status = json.loads(received["status"][-1].data)
            assert status["image_age_s"] > node.options["max_frame_age_s"]
            assert not status["use_sim_time"]
            assert status["state"] == "IDLE"
            assert status["processed_frames"] == 0
            assert not received["goal"] and not received["map"] and not received["camera"]
            # Restore wall timestamps for the remaining checks.
            sensor_sim_time = False

        until(lambda: received["goal"] and received["map"] and received["camera"] and
              received["markers"] and received["frontiers"] and received["robot"] and
              any(m.action == Marker.ADD for m in received["markers"][-1].markers) and
              any(m.text == "ROBOT" for m in received["robot"][-1].markers) and
              any(m.points for m in received["frontiers"][-1].markers))
        assert received["goal"][-1].header.frame_id == "odom_lidar"
        assert 0 in received["map"][-1].data and -1 in received["map"][-1].data
        assert received["camera"][-1].encoding == "bgr8"
        assert received["camera"][-1].step == received["camera"][-1].width * 3
        assert len(received["camera"][-1].data) == received["camera"][-1].height * received["camera"][-1].step
        assert received["active"][-1].data
        assert received["path"][-1].poses
        assert json.loads(received["status"][-1].data)["processed_frames"] > 0
        if use_sim_time:
            image = tmp_path / "map_images" / "latest.png"
            until(image.is_file)
            assert image.stat().st_size > 1000
            until(lambda: len(list((tmp_path / "map_images").glob("frame_*.png"))) >= 2)
            until((tmp_path / "map_images" / "camera" / "latest_camera.jpg").is_file)
        else:
            assert node.image_writer is None
            assert not (tmp_path / "map_images").exists()
        dots = received["frontiers"][-1].markers[-1]
        assert dots.header.frame_id == received["map"][-1].header.frame_id
        assert dots.type == Marker.SPHERE_LIST
        assert all(p.z > 0 for p in dots.points)

        # A late subscriber receives the retained map, even without a new map publication.
        late_maps = []
        subs.append(peer.create_subscription(OccupancyGrid, "/mapping/occupancy", late_maps.append,
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        late_frontiers = []
        subs.append(peer.create_subscription(MarkerArray, "/exploration/frontiers", late_frontiers.append,
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        until(lambda: late_maps and late_frontiers and any(m.points for m in late_frontiers[-1].markers))
        # Stale odometry withholds the goal even while images and scans continue arriving.
        stream_tf = False
        until(lambda: not received["active"][-1].data and not received["path"][-1].poses
              and not json.loads(received["status"][-1].data).get("pose_fresh", True)
              and all(m.action == Marker.DELETEALL for m in received["robot"][-1].markers))
        stream_tf = True
        until(lambda: received["active"][-1].data and
              any(m.text == "ROBOT" for m in received["robot"][-1].markers))
        task_pub.publish(String(data=""))
        until(lambda: received["status"] and json.loads(received["status"][-1].data)["state"] == "FAILED"
              and not received["active"][-1].data and not received["path"][-1].poses
              and all(m.action == Marker.DELETEALL for m in received["markers"][-1].markers))
        assert not received["active"][-1].data
        # Cancelling a task clears its selected goal; computed map frontiers remain visible.
        assert any(m.points for m in received["frontiers"][-1].markers)
        assert any(m.text == "ROBOT" for m in received["robot"][-1].markers)
    finally:
        if node is not None:
            node.stop()
        executor.shutdown()
        if peer is not None:
            peer.destroy_node()
        if node is not None:
            node.destroy_node()
        c.planner.close()
        c.memory.close()
        rclpy.shutdown()
