"""Selected goals become one transformed robot action, including preemption."""
from concurrent.futures import Future
from types import SimpleNamespace as NS
import json
import time

import pytest

pytest.importorskip("rclpy")
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped
from std_msgs.msg import Header, String
from rclpy.action import ActionServer
from rclpy.executors import SingleThreadedExecutor
from tf2_ros import Buffer, TransformException
from vr_common_msgs.action import NavigateToPose

import vlnav_ros.navigation_action as action_module


class FakeClient:
    def __init__(self, *_):
        self.sent = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal, feedback_callback):
        future = Future()
        self.sent.append((goal, feedback_callback, future))
        return future


class FakeHandle:
    accepted = True

    def __init__(self):
        self.result = Future()
        self.cancel = Future()
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel


class FakeNode:
    def __init__(self):
        self.events = []
        self.logs = []

    def create_publisher(self, *_):
        return NS(publish=self.events.append)

    def get_logger(self):
        return NS(
            info=lambda message, **_: self.logs.append(("info", message)),
            warning=lambda message, **_: self.logs.append(("warning", message)),
            error=lambda message, **_: self.logs.append(("error", message)),
        )

    def get_clock(self):
        return NS(now=lambda: NS(to_msg=lambda: Time(sec=42)))


def source_goal(x=1.):
    return PoseStamped(header=Header(frame_id="odom_lidar", stamp=Time(sec=10)),
                       pose=Pose(position=Point(x=x, y=2., z=0.), orientation=Quaternion(w=1.)))


def test_action_transforms_deduplicates_and_preempts(monkeypatch):
    monkeypatch.setattr(action_module, "ActionClient", FakeClient)
    transform = TransformStamped(header=Header(frame_id="map"), child_frame_id="odom_lidar")
    transform.transform.translation.x = 5.
    transform.transform.rotation.w = 1.
    tf = NS(lookup_transform=lambda target, source, stamp: transform)
    node = FakeNode()
    bridge = action_module.NavigationAction(node, tf, name="/navigate_to_pose", frame="map",
                                            waypoint_name="desk", behavior_tree="custom.xml")
    key = (1, 2, "goto", 7)
    bridge.update(source_goal(), key)
    sent = bridge.client.sent
    assert len(sent) == 1
    action_goal = sent[0][0]
    assert action_goal.pose.header.frame_id == "map"
    assert action_goal.pose.header.stamp.sec == 42
    assert action_goal.pose.pose.position.x == 6.
    assert action_goal.pose.pose.position.y == 2.
    assert action_goal.pose.pose.orientation.w == 1.
    assert action_goal.waypoint_name == "desk" and action_goal.behavior_tree == "custom.xml"

    bridge.update(source_goal(), key)
    assert len(sent) == 1
    first_handle = FakeHandle()
    sent[0][2].set_result(first_handle)
    sent[0][1](NS(feedback=NS(
        distance_remaining=1.25,
        number_of_recoveries=2,
        navigation_time=NS(sec=3, nanosec=500_000_000),
        estimated_time_remaining=NS(sec=4, nanosec=250_000_000),
    )))
    bridge.update(source_goal(2.), key)
    assert first_handle.cancel_calls == 1
    assert len(sent) == 1  # replacement waits for cancellation
    first_handle.cancel.set_result(NS(goals_canceling=[1]))
    assert len(sent) == 2
    assert sent[1][0].pose.pose.position.x == 7.

    second_handle = FakeHandle()
    sent[1][2].set_result(second_handle)
    second_handle.result.set_result(NS(status=GoalStatus.STATUS_SUCCEEDED))
    bridge.update(source_goal(2.), key)
    assert len(sent) == 2  # a completed goal is not sent again every timer tick
    assert any('"succeeded": true' in event.data for event in node.events)
    event_data = [json.loads(event.data)["data"] for event in node.events]
    assert any(event.get("status") == "ACCEPTED" for event in event_data)
    assert any(event.get("status") == "EXECUTING" for event in event_data)
    assert any(event.get("status_name") == "SUCCEEDED" for event in event_data)
    log_text = "\n".join(message for _, message in node.logs)
    assert "server /navigate_to_pose is available" in log_text
    assert "position=(6.000, 2.000, 0.000)" in log_text
    assert "received and accepted the goal (status=ACCEPTED)" in log_text
    assert "status=EXECUTING" in log_text and "distance_remaining=1.250 m" in log_text
    assert "status=SUCCEEDED (4)" in log_text


def test_missing_map_transform_does_not_send_mislabeled_goal(monkeypatch):
    monkeypatch.setattr(action_module, "ActionClient", FakeClient)

    def missing(*_):
        raise TransformException("no map transform")

    bridge = action_module.NavigationAction(FakeNode(), NS(lookup_transform=missing),
                                            name="/navigate_to_pose", frame="map")
    bridge.update(source_goal(), (1, 2, "goto", 7))
    assert bridge.client.sent == []


def test_debug_identifies_replacement_and_timeout_cancel(monkeypatch):
    monkeypatch.setattr(action_module, "ActionClient", FakeClient)
    events = []
    bridge = action_module.NavigationAction(
        FakeNode(), None, name="/test", frame="odom_lidar",
        debug=lambda event, **data: events.append({"event": event, **data}))
    key = (1, 2, "frontier", 7)
    bridge.update(source_goal(), key)
    bridge.update(source_goal(), key)
    first = FakeHandle()
    bridge.client.sent[0][2].set_result(first)
    moved = source_goal()
    moved.pose.position.z = .3
    bridge.update(moved, key)
    desired = [e for e in events if e["event"] == "action_goal_desired"]
    assert len(desired) == 2
    assert desired[-1]["reasons"] == ["position_changed"]
    assert desired[-1]["delta"]["distance_xy_m"] == 0
    assert desired[-1]["delta"]["delta_z_m"] == .3
    assert first.cancel_calls == 1
    first.cancel.set_result(NS(goals_canceling=[1]))
    second = FakeHandle()
    bridge.client.sent[1][2].set_result(second)
    bridge.update(None, key, clear_reason="worker_output_timeout")
    assert any(e["event"] == "action_goal_cleared" and e["reason"] == "worker_output_timeout"
               for e in events)
    assert second.cancel_calls == 1


def test_real_action_server_receives_one_goal(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    rclpy.init(domain_id=174)
    executor = SingleThreadedExecutor()
    node = server_node = server = None
    try:
        node = rclpy.create_node("vlnav_action_client_test")
        server_node = rclpy.create_node("vlnav_action_server_test")
        executor.add_node(node)
        executor.add_node(server_node)
        tf = Buffer()
        transform = TransformStamped(header=Header(frame_id="map"), child_frame_id="odom_lidar")
        transform.transform.translation.x = 5.
        transform.transform.rotation.w = 1.
        tf.set_transform_static(transform, "test")
        bridge = action_module.NavigationAction(node, tf, name="/test_vr_navigate_to_pose", frame="map")
        received = []
        events = []
        status_sub = server_node.create_subscription(
            String, "/exploration/navigation_action_status", events.append, 10)

        def execute(handle):
            received.append(handle.request)
            handle.succeed()
            return NavigateToPose.Result()

        server = ActionServer(server_node, NavigateToPose, "/test_vr_navigate_to_pose",
                              execute_callback=execute)
        key = (1, 2, "goto", 7)
        end = time.monotonic() + 5.
        while time.monotonic() < end:
            bridge.update(source_goal(), key)
            executor.spin_once(timeout_sec=.02)
            if received and any(json.loads(msg.data)["type"] == "action_result" for msg in events):
                break
        assert len(received) == 1
        assert any(json.loads(msg.data)["data"].get("succeeded") for msg in events)
        assert received[0].pose.header.frame_id == "map"
        assert received[0].pose.pose.position.x == 6.
        for _ in range(20):
            bridge.update(source_goal(), key)
            executor.spin_once(timeout_sec=.01)
        assert len(received) == 1
    finally:
        if server is not None:
            server.destroy()
        executor.shutdown()
        if server_node is not None:
            server_node.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
