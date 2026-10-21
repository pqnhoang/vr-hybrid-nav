"""Send selected VL-Nav poses to the robot's NavigateToPose action server."""
from __future__ import annotations

import copy
import json
import math

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.time import Time
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import TransformException
from vr_common_msgs.action import NavigateToPose

from vlnav_ros.debug import goal_delta, pose_data


_STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


def _same_goal(old, new) -> bool:
    old_key, old_pose = old
    new_key, new_pose = new
    if old_key != new_key or old_pose.header.frame_id != new_pose.header.frame_id:
        return False
    a, b = old_pose.pose, new_pose.pose
    distance = math.dist((a.position.x, a.position.y, a.position.z),
                         (b.position.x, b.position.y, b.position.z))
    qa, qb = a.orientation, b.orientation
    dot = abs(qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w)
    angle = 2.0 * math.acos(min(1.0, dot))
    return distance < 0.25 and angle < 0.25


class NavigationAction:
    def __init__(self, node, tf_buffer, *, name: str, frame: str,
                 waypoint_name: str = "", behavior_tree: str = "", debug=None):
        self.node = node
        self.tf_buffer = tf_buffer
        self.name = name
        self.frame = frame
        self.waypoint_name = waypoint_name
        self.behavior_tree = behavior_tree
        self.debug = debug
        self.client = ActionClient(node, NavigateToPose, name)
        self.status_pub = node.create_publisher(String, "/exploration/navigation_action_status", 10)
        self._source = None
        self._desired = None
        self._serial = 0
        self._send_future = None
        self._handle = None
        self._active_serial = None
        self._cancel_future = None
        self._finished_serial = None
        self._server_ready = False

    def _event(self, kind: str, **data):
        data.setdefault("server", self.name)
        self.status_pub.publish(String(data=json.dumps({"type": kind, "data": {"action": "nav_goal", **data}})))
        self._debug(kind, **data)

    def _debug(self, event, **data):
        if self.debug is not None:
            self.debug(event, **{"server": self.name, **data})

    def _action_goal(self, pose):
        try:
            if pose.header.frame_id == self.frame:
                converted = copy.deepcopy(pose)
            else:
                transform = self.tf_buffer.lookup_transform(self.frame, pose.header.frame_id, Time())
                converted = do_transform_pose_stamped(pose, transform)
        except TransformException as exc:
            self._debug("action_tf_missing", source_frame=pose.header.frame_id,
                        target_frame=self.frame, error=str(exc))
            self.node.get_logger().warning(f"Navigation action waits for TF to {self.frame}: {exc}",
                                           throttle_duration_sec=5.0)
            return None
        converted.header.frame_id = self.frame
        converted.header.stamp = self.node.get_clock().now().to_msg()
        if self.debug is not None:
            self._debug("action_frame_conversion", source=pose_data(pose), converted=pose_data(converted))
        goal = NavigateToPose.Goal()
        goal.waypoint_name = self.waypoint_name
        goal.pose = converted
        goal.behavior_tree = self.behavior_tree
        return goal

    @staticmethod
    def _goal_summary(goal) -> str:
        pose = goal.pose
        p, q = pose.pose.position, pose.pose.orientation
        stamp = pose.header.stamp
        return (
            f"frame={pose.header.frame_id!r}, stamp={stamp.sec}.{stamp.nanosec:09d}, "
            f"position=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
            f"orientation=({q.x:.5f}, {q.y:.5f}, {q.z:.5f}, {q.w:.5f}), "
            f"waypoint_name={goal.waypoint_name!r}, behavior_tree={goal.behavior_tree!r}"
        )

    def update(self, pose, key=None, *, clear_reason="no_valid_goal"):
        """Track one selected goal; repeated publications do not resend the action."""
        if pose is None:
            if self._source is not None:
                self._debug("action_goal_cleared", reason=clear_reason, serial=self._serial,
                            key=self._source[0])
            self._source = self._desired = None
        elif self._source is None or not _same_goal(self._source, (key, pose)):
            if self.debug is not None:
                reasons, delta = ["initial_goal"], None
                previous = None
                if self._source is not None:
                    old_key, old_pose = self._source
                    previous = {"key": old_key, "pose": pose_data(old_pose)}
                    delta = goal_delta(old_pose, pose)
                    reasons = []
                    if old_key != key:
                        reasons.append("goal_key_changed")
                    if old_pose.header.frame_id != pose.header.frame_id:
                        reasons.append("frame_changed")
                    if delta["distance_xyz_m"] >= 0.25:
                        reasons.append("position_changed")
                    if delta["orientation_delta_rad"] >= 0.25:
                        reasons.append("orientation_changed")
            action_goal = self._action_goal(pose)
            if action_goal is None:
                self._source = self._desired = None
            else:
                self._serial += 1
                self._source = (key, copy.deepcopy(pose))
                self._desired = (self._serial, action_goal)
                if self.debug is not None:
                    self._debug("action_goal_desired", serial=self._serial, key=key, reasons=reasons,
                                previous=previous, source=pose_data(pose),
                                converted=pose_data(action_goal.pose), delta=delta)
        self._reconcile()

    def _reconcile(self):
        desired_serial = self._desired[0] if self._desired is not None else None
        if self._handle is not None:
            if self._active_serial != desired_serial and self._cancel_future is None:
                serial = self._active_serial
                try:
                    self._cancel_future = self._handle.cancel_goal_async()
                    self._debug("action_cancel_requested", serial=serial, desired_serial=desired_serial)
                except Exception as exc:
                    self._debug("action_cancel_error", serial=serial, error=str(exc))
                    self.node.get_logger().warning(f"Navigation action cancel request failed: {exc}",
                                                   throttle_duration_sec=5.0)
                    return
                self._cancel_future.add_done_callback(lambda future: self._cancel_done(future, serial))
            return
        if (self._send_future is not None or self._desired is None or
                desired_serial == self._finished_serial):
            return
        if not self.client.server_is_ready():
            if self._server_ready:
                self._debug("action_server_unavailable")
            self._server_ready = False
            self.node.get_logger().warning(f"Waiting for navigation action server {self.name}",
                                           throttle_duration_sec=5.0)
            return
        if not self._server_ready:
            self._debug("action_server_ready")
            self.node.get_logger().info(f"Navigation action server {self.name} is available")
            self._server_ready = True
        serial, goal = self._desired
        try:
            self._send_future = self.client.send_goal_async(
                goal, feedback_callback=lambda feedback: self._feedback(feedback, serial))
            if self.debug is not None:
                self._debug("action_send", serial=serial, pose=pose_data(goal.pose),
                            waypoint_name=goal.waypoint_name, behavior_tree=goal.behavior_tree)
        except Exception as exc:
            self._debug("action_send_error", serial=serial, error=str(exc))
            self.node.get_logger().warning(f"Navigation action send failed: {exc}",
                                           throttle_duration_sec=5.0)
            return
        self._send_future.add_done_callback(lambda future: self._response(future, serial))
        self.node.get_logger().info(
            f"Sending VR navigation action to {self.name}: {self._goal_summary(goal)}")

    def _response(self, future, serial):
        self._send_future = None
        try:
            handle = future.result()
        except Exception as exc:
            self._debug("action_response_error", serial=serial, error=str(exc))
            self.node.get_logger().error(f"Navigation action goal request failed: {exc}")
            self._finished_serial = serial
            self._reconcile()
            return
        accepted = handle is not None and handle.accepted
        self._event("action_goal_response", serial=serial, accepted=bool(accepted),
                    status="ACCEPTED" if accepted else "REJECTED")
        if not accepted:
            self.node.get_logger().warning(
                f"Navigation action server {self.name} rejected the goal (status=REJECTED)")
            self._finished_serial = serial
        else:
            self.node.get_logger().info(
                f"Navigation action server {self.name} received and accepted the goal "
                f"(status=ACCEPTED)")
            self._handle = handle
            self._active_serial = serial
            handle.get_result_async().add_done_callback(lambda result: self._result(result, serial))
        self._reconcile()

    def _feedback(self, message, serial):
        if serial != self._active_serial:
            return
        fb = message.feedback
        navigation_time = fb.navigation_time.sec + fb.navigation_time.nanosec * 1e-9
        estimated_time = (fb.estimated_time_remaining.sec +
                          fb.estimated_time_remaining.nanosec * 1e-9)
        self._event("action_feedback", serial=serial, status="EXECUTING",
                    distance_remaining=fb.distance_remaining,
                    number_of_recoveries=fb.number_of_recoveries,
                    navigation_time=navigation_time,
                    estimated_time_remaining=estimated_time)
        self.node.get_logger().info(
            f"Navigation action feedback from {self.name} (status=EXECUTING): "
            f"distance_remaining={fb.distance_remaining:.3f} m, "
            f"navigation_time={navigation_time:.3f} s, "
            f"estimated_time_remaining={estimated_time:.3f} s, "
            f"number_of_recoveries={fb.number_of_recoveries}")

    def _result(self, future, serial):
        try:
            result = future.result()
            status = int(result.status)
            status_name = _STATUS_NAMES.get(status, f"UNKNOWN({status})")
            self._event("action_result", serial=serial, status=status,
                        status_name=status_name,
                        succeeded=status == GoalStatus.STATUS_SUCCEEDED)
            self.node.get_logger().info(
                f"Navigation action result from {self.name}: status={status_name} ({status})")
        except Exception as exc:
            self._debug("action_result_error", serial=serial, error=str(exc))
            self.node.get_logger().error(f"Navigation action result failed: {exc}")
        if self._active_serial == serial:
            self._handle = self._cancel_future = None
            self._active_serial = None
        self._finished_serial = serial
        self._reconcile()

    def _cancel_done(self, future, serial):
        try:
            response = future.result()
            accepted = bool(response.goals_canceling)
        except Exception as exc:
            self.node.get_logger().warning(f"Navigation action cancel failed: {exc}")
            accepted = False
            self._debug("action_cancel_error", serial=serial, error=str(exc))
        self._debug("action_cancel_response", serial=serial, accepted=accepted)
        self._cancel_future = None
        if not accepted:
            return  # Keep the handle until a result arrives or cancellation succeeds.
        self.node.get_logger().info("Cancelled VR navigation goal")
        if self._active_serial == serial:
            self._handle = None
            self._active_serial = None
        self._finished_serial = serial
        self._reconcile()
