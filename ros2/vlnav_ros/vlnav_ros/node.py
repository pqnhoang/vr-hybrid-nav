"""Camera + PointCloud2 + TF -> VL-Nav -> proposed goal, occupancy grid and RViz markers."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from vlnav.common.config import deep_merge, load_yaml
from vlnav.common.models import Pose
from vlnav.io.live_source import LiveSource
from vlnav.io.ros_decode import stamp_s, transform_to_matrix
from vlnav.io.sources import load_bag_config, resolve
from vlnav.io.tf_buffer import static_matrix
from vlnav.runtime.builder import build_components
from vlnav.runtime.live_runtime import LiveRuntime, hold
from vlnav_ros.camera_images import annotate_detections
from vlnav_ros.messages import (frontier_messages, header, navigation_messages, occupancy_message,
                                pose_message, robot_messages)


class VlnavNode(Node):
    def __init__(self, *, components=None, **kwargs):
        super().__init__("vlnav", **kwargs)
        defaults = {
            "sensor_config": str(resolve("configs/bags/vr_h3_2_calib_1.yaml")),
            "run_config": str(resolve("configs/runs/vr_h3_2_ros2.yaml")),
            "run_dir": str(resolve("runs/ros2")),
            "instruction": "",
            "max_frame_age_s": 5.0,
            "max_scan_age_s": 0.5,
            "max_pose_age_s": 0.5,
            "goal_timeout_s": 2.0,
            "map_publish_hz": 1.0,
            "save_map_images": False,
            "map_image_every_s": 1.0,
            "map_image_dir": "",
            "publish_extra_static_tf": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.options = {name: self.get_parameter(name).value for name in defaults}
        for name in ("sensor_config", "run_config", "run_dir"):
            self.options[name] = self.options[name] or defaults[name]
        for name in ("max_frame_age_s", "max_scan_age_s", "max_pose_age_s", "goal_timeout_s",
                     "map_publish_hz", "map_image_every_s"):
            if self.options[name] <= 0:
                raise ValueError(f"{name} must be positive")
        cfg = self.sensor_cfg = load_bag_config(resolve(self.options["sensor_config"]))
        if cfg.frame_rate_hz <= 0 or cfg.lidar_window_s <= 0 or cfg.pair_wait_s < 0:
            raise ValueError("Invalid sensor rate/window/wait in sensor_config")
        self.map_frame = cfg.poses.map_frame
        self.source = LiveSource(cfg, self.options["max_frame_age_s"], self.options["max_scan_age_s"])
        rc = deep_merge(load_yaml(resolve(self.options["run_config"])), {"planner": {"blocking": False}})
        self.get_logger().info("Loading perception, memory and planner models...")
        self.runtime = LiveRuntime(components or build_components(rc, Path(self.options["run_dir"])),
                                   goto_object_approach=True)
        self.image_writer = None
        self.camera_writer = None
        if self.options["save_map_images"]:
            from vlnav_ros.camera_images import CameraImageWriter
            from vlnav_ros.map_images import MapImageWriter
            image_dir = self.options["map_image_dir"] or str(Path(self.options["run_dir"]) / "map_images")
            self.image_writer = MapImageWriter(image_dir)
            self.camera_writer = CameraImageWriter(Path(image_dir) / "camera")
            self.get_logger().info(f"Map PNGs: {self.image_writer.directory / 'latest.png'}")
            self.get_logger().info(f"Labeled camera views: {self.camera_writer.directory}")
        self.period = 1.0 / self.runtime.c.exploration.cfg.select_rate_hz
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0), node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        if self.options["publish_extra_static_tf"]:
            transforms = []
            for item in cfg.poses.extra_static_tf:
                pose = Pose.from_matrix(static_matrix(item.xyz, item.rpy_deg), 0.0)
                tr = TransformStamped(header=header(item.parent, 0.0), child_frame_id=item.child)
                p = pose_message(pose, tr.header).pose
                tr.transform.translation.x = p.position.x
                tr.transform.translation.y = p.position.y
                tr.transform.translation.z = p.position.z
                tr.transform.rotation = p.orientation
                self.tf_buffer.set_transform_static(tr, "vlnav_config")
                transforms.append(tr)
            if transforms:
                self.static_broadcaster.sendTransform(transforms)
        self.create_subscription(Image, cfg.color_topic, self._image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, cfg.camera_info_topic, self.source.camera_info, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, cfg.lidar_topic, self.source.lidar, qos_profile_sensor_data)
        self.create_subscription(String, "/task_instruction", self._instruction, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, "/mapping/occupancy", latched)
        self.annotated_pub = self.create_publisher(Image, "/perception/annotated_image", latched)
        self.goal_pub = self.create_publisher(PoseStamped, "/exploration/goal", 1)
        self.point_pub = self.create_publisher(PointStamped, "/exploration/goal_point", 1)
        self.active_pub = self.create_publisher(Bool, "/exploration/goal_active", 1)
        self.path_pub = self.create_publisher(PathMsg, "/exploration/path", latched)
        self.marker_pub = self.create_publisher(MarkerArray, "/exploration/markers", latched)
        self.frontier_pub = self.create_publisher(MarkerArray, "/exploration/frontiers", latched)
        self.robot_pub = self.create_publisher(MarkerArray, "/exploration/robot_markers", latched)
        self.report_pub = self.create_publisher(String, "/planner/report", 10)
        self.status_pub = self.create_publisher(String, "/planner/status", latched)
        self._lock = threading.Lock()
        self._pending_instruction = self.options["instruction"] or None
        self._output, self._output_t = hold(), -float("inf")
        self._map_msg = None
        self._frontier_points = []
        self._robot_pose = None
        self._saved_goal_key = None
        self._latest_image_stamp = None
        self._status = {"state": "IDLE", "input": self.source.wait_reason}
        self._reports = []
        self._stop = threading.Event()
        # Diagnostics and goal invalidation must run even before /clock starts.
        self._wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(self.period, self._publish_navigation, clock=self._wall_clock)
        self.create_timer(1.0 / self.options["map_publish_hz"], self._publish_map, clock=self._wall_clock)
        if self.image_writer is not None:
            self.create_timer(self.options["map_image_every_s"], self._save_map_image,
                              clock=self._wall_clock)
        self._worker = threading.Thread(target=self._run, name="vlnav_pipeline", daemon=True)
        self._worker.start()
        self.get_logger().info(f"Ready: map frame={self.map_frame}; send instructions on /task_instruction")

    def _image(self, msg):
        with self._lock:
            self._latest_image_stamp = stamp_s(msg.header)
        self.source.image(msg, time.monotonic())

    def _clock_issue(self, now, stamp):
        if self.get_parameter("use_sim_time").value and now == 0:
            return "waiting for /clock; replay the bag with --clock"
        if stamp is None:
            return ""
        age = now - stamp
        if age > self.options["max_frame_age_s"]:
            return "image timestamp is too old for ROS time; for bag replay use use_sim_time:=true and --clock"
        if age < 0:
            return "image timestamp is ahead of ROS time; check /clock and sensor timestamps"
        return ""

    def _instruction(self, msg):
        with self._lock:
            self._pending_instruction = msg.data
            self._output = hold()  # immediately invalidate the old goal during preemption

    def _lookup(self, frame_id, stamp):
        try:
            tr = self.tf_buffer.lookup_transform(self.map_frame, frame_id,
                                                 Time(nanoseconds=round(stamp * 1e9)))
            return transform_to_matrix(tr.transform)
        except TransformException:
            return None

    def _robot(self, now):
        try:
            tr = self.tf_buffer.lookup_transform(self.map_frame, self.sensor_cfg.poses.base_frame, Time())
            if abs(now - stamp_s(tr.header)) > self.options["max_pose_age_s"]:
                return None
            return Pose.from_matrix(transform_to_matrix(tr.transform), stamp_s(tr.header))
        except TransformException:
            return None

    def _run(self):
        last_t = None
        last_clock_issue = ""
        processed_frames = 0
        try:
            while not self._stop.is_set():
                now = self.get_clock().now().nanoseconds * 1e-9
                if last_t is not None and now < last_t - 0.01:
                    raise RuntimeError("ROS time moved backwards; restart the node for a new bag playback")
                last_t = now
                with self._lock:
                    image_stamp = self._latest_image_stamp
                clock_issue = self._clock_issue(now, image_stamp)
                frame = self.source.next_frame(self._lookup, time.monotonic()) if not clock_issue else None
                if frame is not None:
                    clock_issue = self._clock_issue(now, frame.stamp)
                if clock_issue and clock_issue != last_clock_issue:
                    self.get_logger().warning(clock_issue)
                last_clock_issue = clock_issue
                frame_valid = frame is not None and not clock_issue
                previous_fresh = (self.runtime.last_frame_stamp is not None and
                                  0 <= now - self.runtime.last_frame_stamp <= self.options["max_frame_age_s"])
                # Start tasks only with fresh geometry. Cancellation needs no sensor input.
                with self._lock:
                    can_submit = frame_valid or previous_fresh or self._pending_instruction == ""
                    instruction = None
                    if can_submit:
                        instruction, self._pending_instruction = self._pending_instruction, None
                if instruction is not None:
                    self.runtime.submit(instruction, now)
                result = None
                if frame_valid:
                    result = self.runtime.ingest(frame)
                    processed_frames += 1
                    map_msg = occupancy_message(self.runtime.c.grid, header(self.map_frame, frame.stamp))
                    with self._lock:
                        self._map_msg = map_msg
                if self._stop.is_set():
                    break
                now = self.get_clock().now().nanoseconds * 1e-9
                fresh = (self.runtime.last_frame_stamp is not None and
                         0 <= now - self.runtime.last_frame_stamp <= self.options["max_frame_age_s"])
                robot = self._robot(now) if fresh else None
                out = self.runtime.step(robot, now)
                state = self.runtime.c.planner.state()
                st = state.current
                if result is not None:
                    try:
                        annotated = annotate_detections(result, state, self.runtime.last_reasoning)
                        h, w = annotated.shape[:2]
                        self.annotated_pub.publish(Image(
                            header=header(frame.camera_id, frame.stamp), height=h, width=w,
                            encoding="bgr8", is_bigendian=0, step=w * 3, data=annotated.tobytes()))
                        if self.camera_writer is not None:
                            try:
                                self.camera_writer.write_frame(annotated, frame.stamp)
                            except Exception as exc:
                                self.get_logger().error(f"Camera image saving disabled: {type(exc).__name__}: {exc}")
                                self.camera_writer = None
                    except Exception as exc:
                        self.get_logger().error(f"Annotated camera publish failed: {type(exc).__name__}: {exc}")
                if self.camera_writer is not None and st is not None and st.type == "goto" and st.node_id is not None:
                    key = (st.task_id, st.id)
                    if key != self._saved_goal_key:
                        try:
                            node = self.runtime.c.memory.view().get_object(st.node_id)
                            self.camera_writer.write_goal(node, *key)
                            self._saved_goal_key = key
                        except Exception as exc:
                            self.get_logger().error(f"Goal camera image saving failed: {type(exc).__name__}: {exc}")
                            self._saved_goal_key = key
                reports = self.runtime.c.planner.pop_reports()
                # Copy on the worker: ROS callbacks must not iterate mutable core state.
                frontier_points = [{"xy": p.xy, "blacklisted": p.blacklisted_until > now}
                                   for p in self.runtime.c.exploration.frontiers.points.values()]
                with self._lock:
                    # An instruction can arrive while inference is running. Don't republish its old goal.
                    self._output = out if self._pending_instruction is None else hold()
                    self._output_t = now
                    self._frontier_points = frontier_points
                    self._robot_pose = robot
                    self._reports.extend(r.model_dump_json() for r in reports)
                    self._status = {"state": state.fsm, "task_id": state.task_id,
                                    "instruction": state.instruction, "done_reason": state.done_reason,
                                    "input": clock_issue or self.source.wait_reason,
                                    "use_sim_time": self.get_parameter("use_sim_time").value,
                                    "ros_time": now, "latest_image_stamp": image_stamp,
                                    "image_age_s": None if image_stamp is None else now - image_stamp,
                                    "processed_frames": processed_frames,
                                    "frontier_points": len(frontier_points), "frame_fresh": fresh,
                                    "pose_fresh": robot is not None}
                self._stop.wait(self.period)
        except Exception as exc:
            self.get_logger().error(f"Pipeline stopped: {type(exc).__name__}: {exc}")
            with self._lock:
                self._output = hold()
                self._robot_pose = None
                self._status = {"state": "ERROR", "error": str(exc)}
        finally:
            self.runtime.close()

    def _publish_navigation(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            out = self._output if 0 <= now - self._output_t <= self.options["goal_timeout_s"] else hold()
            status = dict(self._status)
            frontier_points = self._frontier_points
            robot = self._robot_pose
            reports, self._reports = self._reports, []
        if robot is not None and not (0 <= now - robot.stamp <= self.options["max_pose_age_s"]):
            robot = None
        goal, point, path, markers = navigation_messages(out, header(self.map_frame, now),
                                                        self.options["goal_timeout_s"])
        self.active_pub.publish(Bool(data=goal is not None))
        if goal is not None:
            self.goal_pub.publish(goal)
            self.point_pub.publish(point)
        self.path_pub.publish(path)
        self.marker_pub.publish(markers)
        self.frontier_pub.publish(frontier_messages(frontier_points, header(self.map_frame, now)))
        self.robot_pub.publish(robot_messages(robot, header(self.map_frame, now),
                                              self.options["goal_timeout_s"]))
        status.update(goal_active=goal is not None, goal_kind=out.goal_kind)
        self.status_pub.publish(String(data=json.dumps(status)))
        for report in reports:
            self.report_pub.publish(String(data=report))

    def _publish_map(self):
        with self._lock:
            msg = self._map_msg
        if msg is not None:
            self.map_pub.publish(msg)

    def _save_map_image(self):
        if self.image_writer is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            msg = self._map_msg
            frontiers = self._frontier_points
            robot = self._robot_pose
            out = self._output if 0 <= now - self._output_t <= self.options["goal_timeout_s"] else hold()
        if msg is None:
            return
        try:
            self.image_writer.write(msg, frontiers, robot, out)
        except Exception as exc:
            self.get_logger().error(f"Map image saving disabled: {type(exc).__name__}: {exc}")
            self.image_writer = None

    def stop(self):
        self._stop.set()
        with self._lock:
            self._output = hold()
            self._frontier_points = []
            self._robot_pose = None
        if rclpy.ok():
            self._publish_navigation()
        self._worker.join(timeout=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VlnavNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
