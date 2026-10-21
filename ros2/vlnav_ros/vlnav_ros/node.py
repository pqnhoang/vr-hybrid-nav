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
from vlnav_ros.debug import DebugRecorder, goal_delta, goal_geometry, path_data, pose_data, selection_data
from vlnav_ros.messages import (frontier_messages, header, navigation_messages, occupancy_message,
                                pose_message, robot_messages)
from vlnav_ros.navigation_action import NavigationAction


class VlnavNode(Node):
    def __init__(self, *, components=None, **kwargs):
        super().__init__("vlnav", **kwargs)
        defaults = {
            "sensor_config": str(resolve("configs/bags/vr_h3_2_calib_1.yaml")),
            "run_config": str(resolve("configs/runs/vr_h3_2_ros2.yaml")),
            "run_dir": str(resolve("runs/ros2")),
            "instruction": "",
            "debug": False,
            "nav_action_enabled": True,
            "nav_action_name": "vr_navigate_to_pose",
            "nav_action_frame": "map",
            "nav_waypoint_name": "use_goal_pose",
            "nav_behavior_tree": "",
            "max_frame_age_s": 30.0,
            "max_scan_age_s": 0.5,
            "max_pose_age_s": 30.0,
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
        self.debug_recorder = (DebugRecorder(self.options["run_dir"], self.get_logger().error)
                               if self.options["debug"] else None)
        self._tf_debug = []
        self._pose_debug = {}
        self._output_seq = self._published_cycle_id = None
        self._output_reason = "waiting_for_first_output"
        self._last_publish_reason = None
        if self.debug_recorder is not None:
            self.get_logger().info(f"Debug JSONL: {self.debug_recorder.path.resolve()}")
        for name in ("max_frame_age_s", "max_scan_age_s", "max_pose_age_s", "goal_timeout_s",
                     "map_publish_hz", "map_image_every_s"):
            if self.options[name] <= 0:
                raise ValueError(f"{name} must be positive")
        cfg = self.sensor_cfg = load_bag_config(resolve(self.options["sensor_config"]))
        if cfg.frame_rate_hz <= 0 or cfg.lidar_window_s <= 0 or cfg.pair_wait_s < 0:
            raise ValueError("Invalid sensor rate/window/wait in sensor_config")
        self.map_frame = cfg.poses.map_frame
        self.source = LiveSource(cfg, self.options["max_frame_age_s"], self.options["max_scan_age_s"],
                                 debug=self.options["debug"])
        rc = deep_merge(load_yaml(resolve(self.options["run_config"])), {"planner": {"blocking": False}})
        self.get_logger().info("Loading perception, memory and planner models...")
        self.runtime = LiveRuntime(components or build_components(rc, Path(self.options["run_dir"])),
                                   goto_object_approach=True)
        self.runtime.c.exploration.selector.debug = self.options["debug"]
        if self.debug_recorder is not None:
            self._debug_event("session_start", schema_version=1, options=self.options,
                              sensor_config=cfg.model_dump(), run_config=rc,
                              exploration_config=self.runtime.c.exploration.cfg.model_dump(),
                              mapping_config=self.runtime.c.mapping.model_dump(),
                              navigation_config=self.runtime.c.exploration.planner.cfg.model_dump(),
                              use_sim_time=self.get_parameter("use_sim_time").value)
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
        self.navigation_action = (NavigationAction(
            self, self.tf_buffer, name=self.options["nav_action_name"],
            frame=self.options["nav_action_frame"],
            waypoint_name=self.options["nav_waypoint_name"],
            behavior_tree=self.options["nav_behavior_tree"],
            debug=self._debug_action if self.debug_recorder is not None else None)
            if self.options["nav_action_enabled"] else None)
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

    def _debug_event(self, event, **data):
        if self.debug_recorder is not None:
            self.debug_recorder.emit(event, ros_time=self.get_clock().now().nanoseconds * 1e-9, **data)

    def _debug_action(self, event, **data):
        self._debug_event(event, publish_cycle_id=self._published_cycle_id, **data)
        if event in ("action_goal_desired", "action_goal_cleared"):
            reason = data.get("reasons", data.get("reason"))
            self.get_logger().info(f"[vlnav debug] {event}: reason={reason}, "
                                   f"serial={data.get('serial')}, key={data.get('key')}")

    def _debug_pipeline(self, cycle_id, frame, frame_valid, result, robot, out, status, timing):
        if self.debug_recorder is None:
            return
        try:
            motion = None
            if frame_valid and robot is not None:
                motion = goal_delta(Pose.from_matrix(frame.T_map_base, frame.stamp), robot)
                motion["time_delta_s"] = robot.stamp - frame.stamp
            perception = None
            if result is not None:
                perception = {"detections": len(result.detections), "observations": len(result.observations),
                              "relevant_detections": len(result.relevant_idx),
                              "objects": [{"label": o.label, "score": o.score, "centroid_map": o.centroid_map}
                                          for o in result.observations[:10]]}
                depth = getattr(self.runtime.c.perception, "last_depth", None)
                if depth is not None:
                    perception["depth_pixels"] = int((depth > 0).sum())
                    perception["depth_coverage"] = perception["depth_pixels"] / max(1, depth.size)
            self._debug_event("pipeline", cycle_id=cycle_id, status=status,
                              frame_valid=frame_valid, frame_stamp=frame.stamp if frame is not None else None,
                              sync=self.source.debug_snapshot(), tf_errors=list(self._tf_debug),
                              robot=pose_data(robot, self.map_frame), robot_tf=dict(self._pose_debug),
                              motion_since_image=motion, perception=perception,
                              goal={"kind": out.goal_kind, "id": out.goal_id,
                                    "pose": pose_data(out.goal, self.map_frame)},
                              goal_geometry=goal_geometry(out, robot), path=path_data(out.path),
                              selection=selection_data(self.runtime.c.exploration), timing_ms=timing)
        except Exception as exc:
            self._debug_event("diagnostic_error", cycle_id=cycle_id, error=f"{type(exc).__name__}: {exc}")

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
            if self.get_parameter("use_sim_time").value:
                return "image timestamp is too old for ROS time; for bag replay use --clock"
            return "image timestamp is too old for ROS time; check sensor clock synchronization or max_frame_age_s"
        if age < 0:
            return "image timestamp is ahead of ROS time; check /clock and sensor timestamps"
        return ""

    def _instruction(self, msg):
        with self._lock:
            self._pending_instruction = msg.data
            self._output = hold()  # immediately invalidate the old goal during preemption
            self._output_reason = "new_instruction"
        self._debug_event("instruction_received", instruction=msg.data)

    def _lookup(self, frame_id, stamp):
        try:
            tr = self.tf_buffer.lookup_transform(self.map_frame, frame_id,
                                                 Time(nanoseconds=round(stamp * 1e9)))
            return transform_to_matrix(tr.transform)
        except TransformException as exc:
            if self.debug_recorder is not None and len(self._tf_debug) < 16:
                self._tf_debug.append({"target_frame": self.map_frame, "source_frame": frame_id,
                                       "requested_stamp": stamp, "error": str(exc)})
            return None

    def _robot(self, now):
        try:
            tr = self.tf_buffer.lookup_transform(self.map_frame, self.sensor_cfg.poses.base_frame, Time())
            if self.debug_recorder is not None:
                self._pose_debug = {"stamp": stamp_s(tr.header), "age_s": now - stamp_s(tr.header),
                                    "source_frame": self.sensor_cfg.poses.base_frame,
                                    "target_frame": self.map_frame,
                                    "T_map_base": transform_to_matrix(tr.transform).tolist()}
            if abs(now - stamp_s(tr.header)) > self.options["max_pose_age_s"]:
                if self.debug_recorder is not None:
                    self._pose_debug["reason"] = "pose_too_old_or_future"
                return None
            return Pose.from_matrix(transform_to_matrix(tr.transform), stamp_s(tr.header))
        except TransformException as exc:
            if self.debug_recorder is not None:
                self._pose_debug = {"reason": "pose_tf_missing", "error": str(exc)}
            return None

    def _run(self):
        last_t = None
        last_clock_issue = ""
        processed_frames = 0
        cycle_id = 0
        try:
            while not self._stop.is_set():
                cycle_started = time.monotonic()
                self._tf_debug = []
                ingest_ms = 0.0
                now = self.get_clock().now().nanoseconds * 1e-9
                if last_t is not None and now < last_t - 0.01:
                    raise RuntimeError("ROS time moved backwards; restart the node for a new bag playback")
                last_t = now
                with self._lock:
                    image_stamp = self._latest_image_stamp
                clock_issue = self._clock_issue(now, image_stamp)
                frame = self.source.next_frame(self._lookup, time.monotonic()) if not clock_issue else None
                sync_ms = (time.monotonic() - cycle_started) * 1000
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
                    ingest_started = time.monotonic()
                    result = self.runtime.ingest(frame)
                    ingest_ms = (time.monotonic() - ingest_started) * 1000
                    processed_frames += 1
                    map_msg = occupancy_message(self.runtime.c.grid, header(self.map_frame, frame.stamp))
                    with self._lock:
                        self._map_msg = map_msg
                if self._stop.is_set():
                    break
                now = self.get_clock().now().nanoseconds * 1e-9
                fresh = (self.runtime.last_frame_stamp is not None and
                         0 <= now - self.runtime.last_frame_stamp <= self.options["max_frame_age_s"])
                self._pose_debug = {"reason": "no_fresh_frame"}
                robot = self._robot(now) if fresh else None
                step_started = time.monotonic()
                out = self.runtime.step(robot, now)
                step_ms = (time.monotonic() - step_started) * 1000
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
                    self._output_seq = cycle_id
                    self._output_reason = ("instruction_pending" if self._pending_instruction is not None else
                                           "no_fresh_frame" if not fresh else
                                           "robot_pose_unavailable" if robot is None else
                                           "planner_or_exploration_hold" if out.goal is None else
                                           "no_path" if not out.path else "valid_goal")
                    self._frontier_points = frontier_points
                    self._robot_pose = robot
                    self._reports.extend(r.model_dump_json() for r in reports)
                    self._status = {"state": state.fsm, "task_id": state.task_id,
                                    "subtask_id": st.id if st is not None else None,
                                    "instruction": state.instruction, "done_reason": state.done_reason,
                                    "input": clock_issue or self.source.wait_reason,
                                    "use_sim_time": self.get_parameter("use_sim_time").value,
                                    "ros_time": now, "latest_image_stamp": image_stamp,
                                    "image_age_s": None if image_stamp is None else now - image_stamp,
                                    "processed_frames": processed_frames,
                                    "frontier_points": len(frontier_points), "frame_fresh": fresh,
                                    "pose_fresh": robot is not None}
                    debug_status = dict(self._status) if self.debug_recorder is not None else None
                self._debug_pipeline(cycle_id, frame, frame_valid, result, robot, out, debug_status,
                                     {"sync": sync_ms, "ingest": ingest_ms, "step": step_ms,
                                      "cycle_before_debug": (time.monotonic() - cycle_started) * 1000,
                                      "sleep_period_s": self.period})
                cycle_id += 1
                self._stop.wait(self.period)
        except Exception as exc:
            self.get_logger().error(f"Pipeline stopped: {type(exc).__name__}: {exc}")
            self._debug_event("pipeline_error", error=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._output = hold()
                self._robot_pose = None
                self._status = {"state": "ERROR", "error": str(exc)}
                self._output_reason = "pipeline_error"
        finally:
            self.runtime.close()

    def _publish_navigation(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            out = self._output if 0 <= now - self._output_t <= self.options["goal_timeout_s"] else hold()
            output_age = now - self._output_t
            output_seq, output_reason = self._output_seq, self._output_reason
            status = dict(self._status)
            frontier_points = self._frontier_points
            robot = self._robot_pose
            reports, self._reports = self._reports, []
        if robot is not None and not (0 <= now - robot.stamp <= self.options["max_pose_age_s"]):
            robot = None
        goal, point, path, markers = navigation_messages(out, header(self.map_frame, now),
                                                        self.options["goal_timeout_s"])
        if output_seq is not None and not (0 <= output_age <= self.options["goal_timeout_s"]):
            output_reason = "worker_output_timeout" if output_age > 0 else "output_time_in_future"
        elif goal is None and out.goal is not None and not out.path:
            output_reason = "no_path"
        self._published_cycle_id = output_seq
        if self.debug_recorder is not None:
            self._debug_event("navigation_publish", cycle_id=output_seq, output_age_s=output_age,
                              goal_active=goal is not None, reason=output_reason,
                              task_id=status.get("task_id"), subtask_id=status.get("subtask_id"),
                              goal_kind=out.goal_kind, goal_id=out.goal_id,
                              goal=pose_data(goal), robot=pose_data(robot, self.map_frame))
            if output_reason != self._last_publish_reason:
                self.get_logger().info(f"[vlnav debug] navigation: {output_reason}; output_age_s={output_age:.3f}")
                self._last_publish_reason = output_reason
        self.active_pub.publish(Bool(data=goal is not None))
        if goal is not None:
            self.goal_pub.publish(goal)
            self.point_pub.publish(point)
        if self.navigation_action is not None:
            key = (status.get("task_id"), status.get("subtask_id"), out.goal_kind, out.goal_id)
            self.navigation_action.update(goal, key, clear_reason=output_reason)
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
            self._output_reason = "node_stopping"
            self._frontier_points = []
            self._robot_pose = None
        if rclpy.ok():
            self._publish_navigation()
        self._worker.join(timeout=5.0)
        if self.debug_recorder is not None:
            self._debug_event("session_end", worker_alive=self._worker.is_alive())
            self.debug_recorder.close()


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
