import math

import numpy as np
import pytest

pytest.importorskip("rclpy")
from rclpy.serialization import deserialize_message, serialize_message
from visualization_msgs.msg import Marker, MarkerArray

from vlnav.common.models import Pose
from vlnav.exploration.explorer import ExplorationOutput
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.runtime.live_runtime import hold
from vlnav_ros.messages import (frontier_messages, header, move_messages, navigation_messages,
                                occupancy_message, robot_messages)


def test_ros_map_encoding_origin_and_row_order():
    grid = OccupancyGrid2D.from_array(np.array([[-1, 0, 1], [1, 0, -1]], np.int8), .2, (-3., 4.))
    msg = occupancy_message(grid, header("odom_lidar", 12.25))
    msg = deserialize_message(serialize_message(msg), type(msg))
    assert list(msg.data) == [-1, 0, 100, 100, 0, -1]
    assert (msg.info.width, msg.info.height) == (3, 2)
    assert (msg.info.origin.position.x, msg.info.origin.position.y) == (-3., 4.)
    assert msg.info.origin.orientation.w == 1.
    assert msg.header.stamp.nanosec == 250000000


@pytest.mark.parametrize("goal_z", [-2., .1, 3.])
def test_goal_pose_path_markers_and_clear(goal_z):
    pose = Pose(position=(2., 3., goal_z), orientation_xyzw=(0., 0., math.sqrt(.5), math.sqrt(.5)), stamp=0.)
    out = ExplorationOutput(goal=pose, goal_kind="goto", goal_id=7, path=[(0., 0.), (2., 3.)], events=[])
    h = header("odom_lidar", 1.)
    goal, point, path, markers = navigation_messages(out, h, 2.)
    assert goal.pose.position == point.point
    assert goal.pose.orientation.z == math.sqrt(.5)
    assert path.poses[-1].pose == goal.pose
    markers = deserialize_message(serialize_message(markers), MarkerArray)
    assert [m.type for m in markers.markers[1:]] == [Marker.SPHERE, Marker.ARROW, Marker.TEXT_VIEW_FACING]
    assert markers.markers[-1].text == "GOAL: goto #7"
    assert markers.markers[1].lifetime.sec == 2
    # Visualization offsets must not mutate the actual goal or path position.
    assert goal.pose.position.z == point.point.z == path.poses[-1].pose.position.z == goal_z
    assert all(m.pose.position.z - m.scale.z / 2 > 0 for m in markers.markers[1:])
    arrow = markers.markers[2]
    assert arrow.pose.orientation.z == pytest.approx(math.sqrt(.5))
    assert arrow.pose.orientation.w == pytest.approx(math.sqrt(.5))
    for invalid in (hold(), out.model_copy(update={"path": None})):
        g, p, path, markers = navigation_messages(invalid, h, 2.)
        assert g is p is None and path.poses == []
        assert [m.action for m in markers.markers] == [Marker.DELETEALL]


def test_frontier_overlay_world_coordinates_and_replacement():
    points = [{"xy": (-3., 4.), "blacklisted": False}, {"xy": (2., 1.), "blacklisted": True}]
    hdr = header("odom_lidar", 12.)
    markers = deserialize_message(serialize_message(frontier_messages(points, hdr)), MarkerArray)
    assert markers.markers[0].action == Marker.DELETEALL
    dots = markers.markers[1]
    assert dots.type == Marker.SPHERE_LIST and dots.header.frame_id == "odom_lidar"
    assert [(p.x, p.y) for p in dots.points] == [(-3., 4.), (2., 1.)]
    assert all(p.z - dots.scale.z / 2 > 0 for p in dots.points)
    assert dots.colors[0].b > dots.colors[0].r
    assert dots.colors[1].r == dots.colors[1].g == dots.colors[1].b
    assert all(c.a == 1 for c in dots.colors)
    assert [m.action for m in frontier_messages([], hdr).markers] == [Marker.DELETEALL]


def test_robot_pose_marker_has_heading_and_clears_when_pose_missing():
    hdr = header("odom_lidar", 12.)
    pose = Pose(position=(2., -3., -1.),
                orientation_xyzw=(0., 0., math.sqrt(.5), math.sqrt(.5)), stamp=12.)
    markers = deserialize_message(serialize_message(robot_messages(pose, hdr, 2.)), MarkerArray)
    assert [m.action for m in markers.markers] == [Marker.DELETEALL, Marker.ADD, Marker.ADD]
    arrow, label = markers.markers[1:]
    assert arrow.type == Marker.ARROW and label.type == Marker.TEXT_VIEW_FACING
    assert arrow.header.frame_id == "odom_lidar"
    assert (arrow.pose.position.x, arrow.pose.position.y) == (2., -3.)
    assert arrow.pose.position.z > 0
    assert arrow.pose.orientation.z == pytest.approx(math.sqrt(.5))
    assert arrow.pose.orientation.w == pytest.approx(math.sqrt(.5))
    assert label.text == "ROBOT" and label.pose.position.z > 0
    assert arrow.lifetime.sec == label.lifetime.sec == 2
    assert [m.action for m in robot_messages(None, hdr, 2.).markers] == [Marker.DELETEALL]


def test_move_markers_show_ray_point_and_text_then_clear():
    snap = {"active": True, "last": {"k": 3, "decision": "new_goal", "reason": None,
                               "exchanges": [{"look_down": False, "text": "↓"}, {"look_down": True, "text": "320 400"}],
                               "projection": {"uv": [200.0, 150.0], "cam_xy": [-1.0, -1.0], "p_map": [0.5, 0.5, 0.0],
                                              "floor_xy": [0.5, 0.5]}}}
    msg = deserialize_message(serialize_message(move_messages(snap, header("map", 1.))), MarkerArray)
    clear, ray, point, text = msg.markers
    assert clear.action == Marker.DELETEALL and ray.type == Marker.LINE_STRIP
    assert [(p.x, p.y) for p in ray.points] == [(-1., -1.), (.5, .5)]
    assert (point.pose.position.x, point.pose.position.y) == (.5, .5)
    assert text.text == "S2 #3: ↓ | 320 400"
    for gone in (None, {**snap, "active": False}, {"active": True, "last": None}):
        assert [m.action for m in move_messages(gone, header("map", 2.)).markers] == [Marker.DELETEALL]
