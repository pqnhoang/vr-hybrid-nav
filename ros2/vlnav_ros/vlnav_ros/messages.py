"""Map and navigation output conversions, without any inference or node state."""
from array import array
import math

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


def header(frame_id: str, t: float) -> Header:
    sec, ns = divmod(round(t * 1_000_000_000), 1_000_000_000)
    return Header(frame_id=frame_id, stamp=Time(sec=sec, nanosec=ns))


def pose_message(pose, hdr: Header) -> PoseStamped:
    msg = PoseStamped(header=hdr)
    msg.pose.position = Point(x=float(pose.position[0]), y=float(pose.position[1]), z=float(pose.position[2]))
    x, y, z, w = pose.orientation_xyzw
    msg.pose.orientation = Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
    return msg


def occupancy_message(grid, hdr: Header) -> OccupancyGrid:
    msg = OccupancyGrid(header=hdr)
    msg.info.map_load_time = hdr.stamp
    msg.info.resolution = grid.resolution
    msg.info.height, msg.info.width = grid.shape
    msg.info.origin.position.x, msg.info.origin.position.y = grid.origin_xy
    msg.info.origin.orientation.w = 1.0
    # ROS is row-major [y, x], starting at the LOWER left: do not flip like a PNG.
    cells = grid.as_array()
    msg.data = array("b", np.where(cells == 1, 100, cells).astype(np.int8).tobytes())
    return msg


def frontier_messages(points, hdr: Header) -> MarkerArray:
    """Map overlays in world XY; replace the whole set when frontiers change."""
    markers = MarkerArray(markers=[Marker(header=hdr, action=Marker.DELETEALL)])
    if not points:
        return markers
    m = Marker(header=hdr, ns="frontier_points", id=0, type=Marker.SPHERE_LIST, action=Marker.ADD)
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = 0.25, 0.25, 0.08
    m.color.a = 1.0
    for p in points:
        x, y = p["xy"]
        m.points.append(Point(x=float(x), y=float(y), z=0.08))
        m.colors.append(ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0) if p["blacklisted"] else
                        ColorRGBA(r=0.05, g=0.85, b=1.0, a=1.0))
    # Like the occupancy map, frontiers persist until replaced, even with no active task.
    markers.markers.append(m)
    return markers


def robot_messages(robot, hdr: Header, lifetime_s: float) -> MarkerArray:
    """Show the current robot position and heading, or clear a stale pose."""
    markers = MarkerArray(markers=[Marker(header=hdr, action=Marker.DELETEALL)])
    if robot is None:
        return markers
    x, y = robot.position[:2]
    yaw = robot.yaw()
    lifetime_ns = round(lifetime_s * 1_000_000_000)
    for marker_id, kind in enumerate((Marker.ARROW, Marker.TEXT_VIEW_FACING)):
        marker = Marker(header=hdr, ns="robot_pose", id=marker_id, type=kind, action=Marker.ADD)
        marker.pose.position = Point(x=float(x), y=float(y), z=.25 if kind == Marker.ARROW else .8)
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = .05, .55, 1.0, 1.0
        marker.lifetime.sec, marker.lifetime.nanosec = divmod(lifetime_ns, 1_000_000_000)
        if kind == Marker.ARROW:
            marker.pose.orientation.z = math.sin(yaw / 2)
            marker.pose.orientation.w = math.cos(yaw / 2)
            marker.scale.x, marker.scale.y, marker.scale.z = .9, .25, .16
        else:
            marker.scale.z = .3
            marker.text = "ROBOT"
        markers.markers.append(marker)
    return markers


def navigation_messages(out, hdr: Header, lifetime_s: float):
    """None goal + empty path + DELETEALL explicitly clears a previous selection."""
    path = Path(header=hdr)
    clear = Marker(header=hdr, action=Marker.DELETEALL)
    markers = MarkerArray(markers=[clear])
    if out.goal is None or out.goal_kind == "hold" or not out.path:
        return None, None, path, markers
    goal = pose_message(out.goal, hdr)
    point = PointStamped(header=hdr, point=goal.pose.position)
    for i, (x, y) in enumerate(out.path):
        p = PoseStamped(header=hdr)
        p.pose.position = Point(x=float(x), y=float(y), z=goal.pose.position.z)
        if i + 1 < len(out.path):
            nx, ny = out.path[i + 1]
            yaw = math.atan2(ny - y, nx - x)
            p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        else:
            p.pose.orientation = goal.pose.orientation
        path.poses.append(p)
    for i, kind in enumerate((Marker.SPHERE, Marker.ARROW, Marker.TEXT_VIEW_FACING)):
        m = Marker(header=hdr, ns="selected_goal", id=i, type=kind, action=Marker.ADD)
        m.pose = pose_message(out.goal, hdr).pose
        # Project the visualization onto the z=0 occupancy plane. A recorded goal's
        # actual z (or tilt) can put it underneath the map in RViz's top-down view.
        m.pose.position.z = 0.12
        m.pose.orientation = Quaternion(w=1.0)
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.65, 0.05, 1.0
        ns = round(lifetime_s * 1_000_000_000)
        m.lifetime.sec, m.lifetime.nanosec = divmod(ns, 1_000_000_000)
        if kind == Marker.SPHERE:
            m.scale.x, m.scale.y, m.scale.z = 0.6, 0.6, 0.12
        elif kind == Marker.ARROW:
            m.scale.x, m.scale.y, m.scale.z = 0.9, 0.12, 0.18
            x, y, z, w = out.goal.orientation_xyzw
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.pose.position.z = 0.25
        else:
            m.scale.z = 0.3
            m.pose.position.z = 0.8
            m.text = f"GOAL: {out.goal_kind} #{out.goal_id}"
        markers.markers.append(m)
    return goal, point, path, markers
