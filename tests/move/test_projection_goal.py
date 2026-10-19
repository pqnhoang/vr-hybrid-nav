"""Move M1: pixel -> map projection (§7) and reachable goals (§8) on a synthetic camera and grid."""
import math

import numpy as np
from move_helpers import H, K, W, cam_T, cfg, free_grid, planner, project

from vlnav.mapping.occupancy import OCCUPIED, UNKNOWN, OccupancyGrid2D
from vlnav.move.goal import make_reachable
from vlnav.move.projection import dense_depth_at, floor_fn, pixel_to_map, sparse_depth_at, to_camera_pixel

PC = cfg().projection


def flat(z=0.0):
    return lambda xy: z


def test_round_trip_floor_point_under_one_cm():
    for yaw in (0.0, 0.7, -2.0):
        T = cam_T(1.0, -2.0, yaw)
        for fwd, side in ((2.0, 0.0), (3.0, 0.8), (4.5, -1.0)):
            p = np.array([1.0 + fwd * math.cos(yaw) - side * math.sin(yaw),
                          -2.0 + fwd * math.sin(yaw) + side * math.cos(yaw), 0.0])
            uv = project(T, p)
            pr = pixel_to_map(uv, K, T, (W, H), (1.0, -2.0), flat(), None, PC)
            assert pr.ok and pr.flags == ["no_depth"], pr
            assert np.linalg.norm(np.asarray(pr.p_map) - p) < 0.01


def test_floor_refined_with_ground_height_at_the_hit():
    T = cam_T()
    p = np.array([3.0, 0.0, 0.2])                                # raised floor (ramp) at the goal
    uv = project(T, p)
    pr = pixel_to_map(uv, K, T, (W, H), (0.0, 0.0), lambda xy: 0.0 if np.linalg.norm(xy) < 0.5 else 0.2, None, PC)
    assert pr.ok and np.linalg.norm(np.asarray(pr.p_map) - p) < 0.15    # one refinement step


def test_floor_ignores_grid_ground_next_to_the_robot():
    def ground(xy):                                              # blind zone: body points at base height
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.where(np.linalg.norm(xy, axis=1) < 0.6, 1.0, np.where(xy[:, 0] > 4.0, 0.3, 0.0))

    def const(z):
        return lambda xy: np.full(len(np.asarray(xy).reshape(-1, 2)), z)
    ring = cfg(projection={"base_height_m": None}).projection
    f = floor_fn(ground, (0.0, 0.0, 1.0), ring)
    assert f((0.0, 0.0)) == 0.0 and f((0.5, 0.0)) == 0.0         # ring median, not the blind zone
    assert f((4.5, 0.0)) == 0.3                                  # ramp within max_floor_step_m
    f = floor_fn(const(2.0), (0.0, 0.0, 1.0), PC)                # base_height_m = 0 in the test config
    assert f((0.0, 0.0)) == 1.0 and f((3.0, 0.0)) == 1.0         # a 1 m step is rejected
    assert math.isnan(floor_fn(const(float("nan")), (0, 0, 1), ring)((2.0, 0.0)))
    T = cam_T()
    uv = project(T, (2.0, 0.0, 0.0))
    pr = pixel_to_map(uv, K, T, (W, H), (0, 0), floor_fn(ground, (0.0, 0.0, 1.0), ring), None, PC)
    assert pr.ok and np.linalg.norm(np.asarray(pr.p_map[:2]) - (2.0, 0.0)) < 0.01


def test_rejects_out_of_image_unknown_floor_and_too_close():
    T = cam_T()
    assert pixel_to_map((700, 100), K, T, (W, H), (0, 0), flat(), None, PC).reason == "out_of_image"
    assert pixel_to_map((320, 400), K, T, (W, H), (0, 0), flat(float("nan")), None, PC).reason == "no_floor"
    pr = pixel_to_map((320, 479), K, T, (W, H), (0.9, 0.0), flat(), None, PC)   # hit at x = 0.97 m
    assert pr.reason == "too_close" and not pr.ok and pr.p_map is not None


def test_above_horizon_uses_depth_or_direction_only():
    T = cam_T(pitch_down_deg=15.0)
    uv = (320.0, 60.0)                                           # above the horizon at 15 deg pitch
    pr = pixel_to_map(uv, K, T, (W, H), (0, 0), flat(), None, PC)
    assert pr.ok and pr.flags == ["direction_only"] and abs(np.linalg.norm(pr.p_map[:2]) - 2.0) < 1e-6
    pr = pixel_to_map(uv, K, T, (W, H), (0, 0), flat(), 3.0, PC)
    assert pr.ok and pr.flags == ["depth_point"] and pr.p_map[2] == 0.0


def test_depth_clips_floor_point_behind_a_wall_and_far_points_clamp():
    T = cam_T(pitch_down_deg=15.0)
    p = np.array([4.0, 0.0, 0.0])
    uv = project(T, p)
    s_floor = ((p - T[:3, 3]) @ T[:3, :3])[2]
    s_wall = s_floor * 2.0 / 4.0                                 # a wall at x = 2 m on the same ray
    pr = pixel_to_map(uv, K, T, (W, H), (0, 0), flat(), s_wall, PC)
    assert pr.ok and "clipped_by_depth" in pr.flags
    r_wall = pr.r_meas
    assert abs(r_wall - pr.r_floor / 2) < 1e-6
    hit = np.linalg.norm(np.asarray(pr.p_map[:2]))
    assert abs(hit - 2.0 * (r_wall - PC.obstacle_backoff_m) / r_wall) < 1e-6   # 0.4 m before the wall along the ray
    far = pixel_to_map(project(T, (9.0, 0.0, 0.0)), K, T, (W, H), (0, 0), flat(), None, PC)
    assert far.ok and "clamped" in far.flags and abs(np.linalg.norm(far.p_map[:2]) - PC.max_range_m) < 1e-6


def test_depth_readers_dense_and_sparse():
    d = np.zeros((H, W), np.float32)
    d[200:210, 300:310] = 2.0
    assert dense_depth_at(d, (305, 205), 7, 0.3, 12.0) == 2.0
    assert dense_depth_at(d, (100, 100), 7, 0.3, 12.0) is None
    T = cam_T(pitch_down_deg=0.0)
    pts = np.array([[3.0, 0.0, 1.0], [3.0, 0.01, 1.0], [8.0, 3.0, 1.0]])
    assert abs(sparse_depth_at(pts, K, T, (320, 240), 15, 5, 0.3, 12.0) - 3.0) < 1e-6
    assert sparse_depth_at(pts, K, T, (50, 50), 15, 5, 0.3, 12.0) is None
    assert sparse_depth_at(np.empty((0, 3)), K, T, (320, 240), 15, 5, 0.3, 12.0) is None


def test_to_camera_pixel_scaling():
    assert to_camera_pixel((191.5, 191.5), (384, 384), (640, 480)) == (319.5, 239.5)
    assert to_camera_pixel((0, 0), (640, 480), (640, 480)) == (0.0, 0.0)


def test_goal_in_free_cell_and_yaw_toward_goal():
    g = free_grid()
    r = make_reachable(g, planner(), (0, 0), (3.0, 1.0), 0.0, 1.0, 0.2, 0.3)
    assert r.goal is not None and r.cell_state == "free" and r.flags == []
    assert abs(r.goal.yaw() - math.atan2(1.0, 3.0)) < 0.05


def test_goal_in_unknown_behind_frontier_walks_back():
    arr = np.full((200, 200), UNKNOWN, np.int8)
    arr[:, :120] = 0                                             # free up to x = 2 m
    g = OccupancyGrid2D.from_array(arr, 0.1, (-10.0, -10.0))
    r = make_reachable(g, planner(), (0, 0), (5.0, 0.0), 0.0, 1.0, 0.2, 0.3)
    assert r.cell_state == "unknown" and "walked_back" in r.flags and r.goal.position[0] <= 2.0
    assert r.walkback_m > 2.5


def test_goal_inside_inflated_obstacle_snaps_and_fully_blocked_fails():
    g = free_grid()
    arr = g.as_array().copy()
    arr[100, 130] = OCCUPIED                                      # obstacle cell at (3.05, 0.05)
    g.set_array(arr, 1.0)
    r = make_reachable(g, planner(), (0, 0), (3.05, 0.05), 0.0, 1.0, 0.2, 0.3)
    assert r.goal is not None and r.cell_state == "occupied" and "snapped" in r.flags
    arr = np.full((200, 200), OCCUPIED, np.int8)
    arr[95:105, 95:105] = 0                                      # robot boxed in
    g2 = OccupancyGrid2D.from_array(arr, 0.1, (-10.0, -10.0))
    r = make_reachable(g2, planner(goal_snap_radius_m=0.1), (0, 0), (4.0, 0.0), 0.0, 1.0, 0.2, 0.3)
    assert r.goal is None and r.reason == "unreachable"
