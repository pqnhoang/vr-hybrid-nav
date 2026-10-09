"""S3: occupancy grid from synthetic scans (SYSTEM.md §9 acceptance) + ground models, growth,
ExternalGrid, and rooms segmented by the memory from a LiDAR-built grid."""
import sys
from pathlib import Path

import numpy as np
import pytest

from vlnav.mapping.config import load_mapping_config
from vlnav.mapping.occupancy import FREE, NO_CHANGE, OCCUPIED, UNKNOWN, ExternalGrid, OccupancyGrid2D

STEP = 0.05


def cfg(**over):
    return load_mapping_config(overrides=over)


def floor_pts(x0, y0, x1, y1, z=0.0, slope=0.0):
    xs, ys = np.meshgrid(np.arange(x0 + STEP / 2, x1, STEP), np.arange(y0 + STEP / 2, y1, STEP))
    xs, ys = xs.ravel(), ys.ravel()
    return np.stack([xs, ys, z + slope * xs], axis=1)


def wall_pts(p0, p1, z_lo=0.2, z_hi=2.0, gap=None):
    """Vertical wall from p0 to p1 (xy); `gap` = (a, b) fractions left open (a door)."""
    n = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / STEP) + 1
    f = np.linspace(0, 1, n)
    if gap is not None:
        f = f[(f < gap[0]) | (f > gap[1])]
    xy = np.asarray(p0) + f[:, None] * (np.asarray(p1) - np.asarray(p0))
    zs = np.arange(z_lo, z_hi + 1e-9, 0.2)
    return np.concatenate([np.column_stack([xy, np.full(len(xy), z)]) for z in zs])


def box_room(x0, y0, x1, y1, door=None):
    """Floor + 4 walls of a rectangular room; door=(wall index, (a, b)) leaves an opening."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = [wall_pts(corners[i], corners[(i + 1) % 4], gap=door[1] if door and door[0] == i else None)
             for i in range(4)]
    return np.concatenate([floor_pts(x0, y0, x1, y1), *walls])


def val(grid, xy):
    return int(grid.value(grid.world_to_cell(np.asarray(xy, float)[None]))[0])


def scan_room(n=2, **over):
    g = OccupancyGrid2D.from_config(cfg(**over))
    pts = box_room(-3, -2, 3, 2)
    infos = [g.update(pts, (0.0, 0.0, 1.2), float(t)) for t in range(n)]
    return g, pts, infos


def test_synthetic_room_free_occupied_unknown():
    g, _, _ = scan_room(n=2)
    assert val(g, (0.0, 0.0)) == FREE and val(g, (2.5, -1.5)) == FREE
    assert val(g, (3.0, 0.0)) == OCCUPIED and val(g, (0.0, -2.0)) == OCCUPIED
    assert val(g, (4.0, 0.0)) == UNKNOWN and val(g, (0.0, 3.5)) == UNKNOWN
    after_one, _, _ = scan_room(n=1)
    assert val(after_one, (0.0, 0.0)) == UNKNOWN          # one miss is not enough evidence


def test_ray_casting_clears_a_transient_obstacle():
    g = OccupancyGrid2D.from_config(cfg())
    room = box_room(-3, -2, 3, 2)
    person = wall_pts((1.0, -0.1), (1.0, 0.1), z_lo=0.2, z_hi=1.6)
    g.update(np.concatenate([room, person]), (0.0, 0.0, 1.2), 0.0)
    assert val(g, (1.0, 0.0)) == OCCUPIED
    states = [val(g, (1.0, 0.0)) for t in range(1, 5) if g.update(room, (0.0, 0.0, 1.2), float(t))]
    assert states[0] == UNKNOWN and states[-1] == FREE     # 0.85 -> 0.45 ... -> -0.75


def test_changed_bbox_covers_every_changed_cell():
    g = OccupancyGrid2D.from_config(cfg())
    room = box_room(-3, -2, 3, 2)
    g.update(room, (0.0, 0.0, 1.2), 0.0)
    for t, pts in ((1.0, room), (2.0, np.concatenate([room, wall_pts((1, -0.5), (1, 0.5))]))):
        before = g.as_array().copy()
        info = g.update(pts, (0.0, 0.0, 1.2), t)
        iy, ix = np.nonzero(g.as_array() != before)
        assert len(ix) > 0
        x0, y0, x1, y1 = info.changed_bbox_cells
        assert ix.min() >= x0 and ix.max() <= x1 and iy.min() >= y0 and iy.max() <= y1
    v = g.version
    info = g.update(np.empty((0, 3)), (0.0, 0.0, 1.2), 3.0)
    assert info.changed_bbox_cells == NO_CHANGE and info.version == v + 1 == g.version


def test_grid_grows_and_keeps_content_aligned():
    g = OccupancyGrid2D.from_config(cfg(initial_size_m=8.0, grow_margin_m=2.0))
    g.update(box_room(-3, -2, 3, 2), (0, 0, 1.2), 0.0)
    g.update(box_room(-3, -2, 3, 2), (0, 0, 1.2), 1.0)
    shape0, origin0 = g.shape, g.origin_xy
    assert val(g, (3.0, 0.0)) == OCCUPIED
    far = box_room(6, -1, 9, 1)
    g.update(far, (7.5, 0.0, 1.2), 2.0)
    g.update(far, (7.5, 0.0, 1.2), 3.0)
    assert g.shape[1] > shape0[1] and val(g, (3.0, 0.0)) == OCCUPIED and val(g, (7.5, 0.0)) == FREE
    shift = np.subtract(origin0, g.origin_xy) / g.resolution
    assert np.allclose(shift, np.round(shift))             # cells stay on the same lattice


def test_cell_min_ground_follows_slope_and_hidden_floor():
    slope = 0.1                                            # 10 % ramp: floor z up to 0.3 m
    floor = floor_pts(-3, -2, 3, 2, slope=slope)
    floor = floor[~((np.abs(floor[:, 0] - 2.0) < 0.25) & (np.abs(floor[:, 1]) < 0.25))]   # under table
    table_top = floor_pts(1.8, -0.2, 2.2, 0.2, z=0.75 + slope * 2.0)
    box = wall_pts((-1.0, 1.0), (-1.0, 1.4), z_lo=-0.1 + 0.2, z_hi=-0.1 + 0.5)            # 0.3-0.4 m tall
    pts = np.concatenate([floor, table_top, box])
    g = OccupancyGrid2D.from_config(cfg())
    for t in range(2):
        g.update(pts, (0, 0, 1.2), float(t))
    assert val(g, (2.9, -1.5)) == FREE                     # z 0.29 on the ramp is still floor
    assert val(g, (2.0, 0.0)) == OCCUPIED                  # table top over hidden floor
    assert val(g, (-1.0, 1.2)) == OCCUPIED
    gc = OccupancyGrid2D.from_config(cfg(ground={"mode": "constant", "constant_z": 0.0}))
    for t in range(2):
        gc.update(pts, (0, 0, 1.2), float(t))
    assert val(gc, (2.9, -1.5)) == OCCUPIED                # constant floor misreads the ramp
    # min-smoothing returns the floor `smooth_radius_m` (0.5 m) down-slope: lags by <= 0.05 m
    assert slope * (2.875 - 0.5) - 0.01 <= g.ground_height((2.9, -1.5)) <= slope * 2.875


def test_overhead_points_are_ignored():
    g = OccupancyGrid2D.from_config(cfg())
    ceiling = floor_pts(-2, -2, 2, 2, z=2.6)
    g.update(np.concatenate([ceiling, floor_pts(-0.5, -0.5, 0.5, 0.5)]), (0, 0, 1.2), 0.0)
    assert (g.as_array() == OCCUPIED).sum() == 0
    assert val(g, (1.5, 1.5)) == UNKNOWN                   # no ray to ceiling points


def test_config_validation_and_static_grids_stay_compatible():
    with pytest.raises(ValueError):
        cfg(negative_obstacles=True)
    with pytest.raises(ValueError):
        OccupancyGrid2D(0.1, (0, 0), (5, 5)).update(np.zeros((1, 3)), (0, 0, 0), 0.0)
    arr = np.full((4, 5), FREE, np.int8)
    arr[1, 2] = OCCUPIED
    g = OccupancyGrid2D.from_array(arr, 0.1, (0.0, 0.0))
    assert g.value([[2, 1], [9, 9]]).tolist() == [OCCUPIED, UNKNOWN] and g.inflated(0.1).sum() > 1


def test_external_grid():
    c = cfg()
    g = ExternalGrid(c)
    data = np.array([-1, 0, 10, 30, 70, 100], dtype=np.int8)
    info = g.set_from_occupancy(data, 3, 2, 0.1, (1.0, 2.0), 0.0)
    assert g.as_array().tolist() == [[UNKNOWN, FREE, FREE], [UNKNOWN, OCCUPIED, OCCUPIED]]
    assert info.changed_bbox_cells == (0, 0, 2, 1) and g.origin_xy == (1.0, 2.0)
    data[3] = 90
    assert g.set_from_occupancy(data, 3, 2, 0.1, (1.0, 2.0), 1.0).changed_bbox_cells == (0, 1, 0, 1)
    with pytest.raises(TypeError):
        g.update(np.zeros((1, 3)), (0, 0, 0), 0.0)


def test_memory_segments_rooms_from_the_lidar_grid(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))
    from memory_helpers import QUIET, make_memory, make_obs, make_result
    from vlnav.common.config import deep_merge
    g = OccupancyGrid2D.from_config(cfg())
    door = (1, (0.375, 0.6))                               # 0.9 m door in the shared wall (x = 4)
    room_a = box_room(0, 0, 4, 4, door=door)
    room_b = box_room(4.05, 0, 8.05, 4, door=(3, (0.4, 0.625)))   # same opening in B's west wall
    in_door = floor_pts(3.95, 1.5, 4.1, 2.4)
    for t in range(3):
        g.update(np.concatenate([room_a, in_door]), (2.0, 2.0, 1.2), float(t))
        g.update(np.concatenate([room_b, in_door]), (6.0, 2.0, 1.2), float(t) + 0.5)
    mem = make_memory(tmp_path, deep_merge(QUIET, {"rooms": {"enabled": True}}), grid=g)
    mem.update(make_result(0.0, [make_obs("bed", 0.8, (1.0, 1.0, 0.5), 0.0, base_xy=(2.0, 2.0)),
                                 make_obs("desk", 0.8, (6.5, 3.0, 0.7), 0.0, bbox=(90, 30, 150, 90),
                                          base_xy=(2.0, 2.0))], base_xy=(2.0, 2.0)))
    rooms = mem.rooms()
    assert len(rooms) == 2 and rooms[0].adjacent_room_ids == [rooms[1].id]
    bed, desk = mem.all_objects()
    assert bed.room_id is not None and desk.room_id is not None and bed.room_id != desk.room_id
    assert all(r.area_m2 > 12 for r in rooms)
