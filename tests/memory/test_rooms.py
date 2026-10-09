"""M3: room segmentation on synthetic floor plans + id tracking."""
import numpy as np
import pytest
from memory_helpers import QUIET, RES, floor_plan, make_grid, make_memory, make_obs, make_result

from vlnav.common.config import deep_merge
from vlnav.memory import load_memory_config
from vlnav.memory.room_tracker import RoomTracker
from vlnav.memory.rooms import segment_rooms

CFG = load_memory_config().rooms
SHAPE = (60, 140)

# three 4 m x 4 m rooms in a row, doors of 0.8 / 0.9 / 1.0 m in the shared walls
ROOM_A, ROOM_B, ROOM_C = (2, 2, 42, 42), (43, 2, 83, 42), (84, 2, 124, 42)


def door(x_wall, width_m, y0=15):
    return (x_wall, y0, x_wall + 1, y0 + int(round(width_m / RES)))


def seg(arr, res=RES, cfg=CFG):
    return segment_rooms(arr, res, (0.0, 0.0), cfg)


def region_at(s, x_cell, y_cell):
    return int(s.labels[y_cell, x_cell])


@pytest.mark.parametrize("width_ab,width_bc", [(0.8, 0.9), (0.9, 1.0), (1.0, 0.8)])
def test_rooms_in_a_row_with_doors(width_ab, width_bc):
    arr = floor_plan(SHAPE, [ROOM_A, ROOM_B, ROOM_C], doors=[door(42, width_ab), door(83, width_bc)])
    s = seg(arr)
    assert len(s.region_ids) == 3
    a, b, c = region_at(s, 20, 20), region_at(s, 60, 20), region_at(s, 100, 20)
    assert len({a, b, c}) == 3
    assert s.adjacency[a] == [b] and sorted(s.adjacency[b]) == sorted([a, c]) and s.adjacency[c] == [b]
    assert s.areas_m2[a] == pytest.approx(16.0, abs=0.5)
    assert s.centroids_xy[b] == pytest.approx((6.3, 2.2), abs=0.2)
    assert len(s.polygons[a]) >= 4


def test_l_shaped_room_is_one_room():
    arr = floor_plan(SHAPE, [(2, 2, 42, 22), (2, 22, 22, 52)])
    arr[22, 2:22] = 0                                   # no wall between the two arms
    s = seg(arr)
    assert len(s.region_ids) == 1


def test_open_plan_pair_is_one_room():
    arr = floor_plan(SHAPE, [ROOM_A, ROOM_B], openings=[(42, 8, 43, 33)])     # 2.5 m opening
    assert len(seg(arr).region_ids) == 1


def test_small_closet_is_absorbed():
    arr = floor_plan(SHAPE, [ROOM_A, (43, 2, 55, 17)], doors=[door(42, 0.8, y0=5)])
    s = seg(arr)
    assert len(s.region_ids) == 1
    assert region_at(s, 50, 10) == region_at(s, 20, 20)


def test_downsampled_grid_gives_same_rooms():
    arr = floor_plan(SHAPE, [ROOM_A, ROOM_B, ROOM_C], doors=[door(42, 0.9), door(83, 0.9)])
    fine = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)                    # 0.05 m cells
    s = segment_rooms(fine, RES / 2, (0.0, 0.0), CFG)
    assert len(s.region_ids) == 3
    assert s.labels.shape == fine.shape


def test_ids_stable_under_incremental_growth():
    tracker = RoomTracker(CFG)
    partial = floor_plan(SHAPE, [ROOM_A], doors=[door(42, 0.9)])            # B not seen yet
    tracker.update(seg(partial))
    (rid_a,) = tracker.rooms
    tracker.set_label(rid_a, "bedroom", 0.8, frozenset())
    full = floor_plan(SHAPE, [ROOM_A, ROOM_B], doors=[door(42, 0.9)])
    summary = tracker.update(seg(full))
    assert sorted(tracker.rooms) == [rid_a, rid_a + 1] and summary["new"] == [rid_a + 1]
    assert tracker.room_at((2.0, 2.0), 0.5) == rid_a
    assert tracker.rooms[rid_a].room.label == "bedroom"
    assert tracker.rooms[rid_a + 1].needs_label and not tracker.rooms[rid_a].needs_label
    assert tracker.rooms[rid_a].room.adjacent_room_ids == [rid_a + 1]


def test_merge_keeps_one_id_and_aliases_the_other():
    tracker = RoomTracker(CFG)
    tracker.update(seg(floor_plan(SHAPE, [ROOM_A, ROOM_B], doors=[door(42, 0.9)])))
    assert sorted(tracker.rooms) == [0, 1]
    summary = tracker.update(seg(floor_plan(SHAPE, [ROOM_A, ROOM_B], openings=[(42, 2, 43, 42)])))
    (kept,) = tracker.rooms
    gone = 1 - kept
    assert summary["aliased"] == {str(gone): kept}
    assert tracker.resolve(gone) == kept


def test_split_keeps_id_on_larger_part():
    tracker = RoomTracker(CFG)
    big = (2, 2, 102, 42)
    tracker.update(seg(floor_plan(SHAPE, [big])))
    assert sorted(tracker.rooms) == [0]
    arr = floor_plan(SHAPE, [big])
    arr[2:42, 66] = 1                                   # new wall at 64 % of the length ...
    arr[15:24, 66] = 0                                  # ... with a 0.9 m door
    tracker.update(seg(arr))
    assert sorted(tracker.rooms) == [0, 1]
    assert tracker.room_at((3.0, 2.0), 0.0) == 0 and tracker.room_at((8.0, 2.0), 0.0) == 1


def test_rooms_disabled_gives_no_rooms(tmp_path):
    arr = floor_plan(SHAPE, [ROOM_A, ROOM_B], doors=[door(42, 0.9)])
    over = deep_merge(QUIET, {"rooms": {"enabled": False}})
    mem = make_memory(tmp_path, over, grid=make_grid(arr))
    mem.update(make_result(0.0, [make_obs("bed", 0.8, (2.0, 2.0, 0.5), 0.0)]))
    mem.tick(100.0)
    assert mem.rooms() == [] and mem.all_objects()[0].room_id is None
