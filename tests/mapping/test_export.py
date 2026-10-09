"""Occupancy map export: ROS map_server PGM/YAML round trip, orientation, origin; room map."""
import numpy as np
import yaml

from vlnav.mapping.export import grid_image, load_map, save_map
from vlnav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid2D


def grid():
    arr = np.full((6, 9), UNKNOWN, np.int8)
    arr[1:5, 1:8] = FREE
    arr[4, 7] = OCCUPIED                                  # north-east corner of the free block
    return OccupancyGrid2D.from_array(arr, 0.05, (-1.5, 2.25))


def test_ros_map_round_trip_orientation_and_origin(tmp_path):
    g = grid()
    paths = save_map(g, tmp_path / "map")
    meta = yaml.safe_load(paths["yaml"].read_text())
    assert meta["image"] == "occupancy.pgm" and meta["resolution"] == 0.05
    assert meta["origin"] == [-1.5, 2.25, 0.0] and meta["mode"] == "trinary"
    img = grid_image(g)
    assert img.shape == (6, 9) and img[6 - 1 - 4, 7] == 0      # north up: row 0 = max y
    back = load_map(paths["yaml"])
    assert np.array_equal(back.as_array(), g.as_array()) and back.origin_xy == g.origin_xy
    xy = g.cell_to_world(np.array([[7, 4]]))
    assert back.value(back.world_to_cell(xy)).tolist() == [OCCUPIED]


def test_room_map_written_without_object_dots(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))
    from memory_helpers import QUIET, floor_plan, make_grid, make_memory, make_obs, make_result
    from vlnav.common.config import deep_merge
    from vlnav.viz.map_export import export_maps
    import cv2
    arr = floor_plan((60, 100), [(2, 2, 42, 42), (43, 2, 83, 42)], doors=[(42, 15, 43, 24)])
    mem = make_memory(tmp_path, deep_merge(QUIET, {"rooms": {"enabled": True}}), grid=make_grid(arr))
    mem.update(make_result(0.0, [make_obs("bed", 0.8, (1.0, 1.0, 0.5), 0.0)]))
    paths = export_maps(mem.grid, mem.view(), tmp_path / "map", np.array([[0.5, 0.5], [3.0, 2.0]]))
    assert set(paths) == {"png", "pgm", "yaml", "rooms", "trajectory"}
    assert np.loadtxt(paths["trajectory"], delimiter=",", ndmin=2).shape == (2, 2)
    img = cv2.imread(str(paths["rooms"]))
    g = mem.grid
    assert img.shape[:2] == (2 * g.shape[0], 2 * g.shape[1])
    assert export_maps(OccupancyGrid2D(0.05, (0, 0), (0, 0)), mem.view(), tmp_path / "empty") == {}
