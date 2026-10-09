"""S4: A* finds paths through unknown with cost, never through occupied / inflated cells."""
import numpy as np

from vlnav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid2D
from vlnav.navigation import GridAStar, load_navigation_config


def planner(**over):
    return GridAStar(load_navigation_config(overrides=over))


def grid(arr):
    return OccupancyGrid2D.from_array(np.asarray(arr, np.int8), 0.1, (0.0, 0.0))


def cells_of(g, path):
    return g.world_to_cell(np.asarray(path))


def test_path_through_unknown_never_through_occupied_or_inflated():
    arr = np.full((30, 60), FREE, np.int8)
    arr[:, 30] = OCCUPIED                       # wall, crossing only through unknown cells
    arr[12:18, 30] = UNKNOWN
    g = grid(arr)
    p = planner(robot_radius_m=0.1, unknown_cost=3.0)
    path = p.plan(g, (1.0, 1.5), (5.0, 1.5))
    assert path is not None
    c = cells_of(g, path)
    assert (g.value(c) != OCCUPIED).all() and not g.inflated(0.1)[c[:, 1], c[:, 0]].any()
    assert any(g.value(c) == UNKNOWN) and np.allclose(path[0], (1.05, 1.55)) and np.allclose(path[-1], (5.05, 1.55))
    assert planner(robot_radius_m=0.4).plan(g, (1.0, 1.5), (5.0, 1.5)) is None    # gap < 2 x radius


def test_unknown_cost_prefers_known_free_detour():
    arr = np.full((40, 40), UNKNOWN, np.int8)
    arr[0:3, :] = FREE                          # free detour along the bottom
    arr[:, 0:3] = FREE
    arr[:, 37:] = FREE
    g = grid(arr)
    cheap = planner(robot_radius_m=0.0, unknown_cost=1.0).plan(g, (0.15, 3.5), (3.85, 3.5))
    dear = planner(robot_radius_m=0.0, unknown_cost=10.0).plan(g, (0.15, 3.5), (3.85, 3.5))
    assert (g.value(cells_of(g, cheap)) == UNKNOWN).any()
    assert (g.value(cells_of(g, dear)) == FREE).all()


def test_goal_and_start_snapping_and_costs():
    arr = np.full((20, 20), FREE, np.int8)
    arr[10, 10] = OCCUPIED
    g = grid(arr)
    p = planner(robot_radius_m=0.0, goal_snap_radius_m=0.3)
    path = p.plan(g, (0.5, 0.5), (1.05, 1.05))          # goal on the occupied cell -> neighbor
    assert path is not None and tuple(g.world_to_cell(np.asarray(path[-1]))) != (10, 10)
    assert np.isclose(p.cost_to(g, (0.05, 0.05), (1.05, 0.05)), 1.0)
    arr2 = np.full((20, 20), OCCUPIED, np.int8)
    arr2[0:5, 0:5] = FREE
    g2 = grid(arr2)
    assert p.plan(g2, (0.2, 0.2), (1.5, 1.5)) is None and np.isinf(p.cost_to(g2, (0.2, 0.2), (1.5, 1.5)))
    p.plan(g2, (0.2, 0.2), (0.4, 0.4))
    n = p.n_searches
    p.plan(g2, (0.2, 0.2), (0.3, 0.1))                   # same grid version + start: no new search
    assert p.n_searches == n
