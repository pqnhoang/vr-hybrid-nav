"""E1: FoV wedge gating, 8-connectivity, cluster sampling counts, incremental = full
recomputation, pool identity / inheritance across updates."""
import math

import numpy as np
from expl_helpers import (HFOV, RES, cfg, corridor_grid, floor_pts, full_info, no_change, scan_grid,
                          wall_pts)

from vlnav.exploration.frontiers import FrontierMap, Wedge, frontier_test, representative_points
from vlnav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid2D


def fmap(**over):
    return FrontierMap(cfg(frontier=over).frontier)


def test_only_cells_in_the_wedge_become_frontiers():
    g = corridor_grid(nx=80, ny=60, free_box=(20, 10, 60, 50))
    w = Wedge(4.0, 3.0, 0.0, HFOV, 5.0)                    # facing +x
    fm = fmap()
    fm.update(g, full_info(g), [w], 0.0, None)
    ix = fm.cells()[:, 0]
    assert len(ix) and (ix == 59).all()                  # right edge only, not left/top/bottom
    fz = fmap(outside_fov="zero_score")
    fz.update(g, full_info(g), [w], 0.0, None)
    assert np.array_equal(fz.mask, frontier_test(g.as_array(), 8))
    assert {20, 59} <= set(fz.cells()[:, 0].tolist())


def test_frontier_connectivity():
    arr = np.full((7, 7), OCCUPIED, np.int8)
    arr[3, 3] = FREE
    arr[2, 2] = UNKNOWN                                  # only a diagonal unknown neighbor
    assert frontier_test(arr, 8)[3, 3] and not frontier_test(arr, 4)[3, 3]
    assert frontier_test(np.zeros((1, 1), np.int8), 4)[0, 0]      # outside the grid = unknown


def test_cluster_sampling_counts():
    g = corridor_grid(nx=80, ny=20, free_box=(0, 0, 80, 20))
    fm = fmap()
    fm.update(g, no_change(g), [], 0.0, None)
    fm.mask[:] = False
    fm.mask[5, 2:6] = True                               # 4 cells < min_cluster_cells: dropped
    fm.mask[10, 2:22] = True                             # 20 cells <= small_cluster_cells: centroid
    fm.mask[15, 2:62] = True                             # 60 cells: ceil(60 * 0.1 / 1.5) = 4 points
    fm.recompute_all(g, 0.0)
    by_row = {}
    for p in fm.points.values():
        by_row.setdefault(round(p.xy[1], 2), []).append(p.xy)
    assert sorted((k, len(v)) for k, v in by_row.items()) == [(1.05, 1), (1.55, 4)]
    assert np.allclose(by_row[1.05][0], (1.15, 1.05))   # centroid snapped onto a cluster cell
    xy = np.stack([np.arange(60) * RES, np.zeros(60)], 1)
    pts = representative_points(xy, RES, cfg().frontier)
    assert len(pts) == 4 and {0.0, 5.9} <= set(np.round(pts[:, 0], 2))   # both ends sampled


def scans():
    room = np.concatenate([floor_pts(0, 0, 6, 4), wall_pts((0, 0), (6, 0)), wall_pts((0, 4), (2.5, 4)),
                           wall_pts((0, 0), (0, 4))])
    far = np.concatenate([floor_pts(6, 0, 13, 4), wall_pts((6, 0), (13, 0))])
    return [(room, (1.0, 2.0), 0.0), (room, (2.0, 2.0), 0.3), (room, (3.0, 2.0), -0.3),
            (np.concatenate([room, far]), (5.0, 2.0), 0.0), (far, (10.0, 2.0), 0.0), (far, (11.0, 2.0), 3.0)]


def test_incremental_equals_full_recomputation():
    for mode in ("zero_score", "ignore"):
        g, fm, full = scan_grid(), fmap(outside_fov=mode), fmap(outside_fov=mode)
        for k, (pts, xy, yaw) in enumerate(scans()):
            info = g.update(pts, (*xy, 1.0), float(k))
            w = [Wedge(*xy, yaw, HFOV, 5.0)]
            fm.update(g, info, w, float(k), None)
            full.update(g, info, w, float(k), None, recluster_all=True)
            if mode == "zero_score":
                assert np.array_equal(fm.mask, frontier_test(g.as_array(), 8))
            assert np.array_equal(fm.mask, full.mask)
            assert sorted((p.id, p.xy) for p in fm.points.values()) == \
                sorted((p.id, p.xy) for p in full.points.values())
            assert len(fm.points) > 0 or k == 0
        assert g.shape[1] > 160                           # the grid grew during the sequence
    fresh = fmap()                                        # from scratch: same clusters, same counts
    fresh._align(g)
    fresh.mask = fm.mask.copy()
    fresh.recompute_all(g, 9.0)
    assert len(fresh.points) == len(fm.points)


def test_pool_identity_and_inheritance():
    g = corridor_grid(nx=80, ny=40, free_box=(0, 0, 80, 40))
    fm = fmap()
    fm.update(g, no_change(g), [], 0.0, None)
    fm.mask[:] = False
    fm.mask[10, 10:20] = True                            # cluster A
    fm.mask[30, 50:60] = True                            # cluster B
    fm.recompute_all(g, 0.0, subtask_id=1)
    a = next(p for p in fm.points.values() if p.xy[1] < 2)
    b = next(p for p in fm.points.values() if p.xy[1] > 2)
    a.s_vl, a.s_vl_subtask_id, a.vl_xy = 0.7, 1, a.xy    # as assign() records it
    fm.mask[10, 20:22] = True                            # A grows by 2 cells (touched), B untouched
    created = fm._recluster(g, [(20, 10, 21, 10)], 1.0, 1)
    a2 = next(p for p in fm.points.values() if p.xy[1] < 2)
    assert created == [] and a2.id == a.id and a2.s_vl == 0.7 and a2.xy != a.xy
    assert fm.points[b.id] is b                          # untouched cluster: same object
    fm.mask[10, :] = False                               # A disappears
    fm._recluster(g, [(0, 10, 79, 10)], 2.0, 1)
    assert set(fm.points) == {b.id}
    fm.mask[10, 40:46] = True                            # far from every old point: new id
    created = fm._recluster(g, [(40, 10, 45, 10)], 3.0, 1)
    assert created and created[0] not in (a.id, b.id) and fm.points[created[0]].s_vl == 0.0


def test_inherited_vl_score_stays_anchored_where_measured():
    g = corridor_grid(nx=200, ny=20, free_box=(0, 0, 200, 20))
    fm = fmap()
    fm.update(g, no_change(g), [], 0.0, None)
    fm.mask[:] = False
    fm.mask[10, 10:20] = True                            # 10 cells: one centroid point at x = 1.45
    fm.recompute_all(g, 0.0, subtask_id=1)
    (p,) = fm.points.values()
    p.s_vl, p.s_vl_subtask_id, p.vl_xy = 0.7, 1, p.xy
    for k, end in enumerate(range(22, 40, 2)):           # cluster grows 0.2 m per update (to 28 cells)
        fm.mask[10, 10:end] = True
        fm._recluster(g, [(10, 10, end - 1, 10)], float(k + 1), 1)
    (q,) = fm.points.values()
    assert q.id == p.id and q.xy[0] > p.xy[0] + 0.5      # identity follows the moving centroid
    assert q.s_vl == 0.0 and q.s_vl_subtask_id is None   # but the score stayed where it was measured


def test_points_on_frontier_cells_survive_resampling():
    g = corridor_grid(nx=120, ny=20, free_box=(0, 0, 120, 20))
    fm = fmap()
    fm.update(g, no_change(g), [], 0.0, None)
    fm.mask[:] = False
    fm.mask[10, 10:70] = True                            # 60 cells -> 4 points
    fm.recompute_all(g, 0.0)
    before = {p.id: p.xy for p in fm.points.values()}
    fm.mask[10, 70:85] = True                            # grows by 15 cells -> 5 points
    fm._recluster(g, [(70, 10, 84, 10)], 1.0, None)
    after = {p.id: p.xy for p in fm.points.values()}
    assert len(after) == 5 and all(after[i] == xy for i, xy in before.items())   # 4 kept + 1 new
    fm.mask[10, 10:30] = False                           # shrinks: points off the frontier are dropped
    fm._recluster(g, [(10, 10, 29, 10)], 2.0, None)
    assert all(p.xy[0] >= 3.0 for p in fm.points.values())
