"""E3: instance voxel merge, ranking, approach point, visit rule; unknown-ratio BFS; Eq. 5-8."""
import math

import numpy as np
import pytest
from expl_helpers import HFOV, cfg, corridor_grid, det, frame, result

from vlnav.exploration.curiosity import s_dist, s_unknown, unknown_ratio
from vlnav.exploration.instances import InstancePool
from vlnav.exploration.scoring import s_nesy
from vlnav.exploration.vl_score import RefPose, mixture
from vlnav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN

REF = RefPose(0.0, 0.0, 0.0)


def frame_result(dets, xy, prompts=("sofa",)):
    return result(frame(), dets, xy, prompts=list(prompts))


def test_voxel_merge_tau_and_relevance():
    dets = [det("sofa", 0.6, 80, (75, 50, 85, 70)), det("sofa", 0.9, 82, (77, 50, 87, 70)),
            det("sofa", 0.2, 60, (55, 50, 65, 70)), det("chair", 0.9, 100), det("sofa", 0.5, 90, (85, 50, 95, 70))]
    res = frame_result(dets, [(2.05, 0.05), (2.15, 0.15), (3.0, 0.0), (2.5, 0.5), (3.3, -0.1)])
    pool = InstancePool(cfg().instance)
    new = pool.add_frame(res, REF, HFOV, mixture(res, 0.1), 1, 0.0, "confidence")
    assert len(new) == 2                                   # tau_det drops 0.2; chair is not a prompt
    p = pool.points[new[0]]
    assert p.score == 0.9 and p.rank == 0.9 and p.n_obs == 2
    assert np.allclose(p.xy, ((0.6 * 2.05 + 0.9 * 2.15) / 1.5, (0.6 * 0.05 + 0.9 * 0.15) / 1.5))
    pool.clear()
    assert not pool.points


def test_vl_eq3_rank_is_score_at_own_bearing():
    res = frame_result([det("sofa", 0.8, 80, (75, 50, 85, 70)), det("sofa", 0.4, 30, (25, 50, 35, 70))],
                       [(4.0, 0.0), None])
    pool = InstancePool(cfg().instance)
    (pid,) = pool.add_frame(res, REF, HFOV, mixture(res, 0.1), 1, 0.0, "vl_eq3")
    m = mixture(res, 0.1)
    expected = 0.8 + 0.4 * math.exp(-0.5 * (m.mu[1] / 0.1) ** 2)      # both detections contribute
    assert pool.points[pid].rank == pytest.approx(expected)
    far = frame_result([det("sofa", 0.8, 80, (75, 50, 85, 70))], [(9.0, 0.0)])
    (fid,) = InstancePool(cfg().instance).add_frame(far, REF, HFOV, mixture(far, 0.1), 1, 0.0, "vl_eq3")
    assert fid == 0                                        # beyond R: still ranked (no range gate)


def test_approach_point_and_visit_rule():
    g = corridor_grid(nx=40, ny=40, free_box=(0, 0, 40, 40), walls=[(18, 18, 23, 23)])
    pool = InstancePool(cfg().instance)
    res = frame_result([det("sofa", 0.8, 80)], [(2.05, 2.05)])
    (pid,) = pool.add_frame(res, REF, HFOV, mixture(res, 0.1), 1, 0.0, "confidence")
    opts = pool.approach_options(pool.points[pid], g, g.inflated(0.2), (0.5, 2.0))
    a = opts[0]
    d = math.dist(a, (2.05, 2.05))
    assert 0.6 <= d <= 1.2 and a[0] < 1.2 and g.value(g.world_to_cell(a[None]))[0] == FREE
    assert not g.inflated(0.2)[tuple(g.world_to_cell(a)[::-1])]
    assert pool.update_visits((0.5, 2.0), 0.8) == []
    pool.points[pid].approach_xy = tuple(a)
    assert pool.update_visits(tuple(a), 0.8) == [pid] and pool.points[pid].status == "visited"


def test_unknown_ratio_bfs():
    arr = np.full((11, 11), FREE, np.int8)
    arr[:, 6:] = UNKNOWN
    arr[:, 2] = OCCUPIED                                   # cells left of the wall are not reached
    assert unknown_ratio(arr, (5, 5), 10, np.ones((3, 3), bool)) == pytest.approx(55 / 88)
    assert unknown_ratio(arr, (2, 5), 10, np.ones((3, 3), bool)) == 0.0     # on an obstacle
    small = unknown_ratio(arr, (4, 5), 1, np.ones((3, 3), bool))           # 3x3 disk: no unknown
    assert small == 0.0


def test_equations_5_to_8():
    assert s_dist(1.0) == 0.5
    assert s_unknown(0.5, 4.0) == pytest.approx(1 - math.exp(-2))
    assert s_nesy(0.5, 0.4, 0.6, 1.0, 3.0, "additive_dist") == pytest.approx(0.5 + 3 * 0.4 * 0.6)
    assert s_nesy(0.5, 0.4, 0.6, 1.0, 3.0, "multiplicative_unknown") == pytest.approx((0.5 + 1.2) * 0.6)
    assert s_nesy(0.3, 0.0, 0.9, 1.0, 3.0, "additive_dist") == 0.3          # no VL: nearest frontier
