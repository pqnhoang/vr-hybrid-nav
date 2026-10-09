"""E2: VL score (Eq. 3-4), sign convention, assignment rules per subtask and update modes,
angle-reference validation."""
import math

import numpy as np
import pytest
from expl_helpers import HFOV, K, cfg, det, explore, frame, full_info, planner, result, corridor_grid

from vlnav.exploration import ExplorationCore
from vlnav.exploration.frontiers import FrontierPoint
from vlnav.exploration.vl_score import Mixture, RefPose, assign, bearing, s_vl, view_confidence

REF = RefPose(0.0, 0.0, 0.0)


def mix(mu, alpha, sigma=0.1):
    return Mixture(np.atleast_1d(np.asarray(mu, float)), np.atleast_1d(np.asarray(alpha, float)),
                   np.full(np.size(mu), sigma))


def ring(angles, r=2.0):
    return np.stack([r * np.cos(angles), r * np.sin(angles)], 1)


def test_view_confidence_and_wedge_edges():
    assert view_confidence(0.0, HFOV) == 1.0
    assert view_confidence(HFOV / 2, HFOV) < 1e-12 and view_confidence(HFOV / 2 + 0.01, HFOV) == 0.0
    s, _, inside = s_vl(ring([0.0, HFOV / 2 - 1e-9, HFOV / 2 + 0.05, math.pi]), REF, HFOV, mix(0.0, 0.8), 5.0)
    assert s[0] == pytest.approx(0.8) and s[1] < 1e-9 and s[2] == s[3] == 0.0 and inside.tolist() == [1, 1, 0, 0]
    assert s_vl(ring([0.0], r=6.0), REF, HFOV, mix(0.0, 0.8), 5.0)[0][0] == 0.0          # beyond R
    assert s_vl(ring([0.0], r=6.0), REF, HFOV, mix(0.0, 0.8), None)[0][0] > 0             # no range gate


def test_peak_at_mu_and_clipping():
    a = np.linspace(-0.6, 0.6, 1201)
    assert abs(a[np.argmax(s_vl(ring(a), REF, HFOV, mix(0.0, 0.5), 5.0)[0])]) < 1e-9
    peak = a[np.argmax(s_vl(ring(a), REF, HFOV, mix(0.3, 0.5), 5.0)[0])]
    assert 0.25 < peak <= 0.3                               # C(Δθ) pulls the peak slightly inward
    assert s_vl(ring([0.0]), REF, HFOV, mix([0.0, 0.0], [0.9, 0.9]), 5.0)[0][0] == 1.0


def test_bearing_sign_left_is_positive():
    assert bearing(40, K) > 0 > bearing(120, K) and bearing(80, K) == 0
    m = mix(bearing(40, K), 0.8)
    s = s_vl(np.array([[2.0, 1.0], [2.0, -1.0]]), REF, HFOV, m, 5.0)[0]   # left / right of the robot
    assert s[0] > 0.1 > 0.01 > s[1]
    turned = RefPose(0.0, 0.0, math.pi)                          # wrap: point behind, heading pi
    assert s_vl(np.array([[-2.0, 0.0]]), turned, HFOV, mix(0.0, 0.8), 5.0)[0][0] == pytest.approx(0.8)


def test_assignment_rules_and_update_modes():
    def p():
        return FrontierPoint(id=0, xy=(0, 0), cluster_id=0, created_t=0.0)
    q = p()
    assign(q, 0.4, 1.0, 1, "first")
    assign(q, 0.9, 1.0, 1, "first")
    assert (q.s_vl, q.s_vl_subtask_id, q.n_vl_updates) == (0.4, 1, 1)
    assign(q, 0.1, 1.0, 2, "first")                              # new subtask: reassigned
    assert (q.s_vl, q.s_vl_subtask_id) == (0.1, 2)
    m = p()
    for v in (0.4, 0.9, 0.2):
        assign(m, v, 1.0, 1, "max")
    assert m.s_vl == 0.9 and m.n_vl_updates == 3
    w = p()
    assign(w, 0.8, 1.0, 1, "conf_weighted")
    assign(w, 0.2, 0.5, 1, "conf_weighted")
    assert w.s_vl == pytest.approx((0.8 * 1.0 + 0.2 * 0.5) / 1.5)


def _core(**over):
    g = corridor_grid(nx=80, ny=60, free_box=(20, 10, 60, 50))
    core = ExplorationCore(cfg(**over), g, planner())
    return core, g


def test_scores_assigned_in_wedge_per_subtask():
    core, g = _core()
    core.set_subtask(explore(1))
    core.on_frame(result(frame(4.0, 3.0, 0.0)), full_info(g))           # no relevant detection
    pts = list(core.frontiers.points.values())
    assert pts and all(p.s_vl_subtask_id == 1 and p.s_vl == 0.0 for p in pts)   # scored with 0
    core.on_frame(result(frame(4.0, 3.0, 0.0, 1.0), [det("sofa", 0.9, 80)], prompts=["sofa"]), full_info(g, 1.0))
    assert all(p.s_vl == 0.0 for p in core.frontiers.points.values())   # first: kept within subtask
    core.set_subtask(explore(2))
    core.on_frame(result(frame(4.0, 3.0, 0.0, 2.0), [det("sofa", 0.9, 80)], prompts=["sofa"]), full_info(g, 2.0))
    assert max(p.s_vl for p in core.frontiers.points.values()) > 0.5     # rescored for subtask 2
    core.on_frame(result(frame(4.0, 3.0, 0.0, 3.0), [det("chair", 0.9, 80)], prompts=["sofa"]), full_info(g, 3.0))
    assert max(p.s_vl for p in core.frontiers.points.values()) > 0.5     # irrelevant label ignored


def test_strict_paper_mode_never_rescores():
    core, g = _core(vl={"rescore_on_new_subtask": False})
    core.set_subtask(explore(1))
    core.on_frame(result(frame(4.0, 3.0, 0.0), [det("sofa", 0.9, 80)], prompts=["sofa"]), full_info(g))
    assert max(p.s_vl for p in core.frontiers.points.values()) > 0.5     # scored in the creation frame
    core.set_subtask(explore(2))
    assert all(p.s_vl == 0.0 and p.s_vl_subtask_id == 2 for p in core.frontiers.points.values())
    core.on_frame(result(frame(4.0, 3.0, 0.0, 1.0), [det("sofa", 0.9, 80)], prompts=["sofa"]), full_info(g, 1.0))
    assert all(p.s_vl == 0.0 for p in core.frontiers.points.values())


def test_angle_reference_validation():
    core, g = _core()
    core.set_subtask(explore(1))
    core.on_frame(result(frame()), full_info(g))
    with pytest.raises(ValueError, match="several cameras"):
        core.on_frame(result(frame(camera_id="cam2")), full_info(g))
    yawed, g2 = _core()
    with pytest.raises(ValueError, match="yawed"):
        yawed.on_frame(result(frame(cam_yaw=math.radians(10))), full_info(g2))
    cam, g3 = _core(angle_reference="camera")
    cam.set_subtask(explore(1))
    cam.on_frame(result(frame(4.0, 3.0, 0.0, cam_yaw=math.radians(-90))), full_info(g3))   # looks -y
    assert len(cam.frontiers.points) and all(p.xy[1] < 3.0 for p in cam.frontiers.points.values())
