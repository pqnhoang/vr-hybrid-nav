"""Goal commitment survives map churn but releases invalid goals and preserves recovery."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from expl_helpers import cfg, explore, planner, pose
from pydantic import ValidationError

from vlnav.exploration import ExplorationCore
from vlnav.exploration.config import load_exploration_config
from vlnav.exploration.frontiers import FrontierPoint
from vlnav.exploration.goal_selector import Candidate, GoalSelector
from vlnav.mapping.occupancy import FREE, OCCUPIED, OccupancyGrid2D


class StraightPlanner:
    def __init__(self):
        self.blocked = set()
        self.goals = []

    def plan(self, grid, start, goal):
        self.goals.append(goal)
        return None if goal in self.blocked else [tuple(start), tuple(goal)]


def make_selector(hold=8.0, weight=0.0, **selection):
    config = load_exploration_config(overrides={
        "selection": {"min_goal_hold_s": hold, "switch_margin": 0.0, **selection},
        "scoring": {"w_yaw": weight, "yaw_change_window_s": 8.0},
    })
    result = GoalSelector(config.selection, StraightPlanner(), config.scoring)
    result.debug = True
    return result


def select(selector, candidates, t, robot=(0., 0.), yaw=0., valid=True, instances=()):
    return selector.explore(SimpleNamespace(version=int(t)), robot, t, list(instances), candidates,
                            lambda *_: None, robot_yaw=yaw, current_valid=valid)


def test_commitment_pins_position_and_yaw_through_resampling_and_missing_id():
    selector = make_selector()
    a = Candidate("frontier", 1, .5, [(3., 0.)])
    b = Candidate("frontier", 2, 3., [(0., 3.)])
    first = select(selector, [a], 0.)
    assert selector.last_debug["goal_age_s"] == 0.
    assert selector.last_debug["commitment_remaining_s"] == 8.
    moved = Candidate("frontier", 1, .5, [(3., .4)])
    kept = select(selector, [b, moved], 1., robot=(0., .2))
    assert kept.goal_xy == first.goal_xy and kept.goal_yaw == first.goal_yaw
    assert selector.last_debug["reason"] == "goal_commitment"
    assert selector.s.goal_since == 0.
    # An id disappearing does not immediately cancel a valid committed destination.
    assert select(selector, [b], 7.9).goal_xy == first.goal_xy
    assert selector.planner.goals[-1] == first.goal_xy
    assert select(selector, [b], 8.).cand.id == b.id
    assert selector.s.goal_since == 8.


def test_position_stays_pinned_after_minimum_until_challenger_wins():
    selector = make_selector()
    a = Candidate("frontier", 1, .5, [(3., 0.)])
    select(selector, [a], 0.)
    a.goals = [(3., .4)]
    assert select(selector, [a], 10., robot=(0., .2)).goal_xy == (3., 0.)
    assert selector.s.goal_since == 0.


@pytest.mark.parametrize("cause", ["invalid", "unreachable", "reached"])
def test_commitment_releases_immediately_when_goal_cannot_be_kept(cause):
    selector = make_selector()
    a = Candidate("frontier", 1, 1., [(3., 0.)])
    b = Candidate("frontier", 2, .5, [(0., 3.)])
    select(selector, [a, b], 0.)
    if cause == "unreachable":
        selector.planner.blocked.add((3., 0.))
    robot = (2.5, 0.) if cause == "reached" else (0., 0.)
    chosen = select(selector, [a, b], 1., robot=robot, valid=cause != "invalid")
    assert chosen.cand.id == b.id
    assert selector.s.goal_since == 1.


def test_instance_challenger_waits_for_commitment_then_preempts_frontier():
    selector = make_selector()
    a = Candidate("frontier", 1, 1., [(3., 0.)])
    obj = Candidate("instance", 4, .1, [(0., 2.)], (0., 3.))
    select(selector, [a], 0.)
    assert select(selector, [a], 1., instances=[obj]).cand.kind == "frontier"
    assert select(selector, [a], 8., instances=[obj]).cand.kind == "instance"


def test_stuck_history_resets_for_new_position_but_not_replans_or_id_churn():
    selector = make_selector(hold=3., stuck_window_s=5.)
    a = Candidate("frontier", 1, .5, [(3., 0.)])
    b = Candidate("frontier", 2, .8, [(0., 3.)])
    select(selector, [a], 0.)
    assert not selector.stuck((0., 0.), 2.)
    select(selector, [a], 2.)
    assert selector.s.history[0][0] == 0.
    select(selector, [b, a], 4.)
    assert selector.s.history[0][0] == 4.
    assert not selector.stuck((0., 0.), 5.)
    # New frontier identity at the same physical target must not restart patience.
    duplicate = Candidate("frontier", 3, .8, [(0., 3.)])
    select(selector, [duplicate], 7.)
    assert selector.s.goal_since == 4.
    assert selector.s.history[0][0] == 4.
    assert selector.stuck((0., 0.), 9.)


def test_instance_approach_change_resets_stuck_history_even_with_same_id():
    selector = make_selector(hold=0.)
    obj = Candidate("instance", 1, .8, [(3., 0.)], (4., 0.))
    select(selector, [], 0., instances=[obj])
    assert not selector.stuck((0., 0.), 2.)
    obj.goals = [(3., 1.)]
    select(selector, [], 3., instances=[obj])
    assert selector.s.history[0][0] == 3.


def test_yaw_penalty_discourages_reversal_and_does_not_mutate_base_scores():
    selector = make_selector(hold=0., weight=.4)
    forward = Candidate("frontier", 1, .5, [(3., 0.)])
    reverse = Candidate("frontier", 2, .7, [(-3., 0.)])
    assert select(selector, [reverse, forward], 0.).cand.id == forward.id
    assert selector.frontier_scores[reverse.id] == pytest.approx((.3, .4))
    assert reverse.score == .7 and reverse.yaw_penalty == 0.
    assert selector.last_debug["selected"]["yaw_penalty"] == 0.


def test_yaw_penalty_fades_for_an_old_goal_and_does_not_forbid_reversal():
    selector = make_selector(hold=0., weight=.4)
    a = Candidate("frontier", 1, .5, [(3., 0.)])
    b = Candidate("frontier", 2, .7, [(-3., 0.)])
    select(selector, [a], 0.)
    assert select(selector, [b, a], 1.).cand.id == a.id
    # The reference is the last commanded yaw, even if the robot has started turning.
    assert select(selector, [b, a], 2., yaw=math.pi).cand.id == a.id
    assert select(selector, [b, a], 32.).cand.id == b.id
    assert selector.frontier_scores[b.id] == pytest.approx((.6, .1))


def test_yaw_penalty_wraps_at_pi_and_uses_object_facing_orientation():
    selector = make_selector(hold=0., weight=.4)
    angle = math.radians(-179.)
    a = Candidate("frontier", 1, .5, [(3 * math.cos(angle), 3 * math.sin(angle))])
    chosen = select(selector, [a], 0., yaw=math.radians(179.))
    assert chosen.cand.yaw_penalty == pytest.approx(.4 * 2 / 180)
    selector.reset()
    # Same approach position, different final object-facing headings.
    forward = Candidate("instance", 2, .5, [(3., 0.)], (4., 0.))
    reverse = Candidate("instance", 3, .7, [(3., 0.)], (2., 0.))
    assert select(selector, [], 0., instances=[reverse, forward]).cand.id == forward.id


def make_core(**over):
    grid = OccupancyGrid2D.from_array(np.full((100, 100), FREE, np.int8), .1, (-5., -5.))
    config = cfg(selection={"min_goal_hold_s": 8., "stuck_window_s": 3.}, **over)
    core = ExplorationCore(config, grid, planner(goal_snap_radius_m=.5))
    core.selector.debug = True
    core.set_subtask(explore())
    core.frontiers.points = {i: FrontierPoint(id=i, xy=xy, cluster_id=i, created_t=0.)
                            for i, xy in [(1, (2., 0.)), (2, (-3., 0.))]}
    return core


def test_core_pins_output_yaw_and_scores_the_committed_position():
    core = make_core(scoring={"w_yaw": .4}, curiosity={"dist_metric": "euclidean"})
    first = core.step(pose(0., 0.), 0.)
    assert first.goal_id == 1
    core.frontiers.points[1].xy = (2., .4)
    kept = core.step(pose(0., .3, yaw=math.pi), 1.)
    assert kept.goal.position[:2] == first.goal.position[:2]
    assert kept.goal.yaw() == first.goal.yaw()
    scores = {p["id"]: p["score"] for p in core.snapshot()["points"]}
    assert scores[1]["dist"] == pytest.approx(math.dist((0., .3), (2., 0.)))
    assert scores[1]["yaw_penalty"] == 0.
    assert scores[2]["yaw_penalty"] > .3


def test_core_releases_occupied_pin_even_when_planner_can_snap_nearby():
    core = make_core()
    first = core.step(pose(0., 0.), 0.)
    arr = core.grid.as_array().copy()
    ix, iy = core.grid.world_to_cell(np.asarray(first.goal.position[:2]))
    arr[iy, ix] = OCCUPIED
    core.grid.set_array(arr, 1.)
    assert core.planner.plan(core.grid, (0., 0.), first.goal.position[:2]) is not None
    assert core.step(pose(0., 0.), 1.).goal_id == 2


def test_core_stuck_detection_and_new_subtask_override_commitment():
    core = make_core()
    first = core.step(pose(0., 0.), 0.)
    assert core.step(pose(0., 0.), 2.).goal_id == first.goal_id
    recovered = core.step(pose(0., 0.), 3.)
    assert recovered.goal_id != first.goal_id
    assert any("stuck:" in note for note in core.snapshot()["notes"])
    core.set_subtask(explore(2))
    assert core.selector.s.current is None
    assert not core.selector.s.history and core.selector.s.last_goal_yaw is None


@pytest.mark.parametrize("over", [
    {"selection": {"min_goal_hold_s": -1.}},
    {"selection": {"min_goal_hold_s": float("inf")}},
    {"scoring": {"w_yaw": -1.}},
    {"scoring": {"yaw_change_window_s": 0.}},
])
def test_stabilization_settings_reject_invalid_values(over):
    with pytest.raises(ValidationError):
        load_exploration_config(overrides=over)
