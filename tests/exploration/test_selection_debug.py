from types import SimpleNamespace

import pytest

from vlnav.exploration.config import load_exploration_config
from vlnav.exploration.goal_selector import Candidate, GoalSelector


class StraightPlanner:
    def plan(self, grid, start, goal):
        return [tuple(start), tuple(goal)]


@pytest.mark.parametrize("debug", [False, True])
def test_debug_explains_switches_without_changing_selection(debug):
    selector = GoalSelector(load_exploration_config().selection, StraightPlanner())
    selector.debug = debug
    grid = SimpleNamespace(version=1)
    a = Candidate("frontier", 1, .5, [(1., 0.)])
    b = Candidate("frontier", 2, .55, [(2., 0.)])
    choose = lambda candidates, t: selector.explore(grid, (0., 0.), t, [], candidates, lambda *_: None)
    assert choose([a], 0.).cand.id == 1
    assert choose([b, a], .5).cand.id == 1
    if debug:
        assert selector.last_debug["reason"] == "kept_goal"
    # A missing current candidate bypasses score hysteresis, even for a lower score.
    b.score = .1
    assert choose([b], 1.).cand.id == 2
    if debug:
        assert selector.last_debug["reason"] == "previous_not_selectable"
        assert selector.last_debug["previous"]["id"] == 1
        assert selector.last_debug["plans"][0]["id"] == 2
    b.goals = [(2., 1.)]
    assert choose([b], 1.5).goal_xy == (2., 1.)
    if debug:
        assert selector.last_debug["reason"] == "approach_changed"
        assert selector.last_debug["approach_changed"]
    else:
        assert selector.last_debug == {}
