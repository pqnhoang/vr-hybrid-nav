"""P6: BagEventSim oracle and trajectory modes."""
from vlnav.common.models import Pose, Subtask
from vlnav.sim.bag_event_sim import BagEventSim, load_sim_config


def pose(x, y, t=0.0):
    return Pose(position=(x, y, 0.0), orientation_xyzw=(0, 0, 0, 1), stamp=t)


GOTO = Subtask(id=3, task_id=1, type="goto", target="umbrella", node_id=21, goal_pose=pose(5.0, 0.0))
EXPLORE = Subtask(id=4, task_id=1, type="explore", target="umbrella")


def test_oracle_mode():
    sim = BagEventSim(load_sim_config(overrides={"bag_event_sim": {"mode": "oracle", "oracle_delay_s": 5.0}}))
    sim.set_subtask(GOTO, 10.0)
    assert sim.step(pose(0, 0), 14.0) == []
    (e,) = sim.step(pose(0, 0), 15.0)
    assert (e.type, e.task_id, e.subtask_id) == ("reached", 1, 3)
    assert sim.step(pose(0, 0), 16.0) == [] and sim.finish(20.0) == []


def test_trajectory_mode_and_finish():
    sim = BagEventSim(load_sim_config(overrides={"bag_event_sim": {"mode": "trajectory", "reach_radius_m": 1.0}}))
    sim.set_subtask(GOTO, 0.0)
    assert sim.step(pose(3.9, 0.0), 1.0) == []
    assert sim.step(pose(4.2, 0.3), 2.0)[0].type == "reached"
    sim.set_subtask(GOTO, 3.0)
    assert [e.type for e in sim.finish(9.0)] == ["nav_failed"]
    sim.set_subtask(EXPLORE, 10.0)
    assert sim.step(pose(5, 0), 11.0) == [] and [e.type for e in sim.finish(12.0)] == ["no_frontier"]
