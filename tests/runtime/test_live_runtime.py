import threading
import json
from types import SimpleNamespace

import numpy as np
from expl_helpers import cfg, planner, pose
from move_helpers import cfg as move_cfg, frame, planner as move_planner
from planner_helpers import Router, explore, goto, make_planner, move, plan, target, verify

from vlnav.acquisition.vlm_client import MockVLMClient
from vlnav.exploration import ExplorationCore
from vlnav.mapping.occupancy import OccupancyGrid2D
from vlnav.move import MockS2Client, MoveCore
from vlnav.runtime.live_runtime import LiveRuntime


def runtime(p, mem=None, *, goto_object_approach=False, s2_script=None):
    grid = OccupancyGrid2D.from_array(np.zeros((80, 80), np.int8), .1, (-2., -2.))
    ex = ExplorationCore(cfg(), grid, planner())
    perception = SimpleNamespace(prompts=[])
    perception.set_active_prompts = lambda ps: setattr(perception, "prompts", ps)
    mv = None if s2_script is None else MoveCore(move_cfg(), grid, move_planner(), MockS2Client(s2_script))
    return LiveRuntime(SimpleNamespace(planner=p, exploration=ex, perception=perception, memory=mem, move=mv),
                       goto_object_approach=goto_object_approach)


def test_actual_pose_reaches_goal_and_produces_target_report(tmp_path):
    p, mem = make_planner(tmp_path, Router(decompose=[plan([target("umbrella")], [goto("umbrella", 21)])],
                                          verify=[verify("A")]))
    rt = runtime(p)
    rt.submit("Find umbrella", 0.)
    first = rt.step(pose(0, 0), 1.)
    assert rt.last_reasoning == "r"
    assert first.goal == mem.get_object(21).best_view_pose
    assert first.path and p.pop_reports() == []
    # No bag simulator: moving the real pose to the goal drives the real exploration event.
    stopped = rt.step(first.goal, 2.)
    assert stopped.goal is None and p.state().fsm == "DONE"
    assert p.pop_reports()[0].node_id == 21
    assert rt.c.perception.prompts == []
    assert rt.c.exploration.subtask.type == "stop"


def test_live_goto_uses_memory_centroid_for_robot_goal(tmp_path):
    p, mem = make_planner(tmp_path, Router(decompose=[plan([target("umbrella")], [goto("umbrella", 21)])],
                                          verify=[verify("A")]))
    rt = runtime(p, mem, goto_object_approach=True)
    rt.submit("Find umbrella", 0.)
    out = rt.step(pose(-1.0, 1.0), 1.)
    selected = mem.get_object(21)
    assert out.goal is not None and out.goal != selected.best_view_pose
    assert 0.6 <= np.linalg.norm(np.asarray(out.goal.position[:2]) - selected.centroid[:2]) <= 1.2
    assert out.goal.position[0] > -1.0
    p.close()


def test_async_task_preemption_clears_goal_until_new_plan(tmp_path):
    p, _ = make_planner(tmp_path, Router(decompose=[plan([target("umbrella")], [goto("umbrella", 21)])],
                                        verify=[verify("A")]), blocking=False)
    rt = runtime(p)
    gate = threading.Event()
    try:
        rt.submit("Find umbrella", 0.)
        p.wait_idle(5)
        rt.step(pose(0, 0), .1)  # collect dispatch
        assert rt.step(pose(0, 0), .2).goal is not None

        def slow(messages):
            gate.wait(5)
            return json.dumps(plan([target("sofa")], [explore("sofa")]))

        p.decomposer.vlm = MockVLMClient(slow)
        rt.submit("Find sofa", 1.)
        assert p.state().fsm == "DECOMPOSING"
        assert rt.step(pose(0, 0), 1.1).goal is None
        assert rt.c.exploration.subtask.type == "stop"
        rt.submit("", 1.2)
        gate.set()
        p.wait_idle(5)
        assert rt.step(pose(0, 0), 1.3).goal is None
        assert p.state().fsm == "FAILED"
    finally:
        gate.set()
        p.close()


def test_missing_robot_pose_withholds_navigation(tmp_path):
    p, _ = make_planner(tmp_path, Router(decompose=[plan([target("umbrella")], [goto("umbrella", 21)])],
                                        verify=[verify("A")]))
    rt = runtime(p)
    rt.submit("Find umbrella", 0.)
    assert rt.step(None, 1.).goal is None
    assert p.pop_reports() == []
    assert rt.step(pose(0, 0), 2.).goal is not None


def test_move_subtask_runs_on_move_core_and_stop_ends_the_route(tmp_path):
    route = "walk to the glass door"
    p, _ = make_planner(tmp_path, Router(decompose=[plan([], [move("glass door", route)])],
                                        replan=[{"reasoning": "r", "task_done": True, "target_updates": [],
                                                 "subtasks": []}]), move=True)
    rt = runtime(p, s2_script=["↓", "320 400", "STOP"])
    rt.submit(route, 0.)
    assert rt.move.subtask.type == "move" and rt.c.exploration.subtask.type == "move"
    rt.move.on_frame(frame(stamp=0.5))
    out = rt.step(pose(0, 0), 1.)
    assert out.goal_kind == "move" and out.goal is not None and rt.c.exploration.step(pose(0, 0), 1.).goal is None
    assert [r.instruction for r in rt.move.s2.requests] == [route, route]
    rt.step(pose(*out.goal.position[:2]), 2.)                 # at the goal -> next cycle -> STOP
    assert p.state().fsm == "DONE" and rt.c.exploration.subtask.type == "stop"
    assert p.state().history[0]["outcome"] == "reached"
