"""E4: goal selection (instance-first, δ_reached, hysteresis, blacklist, stuck, hold) and events
(`reached`, `nav_failed`, `no_frontier` exactly once); ablations; no forbidden imports."""
import math
from pathlib import Path

import numpy as np
from expl_helpers import (cfg, det, explore, frame, full_info, goto, planner, pose, result)

from vlnav.exploration import ExplorationCore
from vlnav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid2D

ZERO = {"frontier": {"outside_fov": "zero_score"}}


def gap_grid():
    """Closed corridor (x 0.5..9.5, y 0.5..3.5) with a 1 m gap to unknown at each end."""
    arr = np.full((40, 100), UNKNOWN, np.int8)
    arr[4:36, 4:96] = OCCUPIED
    arr[5:35, 5:95] = FREE
    arr[15:25, 4] = FREE
    arr[15:25, 95] = FREE
    return OccupancyGrid2D.from_array(arr, 0.1, (0.0, 0.0))


def pocket_grid():
    """Closed known room (robot) + a closed pocket whose only frontier is unreachable."""
    arr = np.full((24, 60), UNKNOWN, np.int8)
    arr[1:22, 1:22] = OCCUPIED
    arr[2:21, 2:21] = FREE
    arr[1:22, 29:52] = OCCUPIED
    arr[2:21, 30:41] = FREE
    arr[2:21, 41:51] = UNKNOWN
    return OccupancyGrid2D.from_array(arr, 0.1, (0.0, 0.0))


def core_for(grid, **over):
    c = ExplorationCore(cfg(**{**ZERO, **over}), grid, planner())
    c.set_subtask(explore(1))
    return c


def tick(core, x, y, t, yaw=0.0, dets=(), obs=(), prompts=("sofa",)):
    core.on_frame(result(frame(x, y, yaw, t), dets, obs, prompts=list(prompts)), full_info(core.grid, t))
    return core.step(pose(x, y, yaw, t), t)


LEFT, RIGHT = (0.45, 1.95), (9.55, 1.95)


def test_nearest_frontier_and_vl_steering():
    out = tick(core_for(gap_grid()), 2.0, 2.0, 0.0)
    assert out.goal_kind == "frontier" and np.allclose(out.goal.position[:2], LEFT)
    assert out.path and np.allclose(out.path[-1], LEFT)
    out = tick(core_for(gap_grid()), 4.6, 2.0, 0.0, dets=[det("sofa", 0.9, 80)], obs=[None])
    assert np.allclose(out.goal.position[:2], RIGHT)               # VL beats proximity


def test_instance_first_then_visited():
    core = core_for(gap_grid())
    out = tick(core, 6.0, 2.0, 0.0, dets=[det("sofa", 0.5, 120)], obs=[(8.0, 1.0)])
    assert out.goal_kind == "instance" and out.goal_id == 0
    assert 0.6 <= math.dist(out.goal.position[:2], (8.0, 1.0)) <= 1.2
    out = tick(core, 7.5, 1.5, 1.0)
    assert core.instances.points[0].status == "visited" and out.goal_kind == "frontier"


def test_delta_reached_filter():
    out = tick(core_for(gap_grid()), 1.0, 2.0, 0.0)
    assert np.allclose(out.goal.position[:2], RIGHT)               # left one is within δ_reached


def test_hysteresis_and_instance_preemption():
    core = core_for(gap_grid())
    assert np.allclose(tick(core, 4.9, 2.0, 0.0).goal.position[:2], LEFT)
    assert np.allclose(tick(core, 5.1, 2.0, 1.0).goal.position[:2], LEFT)      # right better by < margin
    assert np.allclose(tick(core, 8.0, 2.0, 2.0).goal.position[:2], RIGHT)     # by more than margin
    assert core.snapshot()["n_switches"] == 1
    out = tick(core, 8.0, 2.0, 3.0, yaw=math.pi, dets=[det("sofa", 0.31, 80)], obs=[(5.0, 2.0)])
    assert out.goal_kind == "instance"                              # low-ranked instance still preempts


def test_unreachable_blacklisted_then_no_frontier_once():
    core = core_for(pocket_grid())
    outs = [tick(core, 1.0, 1.0, float(t)) for t in range(9)]
    assert all(o.goal_kind == "hold" for o in outs)
    pts = core.snapshot()["points"]
    assert pts and all(p["blacklisted"] for p in pts)
    ev = [(t, e) for t, o in enumerate(outs) for e in o.events]
    assert [(t, e.type) for t, e in ev] == [(5, "no_frontier")]
    assert ev[0][1].task_id == 0 and ev[0][1].subtask_id == 1 and ev[0][1].payload["source"] == "exploration"
    core.set_subtask(explore(2))                                    # re-armed per subtask
    assert [e.type for t in range(9, 16) for e in tick(core, 1.0, 1.0, float(t)).events] == ["no_frontier"]


def test_stuck_blacklists_current_goal():
    core = core_for(gap_grid(), selection={"stuck_detection": True, "stuck_window_s": 3.0})
    goals = [tick(core, 4.9, 2.0, float(t)).goal.position[:2] for t in range(5)]
    assert np.allclose(goals[2], LEFT) and np.allclose(goals[3], RIGHT)
    assert any(p["blacklisted"] for p in core.snapshot()["points"])
    off = core_for(gap_grid(), selection={"stuck_detection": False, "stuck_window_s": 3.0})
    assert all(np.allclose(tick(off, 4.9, 2.0, float(t)).goal.position[:2], LEFT) for t in range(5))


def test_goto_reached_and_nav_failed_once():
    core = core_for(gap_grid())
    core.set_subtask(goto(5, 3.0, 2.0))
    out = tick(core, 6.0, 2.0, 0.0)
    assert out.goal_kind == "goto" and out.goal_id == 7 and out.path and not out.events
    out = tick(core, 3.2, 2.0, 1.0)
    assert [e.type for e in out.events] == ["reached"] and out.events[0].subtask_id == 5
    assert out.goal_kind == "hold" and not tick(core, 3.2, 2.0, 2.0).events
    core = core_for(pocket_grid())
    core.set_subtask(goto(6, 3.5, 1.0))
    kinds = [[e.type for e in tick(core, 1.0, 1.0, float(t)).events] for t in range(5)]
    assert kinds == [[], [], ["nav_failed"], [], []]


def test_live_goto_uses_reachable_standoff_in_front_of_robot():
    grid = OccupancyGrid2D.from_array(np.full((100, 100), FREE, np.int8), 0.1, (0.0, 0.0))
    core = core_for(grid)
    core.set_subtask(goto(5, 1.0, 5.0), object_xy=(7.0, 5.0), require_object_approach=True)
    out = tick(core, 3.0, 5.0, 0.0)
    assert out.goal_kind == "goto" and out.goal_id == 7
    assert out.goal.position[0] > 3.0  # old best-view pose is behind the robot
    assert 0.6 <= math.dist(out.goal.position[:2], (7.0, 5.0)) <= 1.2
    assert math.isclose(out.goal.yaw(), math.atan2(5.0 - out.goal.position[1],
                                                  7.0 - out.goal.position[0]), abs_tol=1e-6)
    assert out.path and np.allclose(out.path[-1], out.goal.position[:2])
    arrived = tick(core, *out.goal.position[:2], 1.0)
    assert arrived.goal_kind == "hold" and [e.type for e in arrived.events] == ["reached"]


def test_live_goto_waits_when_no_free_approach_is_mapped():
    grid = OccupancyGrid2D.from_array(np.full((100, 100), UNKNOWN, np.int8), 0.1, (0.0, 0.0))
    core = core_for(grid)
    core.set_subtask(goto(5, 1.0, 5.0), object_xy=(7.0, 5.0), require_object_approach=True)
    out = tick(core, 3.0, 5.0, 0.0)
    assert out.goal_kind == "hold" and out.goal is None and out.path is None
    assert not out.events
    grid.set_array(np.full((100, 100), FREE, np.int8), t=1.0)
    assert tick(core, 3.0, 5.0, 1.0).goal_kind == "goto"


def test_live_goto_waits_if_selected_object_is_missing():
    grid = OccupancyGrid2D.from_array(np.full((100, 100), FREE, np.int8), 0.1, (0.0, 0.0))
    core = core_for(grid)
    core.set_subtask(goto(5, 1.0, 5.0), require_object_approach=True)
    out = tick(core, 3.0, 5.0, 0.0)
    assert out.goal_kind == "hold" and out.goal is None and not out.events


def test_ablations_and_room_bonus():
    core = core_for(gap_grid(), ablation={"nearest_frontier": True})
    out = tick(core, 6.0, 2.0, 0.0, dets=[det("sofa", 0.9, 120)], obs=[(8.0, 1.0)])
    assert out.goal_kind == "frontier" and not core.instances.points        # no IBTP, w_vl = 0
    core = core_for(gap_grid(), ablation={"no_curiosity": True})
    tick(core, 4.6, 2.0, 0.0, dets=[det("sofa", 0.9, 80)], obs=[None])
    for p in core.snapshot()["points"]:
        assert math.isclose(p["score"]["s_nesy"], 3.0 * p["s_vl"]) and p["score"]["s_unknown"] == 1.0
    g = gap_grid()
    core = ExplorationCore(cfg(**ZERO, scoring={"room_bonus": {"enabled": True, "w_room": 0.2}}), g, planner(),
                           room_labels_near=lambda xy: {"office"} if xy[0] > 5 else {"kitchen"})
    core.set_subtask(explore(1, room_hint="office"))
    out = tick(core, 4.9, 2.0, 0.0)
    assert np.allclose(out.goal.position[:2], RIGHT)
    assert {p["score"]["room_bonus"] for p in core.snapshot()["points"]} == {0.0, 0.2}


def test_exploration_never_imports_planner_memory_acquisition():
    src = Path(__file__).resolve().parents[2] / "vlnav" / "exploration"
    for f in src.glob("*.py"):
        text = f.read_text()
        for pkg in ("vlnav.planner", "vlnav.memory", "vlnav.acquisition"):
            assert pkg not in text, (f.name, pkg)


def test_debug_view_and_shadow_recorder(tmp_path):
    from vlnav.exploration.viz import ShadowRecorder, render
    core = core_for(gap_grid())
    recorder = ShadowRecorder(tmp_path / "exploration", core.cfg.viz)
    for t in range(3):
        tick(core, 6.0, 2.0, float(t), dets=[det("sofa", 0.5, 120)], obs=[(8.0, 1.0)])
        img, view = recorder.add(t, core.grid, core.snapshot(), np.array([[5.0, 2.0], [6.0, 2.0]]))
    assert img.shape == (900, 1300, 3)
    assert np.allclose(view.px((view.xmin, view.ymax)), (0, 0))      # top-left corner
    stats = recorder.close()
    out = tmp_path / "exploration"
    assert (out / "shadow.mp4").stat().st_size > 0 and (out / "final.png").exists()
    assert (out / "frames" / "frame_00000.png").exists() and len((out / "shadow_log.jsonl").read_text().splitlines()) == 3
    assert stats["steps"] == 3 and stats["goal_kinds"] == {"instance": 3}
    img2, _ = render(core.grid, core.snapshot(), core.cfg.viz)
    assert img2.shape == img.shape


def test_unreachable_approach_cells_do_not_starve_frontiers():
    g = gap_grid()
    arr = g.as_array().copy()
    arr[5:35, 60] = OCCUPIED                                        # wall + closed right gap:
    arr[15:25, 95] = OCCUPIED                                       # x > 6 m is unreachable
    g = OccupancyGrid2D.from_array(arr, 0.1, (0.0, 0.0))
    core = core_for(g, selection={"max_reach_checks": 3})
    out = tick(core, 4.0, 2.0, 0.0, dets=[det("sofa", 0.9, 80)], obs=[(7.5, 2.0)])    # approach ring behind it
    assert out.goal_kind == "frontier" and np.allclose(out.goal.position[:2], LEFT)
    assert core.snapshot()["reach_checks"] <= 2
    for t in (1.0, 2.0):
        tick(core, 4.0, 2.0, t)
    assert core.instances.points[0].status == "unreachable"


def test_debug_view_shows_the_task(tmp_path):
    import json
    from types import SimpleNamespace
    from vlnav.exploration.viz import ShadowRecorder, render_panel
    from vlnav.runtime.system_runtime import task_context
    st = SimpleNamespace(task_id=0, instruction="Throw away the cup, then find somewhere to relax.", fsm="EXECUTING",
                         done_reason=None, current=explore(2, target="sofa"),
                         targets=[SimpleNamespace(name="trash can", status="reported", node_id=0),
                                  SimpleNamespace(name="sofa", status="exploring", node_id=None)])
    task = task_context(st)
    assert task["current"] == {"type": "explore", "target": "sofa"} and task["targets"][0]["node_id"] == 0
    assert task_context(SimpleNamespace(task_id=-1)) is None and task_context(None) is None
    core = core_for(gap_grid())
    tick(core, 6.0, 2.0, 0.0)
    snap = core.snapshot()
    plain, with_task = render_panel(snap, core.cfg.viz, 900, 0.0, 0), render_panel(snap, core.cfg.viz, 900, 0.0, 0, task)
    assert (plain != with_task).any() and with_task.shape == plain.shape
    long = {**task, "targets": [{"name": f"t{i}", "status": "pending", "node_id": None} for i in range(40)]}
    render_panel(snap, core.cfg.viz, 300, 0.0, 0, long)              # overflow is clipped, no error
    rec = ShadowRecorder(tmp_path / "exploration", core.cfg.viz)
    rec.add(0, core.grid, snap, task=task)
    rec.close()
    line = json.loads((tmp_path / "exploration" / "shadow_log.jsonl").read_text().splitlines()[0])
    assert line["task"]["instruction"].startswith("Throw away")
