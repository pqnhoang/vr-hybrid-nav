"""Move M3: MoveCore loop with a scripted S2 (no server): goals, events, look-down, hysteresis."""
import math
import threading
import time

import numpy as np
from move_helpers import cam_T, cfg, frame, free_grid, planner, project

from vlnav.common.models import Pose, Subtask
from vlnav.mapping.occupancy import OCCUPIED, OccupancyGrid2D
from vlnav.move.core import MoveCore
from vlnav.move.s2_client import MockS2Client


def subtask(i=1):
    return Subtask(id=i, task_id=0, type="move", target="corridor", instruction="walk to the end of the corridor")


def pix(p_map) -> str:
    u, v = project(cam_T(), p_map)
    return f"{int(round(u))} {int(round(v))}"


def robot(x=0.0, y=0.0, yaw=0.0, t=0.0):
    return Pose.from_matrix(frame(x, y, yaw).T_map_base, t)


def core(script, grid=None, **over):
    m = MoveCore(cfg(**over), grid or free_grid(), planner(), MockS2Client(script))
    m.set_subtask(subtask())
    return m


def test_look_down_then_pixel_gives_goal_on_the_projected_floor_point():
    m = core(["↓", pix((3.0, 0.5, 0.0))])
    m.on_frame(frame())
    out = m.step(robot(), 0.0)
    reqs = m.s2.requests
    assert [r.look_down for r in reqs] == [False, True] and reqs[1].prev_output == "↓"
    assert out.goal_kind == "move" and np.linalg.norm(np.asarray(out.goal.position[:2]) - (3.0, 0.5)) < 0.1
    rec = m.pop_records()[0]
    assert rec["decision"] == "new_goal" and rec["projection"]["flags"] == ["no_depth"]
    assert out.path and np.allclose(out.path[-1], out.goal.position[:2], atol=0.06)


def test_stop_emits_reached_and_holds():
    m = core(["STOP"])
    m.on_frame(frame())
    out = m.step(robot(), 0.0)
    assert [e.type for e in out.events] == ["reached"] and out.events[0].payload["reason"] == "vln_stop"
    assert out.goal_kind == "hold" and m.done
    assert m.step(robot(), 1.0).events == [] and len(m.s2.requests) == 1


def test_unreachable_three_times_gives_nav_failed():
    arr = np.full((200, 200), OCCUPIED, np.int8)
    arr[95:105, 95:105] = 0
    g = OccupancyGrid2D.from_array(arr, 0.1, (-10.0, -10.0))
    m = core(["↓", pix((3.0, 0.0, 0.0))], grid=g, look_down_mode="ignore")
    m.s2.script = [pix((3.0, 0.0, 0.0))]                      # pixel on the forward query
    m.cfg.__dict__["main_pixel_ref"] = "view"
    m.on_frame(frame())
    events = []
    for k in range(5):
        events += m.step(robot(), float(k)).events
    assert [e.type for e in events] == ["nav_failed"] and len(m.s2.requests) == 3
    assert [r["reason"] for r in m.pop_records()] == ["unreachable"] * 3


def test_timeout_and_max_queries():
    m = core(["↓", pix((3.0, 0.0, 0.0))], loop={"timeout_s": 5.0})
    m.on_frame(frame())
    m.step(robot(), 0.0)
    assert [e.type for e in m.step(robot(), 6.0).events] == ["timeout"]
    m = core(["←"], loop={"max_queries": 2})
    m.on_frame(frame())
    ev = []
    for k, yaw in enumerate((0.0, 0.27, 0.54, 0.81)):
        ev += m.step(robot(yaw=yaw), float(k)).events
    assert [e.type for e in ev] == ["timeout"] and ev[0].payload["reason"] == "max_queries"


def test_hysteresis_keeps_goal_for_small_jitter_and_period_requery():
    m = core(["↓", pix((3.0, 0.0, 0.0)), "↓", pix((3.1, 0.1, 0.0)), "↓", pix((4.5, 1.5, 0.0))],
             loop={"requery_period_s": 2.0})
    m.on_frame(frame())
    first = m.step(robot(), 0.0)
    g0 = first.goal.position
    assert m.step(robot(), 1.0).goal.position == g0              # no query before the period
    out = m.step(robot(), 2.5)
    assert out.goal.position == g0 and m.pop_records()[-1]["decision"] == "kept"
    assert out.goal_id == first.goal_id                          # kept goal: not resent to navigation
    out = m.step(robot(), 5.0)
    assert np.linalg.norm(np.asarray(out.goal.position[:2]) - (4.5, 1.5)) < 0.1
    assert out.goal_id != first.goal_id


def test_rotation_then_requery_when_turned():
    m = core(["←←", "STOP"])
    m.on_frame(frame())
    out = m.step(robot(), 0.0)
    assert out.goal_kind == "move" and abs(out.goal.yaw() - math.radians(30)) < 1e-3
    assert out.goal.position[:2] == (0.0, 0.0)
    assert len(m.s2.requests) == 1 and m.step(robot(yaw=0.1), 1.0).events == []     # still turning
    out = m.step(robot(yaw=math.radians(30)), 2.0)
    assert [e.type for e in out.events] == ["reached"]


def test_reply_of_an_old_subtask_is_dropped():
    gate = threading.Event()

    def slow(req):
        gate.wait(2.0)
        return "STOP"
    m = core(slow, s2={"blocking": False})
    m.on_frame(frame())
    m.step(robot(), 0.0)
    assert m.in_flight
    m.set_subtask(subtask(2))                                    # new subtask while the old reply is pending
    m.on_frame(frame())
    m.step(robot(), 1.0)                                         # first cycle of subtask 2 (queued behind the old one)
    gate.set()
    events, deadline = [], time.time() + 3.0
    while not events and time.time() < deadline:
        time.sleep(0.05)
        events = m.step(robot(), 2.0).events
    assert [(e.type, e.subtask_id) for e in events] == [("reached", 2)]
    assert [r["k"] for r in m.pop_records()] == [1]              # only subtask 2's cycle was handled
    m.close()


def test_inactive_for_other_subtasks():
    m = MoveCore(cfg(), free_grid(), planner(), MockS2Client(["STOP"]))
    m.set_subtask(Subtask(id=1, task_id=0, type="explore", target="sofa"))
    m.on_frame(frame())
    assert m.step(robot(), 0.0).goal_kind == "hold" and m.s2.requests == []
