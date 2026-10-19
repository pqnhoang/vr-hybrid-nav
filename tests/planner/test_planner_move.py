"""Move subtasks in the planner (TASK_move_subtask.md §10.2): schema, prompts, validation, FSM."""
from planner_helpers import Router, event, explore, goto, make_planner, memory, move, plan, target, verify

from vlnav.planner.decomposer import DECOMPOSE_SCHEMA, REPLAN_SCHEMA, with_move
from vlnav.planner.state import PlannerState
from vlnav.planner.validation import validate_decomposition, validate_replan

ROUTE = "Walk down the corridor and stop at the glass door."
MIXED = "Go down the corridor to the glass door, then find an umbrella."


def replan(subtasks, done=False):
    return {"reasoning": "r", "task_done": done, "target_updates": [], "subtasks": subtasks}


def test_schema_only_changes_when_enabled():
    assert DECOMPOSE_SCHEMA["properties"]["subtasks"]["items"]["properties"]["type"]["enum"] == ["explore", "goto"]
    for schema in (DECOMPOSE_SCHEMA, REPLAN_SCHEMA):
        item = with_move(schema)["properties"]["subtasks"]["items"]
        assert "move" in item["properties"]["type"]["enum"] and "instruction" in item["properties"]
        assert with_move(schema)["required"] == schema["required"]


def test_validation_of_move_subtasks(tmp_path):
    mem = memory(tmp_path)
    data = plan([], [move("glass door", ROUTE), move("x", "  "), {"type": "move", "instruction": "turn left",
                                                                   "node_id": 21}])
    p = validate_decomposition(data, mem, allow_move=True)
    assert p.targets == []                                       # routes are not targets
    assert [(s.type, s.target, s.instruction) for s in p.subtasks] == [
        ("move", "glass door", ROUTE), ("move", "route", "turn left")]
    assert any("without an instruction" in f for f in p.fixes) and any("node_id 21" in f for f in p.fixes)
    off = validate_decomposition(data, mem)
    assert off.subtasks == [] and off.targets == [] and all("disabled" in f for f in off.fixes)
    hist = [{"type": "move", "target": "glass door", "node_id": None, "outcome": o} for o in ("timeout", "nav_failed")]
    st = PlannerState(task_id=0, instruction=ROUTE, fsm="REPLANNING", targets=[], current=None, history=hist,
                      started_t=0.0)
    r = validate_replan(replan([move("glass door", ROUTE)]), st, mem, allow_move=True, max_move_attempts=2)
    assert r.subtasks == [] and any("failed 2 times" in f for f in r.fixes)


def test_route_only_task_moves_then_done(tmp_path):
    r = Router(decompose=[plan([], [move("glass door", ROUTE)])], replan=[replan([], done=True)])
    p, _ = make_planner(tmp_path, r, move=True)
    p.submit_task(ROUTE, 0.0)
    (st,) = p.pop_dispatched()
    assert (st.type, st.target, st.instruction, st.detector_prompts) == ("move", "glass door", ROUTE, [])
    assert p.state().targets == [] and "6. Routes:" in r.log[0][1]
    p.on_event(event(p, "reached", 30.0, reason="vln_stop"))
    s = p.state()
    assert s.fsm == "DONE" and s.done_reason == "planner: task done" and p.pop_reports() == []
    assert s.history[0]["instruction"] == ROUTE and s.history[0]["outcome"] == "reached"
    assert '("Walk down the corridor and stop at the glass door.") -> reached' in r.log[1][1]
    assert p.pop_dispatched()[-1].type == "stop"


def test_route_then_target(tmp_path):
    r = Router(decompose=[plan([target("umbrella")], [move("glass door", ROUTE), explore("umbrella")])],
               replan=[replan([explore("umbrella")])])
    p, _ = make_planner(tmp_path, r, move=True)
    p.submit_task(MIXED, 0.0)
    assert p.pop_dispatched()[0].type == "move"
    assert p.state().targets[0].status == "pending"              # untouched by the move
    p.on_event(event(p, "reached", 20.0))
    st = p.pop_dispatched()[0]
    assert (st.type, st.target) == ("explore", "umbrella") and p.state().fsm == "EXECUTING"


def test_failed_route_without_targets_fails_the_task(tmp_path):
    r = Router(decompose=[plan([], [move("glass door", ROUTE)])], replan=[replan([move("glass door", ROUTE)]),
                                                                           replan([])])
    p, _ = make_planner(tmp_path, r, move=True)
    p.submit_task(ROUTE, 0.0)
    p.pop_dispatched()
    p.on_event(event(p, "nav_failed", 10.0))
    assert p.pop_dispatched()[0].type == "move"                  # one retry
    p.on_event(event(p, "timeout", 20.0))
    s = p.state()
    assert s.fsm == "FAILED" and s.done_reason == "route 'glass door' ended with timeout"


def test_disabled_planner_keeps_old_prompt_and_drops_moves(tmp_path):
    r = Router(decompose=[plan([target("umbrella")], [move("glass door", ROUTE), explore("umbrella")])])
    p, _ = make_planner(tmp_path, r)
    p.submit_task(MIXED, 0.0)
    assert "Routes" not in r.log[0][1]
    (st,) = p.pop_dispatched()
    assert (st.type, st.target) == ("explore", "umbrella")


def test_route_landmarks_are_not_targets(tmp_path):
    instr = "Walk past the beige sofa and stop near the high bar chairs, then take me to an armchair."
    data = plan([target("beige sofa"), target("high bar chairs"), target("armchair"), target("trash can")],
                [explore("beige sofa"), move("bar chairs", "Walk past the beige sofa and stop near the high bar chairs"),
                 goto("high bar chairs", 21), explore("armchair")])
    p = validate_decomposition(data, memory(tmp_path), allow_move=True, instruction=instr)
    assert [t.name for t in p.targets] == ["armchair", "trash can"]      # implicit target (not in the text) kept
    assert [(s.type, s.target) for s in p.subtasks] == [("move", "bar chairs"), ("explore", "armchair")]
    assert sum("landmark" in f for f in p.fixes) == 2
    no_route = validate_decomposition(plan([target("sofa")], [explore("sofa")]), memory(tmp_path), True,
                                      "Walk past the sofa.")
    assert [t.name for t in no_route.targets] == ["sofa"]                # rule only applies with a route


def test_route_after_the_last_target_is_planned_by_the_replanner(tmp_path):
    instr = "Find the umbrella, then walk to the glass door."
    r = Router(decompose=[plan([target("umbrella")], [goto("umbrella", 21), move("glass door", "walk to the glass door")])],
               replan=[replan([move("glass door", "walk to the glass door")]), replan([], done=True)],
               verify=[verify("A")])
    p, _ = make_planner(tmp_path, r, move=True)
    p.submit_task(instr, 0.0)
    assert p.pop_dispatched()[0].type == "goto"
    p.on_event(event(p, "reached", 10.0))
    assert p.pop_reports()[0].target == "umbrella"
    st = p.pop_dispatched()[0]                                   # all targets reported, but the route is left
    assert (st.type, st.instruction) == ("move", "walk to the glass door")
    p.on_event(event(p, "reached", 30.0))
    assert p.state().fsm == "DONE"


def test_move_that_searches_a_target_becomes_explore(tmp_path):
    instr = "Walk past the beige sofa and stop near the bar chairs, then take me to an armchair."
    data = plan([target("armchair")], [move("bar chairs", "Walk past the beige sofa and stop near the bar chairs"),
                                       move("to armchair", "Take me to an armchair")])
    p = validate_decomposition(data, memory(tmp_path), allow_move=True, instruction=instr)
    assert [(s.type, s.target) for s in p.subtasks] == [("move", "bar chairs"), ("explore", "armchair")]
    data["subtasks"].append(explore("armchair"))                  # already planned -> the move is dropped
    p = validate_decomposition(data, memory(tmp_path), allow_move=True, instruction=instr)
    assert [(s.type, s.target) for s in p.subtasks] == [("move", "bar chairs"), ("explore", "armchair")]
    st = PlannerState(task_id=0, instruction=instr, fsm="REPLANNING", targets=p.targets, current=None,
                      history=[], started_t=0.0)
    r = validate_replan(replan([move("armchair", "go to the armchair")]), st, memory(tmp_path), allow_move=True)
    assert [(s.type, s.target) for s in r.subtasks] == [("explore", "armchair")]
