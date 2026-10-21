"""M2-M4: ProgramRuntime with scripted agents and events. The three task types: follow -> move
(DualVLN), locate of a known object -> goto after VLM verification, unknown -> explore."""
import json

from agent_helpers import (AgentRouter, code, decisions, event, make_runtime, observe, plan, verify)

ROUTE = "Go forward to the cartoon box. Turn right at the box and find the umbrella."
ROUTE_PROG = code('''
# Step 1: route
if not api.follow("Go forward to the cartoon box and turn right at the box."):
    return api.fail("route failed")
# Step 2: umbrella
u = api.locate("umbrella", prompts=["umbrella"])
if u is None:
    return api.fail("no umbrella")
if api.go_to(u):
    api.report(u)
''')


def test_route_then_known_object(tmp_path):
    r = AgentRouter(planner=[plan('Step 1: follow "Go forward to the cartoon box and turn right at the box."',
                                  "Step 2: locate the umbrella, go to it and report it.")],
                    coder=[ROUTE_PROG], observer=[observe()], verify=[verify("A")])
    rt, mem = make_runtime(tmp_path, r, move=True)
    rt.submit_task(ROUTE, 0.0)
    (st,) = rt.pop_dispatched()
    assert (st.type, st.instruction) == ("move", "Go forward to the cartoon box and turn right at the box.")
    assert rt.state().fsm == "EXECUTING" and [t.name for t in rt.state().targets] == ["umbrella"]
    rt.on_event(event(rt, "reached", 10.0))                 # DualVLN STOP
    (st2,) = rt.pop_dispatched()                            # verified from the scene graph: no explore
    assert (st2.type, st2.target, st2.node_id) == ("goto", "umbrella", 21)
    assert st2.goal_pose == mem.get_object(21).best_view_pose and st2.detector_prompts == ["umbrella"]
    rt.on_event(event(rt, "reached", 20.0))
    (rep,) = rt.pop_reports()
    assert (rep.target, rep.node_id, rep.label) == ("umbrella", 21, "umbrella")
    s = rt.state()
    assert s.fsm == "DONE" and s.done_reason == "program finished" and s.targets[0].status == "reported"
    assert [h["type"] for h in s.history] == ["move", "goto"]
    assert rt.pop_dispatched()[-1].type == "stop"
    assert r.kinds() == ["planner", "coder", "observer", "verify"]
    lines = decisions(tmp_path)
    dec = [x for x in lines if x["kind"] == "decompose"][-1]
    assert [(p["type"], p["target"]) for p in dec["plan"]] == [
        ("move", "Go forward to the cartoon box and turn right at the box."), ("goto", "umbrella")]
    assert sum(1 for x in lines if x.get("query", {}).get("raw")) == 3       # agent calls (metrics)
    rounds = [json.loads(x) for x in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert len(rounds) == 1 and rounds[0]["valid"] and rounds[0]["trace"][0].startswith("L3: follow(")


def test_explore_candidate_verify_goto(tmp_path):
    prog = code('''
u = api.explore("umbrella")
if u is None:
    return api.fail("no umbrella")
if api.go_to(u):
    api.report(u)
''')
    r = AgentRouter(planner=[plan("Step 1: explore for an umbrella, go to it and report it.")], coder=[prog],
                    observer=[observe()], verify=[verify("A")])
    rt, _ = make_runtime(tmp_path, r)
    rt.submit_task("Find an umbrella.", 0.0)
    (st,) = rt.pop_dispatched()
    assert (st.type, st.target) == ("explore", "umbrella") and rt.state().targets[0].status == "exploring"
    assert r.kinds() == ["planner", "coder", "observer"]     # explore() never verifies known objects
    (cand,) = rt.poll_internal(1.0)                          # umbrella #21 scores 0.40 >= 0.35
    assert cand.type == "candidate_found" and cand.payload["node_id"] == 21
    rt.on_event(cand)
    (st2,) = rt.pop_dispatched()
    assert (st2.type, st2.node_id) == ("goto", 21) and r.kinds()[-1] == "verify"
    assert rt.state().targets[0].status == "verified"
    rt.on_event(event(rt, "reached", 5.0))
    assert rt.pop_reports()[0].node_id == 21 and rt.state().fsm == "DONE"


def test_rejected_candidate_explores_on_and_timeouts(tmp_path):
    prog = code('''
u = api.locate("umbrella", timeout=30)
if u is None:
    return api.fail("no umbrella")
''')
    r = AgentRouter(planner=[plan("Step 1: locate an umbrella.")], coder=[prog], observer=[observe()],
                    verify=[verify("none"), verify("none")])
    rt, _ = make_runtime(tmp_path, r, agent={"max_repairs": 0})
    rt.submit_task("Find an umbrella.", 0.0)
    (st,) = rt.pop_dispatched()                              # the known umbrella was rejected -> explore
    assert st.type == "explore" and rt.blacklisted("umbrella") == {21}
    assert rt.poll_internal(1.0) == []                       # blacklisted: no candidate
    (to,) = rt.poll_internal(31.0)
    assert to.type == "timeout"
    rt.on_event(to)
    s = rt.state()
    assert s.fsm == "FAILED" and "no repairs left" in s.done_reason and s.history[-1]["outcome"] == "timeout"


def test_failed_program_is_repaired(tmp_path):
    first = code('''
u = api.locate("umbrella")
if not api.go_to(u):
    return api.fail("could not reach the umbrella")
api.report(u)
''')
    second = code('''
u = api.explore("umbrella")
if u and api.go_to(u):
    api.report(u)
''')
    r = AgentRouter(planner=[plan("Step 1: locate the umbrella, go to it, report it."),
                             plan("Step 1: explore for another umbrella, go to it, report it.")],
                    coder=[first, second],
                    observer=[observe(), observe(summary="the known umbrella was unreachable: explore"), observe()],
                    verify=[verify("A")])
    rt, _ = make_runtime(tmp_path, r)
    rt.submit_task("Find an umbrella.", 0.0)
    assert rt.pop_dispatched()[0].node_id == 21
    rt.on_event(event(rt, "nav_failed", 5.0))
    (st,) = rt.pop_dispatched()
    assert st.type == "explore" and rt.state().fsm == "EXECUTING"
    assert r.kinds() == ["planner", "coder", "observer", "verify", "observer", "planner", "coder", "observer"]
    repair_prompt = [t for k, t in r.log if k == "planner"][1]
    assert "could not reach the umbrella" in repair_prompt and "unreachable: explore" in repair_prompt
    assert "go_to(umbrella) -> nav_failed" in repair_prompt
    kinds = [x["kind"] for x in decisions(tmp_path)]
    assert "replan" in kinds and kinds.count("decompose") == 1


def test_observer_and_hard_checks_send_feedback_to_planner(tmp_path):
    no_follow = code('''
u = api.locate("umbrella")
if api.go_to(u):
    api.report(u)
''')
    r = AgentRouter(planner=[plan("Step 1: locate the umbrella."),
                             plan('Step 1: follow "Go forward to the cartoon box and turn right at the box."',
                                  "Step 2: locate the umbrella, go to it and report it.")],
                    coder=["no code at all", no_follow, ROUTE_PROG],
                    observer=[observe(valid=True), observe()], verify=[])
    rt, _ = make_runtime(tmp_path, r, move=True)
    rt.submit_task(ROUTE, 0.0)
    (st,) = rt.pop_dispatched()
    assert st.type == "move"
    coder_prompts = [t for k, t in r.log if k == "coder"]
    assert "Your previous program was rejected" in coder_prompts[1]
    second_plan = [t for k, t in r.log if k == "planner"][1]
    assert "never calls api.follow()" in second_plan                 # hard check overrides VALID
    rounds = [json.loads(x) for x in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert [x["valid"] for x in rounds] == [False, True]


def test_no_valid_program_fails_without_fallback(tmp_path):
    r = AgentRouter(planner=["nope", "nope", "nope", "nope", "nope", "nope"])
    rt, _ = make_runtime(tmp_path, r)
    rt.submit_task("Find an umbrella.", 0.0)
    s = rt.state()
    assert s.fsm == "FAILED" and "no valid program in 3 rounds" in s.done_reason
    assert rt.pop_dispatched()[-1].type == "stop"


def test_preemption_cancels_program(tmp_path):
    prog = code('''
api.follow("Go forward.")
api.follow("Turn left.")
''')
    r = AgentRouter(planner=[plan('Step 1: follow "Go forward."', 'Step 2: follow "Turn left."')] * 2,
                    coder=[prog, prog], observer=[observe(), observe()])
    rt, _ = make_runtime(tmp_path, r, move=True)
    rt.submit_task("Go forward and turn left.", 0.0)
    (st,) = rt.pop_dispatched()
    rt.submit_task("Go forward and turn left.", 3.0)
    out = rt.pop_dispatched()
    assert [x.type for x in out] == ["stop", "move"] and rt.state().task_id == 1
    rt.on_event(event(rt, "reached", 4.0).model_copy(update={"subtask_id": st.id}))   # stale: ignored
    assert rt.state().current.id == out[1].id
    rt.on_event(event(rt, "reached", 5.0))
    assert rt.pop_dispatched()[0].instruction == "Turn left."
    rt.on_event(event(rt, "abort", 6.0))
    assert rt.state().fsm == "FAILED" and rt.pop_dispatched()[-1].type == "stop"
    rt.close()


def test_non_blocking_mode(tmp_path):
    r = AgentRouter(planner=[plan("Step 1: locate the umbrella, go to it and report it.")],
                    coder=[code('u = api.locate("umbrella")\nif api.go_to(u):\n    api.report(u)')],
                    observer=[observe()], verify=[verify("A")])
    rt, _ = make_runtime(tmp_path, r, blocking=False)
    rt.submit_task("Find an umbrella.", 0.0)
    rt.wait_idle(10)
    (st,) = rt.pop_dispatched()
    assert st.type == "goto" and not rt.busy()
    rt.on_event(event(rt, "reached", 5.0))
    rt.wait_idle(10)
    assert rt.state().fsm == "DONE" and rt.pop_reports()[0].node_id == 21
    rt.close()


def test_no_valid_program_falls_back_to_nesy(tmp_path):
    nesy = {"reasoning": "r", "targets": [{"name": "umbrella", "detector_prompts": ["umbrella"], "room_priors": []}],
            "subtasks": [{"type": "explore", "target": "umbrella", "node_id": None, "room_hint": None}]}
    r = AgentRouter(planner=["nope"] * 6, nesy=[nesy])
    rt, _ = make_runtime(tmp_path, r, fallback=True)
    rt.submit_task("Find an umbrella.", 0.0)
    (st,) = rt.pop_dispatched()                              # dispatched by the NeSy PlannerCore
    assert (st.type, st.target) == ("explore", "umbrella")
    s = rt.state()
    assert s.fsm == "EXECUTING" and s.targets[0].name == "umbrella" and rt.fallback.state().task_id == 0
    (cand,) = rt.poll_internal(1.0)
    assert cand.type == "candidate_found"
    kinds = [x["kind"] for x in decisions(tmp_path)]
    assert kinds.index("deliberation_failed") < kinds.index("fallback") < kinds.index("decompose")
    # a new task leaves the fallback and deliberates again
    r.q["planner"] += [plan("Step 1: locate the umbrella.")]
    r.q["coder"] += [code('u = api.explore("umbrella")\nif u and api.go_to(u):\n    api.report(u)')]
    r.q["observer"] += [observe()]
    rt.submit_task("Find an umbrella.", 2.0)
    assert [x.type for x in rt.pop_dispatched()] == ["stop", "explore"]
    assert rt.state().task_id == 1 and rt.fallback.state().fsm == "FAILED"


def test_queries_check_and_room_goal(tmp_path):
    prog = code('''
rooms = api.rooms("entryway")
chairs = api.find("chair", room="bedroom")
u = api.locate("umbrella")
if api.check("Is it open?", u) is not True:
    return api.fail("closed: " + str(len(chairs)) + " " + api.side(u))
api.go_to(rooms[0])
''')
    r = AgentRouter(planner=[plan("Step 1: check the umbrella, then go to the entryway.")], coder=[prog],
                    observer=[observe(summary="fine")], verify=[verify("A")], check=[{"answer": "no"}])
    rt, _ = make_runtime(tmp_path, r, agent={"max_repairs": 0})
    rt.submit_task("If the umbrella is open go to the entryway.", 0.0)
    s = rt.state()
    assert s.fsm == "FAILED" and "closed: 1 ahead" in s.done_reason
    assert r.kinds() == ["planner", "coder", "observer", "verify", "check"]
    steps = [x["step"] for x in decisions(tmp_path) if x["kind"] == "program_step"]
    assert steps[-2].endswith("check('Is it open?') -> no")


def test_program_view_and_image(tmp_path):
    from vlnav.agent.validate import extract_code
    from vlnav.agent.viz import render_program
    r = AgentRouter(planner=[plan('Step 1: follow "Go forward to the cartoon box and turn right at the box."',
                                  "Step 2: locate the umbrella, go to it and report it.")],
                    coder=[ROUTE_PROG], observer=[observe()], verify=[verify("A")])
    rt, _ = make_runtime(tmp_path, r, move=True)
    assert rt.program_view() is None
    rt.submit_task(ROUTE, 0.0)
    v = rt.program_view()
    assert v["program"] == extract_code(ROUTE_PROG) and v["rounds"] == 1 and v["planner"] == "cap"
    assert v["current_line"] == 3 and v["steps"] == [] and v["subtask"].startswith("move ")
    assert rt.program_view() == v                            # unchanged -> equal (the node republishes on change)
    rt.on_event(event(rt, "reached", 10.0))
    v2 = rt.program_view()
    assert v2["current_line"] == 9 and v2["steps"][0] == (3, True, "reached")    # go_to(u) runs now
    img = render_program(v2)
    assert img.ndim == 3 and img.shape[1] == 960 and img.shape[0] > 22 * 12
    rt.on_event(event(rt, "reached", 20.0))
    assert rt.program_view()["fsm"] == "DONE" and rt.program_view()["current_line"] is None
