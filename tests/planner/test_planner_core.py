"""P5: PlannerCore FSM with scripted VLM replies and events; P4 decomposer details."""
import json

from planner_helpers import (INSTR, Router, event, explore, goto, make_planner, memory, plan, target,
                             text_of, verify)

UMBRELLA, JACKETS = 21, (17, 19, 18)
DECOMP = plan([target("umbrella"), target("rain jacket", ["raincoat", "jacket"])],
              [goto("umbrella", UMBRELLA), explore("rain jacket")])


def replan(subtasks, done=False, updates=()):
    return {"reasoning": "r", "task_done": done, "target_updates": list(updates), "subtasks": subtasks}


def test_multi_target_task_to_done(tmp_path):
    r = Router(decompose=[DECOMP], replan=[replan([goto("rain jacket", 17)])],
               verify=[verify("A"), verify("B")])
    p, mem = make_planner(tmp_path, r)
    p.submit_task(INSTR, 0.0)
    (st,) = p.pop_dispatched()
    assert (st.type, st.target, st.node_id) == ("goto", "umbrella", UMBRELLA)
    assert st.goal_pose == mem.get_object(UMBRELLA).best_view_pose and st.detector_prompts == ["umbrella"]
    p.on_event(event(p, "reached", 10.0))
    (rep,) = p.pop_reports()
    assert (rep.target, rep.node_id, rep.label) == ("umbrella", UMBRELLA, "umbrella")
    (st2,) = p.pop_dispatched()
    # replanner proposed #17, acquisition verified among [17, 19, 18] and the VLM chose B -> 19
    assert (st2.type, st2.target, st2.node_id) == ("goto", "rain jacket", 19)
    assert st2.detector_prompts == ["raincoat", "jacket"]
    p.on_event(event(p, "reached", 20.0))
    assert p.pop_reports()[0].node_id == 19
    s = p.state()
    assert s.fsm == "DONE" and [t.status for t in s.targets] == ["reported", "reported"]
    assert [t.notes for t in s.targets] == ["reported: reached #21", "reported: reached #19"]
    assert p.pop_dispatched()[-1].type == "stop"
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert kinds == ["new_task", "decompose", "report", "subtask_end", "replan", "report", "subtask_end", "finish"]


def test_goto_failure_blacklists_and_replans(tmp_path):
    r = Router(decompose=[plan([target("rain jacket", ["jacket"])], [goto("rain jacket", 17)])],
               replan=[replan([goto("rain jacket", 17)])], verify=[verify("A"), verify("A")])
    p, _ = make_planner(tmp_path, r)
    p.submit_task(INSTR, 0.0)
    assert p.pop_dispatched()[0].node_id == 17
    p.on_event(event(p, "nav_failed", 5.0))
    st = p.pop_dispatched()[0]
    assert st.node_id == 19 and p.acquirer.blacklisted("rain jacket") == {17}
    assert p.state().targets[0].attempts == 1


def test_explore_candidate_check_and_max_attempts(tmp_path):
    r = Router(decompose=[plan([target("umbrella")], [explore("umbrella")])],
               replan=[replan([explore("umbrella")])] * 3)
    p, _ = make_planner(tmp_path, r, max_attempts_per_target=2, explore_candidate_score=0.35)
    p.submit_task(INSTR, 0.0)
    assert p.pop_dispatched()[0].type == "explore"
    (cand,) = p.poll_internal(1.0)                           # umbrella #21 scores 0.40
    assert cand.type == "candidate_found" and cand.payload["node_id"] == UMBRELLA
    p.on_event(cand)
    assert p.pop_dispatched()[0].type == "explore"          # replanner chose explore again
    assert p.poll_internal(5.0) == []                       # same candidate does not re-trigger
    p.on_event(event(p, "no_frontier", 6.0))
    p.pop_dispatched()
    p.on_event(event(p, "no_frontier", 7.0))
    s = p.state()
    assert s.fsm == "DONE" and s.targets[0].status == "failed" and s.targets[0].attempts == 2


def test_stale_events_abort_and_timeouts(tmp_path):
    r = Router(decompose=[DECOMP, DECOMP], verify=[verify("A"), verify("A")])
    p, _ = make_planner(tmp_path, r, task_timeout_s=100, goto_timeout_s=30)
    p.submit_task(INSTR, 0.0)
    st = p.pop_dispatched()[0]
    p.on_event(event(p, "reached", 1.0).model_copy(update={"subtask_id": st.id + 7}))
    assert p.state().current.id == st.id and p.pop_reports() == []
    (to,) = p.poll_internal(31.0)
    assert to.type == "timeout"
    p.on_event(event(p, "abort", 32.0))
    assert p.state().fsm == "FAILED" and p.pop_dispatched()[-1].type == "stop"
    p.submit_task(INSTR, 40.0)
    p.pop_dispatched()
    assert p.poll_internal(141.0) == [] and p.state().fsm == "FAILED"
    assert p.state().done_reason == "task timeout"


def test_decomposition_and_replan_json_failures(tmp_path):
    p, _ = make_planner(tmp_path, Router(decompose=["nope", "still nope"]))
    p.submit_task(INSTR, 0.0)
    s = p.state()
    assert s.fsm == "FAILED" and "malformed JSON twice" in s.done_reason
    r = Router(decompose=[plan([target("umbrella"), target("shoes")], [explore("umbrella")])],
               replan=["x", "y"])
    p2, _ = make_planner(tmp_path / "b", r)
    p2.submit_task(INSTR, 0.0)
    p2.pop_dispatched()
    p2.on_event(event(p2, "no_frontier", 3.0))
    st = p2.pop_dispatched()[0]
    assert (st.type, st.target) == ("explore", "umbrella")  # fallback: pending targets in order


def test_async_mode_and_stale_result(tmp_path):
    import threading
    gate = threading.Event()
    calls = []

    def slow(messages):
        calls.append(text_of(messages))
        gate.wait(5)
        return json.dumps(plan([target("umbrella")], [explore("umbrella")]))
    from vlnav.acquisition.vlm_client import MockVLMClient
    p, _ = make_planner(tmp_path, Router(), blocking=False)
    p.decomposer.vlm = MockVLMClient(slow)
    p.submit_task(INSTR, 0.0, task_id=0)
    assert p.busy() and p.pop_dispatched() == []
    p.submit_task("second task", 1.0, task_id=1)            # job 0 still running -> its result is stale
    gate.set()
    p.wait_idle(5)
    p.poll_internal(2.0)
    p.wait_idle(5)
    sts = p.pop_dispatched()
    assert [s.task_id for s in sts if s.type == "explore"] == [1]
    assert p.state().task_id == 1 and p.state().fsm == "EXECUTING"
    p.close()


def test_decomposer_attaches_planning_images(tmp_path):
    r = Router(decompose=[DECOMP], verify=[verify("A")])
    p, _ = make_planner(tmp_path, r, planning_images=2)
    p.submit_task(INSTR, 0.0)
    msg = p.decomposer.vlm.calls[0]["messages"][0]["content"]
    assert sum(x["type"] == "image_url" for x in msg) == 2
    assert any(x["type"] == "text" and x["text"].startswith("Image of #") for x in msg)
    assert INSTR in msg[0]["text"] and "ROOMS" in msg[0]["text"]
    r0 = Router(decompose=[DECOMP], verify=[verify("A")])
    p0, _ = make_planner(tmp_path / "noimg", r0, planning_images=0)
    p0.submit_task(INSTR, 0.0)
    assert all(x["type"] == "text" for x in p0.decomposer.vlm.calls[0]["messages"][0]["content"])


def test_task_done_ignored_while_targets_active(tmp_path):
    r = Router(decompose=[plan([target("umbrella"), target("shoes")], [explore("umbrella")])],
               replan=[replan([], done=True)])
    p, _ = make_planner(tmp_path, r)
    p.submit_task(INSTR, 0.0)
    p.pop_dispatched()
    (cand,) = p.poll_internal(1.0)
    p.on_event(cand)
    s = p.state()
    assert s.fsm == "EXECUTING" and s.current.type == "explore"
    assert s.targets[0].notes.startswith("candidate #21 (umbrella, score 0.4) seen")


def test_rejected_candidates_raise_the_retrigger_floor(tmp_path):
    r = Router(decompose=[plan([target("rain jacket", ["jacket"])], [explore("rain jacket")])],
               replan=[replan([goto("rain jacket", 17)])], verify=[verify("none"), verify("none")])
    p, mem = make_planner(tmp_path, r, explore_candidate_score=0.2)
    p.submit_task(INSTR, 0.0)
    p.pop_dispatched()
    (cand,) = p.poll_internal(1.0)
    p.on_event(cand)                                   # replan -> goto 17 -> VLM rejects 2 rounds
    st = p.pop_dispatched()[0]
    assert st.type == "explore" and p._reject_floor["rain jacket"] == 0.62
    assert p.poll_internal(10.0) == []                 # coat #20 (0.30) is not stronger evidence


def test_replan_prompt_lists_one_target_per_line(tmp_path):
    from planner_helpers import matcher
    from vlnav.planner.config import load_planner_config
    from vlnav.planner.decomposer import Replanner
    from vlnav.planner.serializer import GraphSerializer
    from vlnav.planner.state import PlannerState, TargetEntry
    from vlnav.acquisition.vlm_client import MockVLMClient
    cfg = load_planner_config(overrides={"planning_images": 0})
    vlm = MockVLMClient(json.dumps({"reasoning": "r", "task_done": False, "subtasks": []}))
    st = PlannerState(task_id=0, instruction="x", fsm="REPLANNING", current=None, started_t=0.0,
                      targets=[TargetEntry(name="trash can", detector_prompts=["bin"], status="reported", node_id=0,
                                           notes="reported: reached #0"),
                               TargetEntry(name="sofa", detector_prompts=["sofa"], notes="candidate #4 seen")],
                      history=[{"type": "explore", "target": "sofa", "node_id": None, "outcome": "candidate_found"}])
    q = Replanner(cfg, vlm, GraphSerializer(cfg, matcher())).replan(st, "candidate_found", memory(tmp_path))
    lines = q.prompt_text.splitlines()
    assert "- trash can: reported (memory #0), failed attempts 0; reported: reached #0" in lines
    assert "- sofa: pending, failed attempts 0; candidate #4 seen" in lines
    assert "1. explore \"sofa\" -> candidate_found" in lines and "Finished subtasks (oldest first):" in lines
