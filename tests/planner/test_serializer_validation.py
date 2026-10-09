"""P4: serializer token budget / ordering, plan validation and repair rules."""
from planner_helpers import explore, goto, matcher, memory, plan, target

from vlnav.planner.config import load_planner_config
from vlnav.planner.serializer import ExplorationSummary, GraphSerializer
from vlnav.planner.state import PlannerState, TargetEntry
from vlnav.planner.validation import fallback_plan, validate_decomposition, validate_replan


def ser(**over):
    return GraphSerializer(load_planner_config(overrides=over), matcher())


def test_serializer_rooms_relevance_and_ids(tmp_path):
    txt = ser().serialize(memory(tmp_path), ["rain jacket"], ExplorationSummary(142.0, True))
    lines = txt.splitlines()
    assert lines[0] == "ROOMS"
    assert lines[1].startswith("R0 bedroom (0.0 m2) objects: jacket[#17 .62], raincoat[#18 .55], chair[#22 .80]")
    assert lines[2].startswith("R1 hallway (0.0 m2) objects: jacket[#19 .62], coat[#20 .30], jacket[#24 .10]")
    assert lines[-1] == "EXPLORATION: known free area 142 m2; frontiers remaining: yes"


def test_serializer_budget_truncates_least_relevant(tmp_path):
    txt = ser(max_graph_tokens=60).serialize(memory(tmp_path), ["rain jacket"])
    assert "jacket[#17 .62]" in txt and "+" in txt and "more" in txt
    assert "umbrella[#21" not in txt                    # irrelevant objects go first
    assert len(txt) <= 60 * 4


def test_serializer_without_rooms(tmp_path):
    from vlnav.acquisition.memory_api import InMemoryMemory
    mem = memory(tmp_path)
    flat = InMemoryMemory(mem.all_objects())
    assert ser().serialize(flat, []).startswith("OBJECTS: chair[#22 .80], desk[#23 .70]")


def test_decomposition_repairs(tmp_path):
    mem = memory(tmp_path)
    data = plan([target("umbrella"), target("Umbrella"), {"name": ""}, target("rain jacket", ["  raincoat ", "raincoat", "jacket"])],
                [goto("umbrella", 21), goto("rain jacket", 4242), {"type": "goto", "target": "rain jacket"},
                 explore("shoes"), {"type": "fly", "target": "umbrella"}])
    p = validate_decomposition(data, mem)
    assert [t.name for t in p.targets] == ["umbrella", "rain jacket", "shoes"]
    assert p.targets[1].detector_prompts == ["raincoat", "jacket"]
    assert [(s.type, s.target, s.node_id) for s in p.subtasks] == [
        ("goto", "umbrella", 21), ("explore", "rain jacket", None), ("explore", "rain jacket", None),
        ("explore", "shoes", None), ("explore", "umbrella", None)]
    assert len(p.fixes) == 6


def state(**kw):
    t = [TargetEntry(name="umbrella", detector_prompts=["umbrella"], status="reported"),
         TargetEntry(name="rain jacket", detector_prompts=["raincoat"]),
         TargetEntry(name="shoes", detector_prompts=["shoes"])]
    return PlannerState(task_id=0, instruction="x", fsm="REPLANNING", targets=t, current=None,
                        history=[], started_t=0.0, **kw)


def test_replan_repairs(tmp_path):
    mem = memory(tmp_path)
    data = {"reasoning": "r", "task_done": False,
            "target_updates": [{"name": "shoes", "status": "failed", "reason": "no shoes here"},
                               {"name": "hat", "status": "failed"}],
            "subtasks": [goto("umbrella", 21), explore("hat"), goto("shoes", 17), goto("rain jacket", 19)]}
    p = validate_replan(data, state(), mem)
    assert [(s.type, s.target, s.node_id) for s in p.subtasks] == [("goto", "rain jacket", 19)]
    assert {t.name: t.status for t in p.targets} == {"umbrella": "reported", "rain jacket": "pending",
                                                     "shoes": "failed"}
    assert p.targets[2].notes == "no shoes here"


def test_empty_plan_synthesizes_explore_and_fallback(tmp_path):
    mem = memory(tmp_path)
    p = validate_replan({"reasoning": "", "task_done": False, "subtasks": []}, state(), mem)
    assert [(s.type, s.target) for s in p.subtasks] == [("explore", "rain jacket")]
    done = validate_replan({"reasoning": "", "task_done": True, "subtasks": [explore("shoes")]}, state(), mem)
    assert not done.task_done and done.subtasks[0].target == "shoes"
    fb = fallback_plan(state())
    assert [s.target for s in fb.subtasks] == ["rain jacket", "shoes"]
