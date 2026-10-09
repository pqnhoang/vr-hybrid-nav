"""M4: LLM room labeling with a mock client."""
import json
from types import SimpleNamespace

from memory_helpers import QUIET, floor_plan, make_grid, make_memory, make_obs, make_result

from vlnav.common.config import deep_merge
from vlnav.memory import MockLLMClient, RoomLabeler, load_memory_config
from vlnav.memory.room_labeler import RoomLabelRequest

LAB = load_memory_config().labeling


def reply(label, conf=0.8):
    return json.dumps({"label": label, "confidence": conf})


def req(objects=(("bed", 1, 0.8), ("wardrobe", 1, 0.6)), area=16.0):
    return RoomLabelRequest(room_id=0, objects=tuple(objects), area_m2=area, adjacent_labels=())


def test_label_constrained_to_enum():
    lab = RoomLabeler(LAB, MockLLMClient([reply("Bedroom"), reply("spaceship")]))
    assert lab.label(req()).label == "bedroom"
    assert lab.label(req(objects=(("x", 2, 0.5),))).label == "other"
    assert lab.schema()["properties"]["label"]["enum"][-1] == "other"


def test_prompt_contents_and_code_fence_parsing():
    llm = MockLLMClient("```json\n" + reply("kitchen", 0.7) + "\n```")
    lab = RoomLabeler(LAB, llm)
    res = lab.label(req(objects=(("fridge", 1, 0.9), ("oven", 2, 0.5))))
    assert (res.label, res.confidence, res.ok) == ("kitchen", 0.7, True)
    prompt = llm.calls[0][0]["content"]
    assert "fridge: 1, 0.90" in prompt and "oven: 2, 0.50" in prompt and "laundry room" in prompt


def test_min_objects_rule():
    lab = RoomLabeler(LAB, MockLLMClient(reply("bedroom")))
    node = SimpleNamespace(label="bed", best_score=0.8)
    assert lab.build_request(0, [node], 10.0, []) is None
    r = lab.build_request(0, [node, node], 10.0, ["unknown", "hallway"])
    assert r.objects == (("bed", 2, 0.8),) and r.adjacent_labels == ("hallway",)


def test_cache_hit_and_area_bucket():
    llm = MockLLMClient(reply("bedroom"))
    lab = RoomLabeler(LAB, llm)
    assert not lab.label(req(area=16.0)).from_cache
    assert lab.label(req(area=17.0)).from_cache                     # same 5 m2 bucket
    assert not lab.label(req(area=21.0)).from_cache
    assert lab.n_llm_calls == 2 and len(llm.calls) == 2


def test_malformed_or_failing_llm_stays_unknown():
    def boom(messages):
        raise TimeoutError
    for client in (MockLLMClient("not json"), MockLLMClient(boom)):
        res = RoomLabeler(LAB, client).label(req())
        assert (res.label, res.ok) == ("unknown", False)


def test_memory_labels_rooms_and_relabels_on_new_objects(tmp_path):
    shape = (60, 100)
    arr = floor_plan(shape, [(2, 2, 42, 42), (43, 2, 83, 42)], doors=[(42, 15, 43, 24)])
    llm = MockLLMClient(lambda m: reply("bedroom" if "bed" in m[0]["content"] else "kitchen"))
    over = deep_merge(QUIET, {"rooms": {"enabled": True}, "labeling": {"worker_thread": False}})
    cfg = load_memory_config(overrides=over)
    mem = make_memory(tmp_path, over, grid=make_grid(arr), labeler=RoomLabeler(cfg.labeling, llm))

    def add(t, label, x, y):
        mem.update(make_result(t, [make_obs(label, 0.8, (x, y, 0.5), t)]))

    add(0.0, "bed", 1.0, 1.0)
    assert len(mem.rooms()) == 2 and len(llm.calls) == 0          # 1 object < min_objects
    add(1.0, "wardrobe", 3.0, 1.0)
    room_a = mem.get_room(mem.all_objects()[0].room_id)
    assert room_a.label == "bedroom" and room_a.object_ids == [0, 1] and len(llm.calls) == 1
    add(2.0, "lamp", 2.0, 3.0)                                     # 1 new object: no relabel
    assert len(llm.calls) == 1
    add(3.0, "desk", 1.0, 3.0)
    add(4.0, "chair", 3.0, 3.0)                                    # 3 new objects -> relabel
    assert len(llm.calls) == 2
    assert mem.get_room(room_a.id).object_ids == [0, 1, 2, 3, 4]
    assert any(e.type == "room_labeled" for e in mem.events(0))


def test_worker_thread_labels_and_discards_stale_results(tmp_path):
    import threading
    shape = (60, 100)
    arr = floor_plan(shape, [(2, 2, 42, 42), (43, 2, 83, 42)], doors=[(42, 15, 43, 24)])
    gate = threading.Event()

    def slow(messages):
        gate.wait(5)
        return reply("bedroom")
    over = deep_merge(QUIET, {"rooms": {"enabled": True}, "labeling": {"worker_thread": True}})
    cfg = load_memory_config(overrides=over)
    llm = MockLLMClient(slow)
    mem = make_memory(tmp_path, over, grid=make_grid(arr), labeler=RoomLabeler(cfg.labeling, llm))

    def add(t, label, x, y):
        mem.update(make_result(t, [make_obs(label, 0.8, (x, y, 0.5), t)]))
    add(0.0, "bed", 1.0, 1.0)
    add(1.0, "wardrobe", 3.0, 1.0)                     # request sent, reply blocked
    assert len(mem._label_jobs) == 1 and mem.rooms()[0].label == "unknown"
    for k, (x, y) in enumerate([(2.0, 3.0), (1.0, 3.0), (3.0, 3.0)]):
        add(2.0 + k, "lamp", x, y)                     # 3 new objects while in flight
    gate.set()
    mem.wait_labels(5)                                 # 1st reply stale (objects changed) -> re-queued
    labeled = [e for e in mem.events(0) if e.type == "room_labeled"]
    assert mem.rooms()[0].label == "bedroom" and len(llm.calls) == 2 and len(labeled) == 1
    assert not mem._label_jobs
    mem.close()


def test_room_that_loses_its_objects_goes_back_to_unknown(tmp_path):
    shape = (60, 100)
    arr = floor_plan(shape, [(2, 2, 42, 42), (43, 2, 83, 42)], doors=[(42, 15, 43, 24)])
    over = deep_merge(QUIET, {"rooms": {"enabled": True}, "labeling": {"worker_thread": False}})
    cfg = load_memory_config(overrides=over)
    mem = make_memory(tmp_path, over, grid=make_grid(arr),
                      labeler=RoomLabeler(cfg.labeling, MockLLMClient(lambda m: reply("bedroom"))))
    mem.update(make_result(0.0, [make_obs("bed", 0.8, (1.0, 1.0, 0.5), 0.0)]))
    mem.update(make_result(1.0, [make_obs("wardrobe", 0.8, (3.0, 1.0, 0.5), 1.0)]))
    rid = mem.rooms()[0].id
    assert mem.rooms()[0].label == "bedroom"
    rs = mem.tracker.rooms[rid]                        # e.g. objects re-assigned after a split
    rs.room = rs.room.model_copy(update={"object_ids": [rs.room.object_ids[0]]})
    mem.tick(2.0)
    assert mem.rooms()[0].label == "unknown" and mem.tracker.rooms[rid].needs_label
