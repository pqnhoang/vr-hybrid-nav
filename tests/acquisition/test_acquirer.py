"""A5: TargetAcquirer orchestration."""
from acq_helpers import cfg, matcher, memory, reply

from vlnav.acquisition.acquirer import TargetAcquirer
from vlnav.acquisition.vlm_client import MockVLMClient

INSTR = "It is raining. Find Rob's rain jacket."


def acquirer(replies, **over):
    vlm = MockVLMClient(replies) if replies is not None else None
    return TargetAcquirer(cfg(**over), matcher(), vlm), vlm


def test_selected_in_round_0_returns_best_view_pose(tmp_path):
    mem = memory(tmp_path)
    acq, vlm = acquirer([reply("B")])
    r = acq.acquire(INSTR, "rain jacket", mem)
    assert (r.status, r.node_id, r.candidates) == ("selected", 19, [17, 19, 18])
    assert r.goal_pose == mem.get_object(19).best_view_pose
    assert set(r.latency_s) == {"filter", "vlm", "total"}


def test_rejected_then_next_k_then_rejected_all(tmp_path):
    mem = memory(tmp_path)
    acq, vlm = acquirer([reply("none"), reply("A")])
    r = acq.acquire(INSTR, "rain jacket", mem)
    assert (r.status, r.node_id, r.candidates) == ("selected", 20, [17, 19, 18, 20])
    acq2, _ = acquirer([reply("none"), reply("none")])
    r2 = acq2.acquire(INSTR, "rain jacket", mem)
    assert r2.status == "rejected_all" and r2.node_id is None and r2.goal_pose is None


def test_no_candidates_and_error(tmp_path):
    mem = memory(tmp_path)
    acq, vlm = acquirer([reply("A")])
    assert acq.acquire(INSTR, "bicycle", mem).status == "no_candidates" and vlm.calls == []
    acq_err, _ = acquirer([TimeoutError()])
    assert acq_err.acquire(INSTR, "rain jacket", mem).status == "error"


def test_blacklist_persists_per_target_and_resets(tmp_path):
    mem = memory(tmp_path)
    acq, _ = acquirer(lambda m: reply("A"))
    acq.blacklist("rain jacket", 17)
    assert acq.acquire(INSTR, "rain jacket", mem).node_id == 19
    assert acq.acquire(INSTR, "Rain Jacket", mem).candidates[0] == 19          # still excluded
    assert acq.acquire("find a coat", "coat", mem).candidates[0] == 17         # other target
    acq.reset_blacklist("rain jacket")
    assert acq.acquire(INSTR, "rain jacket", mem).node_id == 17
    acq.blacklist("rain jacket", 17)
    acq.reset_blacklist()
    assert acq.blacklisted("rain jacket") == set()


def test_rejected_candidates_are_blacklisted_across_calls(tmp_path):
    mem = memory(tmp_path)
    acq, _ = acquirer([reply("none"), reply("none"), reply("A")], acquirer={"max_rounds": 2})
    assert acq.acquire(INSTR, "rain jacket", mem).status == "rejected_all"
    assert acq.acquire(INSTR, "rain jacket", mem).status == "no_candidates"


def test_force_include_and_hallucinated_ids(tmp_path):
    mem = memory(tmp_path)
    acq, vlm = acquirer([reply("D")], symbolic={"k": 3})
    r = acq.acquire(INSTR, "rain jacket", mem, force_include=[21, 12345])
    assert r.candidates == [17, 19, 18, 21] and r.node_id == 21               # k+1, verified
    letters = vlm.calls[0]["schema"]["properties"]["choice"]["enum"]
    assert letters == ["A", "B", "C", "D", "none"]


def test_no_vlm_ablation_takes_symbolic_top1(tmp_path):
    acq, _ = acquirer(None)
    assert acq.acquire(INSTR, "rain jacket", memory(tmp_path)).node_id == 17


def test_deterministic(tmp_path):
    mem = memory(tmp_path)
    runs = [acquirer(lambda m: reply("C"))[0].acquire(INSTR, "rain jacket", mem) for _ in range(2)]
    assert runs[0].model_dump(exclude={"latency_s"}) == runs[1].model_dump(exclude={"latency_s"})
