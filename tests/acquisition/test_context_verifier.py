"""A3 (context builder) + A4 (VLM client contract, neural verifier)."""
import json

import numpy as np
import pytest
from acq_helpers import cfg, matcher, memory, reply

from vlnav.acquisition.context_builder import ContextBuilder, letters
from vlnav.acquisition.symbolic_filter import SymbolicFilter
from vlnav.acquisition.verifier import NeuralVerifier
from vlnav.acquisition.vlm_client import MockVLMClient, extract_json, strip_images


def cands(tmp_path, k=3):
    mem = memory(tmp_path)
    return mem, SymbolicFilter(cfg().symbolic, matcher()).topk("rain jacket", mem, k)


def test_context_crops_frames_and_text(tmp_path):
    mem, cs = cands(tmp_path)
    c = cfg(context={"crop_max_px": 64, "frame_max_px": 100})
    ctx = ContextBuilder(c.context).build(cs, mem)
    assert [x.letter for x in ctx] == ["A", "B", "C"] and [x.node_id for x in ctx] == [17, 19, 18]
    assert max(ctx[0].crop.shape[:2]) <= 64 and max(ctx[0].frame.shape[:2]) == 100
    # node 18's bbox touches the top-left border: the margin is clamped, not padded
    assert ctx[2].crop.shape[:2] == (64, 43)
    frame = ctx[0].frame
    red = (frame[..., 0] > 200) & (frame[..., 1] < 60) & (frame[..., 2] < 60)
    assert red.sum() > 50                                   # bbox drawn
    assert ctx[0].text.startswith("detector label: jacket; detection confidence: 0.62; room: bedroom")
    assert ctx[0].text.endswith("on/near: raincoat (0.4 m), desk (0.5 m), chair (0.7 m)")
    no_frame = ContextBuilder(c.context).build(cs, mem, include_full_frame=False, use_neighbors=False)
    assert no_frame[0].frame is None and "on/near" not in no_frame[0].text
    with pytest.raises(ValueError):
        letters(27)


def test_extract_json_and_strip_images():
    assert extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```') == {"a": {"b": "}"}}
    assert extract_json('<think>{"x":1}</think> answer {"choice": "A"} trailing') == {"choice": "A"}
    with pytest.raises(ValueError):
        extract_json("no json here")
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
    assert strip_images(msgs)[0]["content"][0]["image_url"]["url"].startswith("<image sha1=")


def verifier(tmp_path, replies, log=True, **over):
    c = cfg(**over)
    vlm = MockVLMClient(replies)
    return NeuralVerifier(c.verifier, c.context, vlm, tmp_path / "debug.jsonl" if log else None), vlm


INSTR = "Today's weather report indicates rain. Help Rob find a jacket."


@pytest.mark.parametrize("replies,status,node", [
    ([reply("B")], "selected", 19),
    ([reply("none", 0.8)], "none", None),
    ([reply("A", 0.3)], "none", None),                      # below min_vlm_confidence
    (["not json", reply("C")], "selected", 18),             # repair retry succeeds
    (["not json", "still not"], "error", None),
    ([reply("D")], "error", None),                          # letter not offered
    ([TimeoutError("slow")], "error", None),
])
def test_verifier_outcomes(tmp_path, replies, status, node):
    mem, cs = cands(tmp_path)
    v, vlm = verifier(tmp_path, replies)
    out = v.verify(INSTR, "rain jacket", cs, mem)
    assert (out.status, out.node_id) == (status, node)
    assert len(vlm.calls) == min(len(replies), 2)


def test_prompt_contents_schema_and_debug_log(tmp_path):
    mem, cs = cands(tmp_path)
    v, vlm = verifier(tmp_path, [reply("A")])
    v.verify(INSTR, "rain jacket", cs, mem)
    content = vlm.calls[0]["messages"][0]["content"]
    texts = " ".join(p["text"] for p in content if p["type"] == "text")
    assert INSTR in texts and '"rain jacket"' in texts and "none" in texts
    assert texts.index("Candidate A") < texts.index("Candidate B") < texts.index("Candidate C")
    assert sum(p["type"] == "image_url" for p in content) == 6          # crop + frame per candidate
    assert vlm.calls[0]["schema"]["properties"]["choice"]["enum"] == ["A", "B", "C", "none"]
    rec = json.loads((tmp_path / "debug.jsonl").read_text().splitlines()[0])
    assert rec["candidates"]["B"]["node_id"] == 19 and "base64" not in json.dumps(rec)
    assert rec["raw"] == [reply("A")] and rec["result"]["status"] == "selected"
    assert np.isfinite(rec["result"]["latency_s"])
