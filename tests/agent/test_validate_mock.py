"""M0/M1: program validation, NavAPI doc, dry run and hard checks on the fixture memory."""
import pytest
from agent_helpers import matcher, memory, snapshot

from vlnav.acquisition.symbolic_filter import SymbolicFilter
from vlnav.agent.api import api_doc
from vlnav.agent.mock_world import dry_run, movement_phrases
from vlnav.agent.validate import ProgramError, extract_code, load_program, validate
from vlnav.navigation import GridAStar, load_navigation_config

from acq_helpers import cfg as acq_cfg

GOOD = '''def task(api):
    # Step 1: route
    if not api.follow("Go forward to the box and turn right."):
        return api.fail("route failed")
    # Step 2: umbrella
    u = api.locate("umbrella", prompts=["umbrella"])
    if u is None:
        return api.fail("no umbrella")
    for c in api.find("chair", room="bedroom")[:2]:
        print_me = c.name.lower()
    if api.go_to(u):
        api.report(u)
'''


def run(code, instruction="Find the umbrella.", move=True, tmp_path=None, **kw):
    mem = memory(tmp_path)
    filt = SymbolicFilter(acq_cfg().symbolic, matcher())
    return dry_run(code, snapshot(mem, instruction, move=move, **kw), filt, GridAStar(load_navigation_config()), 40)


def test_validate_accepts_plain_programs():
    validate(GOOD)
    assert callable(load_program(GOOD))


@pytest.mark.parametrize("body,msg", [
    ("import os", "Import"),
    ("while True:\n        api.here()", "While"),
    ("try:\n        api.here()\n    except Exception:\n        pass", "Try"),
    ("api.teleport()", "api.teleport"),
    ("x = api.here()._id", "_id"),
    ("open('f')", "open"),
    ("n = 3\n    for i in range(n):\n        api.here()", "literal bounds"),
    ("f = lambda: 1", "Lambda"),
    ("api.follow()", "text argument"),
])
def test_validate_rejects(body, msg):
    with pytest.raises(ProgramError, match=msg):
        validate(f"def task(api):\n    {body}\n")


def test_validate_shape_and_extract():
    with pytest.raises(ProgramError, match="exactly one function"):
        validate("x = 1\ndef task(api):\n    pass\n")
    with pytest.raises(ProgramError, match="def task"):
        validate("def task(robot):\n    pass\n")
    reply = "<think>hm</think>Here:\n```python\ndef task(api):\n    api.here()\n```\nDone."
    assert extract_code(reply) == "def task(api):\n    api.here()\n"
    with pytest.raises(ProgramError):
        extract_code("no code here")


def test_api_doc_without_move():
    assert "api.follow(" in api_doc(True)
    doc = api_doc(False)
    assert "follow(" not in doc and "api.locate(" in doc and "api.go_to(" in doc


def test_movement_phrases():
    assert movement_phrases("Go forward to the box. Turn right at the box and find the sofa.") == \
        ["Go forward", "Turn right"]
    assert movement_phrases("Find an umbrella and a rain jacket.") == []


def test_dry_run_known_object_is_goto_unknown_is_explore(tmp_path):
    code = ('def task(api):\n    u = api.locate("umbrella")\n    s = api.locate("shoes", prompts=["shoes", "boots"])\n'
            '    if api.go_to(u):\n        api.report(u)\n    if api.go_to(s):\n        api.report(s)\n')
    r = run(code, "Find the umbrella and my shoes.", tmp_path=tmp_path)
    assert r.finished and r.hard_failures == []
    assert [(a["type"], a["target"]) for a in r.actions] == [("goto", "umbrella"), ("explore", "shoes")]
    assert r.actions[1]["prompts"] == ["shoes", "boots"]
    assert "known in the scene graph: umbrella #21" in r.trace[0].result and r.trace[0].line == 2
    assert "will explore" in r.trace[1].result


def test_dry_run_route_order_and_hard_checks(tmp_path):
    instr = "Go forward to the cartoon box. Turn right at the box and find the umbrella."
    r = run(GOOD, instr, tmp_path=tmp_path)
    assert r.hard_failures == [] and [a["type"] for a in r.actions] == ["move", "goto"]
    # movement command but no follow()
    r = run('def task(api):\n    u = api.locate("umbrella")\n    if api.go_to(u):\n        api.report(u)\n',
            instr, tmp_path=tmp_path)
    assert any("never calls api.follow" in h for h in r.hard_failures)
    # follow() while move is disabled
    r = run(GOOD, instr, move=False, tmp_path=tmp_path)
    assert any("move (DualVLN) is disabled" in h for h in r.hard_failures)
    # report before go_to, and fail() on the success path
    r = run('def task(api):\n    u = api.locate("umbrella")\n    api.report(u)\n    api.fail("x")\n',
            tmp_path=tmp_path)
    assert any("before go_to" in h for h in r.hard_failures) and any("fail(" in h for h in r.hard_failures)


def test_dry_run_flags_route_landmarks(tmp_path):
    code = ('def task(api):\n    api.follow("Go forward to the umbrella and turn right.")\n'
            '    u = api.locate("umbrella")\n    if api.go_to(u):\n        api.report(u)\n')
    r = run(code, "Go forward to the umbrella and turn right.", tmp_path=tmp_path)
    assert any("'umbrella' is a landmark" in h for h in r.hard_failures)
    r = run(GOOD, "Go forward to the cartoon box. Turn right at the box and find the umbrella.", tmp_path=tmp_path)
    assert r.hard_failures == []


def test_dry_run_exception_line_and_budget(tmp_path):
    r = run('def task(api):\n    u = api.locate("umbrella")\n    x = u.xy[5]\n', tmp_path=tmp_path)
    assert r.error.startswith("line 3: IndexError") and not r.finished
    r = run('def task(api):\n    for i in range(50):\n        api.check("q?")\n', tmp_path=tmp_path)
    assert any(h.startswith("budget") for h in r.hard_failures)
    r = run("def task(api):\n    api.here()\n", tmp_path=tmp_path)
    assert "the program does no action" in r.hard_failures
