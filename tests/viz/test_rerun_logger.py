"""Smoke test: the Rerun logger records synthetic frames without errors (no viewer needed)."""
import sys
from pathlib import Path

import pytest

rr = pytest.importorskip("rerun")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))
from memory_helpers import build_fixture_memory, make_obs, make_result  # noqa: E402

from vlnav.viz.rerun_logger import RerunLogger, load_viz_config, start  # noqa: E402


def test_logger_records_frames(tmp_path):
    mem = build_fixture_memory(tmp_path)
    path = tmp_path / "rec.rrd"
    assert start("vlnav_test", ["save"], str(path), 0, 0, "127.0.0.1") is None
    logger = RerunLogger(load_viz_config(overrides={"map_every_frames": 1}))
    for k, t in enumerate((11.0, 12.0)):
        res = make_result(t, [make_obs("bed", 0.9, (1.0, 1.0, 0.5), t, base_xy=(2.0, 2.0))],
                          base_xy=(2.0, 2.0))
        v = mem.version()
        mem.update(res)
        logger.log_frame(k, res.frame, res, ["lifted"], mem.view(), mem.events(v),
                         {"perception": 1.0, "memory_update": 0.1})
        logger.log_map(k, mem.grid, mem.view())                 # grid image + room outlines
    rr.disconnect()
    assert path.exists() and path.stat().st_size > 1000
    assert len(mem.rooms()) == 2 and logger._rooms_key


def test_blueprint_has_only_the_views_a_run_logs():
    from vlnav.viz.rerun_logger import blueprint

    def origins(bp):
        found = []

        def walk(x):
            if hasattr(x, "origin") and x.origin is not None:
                found.append(str(x.origin).lstrip("/"))
            for c in getattr(x, "contents", None) or []:
                walk(c)
        walk(bp.root_container)
        return set(found)
    full = origins(blueprint(exploration=True))
    expl = origins(blueprint(memory=False, planner=False, exploration=True))
    mem = origins(blueprint(planner=False))
    assert {"memory/latest_best_view", "map", "planner", "exploration/view"} <= full
    assert "exploration/view" in expl and not expl & {"memory/latest_best_view", "map", "events", "planner", "task"}
    assert "map" in mem and not mem & {"planner", "task", "exploration/view"}
