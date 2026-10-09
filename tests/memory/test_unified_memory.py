"""M5: UnifiedMemory orchestration, queries, snapshots and checkpoints."""
import json
from pathlib import Path

import pytest
from memory_helpers import (QUIET, TWO_ROOMS, build_fixture_memory, make_grid, make_memory,
                            make_obs, make_result, synonym_embedder)

from vlnav.common.config import deep_merge
from vlnav.memory.checkpoint import load_checkpoint, load_grid
from vlnav.memory.serialize import snapshot_to_dict


def test_object_room_edges_are_consistent(tmp_path):
    mem = build_fixture_memory(tmp_path)
    for room in mem.rooms():
        for nid in room.object_ids:
            assert mem.get_object(nid).room_id == room.id
    for node in mem.all_objects():
        assert node.id in mem.get_room(node.room_id).object_ids


def test_room_assignment_fallback(tmp_path):
    mem = make_memory(tmp_path, deep_merge(QUIET, {"rooms": {"enabled": True}}),
                      grid=make_grid(TWO_ROOMS))
    cam = (2.0, 2.0)
    obs = [make_obs("picture", 0.8, (0.15, 2.0, 1.5), 0.0, bbox=(0, 0, 40, 40), base_xy=cam),
           make_obs("tree", 0.8, (9.0, 5.0, 1.0), 0.0, bbox=(60, 0, 100, 40), base_xy=cam)]
    # picture: centroid in the outer wall of room A; tree: 1.1 m from the nearest room cell
    mem.update(make_result(0.0, obs, base_xy=cam))
    picture, tree = mem.all_objects()
    assert picture.room_id is not None
    assert picture.room_id == mem.tracker.room_at((1.0, 1.0), 0.0)
    assert tree.room_id is None


def test_view_is_immutable_and_cached(tmp_path):
    mem = build_fixture_memory(tmp_path)
    v1 = mem.view()
    assert mem.view() is v1
    before = [(n.id, n.n_obs, n.centroid) for n in v1.all_objects(include_tentative=True)]
    mem.update(make_result(11.0, [make_obs("bed", 0.9, (1.0, 1.0, 0.5), 11.0),
                                  make_obs("plant", 0.9, (2.0, 3.0, 0.5), 11.0,
                                           bbox=(100, 50, 140, 90))], base_xy=(2.0, 2.0)))
    after = [(n.id, n.n_obs, n.centroid) for n in v1.all_objects(include_tentative=True)]
    assert after == before and v1.version() < mem.version()
    assert mem.get_object(0).n_obs == 2 and len(mem.all_objects()) == 6
    with pytest.raises(TypeError):
        v1.objects[0] = None


def test_snapshot_returns_historical_state_not_filtered_live_state(tmp_path):
    """AUDIT leakage pitfall: a node first seen before t must not show its later view / score."""
    over = deep_merge(QUIET, {"checkpoints": {"checkpoint_every_s": 10}})
    mem = make_memory(tmp_path, over)
    jacket = (2.0, 0.0, 0.5)
    mem.update(make_result(0.0, [make_obs("jacket", 0.3, jacket, 0.0)]))              # ckpt@0
    mem.update(make_result(5.0, [make_obs("jacket", 0.35, jacket, 5.0)]))
    mem.update(make_result(10.0, []))                                                    # ckpt@10
    mem.update(make_result(12.0, [make_obs("jacket", 0.9, jacket, 12.0, base_xy=(1.0, 0.0)),
                                  make_obs("umbrella", 0.9, (3.0, 1.0, 0.5), 12.0,
                                           bbox=(100, 10, 150, 60))], base_xy=(1.0, 0.0)))

    live = {n.label: n for n in mem.all_objects()}
    assert live["jacket"].best_score == 0.9 and live["jacket"].first_seen_t <= 11.0

    snap = mem.snapshot(11.0)
    assert snap.stamp == 10.0
    (old,) = snap.all_objects()
    assert old.label == "jacket" and old.best_score == 0.35 and old.n_obs == 2
    assert old.best_view_pose.position[0] == 0.0
    assert old.best_view_image_path.endswith("cam0_5.000.jpg") and Path(old.best_view_image_path).exists()
    assert [n.best_score for n in mem.all_objects(until_t=11.0)] == [0.35]
    assert snap.best_view_crop(old.id, 0.1, 32).shape[2] == 3

    early = mem.snapshot(3.0)                                    # checkpoint at the first frame
    assert early.all_objects() == [] and len(early.all_objects(include_tentative=True)) == 1
    assert mem.snapshot(-1.0).all_objects(include_tentative=True) == []
    assert snap.snapshot(3.0).stamp == 0.0                       # snapshots reach history too


def test_checkpoints_on_disk_round_trip(tmp_path):
    mem = build_fixture_memory(tmp_path, checkpoint_dir=tmp_path / "ckpts")
    dirs = sorted((tmp_path / "ckpts").iterdir())
    assert [d.name for d in dirs] == ["ckpt_00000000.000", "ckpt_00000010.000"]
    for name in ("graph.json", "objects.npz", "rooms.npz", "grid.npz"):
        assert (dirs[-1] / name).exists()
    loaded = load_checkpoint(dirs[-1], image_root=mem.images.root, max_neighbors=6)
    assert snapshot_to_dict(loaded, include_geometry=True) == \
        json.loads(json.dumps(snapshot_to_dict(mem.view(), include_geometry=True)))
    assert (load_grid(dirs[-1]).as_array() == mem.grid.as_array()).all()


def test_rate_limit_per_camera(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    for k in range(11):
        t = k / 10
        for cam in ("cam0", "cam1"):
            mem.update(make_result(t, [make_obs("box", 0.6, (2.0, 0, 0.5), t, cam=cam)], cam=cam))
    st = mem.stats()
    assert (st.n_frames_processed, st.n_frames_skipped) == (6, 16)
    assert mem.all_objects()[0].n_obs == 6


def test_gating(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    bad = [make_obs("box", 0.05, (2.0, 0, 0.5), 0.0, bbox=(0, 0, 40, 40)),             # score
           make_obs("floor", 0.6, (2.0, 1, 0.0), 0.0, bbox=(40, 0, 80, 40)),           # ignored
           make_obs("pen", 0.6, (2.0, 2, 0.5), 0.0, bbox=(80, 0, 120, 40),
                    points=[[2.0, 2.0, 0.5]] * 3),                                      # points
           make_obs("car", 0.6, (9.0, 0, 0.5), 0.0, bbox=(120, 0, 160, 40)),           # range
           make_obs("dot", 0.6, (2.0, -1, 0.5), 0.0, bbox=(0, 60, 10, 70))]            # bbox
    good = make_obs("book", 0.6, (3.0, 0, 0.5), 0.0, bbox=(40, 60, 100, 110))
    mem.update(make_result(0.0, [*bad, good]))
    assert [n.label for n in mem.all_objects(include_tentative=True)] == ["book"]
    assert mem.stats().n_obs_gated == 5


def test_find_by_text_with_embedder(tmp_path):
    emb = synonym_embedder({("sofa", "couch"): 0.9})
    mem = make_memory(tmp_path, QUIET, embedder=emb)
    obs = [make_obs("sofa", 0.6, (2.0, 0, 0.5), 0.0, bbox=(0, 0, 40, 40)),
           make_obs("sofa", 0.9, (4.0, 0, 0.5), 0.0, bbox=(60, 0, 100, 40)),
           make_obs("table", 0.9, (3.0, 2, 0.5), 0.0, bbox=(110, 0, 150, 40))]
    mem.update(make_result(0.0, obs))
    hits = mem.find_by_text("couch", min_sim=0.8, top_n=5)
    assert [(n.id, round(s, 2)) for n, s in hits] == [(1, 0.9), (0, 0.9)]   # tie -> best_score
    assert mem.find_by_text("couch", min_sim=0.95, top_n=5) == []
    assert len(mem.find_by_text("couch", 0.8, top_n=1)) == 1


def test_events_and_stats(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    v0 = mem.version()
    mem.update(make_result(0.0, [make_obs("box", 0.6, (2.0, 0, 0.5), 0.0)]))
    assert [e.type for e in mem.events(v0)] == ["node_created"]
    assert mem.events(mem.version()) == []
    st = mem.stats()
    assert st.n_objects == st.n_confirmed == 1 and st.update_ms_p95 is not None
