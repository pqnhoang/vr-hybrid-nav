"""MemoryView conformance suite, parametrized over implementations (TASK_unified_memory.md §5).

Includes the acquisition test-only `InMemoryMemory` (A1); `RemoteMemoryMirror` (M6) joins when it
exists. Every implementation must pass the same tests.
"""
import numpy as np
import pytest
from memory_helpers import build_fixture_memory

from vlnav.acquisition.memory_api import InMemoryMemory, MemoryView
from vlnav.memory.checkpoint import load_checkpoint, save_checkpoint
from vlnav.memory.serialize import snapshot_from_json, snapshot_to_json

BED, WARDROBE, LAMP, SOFA, CUP, CHAIR, CHAIR_DUP = range(7)


def _live_view(mem, tmp_path):
    return mem.view()


def _unified_memory(mem, tmp_path):
    return mem


def _json_roundtrip(mem, tmp_path):
    return snapshot_from_json(snapshot_to_json(mem.view()), image_root=mem.images.root,
                              max_neighbors=mem.cfg.queries.max_neighbors, image_store=mem.images)


def _checkpoint_load(mem, tmp_path):
    out = save_checkpoint(mem.view(), tmp_path / "ckpt", mem.grid)
    return load_checkpoint(out, image_root=mem.images.root,
                           max_neighbors=mem.cfg.queries.max_neighbors, image_store=mem.images)


def _in_memory_fixture(mem, tmp_path):
    """Acquisition's fixture-backed InMemoryMemory, loaded from the scene-graph JSON."""
    path = tmp_path / "fixture.json"
    path.write_text(snapshot_to_json(mem.view()))
    return InMemoryMemory.from_json(path, max_neighbors=mem.cfg.queries.max_neighbors,
                                    image_root=mem.images.root)


IMPLEMENTATIONS = [_live_view, _unified_memory, _json_roundtrip, _checkpoint_load, _in_memory_fixture]


@pytest.fixture(params=IMPLEMENTATIONS, ids=lambda f: f.__name__.strip("_"))
def view(request, tmp_path) -> MemoryView:
    return request.param(build_fixture_memory(tmp_path), tmp_path)


def ids(nodes):
    return [n.id for n in nodes]


def test_is_a_memory_view(view):
    assert isinstance(view, MemoryView)
    assert isinstance(view.version(), int) and view.version() > 0


def test_all_objects_hides_tentative(view):
    assert ids(view.all_objects()) == [BED, WARDROBE, LAMP, SOFA, CHAIR]
    assert ids(view.all_objects(include_tentative=True)) == [BED, WARDROBE, LAMP, SOFA, CUP, CHAIR]


def test_get_object_follows_aliases(view):
    assert view.get_object(CHAIR_DUP).id == CHAIR
    assert view.get_object(CHAIR).n_obs == 2
    assert view.get_object(CUP).label == "cup"                      # tentative still addressable
    with pytest.raises(KeyError):
        view.get_object(99)


def test_rooms_and_edges(view):
    rooms = view.rooms()
    assert len(rooms) == 2 and ids(rooms) == sorted(ids(rooms))
    room_a = view.get_room(view.get_object(BED).room_id)
    room_b = view.get_room(view.get_object(SOFA).room_id)
    assert room_a.id != room_b.id and room_a.label == "unknown"
    assert room_a.object_ids == [BED, WARDROBE, LAMP] and room_b.object_ids == [SOFA, CHAIR]
    assert ids(view.objects_in_room(room_a.id)) == [BED, WARDROBE, LAMP]
    assert room_a.adjacent_room_ids == [room_b.id]
    assert view.get_room(999) is None and view.objects_in_room(999) == []


def test_neighbors(view):
    assert ids(view.neighbors(BED, 1.0, same_room=False)) == [LAMP]
    assert ids(view.neighbors(BED, 3.0, same_room=False)) == [LAMP, WARDROBE]
    assert ids(view.neighbors(BED, 6.0, same_room=False)) == [LAMP, WARDROBE, SOFA]
    assert ids(view.neighbors(BED, 6.0, same_room=True)) == [LAMP, WARDROBE]
    assert ids(view.neighbors(CHAIR_DUP, 3.0, same_room=True)) == [SOFA]       # via alias


def test_find_by_text_exact_without_embedder(view):
    (hit,) = view.find_by_text("Sofa", min_sim=0.5, top_n=5)
    assert hit[0].id == SOFA and hit[1] == 1.0
    assert view.find_by_text("cup", 0.5, 5) == []                    # tentative hidden
    assert view.find_by_text("table", 0.5, 5) == []


def test_best_view_crop(view):
    crop = view.best_view_crop(BED, margin=0.25, max_px=16)
    assert isinstance(crop, np.ndarray) and crop.ndim == 3 and crop.shape[2] == 3
    assert max(crop.shape[:2]) <= 16


def test_snapshot_at_or_after_now_is_current(view):
    snap = view.snapshot(1e9)
    assert ids(snap.all_objects()) == ids(view.all_objects())
    assert snap.version() == view.version()
