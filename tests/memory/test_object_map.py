"""M2: node lifecycle, best view, image store, dedup, prune."""
from pathlib import Path

import numpy as np
import pytest
from memory_helpers import QUIET, make_frame, make_memory, make_obs, make_result

from vlnav.common.config import deep_merge
from vlnav.memory import ImageStore

C = (2.0, 0.0, 0.5)


def obs_at(t, label="jacket", score=0.4, center=C, **kw):
    return make_result(t, [make_obs(label, score, center, t, **kw)], base_xy=kw.get("base_xy", (0, 0)))


def only(mem):
    nodes = mem.all_objects(include_tentative=True)
    assert len(nodes) == 1
    return nodes[0]


def test_label_voting_flips_on_accumulated_votes(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    # one object detected as "jacket" and "coat" in the same frame -> one node with both votes
    mem.update(make_result(0.0, [make_obs("jacket", 0.4, C, 0.0), make_obs("coat", 0.3, C, 0.0)]))
    assert only(mem).label == "jacket"
    mem.update(obs_at(1.0, "coat", 0.3))                             # shares "coat" -> merges
    node = only(mem)
    assert node.label == "coat"
    assert node.label_votes == pytest.approx({"jacket": 0.4, "coat": 0.6})
    assert node.n_obs == 2 and node.first_seen_t == 0.0 and node.last_seen_t == 1.0


def test_tentative_until_min_obs_unless_confident(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    mem.update(obs_at(0.0, score=0.3))
    assert mem.all_objects() == [] and len(mem.all_objects(include_tentative=True)) == 1
    mem.update(obs_at(1.0, score=0.3))
    assert len(mem.all_objects()) == 1                              # n_obs >= min_obs
    mem2 = make_memory(tmp_path / "b", QUIET)
    mem2.update(obs_at(0.0, score=0.6))                              # >= confirm_score
    assert len(mem2.all_objects()) == 1


def test_best_view_follows_highest_score(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    mem.update(obs_at(0.0, score=0.3, base_xy=(0.0, 0.0)))
    first = only(mem).best_view_image_path
    mem.update(obs_at(1.0, score=0.5, base_xy=(0.5, 0.0)))
    node = only(mem)
    assert node.best_view_image_path != first and node.best_view_image_path.endswith("1.000.jpg")
    assert node.best_view_pose.position[0] == pytest.approx(0.5)
    assert node.best_view_cam_pose.position[2] == pytest.approx(1.0)
    mem.update(obs_at(2.0, score=0.4, base_xy=(1.0, 0.0)))           # lower score: keep view
    node = only(mem)
    assert node.best_view_pose.position[0] == pytest.approx(0.5) and node.best_score == 0.5
    assert Path(node.best_view_image_path).exists()


def test_border_penalty_blocks_view_but_not_best_score(tmp_path):
    mem = make_memory(tmp_path, deep_merge(QUIET, {"best_view": {"border_penalty": 0.5}}))
    mem.update(obs_at(0.0, score=0.5))
    view = only(mem).best_view_image_path
    mem.update(obs_at(1.0, score=0.6, bbox=(0, 30, 60, 90)))         # touches the left border
    node = only(mem)
    assert node.best_view_image_path == view and node.best_score == 0.6


def test_best_view_rejected_in_occupied_cell(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    mem.update(obs_at(0.0, score=0.3, base_xy=(0.0, 0.0)))
    arr = np.zeros((100, 100), dtype=np.int8)
    cell = mem.grid.world_to_cell((1.0, 0.0))
    arr[cell[1], cell[0]] = 1
    mem.grid.set_array(arr, t=0.5)
    mem.update(obs_at(1.0, score=0.9, base_xy=(1.0, 0.0)))           # pose inside an obstacle
    assert only(mem).best_view_pose.position[0] == pytest.approx(0.0)
    # the current view becomes occupied -> any later view replaces it
    arr[:] = 0
    cell = mem.grid.world_to_cell((0.0, 0.0))
    arr[cell[1], cell[0]] = 1
    mem.grid.set_array(arr, t=1.5)
    mem.update(obs_at(2.0, score=0.2, base_xy=(2.0, 1.0)))
    assert only(mem).best_view_pose.position[0] == pytest.approx(2.0)


def test_image_store_refcount_and_gc(tmp_path):
    store = ImageStore(tmp_path, 90, 4)
    p = store.put_frame(make_frame(1.0))
    assert store.put_frame(make_frame(1.0)) == p                    # idempotent
    store.acquire(p)
    store.acquire(p)
    store.release(p)
    assert store.gc() == [] and Path(p).exists()
    store.release(p)
    assert store.gc() == [p] and not Path(p).exists()
    with pytest.raises(RuntimeError):
        store.release(p)


def test_crop_clamps_and_limits_size(tmp_path):
    store = ImageStore(tmp_path, 90, 4)
    p = store.put_frame(make_frame(1.0))
    crop = store.crop(p, (0, 0, 40, 20), margin=0.5, max_px=1000)
    assert crop.shape == (30, 60, 3)                                # clamped at the top-left
    small = store.crop(p, (10, 10, 150, 110), margin=0.0, max_px=70)
    assert max(small.shape[:2]) == 70
    assert store.crop(p, (10, 10, 150, 110), margin=0.0, max_px=70) is small   # LRU hit


def test_replaced_view_deleted_only_when_unreferenced(tmp_path):
    over = deep_merge(QUIET, {"checkpoints": {"checkpoint_every_s": 0.0, "max_checkpoints": 1}})
    mem = make_memory(tmp_path, over)
    mem.update(obs_at(0.0, score=0.3))                               # checkpoint holds frame 0
    first = only(mem).best_view_image_path
    mem.update(obs_at(1.0, score=0.5))                               # replaced; ckpt@0 evicted
    assert not Path(first).exists()
    over = deep_merge(QUIET, {"checkpoints": {"checkpoint_every_s": 0.0, "max_checkpoints": 0}})
    mem = make_memory(tmp_path / "all", over)
    mem.update(obs_at(0.0, score=0.3))
    first = only(mem).best_view_image_path
    mem.update(obs_at(1.0, score=0.5))
    assert Path(first).exists()                                      # still used by ckpt@0


def test_one_frame_shared_by_two_nodes(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    obs = [make_obs("cup", 0.4, C, 0.0, bbox=(10, 10, 50, 50)),
           make_obs("book", 0.4, (3.0, 1.0, 0.5), 0.0, bbox=(80, 10, 140, 50))]
    mem.update(make_result(0.0, obs))
    a, b = mem.all_objects(include_tentative=True)
    assert a.best_view_image_path == b.best_view_image_path
    assert mem.images.refcount(a.best_view_image_path) == 4      # 2 nodes + checkpoint@0


def test_dedup_merges_into_older_id_with_alias(tmp_path):
    mem = make_memory(tmp_path, deep_merge(QUIET, {"maintenance": {"dedup_every_s": 5}}))
    obs = [make_obs("chair", 0.6, C, 0.0, bbox=(20, 30, 60, 90)),
           make_obs("chair", 0.7, (2.2, 0.0, 0.5), 0.0, bbox=(90, 30, 130, 90))]
    mem.update(make_result(0.0, obs))           # one-to-one: two nodes, dedup not due yet
    assert len(mem.all_objects()) == 2
    mem.tick(6.0)
    (node,) = mem.all_objects()
    assert node.id == 0 and node.n_obs == 2 and node.best_score == 0.7
    assert mem.get_object(1).id == 0
    assert any(e.type == "nodes_deduped" for e in mem.events(0))
    assert mem.images.refcount(node.best_view_image_path) == 3      # node + ckpt@0 x2 nodes


def test_prune_only_tentative_nodes(tmp_path):
    mem = make_memory(tmp_path, deep_merge(QUIET, {"maintenance": {"prune_every_s": 1}}))
    obs = [make_obs("cup", 0.2, C, 0.0, bbox=(10, 10, 50, 50)),
           make_obs("book", 0.6, (3.0, 1.0, 0.5), 0.0, bbox=(80, 10, 140, 50))]
    mem.update(make_result(0.0, obs))
    mem.tick(30.0)
    assert len(mem.all_objects(include_tentative=True)) == 2
    mem.tick(61.0)
    assert [n.label for n in mem.all_objects(include_tentative=True)] == ["book"]
    with pytest.raises(KeyError):
        mem.get_object(0)


def test_dedup_chain_returns_only_existing_nodes(tmp_path):
    """Real-bag crash: 2 merges into 1, then 1 into 0 -> callers must not see id 1."""
    mem = make_memory(tmp_path, deep_merge(QUIET, {"maintenance": {"dedup_every_s": 5}}))
    obs = [make_obs("chair", 0.6, (2.0, 0, 0.5), 0.0, bbox=(0, 30, 40, 90), half=0.05),
           make_obs("chair", 0.6, (2.2, 0, 0.5), 0.0, bbox=(60, 30, 100, 90), half=0.05),
           make_obs("chair", 0.6, (2.3, 0, 0.5), 0.0, bbox=(120, 30, 160, 90), half=0.05)]
    mem.update(make_result(0.0, obs))
    assert len(mem.all_objects()) == 3
    assert mem.objects.dedup() == {0}
    assert [n.id for n in mem.all_objects()] == [0] and mem.get_object(2).id == 0
