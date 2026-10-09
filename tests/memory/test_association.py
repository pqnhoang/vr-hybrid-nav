"""M1: consolidation + association."""
import numpy as np
import pytest
from memory_helpers import QUIET, bbox_mask, make_memory, make_obs, make_result, synonym_embedder

from vlnav.common.config import deep_merge
from vlnav.memory.consolidate import bbox_iou, consolidate

C = (2.0, 0.0, 0.5)


def objs(mem):
    return mem.all_objects(include_tentative=True)


def test_consolidate_groups_by_mask_iou_and_falls_back_to_bbox():
    a = make_obs("jacket", 0.4, C, 0.0, bbox=(40, 30, 100, 90))
    b = make_obs("raincoat", 0.3, C, 0.0, bbox=(42, 30, 100, 90))
    c = make_obs("chair", 0.5, (3, 0, 0.5), 0.0, bbox=(110, 30, 150, 90))
    groups = consolidate([a, b, c], [bbox_mask(a.bbox), bbox_mask(b.bbox), None], (120, 160), 0.7)
    assert len(groups) == 2
    assert groups[0].label_votes == {"jacket": 0.4, "raincoat": 0.3}
    assert groups[0].label == "jacket" and groups[0].score == 0.4 and groups[0].n_members == 2
    # no masks at all -> bbox IoU decides
    assert bbox_iou(a.bbox, b.bbox) > 0.7
    assert len(consolidate([a, b], [None, None], (120, 160), 0.7)) == 1


def test_duplicate_labels_in_one_frame_become_one_node(tmp_path):
    mem = make_memory(tmp_path, QUIET)
    obs = [make_obs("jacket", 0.4, C, 0.0), make_obs("raincoat", 0.3, C, 0.0)]
    mem.update(make_result(0.0, obs))
    nodes = objs(mem)
    assert len(nodes) == 1
    assert nodes[0].label == "jacket"
    assert nodes[0].label_votes == {"jacket": 0.4, "raincoat": 0.3}


def test_two_objects_in_one_frame_stay_two_nodes(tmp_path):
    """Both observations pass the gates for node 0; one-to-one matching keeps them apart."""
    mem = make_memory(tmp_path, QUIET)
    mem.update(make_result(0.0, [make_obs("chair", 0.4, C, 0.0, bbox=(20, 30, 60, 90))]))
    second = [make_obs("chair", 0.4, C, 1.0, bbox=(20, 30, 60, 90)),
              make_obs("chair", 0.4, (2.2, 0.0, 0.5), 1.0, bbox=(90, 30, 130, 90))]
    mem.update(make_result(1.0, second))
    nodes = objs(mem)
    assert len(nodes) == 2
    assert nodes[0].n_obs == 2 and nodes[1].n_obs == 1
    assert np.allclose(nodes[0].centroid, C, atol=0.03)
    # next frame: each chair goes back to its own node
    mem.update(make_result(2.0, [make_obs("chair", 0.4, (2.2, 0.0, 0.5), 2.0, bbox=(90, 30, 130, 90)),
                                 make_obs("chair", 0.4, C, 2.0, bbox=(20, 30, 60, 90))]))
    assert [n.n_obs for n in objs(mem)] == [3, 2]


def test_semantic_gate_uses_embedding_similarity(tmp_path):
    emb = synonym_embedder({("sofa", "couch"): 0.9, ("chair", "table"): 0.3})
    for embedder, expected in ((emb, 1), (None, 2)):
        mem = make_memory(tmp_path / str(expected), QUIET, embedder=embedder)
        mem.update(make_result(0.0, [make_obs("sofa", 0.4, C, 0.0)]))
        mem.update(make_result(1.0, [make_obs("couch", 0.4, C, 1.0)]))
        assert len(objs(mem)) == expected
    # dissimilar labels never merge even with full overlap
    mem = make_memory(tmp_path / "tc", QUIET, embedder=emb)
    mem.update(make_result(0.0, [make_obs("chair", 0.4, C, 0.0)]))
    mem.update(make_result(1.0, [make_obs("table", 0.4, C, 1.0)]))
    assert len(objs(mem)) == 2


def test_geometric_gate_overlap_or_close_distance(tmp_path):
    def run(c0, c1, half, name):
        mem = make_memory(tmp_path / name, QUIET)
        mem.update(make_result(0.0, [make_obs("box", 0.4, c0, 0.0, half=half)]))
        mem.update(make_result(1.0, [make_obs("box", 0.4, c1, 1.0, half=half)]))
        return len(objs(mem))

    # dist 0.3 > assoc_close_m but voxel overlap 0.5 >= assoc_overlap_min -> merge
    assert run((2.0, 0, 0.5), (2.3, 0, 0.5), 0.3, "overlap") == 1
    # no overlap but dist 0.2 < assoc_close_m -> merge
    assert run((2.0, 0, 0.5), (2.2, 0, 0.5), 0.05, "close") == 1
    # no overlap, dist 0.4 -> new node
    assert run((2.0, 0, 0.5), (2.4, 0, 0.5), 0.05, "far") == 2


def test_assoc_thresholds_come_from_config(tmp_path):
    cfg = deep_merge(QUIET, {"association": {"assoc_close_m": 0.1}})
    mem = make_memory(tmp_path, cfg)
    mem.update(make_result(0.0, [make_obs("box", 0.4, (2.0, 0, 0.5), 0.0, half=0.05)]))
    mem.update(make_result(1.0, [make_obs("box", 0.4, (2.2, 0, 0.5), 1.0, half=0.05)]))
    assert len(objs(mem)) == 2


def test_partial_view_then_close_view_merges_with_min_overlap(tmp_path):
    """Real-bag case: small partial node, then a much larger observation of the same object."""
    small = make_obs("umbrella", 0.6, (2.0, 0, 0.5), 0.0, half=0.05)                  # 8 voxels
    big = make_obs("umbrella", 0.6, (2.25, 0, 0.5), 1.0, half=0.3)                   # 1728 voxels
    for norm, expected in (("observation", 2), ("min", 1)):
        mem = make_memory(tmp_path / norm, deep_merge(QUIET, {"association": {"overlap_norm": norm}}))
        mem.update(make_result(0.0, [small]))
        mem.update(make_result(1.0, [big]))
        assert len(objs(mem)) == expected


def test_mask_iou_box_crop_matches_full_image():
    from vlnav.memory.consolidate import mask_iou
    rng = np.random.default_rng(0)
    for _ in range(50):
        boxes = []
        masks = []
        for _ in range(2):
            x1, y1 = rng.integers(0, 120, 2)
            x2, y2 = x1 + rng.integers(1, 40), y1 + rng.integers(1, 40)
            m = np.zeros((120, 160), bool)
            m[y1:y2, x1:x2] = rng.random((y2 - y1, x2 - x1))[: 120 - y1, : 160 - x1] > 0.3
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            masks.append(m)
        assert mask_iou(*masks, *boxes) == pytest.approx(mask_iou(*masks))
