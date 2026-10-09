"""A1 (fixture memory) + A2 (label matcher, symbolic filter)."""
import pytest
from acq_helpers import cfg, matcher, memory

from vlnav.acquisition.memory_api import MemoryView
from vlnav.acquisition.symbolic_filter import SymbolicFilter


def ids(xs):
    return [x.node.id if hasattr(x, "node") else x.id for x in xs]


def test_fixture_memory_until_t_and_neighbors(tmp_path):
    mem = memory(tmp_path)
    assert isinstance(mem, MemoryView)
    assert ids(mem.all_objects(until_t=100.0)) == [19, 20, 21, 22, 23, 24]
    assert ids(mem.snapshot(100.0).all_objects()) == ids(mem.all_objects(until_t=100.0))
    assert ids(mem.neighbors(17, 1.0, same_room=False)) == [18, 23, 22]       # 0.36, 0.51, 0.71 m
    assert ids(mem.neighbors(19, 1.0, same_room=True)) == [20]
    assert mem.get_object(99).id == 17 and mem.get_room(0).label == "bedroom"
    assert mem.get_room(0).object_ids == [17, 18, 22, 23]
    assert mem.best_view_crop(17, 0.0, 50).shape == (50, 36, 3)


def test_label_matcher_synonyms_threshold_and_margin():
    m = matcher(threshold=0.9, synonyms={"rain jacket": ["umbrella"]})
    s = m.score("rain jacket", ["raincoat", "umbrella", "chair"])
    assert s["umbrella"] == 1.0 and s["raincoat"] > 0.9 > s["chair"]
    assert set(m.matches("rain jacket", ["raincoat", "jacket", "chair"])) == {"raincoat", "jacket"}
    exact = matcher(threshold=0.5, rel_margin=0.01)
    assert set(exact.matches("jacket", ["jacket", "coat", "chair"])) == {"jacket"}
    assert matcher(threshold=0.9).score("Rain  Jacket", ["rain jacket"])["rain jacket"] == 1.0


@pytest.mark.parametrize("strategy,expected", [("confidence", [17, 19, 18]),
                                               ("confidence_x_similarity", [17, 19, 18])])
def test_topk_ranking_tiebreak_and_score_gate(tmp_path, strategy, expected):
    f = SymbolicFilter(cfg(symbolic={"rank_strategy": strategy}).symbolic, matcher())
    top = f.topk("rain jacket", memory(tmp_path), k=3)
    assert ids(top) == expected                     # 17 and 19 tie at 0.62 -> more n_obs first
    assert all(c.match_score >= 0.9 for c in top)
    assert 24 not in ids(f.candidates("rain jacket", memory(tmp_path)))     # min_det_score


def test_topk_blacklist_and_until_t(tmp_path):
    f = SymbolicFilter(cfg().symbolic, matcher())
    assert ids(f.topk("rain jacket", memory(tmp_path), 3, exclude={17, 19})) == [18, 20]
    assert ids(f.topk("rain jacket", memory(tmp_path), 3, until_t=100.0)) == [19, 20]
    assert f.topk("bicycle", memory(tmp_path), 3) == []
