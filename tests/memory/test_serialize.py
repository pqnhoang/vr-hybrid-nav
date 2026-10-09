"""M6 (first part): scene-graph JSON for /memory/scene_graph and the mirror."""
import json

from memory_helpers import build_fixture_memory

from vlnav.memory.serialize import snapshot_from_json, snapshot_to_dict, snapshot_to_json

OBJECT_KEYS = {"id", "label", "label_votes", "centroid", "best_score", "best_view_pose",
               "best_view_cam_pose", "best_view_image", "best_view_bbox", "room_id", "n_obs",
               "first_seen_t", "last_seen_t", "confirmed"}
ROOM_KEYS = {"id", "label", "label_confidence", "centroid_xy", "area_m2", "adjacent_room_ids",
             "object_ids", "polygon"}


def test_json_schema_without_geometry(tmp_path):
    mem = build_fixture_memory(tmp_path)
    d = json.loads(snapshot_to_json(mem.view()))
    assert {"version", "stamp", "frame", "rooms", "objects", "aliases"} <= d.keys()
    assert d["frame"] == "map" and d["aliases"] == {"6": 5}
    assert all(o.keys() == OBJECT_KEYS for o in d["objects"])
    assert all(r.keys() == ROOM_KEYS for r in d["rooms"])
    assert d["objects"][0]["best_view_image"] == "cam0_0.000.jpg"       # name, not a path
    assert all(len(r["polygon"]) >= 4 for r in d["rooms"])


def test_round_trip_is_lossless(tmp_path):
    mem = build_fixture_memory(tmp_path)
    for geometry in (False, True):
        text = snapshot_to_json(mem.view(), include_geometry=geometry)
        back = snapshot_from_json(text, image_root=mem.images.root, max_neighbors=6)
        assert snapshot_to_dict(back, include_geometry=geometry) == json.loads(text)
