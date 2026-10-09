"""S2: lifting accuracy, mask assignment, vocabulary switching, dynamic-object filter."""
import numpy as np
import pytest

from vlnav.common.models import Detection2D, Frame
from vlnav.perception.config import DynamicCfg, load_perception_config
from vlnav.perception.detector import MockDetector, assign_masks, box_mask
from vlnav.perception.lifting import lift_mask
from vlnav.perception.pipeline import PerceptionPipeline, VocabularyManager

LIFT = load_perception_config().lifting
K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
H, W = 240, 320


def cam_pose():
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]      # optical z forward = map x
    T[:3, 3] = (0, 0, 1.0)
    return T


def box_scene(bg_depth=6.0, box_depth=2.0, box=(120, 80, 200, 160)):
    depth = np.full((H, W), bg_depth, np.float32)
    x1, y1, x2, y2 = box
    depth[y1:y2, x1:x2] = box_depth
    return depth, box_mask(box, (H, W))


def test_lift_box_centroid_within_2cm():
    depth, mask = box_scene()
    lifted = lift_mask(mask, depth, K, cam_pose(), LIFT, np.random.default_rng(0))
    # box center pixel (160, 120) is on the optical axis -> map point (2.0, 0, 1.0)
    assert np.allclose(lifted.centroid_map, (2.0, 0.0, 1.0), atol=0.02)


def test_mad_rejects_leaking_background():
    depth, mask = box_scene()
    leaky = box_mask((100, 60, 220, 180), (H, W))           # mask bleeds onto the background
    lifted = lift_mask(leaky, depth, K, cam_pose(), LIFT, np.random.default_rng(0))
    assert lifted.centroid_map[0] == pytest.approx(2.0, abs=0.02)
    assert lifted.points_map[:, 0].max() < 2.1


def test_assign_masks_pairs_each_box_with_its_mask():
    m0, m1 = box_mask((10, 10, 50, 50), (H, W)), box_mask((100, 100, 180, 200), (H, W))
    boxes = np.array([[95, 98, 182, 201], [8, 12, 52, 49], [250, 10, 300, 40]])
    assert assign_masks(boxes, np.stack([m0, m1]), 0.3) == [1, 0, None]


def make_frame():
    depth, _ = box_scene(bg_depth=4.0)                    # background within lifting range
    return Frame(stamp=1.0, camera_id="c", rgb=np.zeros((H, W, 3), np.uint8), K=K, hfov=1.0,
                 T_map_cam=cam_pose(), T_map_base=np.eye(4), depth=depth,
                 sensor_origin_map=(0, 0, 1.0))


def det(label, box, score=0.6):
    return Detection2D(label=label, score=score, bbox=box, mask=box_mask(box, (H, W)),
                       mask_centroid_uv=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


def test_vocabulary_set_only_on_change_and_relevance():
    detector = MockDetector()
    pipe = PerceptionPipeline(detector, VocabularyManager(["chair", "table"]), LIFT)
    pipe.set_active_prompts(["umbrella"])
    pipe.set_active_prompts(["umbrella"])
    assert detector.n_set_vocabulary == 2 and detector.vocab == ["chair", "table", "umbrella"]
    detector.script = {"c_1.000": [det("chair", (120, 80, 200, 160)), det("umbrella", (0, 0, 40, 40))]}
    res = pipe.process(make_frame())
    assert res.relevant_idx == [1] and [o.label for o in res.observations][0] == "chair"


def test_dynamic_filter_drops_people_and_what_they_carry():
    detector = MockDetector({"c_1.000": [det("person", (100, 40, 220, 230)),
                                         det("backpack", (120, 80, 200, 160)),
                                         det("bench", (230, 80, 310, 160))]})
    pipe = PerceptionPipeline(detector, VocabularyManager(["backpack", "bench"], dynamic=["person"]),
                              LIFT, DynamicCfg(labels=["person"], max_bbox_overlap=0.5))
    res = pipe.process(make_frame())
    assert len(res.detections) == 3
    assert [o.label for o in res.observations] == ["bench"] and pipe.n_dynamic_dropped == 1
