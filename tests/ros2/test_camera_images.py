"""Camera snapshots show the actual detection and selected memory view."""
from types import SimpleNamespace

import cv2
import numpy as np

from vlnav_ros.camera_images import (MOVE_COLOR, CameraImageWriter, _fit, _move_line, _planner_lines,
                                     annotate_detections)


def test_camera_detections_and_selected_goal_are_labeled(tmp_path):
    writer = CameraImageWriter(tmp_path)
    rgb = np.full((240, 320, 3), 255, np.uint8)
    detection = SimpleNamespace(label="sofa", score=.75, bbox=(40, 50, 180, 170))
    other = SimpleNamespace(label="chair", score=.6, bbox=(200, 50, 300, 170))
    result = SimpleNamespace(frame=SimpleNamespace(rgb=rgb, stamp=123.456),
                             detections=[detection, other], relevant_idx=[0])
    image = annotate_detections(result)
    assert tuple(image[100, 40]) == (0, 200, 255)   # task match: yellow
    assert tuple(image[100, 200]) == (255, 220, 0)  # other detection: cyan
    assert writer.write_frame(image, result.frame.stamp) == tmp_path / "camera_123.456.jpg"
    assert (tmp_path / "latest_camera.jpg").is_file()
    annotated = cv2.imread(str(tmp_path / "latest_camera.jpg"))
    assert annotated.shape == (240, 320, 3)
    assert annotated[100, 40, 2] > 200  # colored box on the full camera frame
    assert writer.write_frame(annotate_detections(SimpleNamespace(**{**vars(result), "relevant_idx": []})),
                              123.956) == tmp_path / "camera_123.956.jpg"

    original = tmp_path / "stored.jpg"
    cv2.imwrite(str(original), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    node = SimpleNamespace(id=4, label="sofa", best_view_image_path=str(original),
                           best_view_bbox=(40, 50, 180, 170))
    assert writer.write_goal(node, 1, 2) == tmp_path / "goal_1_2_node_4.jpg"
    selected = cv2.imread(str(tmp_path / "latest_goal.jpg"))
    assert selected.shape == (240, 320, 3)
    assert selected[100, 40, 2] > 200


def test_camera_overlay_shows_current_task_mode_and_reason():
    rgb = np.full((480, 640, 3), 255, np.uint8)
    result = SimpleNamespace(frame=SimpleNamespace(rgb=rgb), detections=[], relevant_idx=[])
    state = SimpleNamespace(fsm="EXECUTING", instruction="Find a sofa, then reach the blue table",
                            current=SimpleNamespace(type="explore", target="blue table", node_id=None),
                            done_reason=None)
    lines = _planner_lines(state, "The sofa was reached; search for a table that is blue.")
    assert lines == ["Task: Find a sofa, then reach the blue table", "Mode: EXPLORE blue table",
                     "Reason: The sofa was reached; search for a table that is blue."]
    image = annotate_detections(result, state, lines[-1][8:])
    assert (image[:90] != 255).any() and (image[110:] == 255).all()
    assert _planner_lines(SimpleNamespace(**{**vars(state), "fsm": "REPLANNING", "current": None}),
                          "old reason")[1:] == ["Mode: REPLAN", "Reason: Waiting for planner decision"]
    assert _fit("Task: " + "long text " * 60, 300, cv2.FONT_HERSHEY_SIMPLEX, .5).endswith("...")


def test_camera_overlay_shows_s2_pixel_goal_and_texts_during_move():
    rgb = np.full((480, 640, 3), 255, np.uint8)
    result = SimpleNamespace(frame=SimpleNamespace(rgb=rgb), detections=[], relevant_idx=[])
    state = SimpleNamespace(fsm="EXECUTING", instruction="Walk to the door",
                            current=SimpleNamespace(type="move", target="door", node_id=None,
                                                    instruction="walk to the door"), done_reason=None)
    snap = {"active": True, "last": {"k": 3, "decision": "new_goal", "reason": None,
                               "exchanges": [{"look_down": False, "text": "↓"}, {"look_down": True, "text": "320 400"}],
                               "projection": {"uv": [200.0, 150.0], "cam_xy": [-1.0, -1.0], "p_map": [0.5, 0.5, 0.0],
                                              "floor_xy": [0.5, 0.5]}}}
    assert _planner_lines(state, "r")[2] == "Reason: route: walk to the door"
    assert _move_line(snap["last"]) == "S2 #3: v | LD 320 400 -> new_goal"
    image = annotate_detections(result, state, "r", snap["last"])
    assert tuple(image[150, 200 + 8]) == MOVE_COLOR                  # ring around the projected pixel
    assert (image[150:, 400:] == 255).all()
    plain = annotate_detections(result, state, "r")
    assert (image[:140] != plain[:140]).any()                         # fourth (S2) panel line
