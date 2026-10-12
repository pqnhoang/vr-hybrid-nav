"""Camera snapshots show the actual detection and selected memory view."""
from types import SimpleNamespace

import cv2
import numpy as np

from vlnav_ros.camera_images import CameraImageWriter, annotate_detections


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
