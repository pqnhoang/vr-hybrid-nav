"""Draw detector boxes once for the ROS camera topic and optional JPEG snapshots."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _box(image: np.ndarray, bbox, label: str, color=(0, 200, 255)) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, baseline = cv2.getTextSize(label, font, .7, 2)
    top = max(0, y1 - size[1] - baseline - 8)
    cv2.rectangle(image, (x1, top), (x1 + size[0] + 8, top + size[1] + baseline + 8), color, -1)
    cv2.putText(image, label, (x1 + 4, top + size[1] + 4), font, .7, (0, 0, 0), 2,
                cv2.LINE_AA)


def annotate_detections(result) -> np.ndarray:
    """Return the processed camera frame with every detector box and label."""
    image = cv2.cvtColor(result.frame.rgb, cv2.COLOR_RGB2BGR)
    relevant = set(result.relevant_idx)
    for i, det in enumerate(result.detections):
        color = (0, 200, 255) if i in relevant else (255, 220, 0)
        _box(image, det.bbox, f"{det.label} {det.score:.2f}", color)
    return image


class CameraImageWriter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _save(self, image: np.ndarray, name: str, latest: str) -> Path:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError("OpenCV could not encode the camera JPEG")
        data = encoded.tobytes()
        path = self.directory / name
        for destination in (path, self.directory / latest):
            temporary = self.directory / f".{destination.name}.tmp"
            temporary.write_bytes(data)
            temporary.replace(destination)
        return path

    def write_frame(self, image: np.ndarray, stamp: float) -> Path:
        """Save the image published on /perception/annotated_image."""
        return self._save(image, f"camera_{stamp:.3f}.jpg", "latest_camera.jpg")

    def write_goal(self, node, task_id: int, subtask_id: int) -> Path:
        """Keep the full best-view frame used for a selected object goal."""
        image = cv2.imread(node.best_view_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(node.best_view_image_path)
        _box(image, node.best_view_bbox, f"GOAL {node.label} #{node.id}")
        return self._save(image, f"goal_{task_id}_{subtask_id}_node_{node.id}.jpg",
                          "latest_goal.jpg")
