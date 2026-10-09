"""Image helpers shared by memory, acquisition and the planner."""
from __future__ import annotations

import cv2
import numpy as np


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def limit_size(img: np.ndarray, max_px: int) -> np.ndarray:
    """Resize so the long side is <= max_px (INTER_AREA), unchanged if already smaller."""
    long_side = max(img.shape[:2])
    if long_side <= max_px:
        return img
    s = max_px / long_side
    return cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def crop_bbox(img: np.ndarray, bbox, margin: float, max_px: int) -> np.ndarray:
    """Crop of `bbox` grown by `margin` x its size, clamped to the image, long side <= max_px."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    x1, x2 = int(max(0, np.floor(x1 - mx))), int(min(w, np.ceil(x2 + mx)))
    y1, y2 = int(max(0, np.floor(y1 - my))), int(min(h, np.ceil(y2 + my)))
    out = img[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    return np.ascontiguousarray(limit_size(out, max_px))


def draw_bbox(img: np.ndarray, bbox, color, thickness: int) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), tuple(int(c) for c in color), thickness)
    return out
