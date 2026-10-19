"""Virtual Habitat camera for DualVLN (TASK_move_subtask.md §7.1, settled in M0).

DualVLN was trained on Habitat images: 640×480, hfov 79°, principal point at the center; pixel goals
are predicted on the 640×480 look-down image and come back in its coordinates. A robot camera with
another size / field of view is shown to the model through a crop centered on the principal point
with the training field of view and aspect, resized to 640×480 (`mode: habitat`). `mode: full` sends
the whole image unchanged (the pixel then refers to the original image).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelView:
    x0: float                          # crop box in the camera image (pixels, float edges)
    y0: float
    w: float
    h: float
    out_w: int                         # image size shown to the model
    out_h: int

    @property
    def scale(self) -> tuple[float, float]:
        return self.out_w / self.w, self.out_h / self.h

    def image(self, rgb: np.ndarray) -> np.ndarray:
        """Crop + resize the camera image to the model view (identity for a full view)."""
        H, W = rgb.shape[:2]
        if (self.x0, self.y0, self.w, self.h, self.out_w, self.out_h) == (0, 0, W, H, W, H):
            return rgb
        sx, sy = self.scale
        M = np.array([[sx, 0.0, -self.x0 * sx], [0.0, sy, -self.y0 * sy]])
        return cv2.warpAffine(np.ascontiguousarray(rgb), M, (self.out_w, self.out_h), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_CONSTANT)

    def to_camera(self, xy) -> tuple[float, float]:
        """Pixel in the model view -> pixel in the camera image."""
        sx, sy = self.scale
        return self.x0 + xy[0] / sx, self.y0 + xy[1] / sy

    def to_view(self, uv) -> tuple[float, float]:
        sx, sy = self.scale
        return (uv[0] - self.x0) * sx, (uv[1] - self.y0) * sy

    def K(self, K_cam: np.ndarray) -> np.ndarray:
        """Intrinsics of the model view."""
        sx, sy = self.scale
        K = np.asarray(K_cam, float).copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2] - self.x0) * sx
        K[1, 2] = (K[1, 2] - self.y0) * sy
        return K


def make_view(K: np.ndarray, image_wh, mode: str, out_wh=(640, 480), hfov_deg: float = 79.0) -> ModelView:
    W, H = image_wh
    if mode == "full":
        return ModelView(0, 0, W, H, W, H)
    fx, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    aspect = out_wh[1] / out_wh[0]
    half_w = fx * math.tan(math.radians(hfov_deg) / 2)
    half_w = min(half_w, cx, W - cx)                       # stay inside the image horizontally
    half_h = min(half_w * aspect, cy, H - cy)              # and vertically, keeping the aspect
    half_w = half_h / aspect
    return ModelView(cx - half_w, cy - half_h, 2 * half_w, 2 * half_h, out_wh[0], out_wh[1])
