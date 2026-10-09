"""Object-centric image memory (TASK_unified_memory.md §7).

Full frames are stored only when they become the best view of some node, as
`root/{camera_id}_{stamp:.3f}.jpg`. References are counted (current nodes + retained checkpoints);
frames whose count drops to zero are deleted by `gc()`, which memory calls at checkpoint time.
Crops are generated on demand with an LRU cache. Writes are synchronous in this first build.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from vlnav.common.images import crop_bbox, load_rgb
from vlnav.common.models import Frame


class ImageStore:
    def __init__(self, root: str | Path, jpeg_quality: int, crop_cache_size: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = jpeg_quality
        self.crop_cache_size = crop_cache_size
        self._refs: dict[str, int] = {}
        self._garbage: set[str] = set()
        self._crops: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()     # crops may be requested from planner threads

    @staticmethod
    def frame_name(camera_id: str, stamp: float) -> str:
        return f"{camera_id}_{stamp:.3f}.jpg"

    def path_of(self, name: str) -> str:
        return str(self.root / name)

    def put_frame(self, frame: Frame) -> str:
        """Write the frame once (idempotent per camera/stamp) and return its path."""
        path = self.path_of(self.frame_name(frame.camera_id, frame.stamp))
        if not Path(path).exists():
            bgr = cv2.cvtColor(np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                raise OSError(f"failed to write {path}")
        return path

    def acquire(self, path: str) -> None:
        self._refs[path] = self._refs.get(path, 0) + 1
        self._garbage.discard(path)

    def release(self, path: str) -> None:
        n = self._refs.get(path, 0) - 1
        if n < 0:
            raise RuntimeError(f"release of unreferenced image {path}")
        if n == 0:
            del self._refs[path]
            self._garbage.add(path)
        else:
            self._refs[path] = n

    def refcount(self, path: str) -> int:
        return self._refs.get(path, 0)

    def gc(self) -> list[str]:
        """Delete frames that are no longer referenced. Returns deleted paths."""
        deleted = []
        for path in sorted(self._garbage):
            if self._refs.get(path, 0) == 0:
                Path(path).unlink(missing_ok=True)
                deleted.append(path)
        self._garbage.clear()
        if deleted:
            gone = set(deleted)
            with self._lock:
                for key in [k for k in self._crops if k[0] in gone]:
                    del self._crops[key]
        return deleted

    def load_rgb(self, path: str) -> np.ndarray:
        return load_rgb(path)

    def crop(self, path: str, bbox: tuple[int, int, int, int], margin: float,
             max_px: int) -> np.ndarray:
        """RGB crop of `bbox` grown by `margin` x its size, clamped, long side <= max_px."""
        key = (path, tuple(bbox), margin, max_px)
        with self._lock:
            if key in self._crops:
                self._crops.move_to_end(key)
                return self._crops[key]
        out = crop_bbox(load_rgb(path), bbox, margin, max_px)
        out.flags.writeable = False
        with self._lock:
            self._crops[key] = out
            while len(self._crops) > self.crop_cache_size:
                self._crops.popitem(last=False)
        return out
