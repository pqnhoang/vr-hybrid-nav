"""Open-vocabulary detection + segmentation (SYSTEM.md §4.2).

`YoloWorldFastSam`: YOLO-World boxes (paper's detector), FastSAM "segment everything" masks, each
box takes the mask with the best box/mask IoU (the rule of ultralytics' FastSAM box prompt, kept
per box so detections and masks stay paired). Boxes without a good mask fall back to the box.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vlnav.common.models import Detection2D
from vlnav.perception.config import CheckpointPaths, DetectorCfg


class OpenVocabDetector(Protocol):
    def set_vocabulary(self, labels: list[str]) -> None: ...

    def detect(self, rgb: np.ndarray, key: str | None = None) -> list[Detection2D]: ...


def vocab_hash(labels: list[str]) -> str:
    return hashlib.sha1(json.dumps(labels).encode()).hexdigest()[:12]


def _detection(label: str, score: float, bbox, mask: np.ndarray) -> Detection2D:
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean())) if len(xs) else (
        (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return Detection2D(label=label, score=float(score), bbox=tuple(int(v) for v in bbox),
                       mask=mask, mask_centroid_uv=centroid)


def box_mask(bbox, shape) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = (int(v) for v in bbox)
    m[y1:y2, x1:x2] = True
    return m


def assign_masks(bboxes: np.ndarray, masks: np.ndarray, min_iou: float) -> list[int | None]:
    """For each box, index of the mask maximizing |mask ∩ box| / |box ∪ mask| (None if < min_iou)."""
    out: list[int | None] = []
    if len(masks) == 0:
        return [None] * len(bboxes)
    full = masks.reshape(len(masks), -1).sum(1).astype(float)
    for x1, y1, x2, y2 in bboxes.astype(int):
        inside = masks[:, y1:y2, x1:x2].reshape(len(masks), -1).sum(1).astype(float)
        union = (x2 - x1) * (y2 - y1) + full - inside
        iou = np.divide(inside, union, out=np.zeros_like(inside), where=union > 0)
        j = int(np.argmax(iou))
        out.append(j if iou[j] >= min_iou else None)
    return out


class DetectionCache:
    """Detections on disk keyed by (frame key, vocab hash); masks bit-packed + compressed."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, vhash: str) -> Path:
        return self.root / vhash / f"{key}.npz"

    def get(self, key: str, vhash: str) -> list[Detection2D] | None:
        p = self._path(key, vhash)
        if not p.exists():
            return None
        with np.load(p) as d:
            shape = tuple(d["shape"])
            labels = json.loads(str(d["labels"]))
            masks = np.unpackbits(d["masks"], axis=1)[:, :shape[0] * shape[1]]
            return [_detection(lbl, s, b, m.reshape(shape).astype(bool))
                    for lbl, s, b, m in zip(labels, d["scores"], d["bboxes"], masks)]

    def put(self, key: str, vhash: str, dets: list[Detection2D], shape) -> None:
        p = self._path(key, vhash)
        p.parent.mkdir(parents=True, exist_ok=True)
        masks = np.stack([d.mask.ravel() for d in dets]) if dets else np.zeros((0, shape[0] * shape[1]), bool)
        np.savez_compressed(p, shape=np.array(shape), labels=json.dumps([d.label for d in dets]),
                            scores=np.array([d.score for d in dets], dtype=float),
                            bboxes=np.array([d.bbox for d in dets], dtype=int).reshape(-1, 4),
                            masks=np.packbits(masks, axis=1))


class YoloWorldFastSam:
    def __init__(self, cfg: DetectorCfg, paths: CheckpointPaths):
        from ultralytics import FastSAM, YOLOWorld          # heavy import, only when used
        for p in (paths.yolo_world, paths.fastsam, paths.clip_text):
            if not Path(p).exists():
                raise FileNotFoundError(f"checkpoint missing: {p} (see configs/checkpoints.yaml)")
        clip_cache = Path.home() / ".cache" / "clip" / Path(paths.clip_text).name
        if not clip_cache.exists():
            raise FileNotFoundError(f"YOLO-World loads its CLIP text encoder from {clip_cache}; "
                                    f"symlink {paths.clip_text} there")
        self.cfg = cfg
        self.yolo = YOLOWorld(paths.yolo_world)
        self.fastsam = FastSAM(paths.fastsam)
        self.vocab: list[str] = []
        self.n_set_vocabulary = 0
        self.cache = DetectionCache(cfg.cache_dir) if cfg.cache_dir else None

    def set_vocabulary(self, labels: list[str]) -> None:
        if list(labels) == self.vocab:
            return
        self.vocab = list(labels)
        # ultralytics caches the CLIP text model with the device it was built on; after predict()
        # moved YOLO-World (and that submodule) to the GPU the cached `device` is stale -> sync it
        inner = self.yolo.model
        clip_model = getattr(inner, "clip_model", None)
        if clip_model is not None:
            clip_model.device = next(inner.parameters()).device
        self.yolo.set_classes(list(labels))
        self.n_set_vocabulary += 1

    def detect(self, rgb: np.ndarray, key: str | None = None) -> list[Detection2D]:
        # cache key covers everything that changes the output: vocabulary + detector settings
        vhash = vocab_hash([*self.vocab, self.cfg.model_dump_json(exclude={"cache_dir", "device"})])
        if self.cache is not None and key is not None:
            cached = self.cache.get(key, vhash)
            if cached is not None:
                return cached
        c = self.cfg
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)          # ultralytics expects BGR arrays
        det = self.yolo.predict(bgr, imgsz=c.yolo_imgsz, conf=c.conf, iou=c.iou, max_det=c.max_det,
                                device=c.device, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        out: list[Detection2D] = []
        if len(boxes):
            seg = self.fastsam.predict(bgr, imgsz=c.fastsam_imgsz, conf=c.fastsam_conf,
                                       iou=c.fastsam_iou, retina_masks=True, device=c.device,
                                       verbose=False)[0]
            masks = (seg.masks.data.cpu().numpy() > 0.5) if seg.masks is not None else \
                np.zeros((0, *rgb.shape[:2]), bool)
            names = det.names
            for b, cls, s, j in zip(boxes, det.boxes.cls.cpu().numpy().astype(int),
                                    det.boxes.conf.cpu().numpy(),
                                    assign_masks(boxes, masks, c.mask_min_iou)):
                m = masks[j] & box_mask(b, rgb.shape[:2]) if j is not None else box_mask(b, rgb.shape[:2])
                out.append(_detection(names[int(cls)], s, b, m))
        if self.cache is not None and key is not None:
            self.cache.put(key, vhash, out, rgb.shape[:2])
        return out


class MockDetector:
    """Returns scripted detections (tests)."""

    def __init__(self, script: dict[str, list[Detection2D]] | None = None):
        self.script = script or {}
        self.vocab: list[str] = []
        self.n_set_vocabulary = 0

    def set_vocabulary(self, labels: list[str]) -> None:
        if list(labels) != self.vocab:
            self.vocab = list(labels)
            self.n_set_vocabulary += 1

    def detect(self, rgb: np.ndarray, key: str | None = None) -> list[Detection2D]:
        return [d for d in self.script.get(key, []) if d.label in self.vocab]
