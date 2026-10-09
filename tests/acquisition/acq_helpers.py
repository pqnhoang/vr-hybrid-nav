"""Fixture memory with real image files, stub embedder and scripted VLM replies."""
from __future__ import annotations

import json
import zlib

import cv2
import numpy as np

from vlnav.acquisition.config import load_acquisition_config
from vlnav.acquisition.label_matcher import LabelMatcher
from vlnav.acquisition.memory_api import InMemoryMemory

W, H = 320, 240


class TableEmbedder:
    """Deterministic vectors: texts in the same group are similar (cos 0.95), others 0.5."""

    def __init__(self, groups: list[list[str]]):
        self.group = {t: g for g, members in enumerate(groups) for t in members}

    def encode(self, texts):
        out = []
        for t in texts:
            t = t.replace("a photo of a ", "")
            v = np.zeros(64)
            g = self.group.get(t)
            v[0] = 1.0
            if g is not None:
                v[1 + g] = 1.6                      # same group -> high cosine
            v[40 + zlib.crc32(t.encode()) % 20] = 0.4   # every text slightly distinct (stable hash)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


GROUPS = [["rain jacket", "raincoat", "jacket", "coat"], ["umbrella"], ["shoes", "boots"],
          ["chair", "office chair"], ["desk", "table"]]


def matcher(threshold=0.9, synonyms=None, rel_margin=None):
    return LabelMatcher(TableEmbedder(GROUPS), "a photo of a {}", threshold, synonyms or {}, rel_margin)


def write_image(path, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


OBJECTS = [  # id, label, score, centroid, room, n_obs, first_seen_t, bbox
    (17, "jacket", 0.62, (3.1, 4.2, 1.0), 0, 4, 118.0, (100, 60, 200, 200)),
    (18, "raincoat", 0.55, (3.4, 4.0, 1.0), 0, 2, 140.0, (0, 0, 80, 120)),       # touches border
    (19, "jacket", 0.62, (8.0, 1.0, 1.0), 1, 3, 90.0, (150, 100, 300, 230)),
    (20, "coat", 0.30, (8.5, 1.2, 1.0), 1, 2, 95.0, (10, 10, 60, 60)),
    (21, "umbrella", 0.40, (1.0, 1.0, 0.8), 2, 5, 50.0, (40, 40, 120, 200)),
    (22, "chair", 0.80, (3.5, 4.5, 0.5), 0, 6, 60.0, (20, 120, 120, 230)),
    (23, "desk", 0.70, (3.0, 4.6, 0.7), 0, 6, 61.0, (150, 20, 310, 120)),
    (24, "jacket", 0.10, (9.0, 9.0, 1.0), 1, 2, 70.0, (10, 10, 40, 40)),         # below min score
]


def fixture(tmp_path, objects=OBJECTS):
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    objs = []
    for i, (nid, lbl, s, c, room, n, t, bbox) in enumerate(objects):
        write_image(img_dir / f"{nid}.jpg", nid)
        objs.append({"id": nid, "label": lbl, "centroid": list(c), "best_score": s,
                     "best_view_pose": {"position": [c[0] - 1, c[1], 0.0],
                                        "orientation_xyzw": [0, 0, 0, 1], "stamp": t},
                     "best_view_image_path": f"images/{nid}.jpg", "best_view_bbox": list(bbox),
                     "room_id": room, "n_obs": n, "first_seen_t": t})
    data = {"rooms": [{"id": 0, "label": "bedroom"}, {"id": 1, "label": "hallway"},
                      {"id": 2, "label": "entryway"}],
            "objects": objs, "aliases": {"99": 17}}
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(data))
    return path


def memory(tmp_path, **kw) -> InMemoryMemory:
    return InMemoryMemory.from_json(fixture(tmp_path, **kw))


def cfg(**overrides):
    base = {"logging": {"debug_jsonl": None}}
    for k, v in overrides.items():
        base.setdefault(k, {}).update(v)
    return load_acquisition_config(overrides=base)


def reply(choice, confidence=0.9, reasoning="looks right"):
    return json.dumps({"reasoning": reasoning, "choice": choice, "confidence": confidence})
