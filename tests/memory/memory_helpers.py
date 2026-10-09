"""Synthetic frames, observations, floor plans and stubs for the memory tests."""
from __future__ import annotations

import numpy as np

from vlnav.common.models import Detection2D, Frame, FrameResult, Observation
from vlnav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid2D
from vlnav.memory import ImageStore, UnifiedMemory, load_memory_config

H, W = 120, 160
VOX = 0.05
CAM_HEIGHT = 1.0


def box_points(center, half=0.1, voxel=VOX) -> np.ndarray:
    """Voxel centers filling an axis-aligned cube (exact voxel keys, no rounding ambiguity)."""
    c = np.asarray(center, dtype=float)
    lo = np.round((c - half) / voxel).astype(int)
    hi = np.round((c + half) / voxel).astype(int)
    axes = [np.arange(a, b) for a, b in zip(lo, hi)]
    keys = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    return (keys + 0.5) * voxel


def base_T(xy=(0.0, 0.0), yaw=0.0) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = xy
    return T


def cam_T(xy=(0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]      # optical: z fwd = base x
    T[:3, 3] = (xy[0], xy[1], CAM_HEIGHT)
    return T


def make_frame(t: float, cam: str = "cam0", base_xy=(0.0, 0.0)) -> Frame:
    rng = np.random.default_rng(int(t * 1000) % (2 ** 32))
    rgb = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    K = np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])
    return Frame(stamp=t, camera_id=cam, rgb=rgb, K=K, hfov=2 * np.arctan(W / 200.0),
                 T_map_cam=cam_T(base_xy), T_map_base=base_T(base_xy),
                 sensor_origin_map=(base_xy[0], base_xy[1], CAM_HEIGHT))


def make_obs(label: str, score: float, center, t: float, bbox=(40, 30, 100, 90), half=0.1,
             cam: str = "cam0", base_xy=(0.0, 0.0), points=None) -> Observation:
    pts = box_points(center, half) if points is None else np.asarray(points, dtype=float)
    return Observation(stamp=t, camera_id=cam, label=label, score=score,
                       centroid_map=tuple(float(v) for v in np.median(pts, axis=0)),
                       points_map=pts, bbox=bbox, T_map_cam=cam_T(base_xy),
                       T_map_base=base_T(base_xy), frame_ref=f"{cam}_{t:.3f}")


def bbox_mask(bbox) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = bbox
    m[y1:y2, x1:x2] = True
    return m


def make_result(t: float, obs: list[Observation], cam: str = "cam0", base_xy=(0.0, 0.0),
                masks: list[np.ndarray] | None = None) -> FrameResult:
    dets = []
    for i, o in enumerate(obs):
        m = masks[i] if masks is not None else bbox_mask(o.bbox)
        x1, y1, x2, y2 = o.bbox
        dets.append(Detection2D(label=o.label, score=o.score, bbox=o.bbox, mask=m,
                                mask_centroid_uv=((x1 + x2) / 2, (y1 + y2) / 2)))
    return FrameResult(frame=make_frame(t, cam, base_xy), detections=dets, observations=obs,
                       relevant_idx=[])


class StubEmbedder:
    """Hand-set vectors for chosen texts; every other text gets its own orthogonal axis."""

    DIM = 128
    FIRST_FREE_AXIS = 16

    def __init__(self, vectors: dict[str, np.ndarray] | None = None):
        self.vectors = {k: np.asarray(v, dtype=float) for k, v in (vectors or {}).items()}
        self._next_axis = self.FIRST_FREE_AXIS
        self.n_calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.n_calls += 1
        out = []
        for t in texts:
            if t not in self.vectors:
                v = np.zeros(self.DIM)
                v[self._next_axis] = 1.0
                self._next_axis += 1
                self.vectors[t] = v
            v = np.zeros(self.DIM)
            v[:len(self.vectors[t])] = self.vectors[t]
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def synonym_embedder(pairs: dict[tuple[str, str], float]) -> StubEmbedder:
    """Embedder where each (a, b) pair has exactly the given cosine similarity."""
    vecs = {}
    for k, ((a, b), sim) in enumerate(pairs.items()):
        e1, e2 = np.zeros(StubEmbedder.DIM), np.zeros(StubEmbedder.DIM)
        e1[2 * k], e2[2 * k + 1] = 1.0, 1.0
        vecs[a] = e1
        vecs[b] = sim * e1 + np.sqrt(1 - sim ** 2) * e2
    return StubEmbedder(vecs)


# ---------------------------------------------------------------- floor plans (cells at 0.1 m)
RES = 0.1


def floor_plan(shape, rooms, doors=(), openings=()) -> np.ndarray:
    """rooms: (x0, y0, x1, y1) free interiors in cells, surrounded by 1-cell occupied walls;
    doors / openings: free rectangles cut into walls afterwards."""
    g = np.full(shape, UNKNOWN, dtype=np.int8)
    for x0, y0, x1, y1 in rooms:
        g[y0 - 1:y1 + 1, x0 - 1:x1 + 1] = OCCUPIED
    for x0, y0, x1, y1 in rooms:
        g[y0:y1, x0:x1] = FREE
    for x0, y0, x1, y1 in (*doors, *openings):
        g[y0:y1, x0:x1] = FREE
    return g


def make_grid(arr: np.ndarray, res: float = RES, origin=(0.0, 0.0)) -> OccupancyGrid2D:
    return OccupancyGrid2D.from_array(arr, res, origin)


def make_memory(tmp_path, overrides=None, grid=None, embedder=None, labeler=None,
                checkpoint_dir=None) -> UnifiedMemory:
    cfg = load_memory_config(overrides=overrides)
    if grid is None:
        grid = make_grid(np.zeros((100, 100), dtype=np.int8), origin=(-5.0, -5.0))
    store = ImageStore(tmp_path / "images", cfg.images.image_jpeg_quality,
                       cfg.images.crop_cache_size)
    return UnifiedMemory(cfg, grid, store, embedder=embedder, labeler=labeler,
                         checkpoint_dir=checkpoint_dir)


# no periodic maintenance unless a test asks for it
QUIET = {"maintenance": {"dedup_every_s": 1e9, "prune_every_s": 1e9},
         "rooms": {"enabled": False}, "checkpoints": {"checkpoint_every_s": 1e9}}


# ---------------------------------------------------------------- fixture memory (two rooms)
TWO_ROOMS = floor_plan((60, 100), [(2, 2, 42, 42), (43, 2, 83, 42)], doors=[(42, 15, 43, 24)])

# label, score, centroid; ids follow this order. Room A spans x 0.2-4.2 m, room B x 4.3-8.3 m.
FIXTURE_OBJECTS = [
    ("bed", 0.8, (1.0, 1.0, 0.5)),         # 0  A
    ("wardrobe", 0.6, (3.0, 1.0, 1.0)),    # 1  A
    ("lamp", 0.7, (1.5, 1.0, 0.8)),        # 2  A, 0.5 m from the bed
    ("sofa", 0.9, (5.0, 2.0, 0.4)),        # 3  B
    ("cup", 0.2, (6.0, 3.0, 0.8)),         # 4  B, tentative
    ("chair", 0.6, (7.0, 1.0, 0.5)),       # 5  B
    ("chair", 0.7, (7.2, 1.0, 0.5)),       # 6  B, deduped into 5
]


def build_fixture_memory(tmp_path, overrides=None, **kw) -> UnifiedMemory:
    mem = make_memory(tmp_path, overrides, grid=make_grid(TWO_ROOMS), **kw)
    cam = (2.0, 2.0)
    obs = [make_obs(lbl, s, c, 0.0, bbox=(20 * i, 10, 20 * i + 20, 40), base_xy=cam)
           for i, (lbl, s, c) in enumerate(FIXTURE_OBJECTS)]
    mem.update(make_result(0.0, obs, base_xy=cam))
    mem.tick(10.0)                                       # dedup + checkpoint
    return mem
